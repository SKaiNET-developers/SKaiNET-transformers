package sk.ainet.models.bitnet

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.I2sGgufLayout
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata
import sk.ainet.lang.nn.dsl.decoder.decoderMetadataFromGguf

/**
 * The **packed** BitNet load path (transformers#337): a ternary **I2_S** GGUF loads through the
 * SKaiNET engine's [StreamingGgufParametersLoader], so the ternary projections arrive as packed
 * `BITNET_B1_58` tensors — 0.25 bytes per weight instead of FP32 widening — and the lm_head
 * (when the file carries `output.weight`) as the multi-plane `BITNET_PLANES` format.
 *
 * From there, nothing BitNet-specific happens: the weights bind into the stock [bitnetNetwork]
 * via [BitNetNetworkLoader.fromWeights], `linearProject`'s `matmul(x, transpose(W))` unwraps the
 * transpose marker, and `KernelDispatch` selects by the weight's storage format — the exact
 * FP32×`BITNET_B1_58` key when the ternary kernel pack is installed (the vendored NeoGPU NEON
 * kernel, SKaiNET#1136), the decoding reference otherwise. Correctness never depends on the pack.
 *
 * Notes:
 * - [i2sLayout] is the converter-flavor knob (BitNet.cpp x86 = `GROUP_128`, ARM = `GROUP_64`,
 *   NeoGPU = `SEQUENTIAL`) — see the engine's `I2sGgufLayout` docs; a wrong flavor fails fast on
 *   code 3 where possible.
 * - [planesLmHead] serves the lm_head in `BITNET_PLANES` (the fused-lm_head format), covering
 *   both head flavors (transformers#337, #357):
 *   - A file carrying `output.weight` has it requantized at load. For a source whose weights are
 *     exactly ternary this is lossless: the per-row absmax normalizes codes to {−1, 0, +1} and
 *     plane 0 captures them with zero residual.
 *   - A 2B4T-style file with **tied** embeddings has no `output.weight`; the head is then
 *     materialized from the loaded `token_embd` rows (transformers#357) — NeoGPU's lm_head
 *     design: the non-ternary embedding rows decompose into 8 trit planes against an FP16
 *     per-row absmax scale, a *bounded* requantization (per-weight error ≤ ~`0.5·3⁻⁷` of the
 *     row scale; `KernelDispatch`'s planes matmul is always the exact 8-plane product of that
 *     encoding). [BitNetTwoStageDecode] rescoring is exact w.r.t. this stored format.
 *   Memory trade-off, decided here deliberately: the embedding keeps its as-stored (MAPPED
 *   where servable) form for gathers, and the planes head is a **separate heap copy** at
 *   2 B/weight + FP16 row scales — half an FP16 head, but additional to the embedding, one-time
 *   at load. Pass `planesLmHead = false` for the exact dense tied head (the pre-#357 behavior).
 * - Embeddings must be a non-ternary type (F32/F16/BF16) — true of every BitNet checkpoint; a
 *   packed ternary `token_embd` would gather raw codes.
 */
public object BitNetWeightLoader {

    /** A loaded model together with the GGUF metadata it was built from. */
    public data class Loaded(
        public val model: Module<FP32, Float>,
        public val metadata: GgufDecoderMetadata,
    )

    public suspend fun load(
        ctx: ExecutionContext,
        sourceProvider: () -> RandomAccessSource,
        i2sLayout: I2sGgufLayout = I2sGgufLayout.GROUP_128,
        planesLmHead: Boolean = true,
        debug: Boolean = false,
    ): Module<FP32, Float> =
        loadWithMetadata(ctx, sourceProvider, i2sLayout, planesLmHead, debug).model

    /** [load], returning the parsed [GgufDecoderMetadata] alongside the model (bos/eos, dims). */
    public suspend fun loadWithMetadata(
        ctx: ExecutionContext,
        sourceProvider: () -> RandomAccessSource,
        i2sLayout: I2sGgufLayout = I2sGgufLayout.GROUP_128,
        planesLmHead: Boolean = true,
        debug: Boolean = false,
    ): Loaded {
        val weights = loadRuntimeWeights(ctx, sourceProvider, i2sLayout, planesLmHead)
        return Loaded(weights.toModule(debug), weights.metadata)
    }

    /**
     * The weights-only load (#346's `<F>WeightLoader` contract): both passes and the tied-head
     * materialization, stopping short of network binding — [BitNetRuntimeWeights.toModule] does
     * that, and [loadWithMetadata] is the two composed.
     */
    public suspend fun loadRuntimeWeights(
        ctx: ExecutionContext,
        sourceProvider: () -> RandomAccessSource,
        i2sLayout: I2sGgufLayout = I2sGgufLayout.GROUP_128,
        planesLmHead: Boolean = true,
    ): BitNetRuntimeWeights {
        // Pass 1 — metadata from the GGUF KV directory, via the shared
        // decoder-family parser (#346).
        val metadata = StreamingGGUFReader.open(sourceProvider()).use { reader ->
            decoderMetadataFromGguf(reader.fields, reader.tensors)
        }
        require(metadata.architecture in BITNET_ARCHITECTURES) {
            "BitNetWeightLoader: architecture '${metadata.architecture}' is not a BitNet " +
                "architecture ($BITNET_ARCHITECTURES)"
        }

        // Pass 2 — tensors through the engine loader: keep everything as stored (ternary stays
        // packed), shapes in [out, in] orientation, and the lm_head requantized to planes.
        // MAPPED residency is a *request* like everywhere else in the arc: the 0.51 engine
        // serves a trailer-scaled SEQUENTIAL I2_S file zero-copy from the mapping (#1203),
        // while a GROUP-flavor file (stock BitNet.cpp) must be repacked and heap-stages until
        // SKaiNET#1198's sidecar — the loader decides, this loader just states the form.
        val keepPacked = WeightForm(
            encoding = EncodingRequest.KeepAsStored,
            shape = WeightShapeOrientation.OUT_IN,
            residency = WeightResidency.MAPPED,
        )
        // No MAPPED here, deliberately: a requantization can never be served from the file's
        // pages, and `BitNetPlanesTensorData` is heap-only by design — which is exactly what
        // keeps `BitNetTwoStageDecode` type-safe (it branches on that concrete type).
        val planes = WeightForm(
            encoding = EncodingRequest.RequantizeTo(TensorEncoding.BITNET_PLANES),
            shape = WeightShapeOrientation.OUT_IN,
        )
        val tensors = LinkedHashMap<String, Tensor<FP32, Float>>()
        StreamingGgufParametersLoader(
            sourceProvider = sourceProvider,
            weightForm = keepPacked,
            weightFormFor = { name -> if (planesLmHead && name == BitNetTensorNames.OUTPUT) planes else null },
            i2sLayout = i2sLayout,
        ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> tensors[name] = tensor }

        // Tied-embeddings head (transformers#357): 2B4T ships no output.weight, and without this
        // the planes lane above never engages on the flagship checkpoint — the tied fallback in
        // BitNetNetworkLoader would serve the head as the dense embedding tensor. Materialize the
        // planes head from the loaded embedding instead; the embedding itself keeps its as-stored
        // form for gathers.
        if (planesLmHead && BitNetTensorNames.OUTPUT !in tensors) {
            val embedding = requireNotNull(tensors[BitNetTensorNames.TOKEN_EMBEDDING]) {
                "BitNetWeightLoader: file carries neither '${BitNetTensorNames.OUTPUT}' nor " +
                    "'${BitNetTensorNames.TOKEN_EMBEDDING}'"
            }
            tensors[BitNetTensorNames.OUTPUT] = planesHeadFromEmbedding(ctx, embedding)
        }

        return BitNetRuntimeWeights(metadata, tensors)
    }

    /**
     * Encode the `[vocab, dim]` [embedding] rows as a [TensorEncoding.BITNET_PLANES] lm_head
     * (transformers#357). Reads the rows through the tensor's element accessor — the embedding
     * arrived in whatever form the loader materialized (dense heap or mapped) — into the same
     * FP32 staging the engine's own `RequantizeTo(BITNET_PLANES)` lane uses, so the transient
     * cost matches the non-tied `output.weight` path: one dense-FP32 copy of the head, once,
     * at load.
     */
    @OptIn(ExperimentalMemoryApi::class)
    private fun planesHeadFromEmbedding(
        ctx: ExecutionContext,
        embedding: Tensor<FP32, Float>,
    ): Tensor<FP32, Float> {
        val shape = embedding.shape
        require(shape.rank == 2) {
            "BitNetWeightLoader: tied lm_head needs a 2-D token_embd, got rank ${shape.rank}"
        }
        val rows = shape.dimensions[0]
        val cols = shape.dimensions[1]
        val values = FloatArray(rows * cols)
        val data = embedding.data
        for (r in 0 until rows) {
            val base = r * cols
            for (c in 0 until cols) values[base + c] = data.get(r, c)
        }
        val planes = BitNetPlanesTensorData.fromFloats(shape, values)
        @Suppress("UNCHECKED_CAST")
        return ctx.fromData(planes as sk.ainet.lang.tensor.data.TensorData<FP32, Float>, FP32::class)
    }

}

/** Pre-#346-template name of [BitNetWeightLoader]; kept one release for source compatibility. */
@Deprecated(
    "Renamed to BitNetWeightLoader — the #346 family-template name (transformers#359).",
    ReplaceWith("BitNetWeightLoader"),
)
public typealias BitNetPackedGgufLoader = BitNetWeightLoader
