package sk.ainet.models.bitnet

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.I2sGgufLayout
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.nn.Module
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeights
import sk.ainet.models.llama.LlamaModelMetadata
import sk.ainet.models.llama.decoderMetadataFromGguf

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
 * - [planesLmHead] requantizes `output.weight` to `BITNET_PLANES` at load (the fused-lm_head
 *   format). For a source whose weights are exactly ternary this is lossless: the per-row absmax
 *   normalizes codes to {−1, 0, +1} and plane 0 captures them with zero residual. A 2B4T-style
 *   file with **tied** embeddings has no `output.weight`; the tied fallback then serves the
 *   lm_head from `token_embd` as-is, and this knob has no effect.
 * - Embeddings must be a non-ternary type (F32/F16/BF16) — true of every BitNet checkpoint; a
 *   packed ternary `token_embd` would gather raw codes.
 */
public object BitNetPackedGgufLoader {

    /** A loaded model together with the GGUF metadata it was built from. */
    public data class Loaded(
        public val model: Module<FP32, Float>,
        public val metadata: LlamaModelMetadata,
    )

    public suspend fun load(
        ctx: ExecutionContext,
        sourceProvider: () -> RandomAccessSource,
        i2sLayout: I2sGgufLayout = I2sGgufLayout.GROUP_128,
        planesLmHead: Boolean = true,
        debug: Boolean = false,
    ): Module<FP32, Float> =
        loadWithMetadata(ctx, sourceProvider, i2sLayout, planesLmHead, debug).model

    /** [load], returning the parsed [LlamaModelMetadata] alongside the model (bos/eos, dims). */
    public suspend fun loadWithMetadata(
        ctx: ExecutionContext,
        sourceProvider: () -> RandomAccessSource,
        i2sLayout: I2sGgufLayout = I2sGgufLayout.GROUP_128,
        planesLmHead: Boolean = true,
        debug: Boolean = false,
    ): Loaded {
        // Pass 1 — metadata from the GGUF KV directory, via the shared
        // decoder-family parser (#346).
        val metadata = StreamingGGUFReader.open(sourceProvider()).use { reader ->
            decoderMetadataFromGguf(reader.fields, reader.tensors)
        }
        require(metadata.architecture in BITNET_ARCHITECTURES) {
            "BitNetPackedGgufLoader: architecture '${metadata.architecture}' is not a BitNet " +
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
            weightFormFor = { name -> if (planesLmHead && name == "output.weight") planes else null },
            i2sLayout = i2sLayout,
        ).load<FP32, Float>(ctx, FP32::class) { name, tensor -> tensors[name] = tensor }

        val model = BitNetNetworkLoader.fromWeights(DecoderGgufWeights(metadata, tensors), debug)
        return Loaded(model, metadata)
    }

}
