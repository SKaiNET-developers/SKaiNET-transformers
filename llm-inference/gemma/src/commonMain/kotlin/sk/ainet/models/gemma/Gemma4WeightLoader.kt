package sk.ainet.models.gemma

import kotlinx.io.Source
import kotlinx.io.buffered
import sk.ainet.context.ExecutionContext
import sk.ainet.apps.llm.DTypePolicyValidation
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.ReaderField
import sk.ainet.io.gguf.ReaderTensor
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.io.gguf.StreamingTensorInfo
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.nn.quant.PackedRowDequantTensorData
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

/**
 * [WeightForm] requesting every tensor fully dequantized to dense FP32 with
 * logical `[out, in]` shapes — the form the trace/export harnesses
 * (FunctionGemma StableHLO export) and FP32-parity tests consume.
 */
@ExperimentalMemoryApi
public val GEMMA_DEQUANTIZE_ALL: WeightForm = WeightForm(
    encoding = EncodingRequest.DequantizeTo(FP32),
    shape = WeightShapeOrientation.OUT_IN
)

/**
 * Adapter that loads Gemma 4 weights from GGUF files.
 *
 * Recognizes architecture prefixes: "gemma3", "gemma4", "gemma", "llama".
 * Extracts Gemma 4 specific metadata: global_head_dim, proportional RoPE, layer types.
 *
 * The random-access path delegates tensor materialization to the engine's
 * [StreamingGgufParametersLoader]: quantized projection matrices keep their
 * stored block encoding as packed
 * [sk.ainet.lang.tensor.storage.PackedBlockStorage] tensors with logical
 * `[out, in]` shapes, ready for the packed matmul kernels; pass
 * [GEMMA_DEQUANTIZE_ALL] as [weightForm] for a fully dense FP32 load (the
 * export/tracing path). Two Gemma-specific overrides always apply:
 *
 * - `token_embd.weight` is dequantized to a dense FP32 `[vocab, dim]`
 *   tensor — `Embedding.gather()` needs real element access.
 * - `per_layer_token_embd.weight` (the PLE table — Q6_K, ~1.9 GB packed and
 *   ~9 GB as FP32 on Gemma 4 E2B) stays packed and is wrapped as a
 *   [GemmaPerLayerTokenEmbedTensorData] row-dequant source, so
 *   [PerLayerEmbedding.compute] dequantizes only the rows it gathers.
 *
 * The sequential [Source] path (non-seekable inputs) dequantizes everything
 * to dense floats.
 */
@OptIn(ExperimentalMemoryApi::class)
public class Gemma4WeightLoader private constructor(
    private val sourceProvider: (() -> Source)?,
    private val randomAccessProvider: (() -> RandomAccessSource)?,
    private val loadTensorData: Boolean = true,
    private val weightForm: WeightForm? = null,
    private val dtypePolicy: DTypePolicy = DTypePolicy.Any,
) {
    /**
     * Keep `F16` source tensors in their on-disk 2-bytes-per-element layout instead of widening
     * them to FP32. Resolved from [dtypePolicy] exactly as llama's `DecoderGgufWeightLoader` and
     * the engine's `StreamingGgufParametersLoader.keepsNative` do, so a policy means the same thing
     * whichever family loads the checkpoint.
     */
    private val keepF16Native: Boolean = DTypePolicyValidation.keepsNative(dtypePolicy, FP16)

    /** As [keepF16Native], for `BF16` sources. Resolved independently. */
    private val keepBf16Native: Boolean = DTypePolicyValidation.keepsNative(dtypePolicy, BF16)

    public constructor(
        sourceProvider: () -> Source,
        loadTensorData: Boolean = true,
    ) : this(
        sourceProvider = sourceProvider,
        randomAccessProvider = null,
        loadTensorData = loadTensorData,
    )

    /**
     * @param dtypePolicy narrow-float handling. Default [DTypePolicy.Any] widens F16/BF16 sources
     *   to FP32; a policy naming BF16 or FP16 keeps that format packed at 2 bytes per element,
     *   which halves the bytes a decode step reads for a checkpoint stored that way — the gemma3
     *   family ships BF16, and Gemma 4 E2B stores `per_layer_model_proj` that way.
     */
    public constructor(
        randomAccessProvider: () -> RandomAccessSource,
        weightForm: WeightForm? = null,
        loadTensorData: Boolean = true,
        dtypePolicy: DTypePolicy = DTypePolicy.Any,
    ) : this(
        sourceProvider = null,
        randomAccessProvider = randomAccessProvider,
        loadTensorData = loadTensorData,
        weightForm = weightForm,
        dtypePolicy = dtypePolicy,
    )

    public suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma4ModelMetadata {
        return loadFromGguf(ctx, dtype, onTensorLoaded)
    }

    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma4ModelMetadata = load(ctx, T::class, onTensorLoaded)

    public suspend fun <T : DType, V> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Gemma4Weights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val meta = loadFromGguf(ctx, dtype) { name, tensor -> byName[name] = tensor }
        return Gemma4Weights(meta, byName)
    }

    public suspend inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext
    ): Gemma4Weights<T, V> = loadToMap(ctx, T::class)

    // Streaming API

    public suspend fun <T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma4ModelMetadata {
        return loadFromStreamingGguf(ctx, dtype, onTensorLoaded)
    }

    public suspend inline fun <reified T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): Gemma4ModelMetadata = loadStreaming(ctx, T::class, onTensorLoaded)

    public suspend fun <T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): Gemma4Weights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val meta = loadFromStreamingGguf(ctx, dtype) { name, tensor -> byName[name] = tensor }
        return Gemma4Weights(meta, byName)
    }

    public suspend inline fun <reified T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext
    ): Gemma4Weights<T, V> = loadToMapStreaming(ctx, T::class)

    // ============== Sequential loading (dequantize everything) ==============

    private fun <T : DType, V> loadFromGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
    ): Gemma4ModelMetadata {
        require(dtype == FP32::class || dtype == FP16::class) {
            "Gemma 4 GGUF loader supports FP32 and FP16 tensors (got ${dtype.simpleName})"
        }
        requireNotNull(sourceProvider) {
            "Sequential loading requires sourceProvider constructor."
        }

        val reader = sourceProvider.invoke().buffered().use { src ->
            GGUFReader(src, loadTensorData = loadTensorData)
        }

        val metadata = metadataFromGguf(reader.fields, reader.tensors)
        validateMetadata(metadata)

        val required = requiredTensorNames(metadata)
        val tensorByName = reader.tensors.associateBy { it.name }

        // Retained so the tied-output fallback below can ALIAS the already-
        // loaded embedding instead of re-reading the bytes into a second
        // tensor. Two independent tensors mean two BufferHandles, and the
        // compiled export then externalizes the 262153x640 tied weight twice
        // — 77% of the weight archive (#260).
        var embedTensor: Tensor<T, V>? = null

        required.forEach { name ->
            val rt = tensorByName[name]
                ?: error("Missing required tensor in GGUF payload: $name")
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt)
            if (name == Gemma4TensorNames.TOKEN_EMBEDDINGS) embedTensor = tensor
            onTensorLoaded(name, tensor)
        }

        // Output weight with weight tying fallback
        val outputRt = tensorByName[Gemma4TensorNames.OUTPUT_WEIGHT]
        if (outputRt != null) {
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, outputRt)
            onTensorLoaded(Gemma4TensorNames.OUTPUT_WEIGHT, tensor)
        } else {
            val embedRt = tensorByName[Gemma4TensorNames.TOKEN_EMBEDDINGS]
                ?: error("Missing both output.weight and token_embd.weight")
            // Tied output: alias the SAME tensor (same BufferHandle) as
            // token_embd — no second read. The trace then sees one weight
            // and the export emits one global / one archive blob (#260).
            val tensor: Tensor<T, V> = embedTensor
                ?: readerTensorToTensor(ctx, dtype, reader, embedRt)
            onTensorLoaded(Gemma4TensorNames.OUTPUT_WEIGHT, tensor)
        }

        // Optional tensors
        optionalTensorNames(metadata).forEach { name ->
            val rt = tensorByName[name] ?: return@forEach
            onTensorLoaded(name, readerTensorToTensor(ctx, dtype, reader, rt))
        }

        return metadata
    }

    // ============== Streaming loading (engine loader) ==============

    private suspend fun <T : DType, V> loadFromStreamingGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
    ): Gemma4ModelMetadata {
        require(dtype == FP32::class) {
            "Gemma 4 engine-backed GGUF loading delivers FP32-typed tensors (got ${dtype.simpleName})"
        }
        requireNotNull(randomAccessProvider) {
            "Streaming loading requires randomAccessProvider constructor."
        }

        // Header pass: metadata, required-tensor check, tied-embedding
        // detection, and the wanted-name set (the engine loader delivers
        // every tensor in the file; we keep only the ones the gemma network
        // consumes).
        val metadata: Gemma4ModelMetadata
        val wanted: Set<String>
        val tiedEmbeddings: Boolean
        val ggufTypeByName: Map<String, GGMLQuantizationType>
        StreamingGGUFReader.open(randomAccessProvider.invoke()).use { reader ->
            metadata = metadataFromStreamingGguf(reader.fields, reader.tensors)
            validateMetadata(metadata)
            if (!loadTensorData) return metadata

            val required = requiredTensorNames(metadata)
            val tensorByName = reader.tensors.associateBy { it.name }
            ggufTypeByName = reader.tensors.associate { it.name to it.tensorType }

            val missing = required.filter { it !in tensorByName }
            require(missing.isEmpty()) {
                "Missing required tensor(s) in GGUF payload: ${missing.joinToString()}"
            }

            tiedEmbeddings = tensorByName[Gemma4TensorNames.OUTPUT_WEIGHT] == null
            wanted = buildSet {
                addAll(required)
                add(Gemma4TensorNames.OUTPUT_WEIGHT)
                optionalTensorNames(metadata).forEach { if (it in tensorByName) add(it) }
            }
        }

        // Payload pass through the engine loader. Default: keep quantized
        // tensors packed with logical [out, in] shapes; the caller-supplied
        // [weightForm] (e.g. GEMMA_DEQUANTIZE_ALL) overrides. The token
        // embedding and PLE table overrides always win — see class kdoc.
        val defaultForm = weightForm ?: WeightForm(
            shape = WeightShapeOrientation.OUT_IN,
            // MAPPED residency default (#342 arc, P5): servable encodings are
            // served zero-copy from file-backed pages; the engine heap-stages
            // the rest. The PLE override below stays HEAP — its row-dequant
            // wrapper needs a heap byte view of the packed table.
            residency = WeightResidency.MAPPED,
        )
        val engineLoader = StreamingGgufParametersLoader(
            sourceProvider = randomAccessProvider,
            keepF16Native = keepF16Native && dtype == FP32::class,
            keepBf16Native = keepBf16Native && dtype == FP32::class,
            weightForm = defaultForm,
            weightFormFor = { name ->
                when (name) {
                    // Keep the PLE table packed regardless of the caller's
                    // form: dense FP32 is ~9 GB on E2B (and overflows the
                    // JVM array cap on E4B). Wrapped as a row-dequant
                    // source below.
                    Gemma4TensorNames.PER_LAYER_TOKEN_EMBD ->
                        WeightForm(shape = WeightShapeOrientation.OUT_IN)
                    else -> null
                }
            }
        )
        var tokenEmbedding: Tensor<T, V>? = null
        engineLoader.load(ctx, dtype) { name: String, tensor: Tensor<T, V> ->
            if (name !in wanted) return@load
            val delivered = when (name) {
                Gemma4TensorNames.PER_LAYER_TOKEN_EMBD ->
                    wrapGemmaPleIfPacked(ctx, dtype, tensor, ggufTypeByName[name])
                // token_embd rides the default keep-packed form like every other
                // tensor, then is rewrapped as a row-dequant source (llama's
                // embeddingReady pattern): Embedding gathers rows via dequantRow,
                // and the *tied* output head still sees PackedBlockStorage for
                // the packed matmul chain instead of a ~2.4 GB dense FP32 table
                // running off the fast-kernel path every decode step.
                Gemma4TensorNames.TOKEN_EMBEDDINGS ->
                    embeddingReady(ctx, dtype, tensor)
                else -> tensor
            }
            if (name == Gemma4TensorNames.TOKEN_EMBEDDINGS) tokenEmbedding = delivered
            onTensorLoaded(name, delivered)
        }
        if (tiedEmbeddings) {
            // Tied output: alias the SAME tensor (same BufferHandle) as
            // token_embd — one weight in the trace, one export blob (#260).
            onTensorLoaded(
                Gemma4TensorNames.OUTPUT_WEIGHT,
                tokenEmbedding
                    ?: error("Tied embeddings detected but ${Gemma4TensorNames.TOKEN_EMBEDDINGS} was not delivered")
            )
        }

        return metadata
    }

    /**
     * The token-embedding tensor in a form `Embedding.gather` can read
     * (mirrors llama's `DecoderGgufWeightLoader.embeddingReady`): packed
     * deliveries are rewrapped as a [PackedRowDequantTensorData] row-dequant
     * source; a packed table whose rows do not lie on block boundaries is
     * dequantized outright rather than left where element access would
     * return raw quantization codes. Dense deliveries (e.g. under a caller's
     * [GEMMA_DEQUANTIZE_ALL] export form) pass through untouched.
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> embeddingReady(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        tensor: Tensor<T, V>,
    ): Tensor<T, V> {
        val data = tensor.data
        if (data !is PackedBlockStorage) return tensor
        val wrapped = PackedRowDequantTensorData.wrapIfRowDequantable(data as TensorData<FP32, Float>)
        return if (wrapped !== data) {
            ctx.fromData(wrapped as TensorData<T, V>, dtype)
        } else {
            ctx.fromFloatArray<T, V>(tensor.shape, dtype, data.toFloatArray())
        }
    }

    // ============== Metadata extraction ==============

    private fun metadataFromGguf(
        fields: Map<String, ReaderField>,
        tensors: List<ReaderTensor>
    ): Gemma4ModelMetadata {
        val arch = fields["general.architecture"]?.stringValue() ?: "unknown"
        val prefix = findArchPrefix(fields, listOf("gemma3", "gemma4", "gemma", "llama"))

        val embeddingLength = fields["$prefix.embedding_length"]?.scalarInt()
            ?: inferEmbeddingFromTensor(tensors)
        val contextLength = fields["$prefix.context_length"]?.scalarInt() ?: 131072
        val blockCount = fields["$prefix.block_count"]?.scalarInt() ?: 34
        val headCount = fields["$prefix.attention.head_count"]?.scalarInt() ?: 8
        val kvHeadCount = fields["$prefix.attention.head_count_kv"]?.scalarInt() ?: 4
        // GGUF uses attention.key_length_swa for sliding head dim, attention.key_length for global
        val headDim = fields["$prefix.attention.key_length_swa"]?.scalarInt()
            ?: fields["$prefix.attention.head_dim"]?.scalarInt()
            ?: 256
        val globalHeadDim = fields["$prefix.attention.key_length"]?.scalarInt()
            ?: fields["$prefix.attention.global_head_dim"]?.scalarInt()
            ?: headDim
        val vocabSize = fields["$prefix.vocab_size"]?.scalarInt()
            ?: inferVocabFromTensor(tensors)
        val ffnField = fields["$prefix.feed_forward_length"]
        val perLayerIntermediateSize: List<Int> = ffnField
            ?.let { runCatching { it.intListValue() }.getOrNull() }
            ?: emptyList()
        val intermediateSize = ffnField?.scalarInt()
            ?: perLayerIntermediateSize.firstOrNull()
            ?: (embeddingLength * 4)
        val slidingWindow = fields["$prefix.attention.sliding_window"]?.scalarInt()
            ?: Gemma4ModelMetadata.DEFAULT_SLIDING_WINDOW
        // See streaming-path note: absent KV-sharing key ⇒ no sharing (0).
        val kvSharedLayers = fields["$prefix.attention.shared_kv_layers"]?.scalarInt()
            ?: fields["$prefix.kv_shared_layers"]?.scalarInt()
            ?: 0
        val perLayerEmbeddingLength = fields["$prefix.embedding_length_per_layer_input"]?.scalarInt() ?: 0

        val layerTypes = extractLayerTypes(fields, prefix, blockCount)

        val ropeBase = fields["$prefix.rope.freq_base"]?.scalarFloat() ?: 1000000f
        // GGUF uses rope.freq_base_swa for sliding window RoPE base
        val ropeBaseLocal = fields["$prefix.rope.freq_base_swa"]?.scalarFloat()
            ?: fields["$prefix.rope.freq_base_local"]?.scalarFloat()
            ?: 10000f
        val ropeFactor = fields["$prefix.rope.factor"]?.scalarFloat() ?: 1.0f
        // GGUF does not carry partial_rotary_factor directly for Gemma 4, but it DOES carry
        // rope.dimension_count — the actual number of dims rotated for the global/full scheme
        // (this checkpoint: 512, equal to key_length=512, i.e. full rotation, factor=1.0 — NOT
        // the previously-hardcoded 0.25 "HF config" guess, which silently rotated only 128 of
        // 512 dims on every global attention layer and corrupted decode, worsening with
        // sequence position. Derive the factor from the real field when present; only fall back
        // to the guessed default when the checkpoint omits rope.dimension_count entirely.
        val partialRotaryDefault = if (arch.startsWith("gemma")) 0.25f else 1.0f
        val ropeDimensionCount = fields["$prefix.rope.dimension_count"]?.scalarInt()
        val partialRotaryFactor = fields["$prefix.rope.partial_rotary_factor"]?.scalarFloat()
            ?: ropeDimensionCount?.takeIf { globalHeadDim > 0 }?.let { it.toFloat() / globalHeadDim }
            ?: partialRotaryDefault
        val finalLogitSoftcapping = fields["$prefix.final_logit_softcapping"]?.scalarFloat() ?: 0f
        val rmsNormEps = fields["$prefix.attention.layer_norm_rms_epsilon"]?.scalarFloat() ?: 1e-6f
        // Token ids: without these, downstream defaults prefill BOS=1 — which is
        // Gemma's <eos> — and generation collapses into turn-token spam (#325 arc).
        val bosTokenId = fields["tokenizer.ggml.bos_token_id"]?.scalarInt() ?: 2
        val eosTokenId = fields["tokenizer.ggml.eos_token_id"]?.scalarInt() ?: 1
        val padTokenId = fields["tokenizer.ggml.padding_token_id"]?.scalarInt() ?: 0

        return Gemma4ModelMetadata(
            architecture = arch,
            embeddingLength = embeddingLength,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            intermediateSize = intermediateSize,
            headDim = headDim,
            globalHeadDim = globalHeadDim,
            vocabSize = vocabSize,
            slidingWindow = slidingWindow,
            kvSharedLayers = kvSharedLayers,
            layerTypes = layerTypes,
            ropeParametersFull = Gemma4RopeConfig(
                base = ropeBase,
                ropeType = "proportional",
                factor = ropeFactor,
                partialRotaryFactor = partialRotaryFactor
            ),
            ropeParametersSliding = Gemma4RopeConfig(
                base = ropeBaseLocal,
                ropeType = "default"
            ),
            maxPositionEmbeddings = contextLength,
            perLayerEmbeddingLength = perLayerEmbeddingLength,
            perLayerIntermediateSize = perLayerIntermediateSize,
            bosTokenId = bosTokenId,
            eosTokenId = eosTokenId,
            padTokenId = padTokenId,
            rmsNormEps = rmsNormEps,
            finalLogitSoftcapping = finalLogitSoftcapping
        )
    }

    private fun metadataFromStreamingGguf(
        fields: Map<String, Any?>,
        tensors: List<StreamingTensorInfo>
    ): Gemma4ModelMetadata {
        val arch = (fields["general.architecture"] as? String) ?: "unknown"
        val prefix = findStreamingArchPrefix(fields, listOf("gemma3", "gemma4", "gemma", "llama"))

        val embeddingLength = fields["$prefix.embedding_length"]?.toIntValue()
            ?: inferEmbeddingFromStreamingTensor(tensors)
        val contextLength = fields["$prefix.context_length"]?.toIntValue() ?: 131072
        val blockCount = fields["$prefix.block_count"]?.toIntValue() ?: 34
        val headCount = fields["$prefix.attention.head_count"]?.toIntValue() ?: 8
        val kvHeadCount = fields["$prefix.attention.head_count_kv"]?.toIntValue() ?: 4
        val headDim = fields["$prefix.attention.key_length_swa"]?.toIntValue()
            ?: fields["$prefix.attention.head_dim"]?.toIntValue()
            ?: 256
        val globalHeadDim = fields["$prefix.attention.key_length"]?.toIntValue()
            ?: fields["$prefix.attention.global_head_dim"]?.toIntValue()
            ?: headDim
        val vocabSize = fields["$prefix.vocab_size"]?.toIntValue()
            ?: inferVocabFromStreamingTensor(tensors)
        val ffnField = fields["$prefix.feed_forward_length"]
        val perLayerIntermediateSize: List<Int> = (ffnField as? List<*>)
            ?.mapNotNull { (it as? Number)?.toInt() }
            ?: emptyList()
        val intermediateSize = ffnField?.toIntValue()
            ?: perLayerIntermediateSize.firstOrNull()
            ?: (embeddingLength * 4)
        val slidingWindow = fields["$prefix.attention.sliding_window"]?.toIntValue()
            ?: Gemma4ModelMetadata.DEFAULT_SLIDING_WINDOW
        // KV-sharing only applies when the gguf explicitly declares it (gemma3n
        // / gemma-4). A plain gemma3 checkpoint (e.g. FunctionGemma-270M) omits
        // the key and uses no KV-sharing — default 0, NOT DEFAULT_KV_SHARED_LAYERS
        // (20), which on an 18-layer model gives firstSharedLayer = 18-20 = -2
        // and crashes GemmaNetworkDef.
        val kvSharedLayers = fields["$prefix.attention.shared_kv_layers"]?.toIntValue()
            ?: fields["$prefix.kv_shared_layers"]?.toIntValue()
            ?: 0
        val perLayerEmbeddingLength = fields["$prefix.embedding_length_per_layer_input"]?.toIntValue() ?: 0

        val layerTypes = extractStreamingLayerTypes(fields, prefix, blockCount)

        val ropeBase = fields["$prefix.rope.freq_base"]?.toFloatValue() ?: 1000000f
        val ropeBaseLocal = fields["$prefix.rope.freq_base_swa"]?.toFloatValue()
            ?: fields["$prefix.rope.freq_base_local"]?.toFloatValue()
            ?: 10000f
        val ropeFactor = fields["$prefix.rope.factor"]?.toFloatValue() ?: 1.0f
        // See the non-streaming metadata parse above — derive from the real
        // rope.dimension_count field rather than guessing 0.25.
        val partialRotaryDefault = if (arch.startsWith("gemma")) 0.25f else 1.0f
        val ropeDimensionCount = fields["$prefix.rope.dimension_count"]?.toIntValue()
        val partialRotaryFactor = fields["$prefix.rope.partial_rotary_factor"]?.toFloatValue()
            ?: ropeDimensionCount?.takeIf { globalHeadDim > 0 }?.let { it.toFloat() / globalHeadDim }
            ?: partialRotaryDefault
        val finalLogitSoftcapping = fields["$prefix.final_logit_softcapping"]?.toFloatValue() ?: 0f
        val rmsNormEps = fields["$prefix.attention.layer_norm_rms_epsilon"]?.toFloatValue() ?: 1e-6f
        // Token ids: without these, downstream defaults prefill BOS=1 — which is
        // Gemma's <eos> — and generation collapses into turn-token spam (#325 arc).
        val bosTokenId = fields["tokenizer.ggml.bos_token_id"]?.toIntValue() ?: 2
        val eosTokenId = fields["tokenizer.ggml.eos_token_id"]?.toIntValue() ?: 1
        val padTokenId = fields["tokenizer.ggml.padding_token_id"]?.toIntValue() ?: 0

        return Gemma4ModelMetadata(
            architecture = arch,
            embeddingLength = embeddingLength,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            intermediateSize = intermediateSize,
            headDim = headDim,
            globalHeadDim = globalHeadDim,
            vocabSize = vocabSize,
            slidingWindow = slidingWindow,
            kvSharedLayers = kvSharedLayers,
            layerTypes = layerTypes,
            ropeParametersFull = Gemma4RopeConfig(
                base = ropeBase,
                ropeType = "proportional",
                factor = ropeFactor,
                partialRotaryFactor = partialRotaryFactor
            ),
            ropeParametersSliding = Gemma4RopeConfig(
                base = ropeBaseLocal,
                ropeType = "default"
            ),
            perLayerEmbeddingLength = perLayerEmbeddingLength,
            perLayerIntermediateSize = perLayerIntermediateSize,
            bosTokenId = bosTokenId,
            eosTokenId = eosTokenId,
            padTokenId = padTokenId,
            rmsNormEps = rmsNormEps,
            finalLogitSoftcapping = finalLogitSoftcapping,
            maxPositionEmbeddings = contextLength
        )
    }

    // ============== Helpers ==============

    private fun findArchPrefix(fields: Map<String, ReaderField>, prefixes: List<String>): String {
        for (prefix in prefixes) {
            if (fields["$prefix.embedding_length"] != null || fields["$prefix.block_count"] != null) {
                return prefix
            }
        }
        return prefixes.first()
    }

    private fun findStreamingArchPrefix(fields: Map<String, Any?>, prefixes: List<String>): String {
        for (prefix in prefixes) {
            if (fields["$prefix.embedding_length"] != null || fields["$prefix.block_count"] != null) {
                return prefix
            }
        }
        return prefixes.first()
    }

    private fun extractLayerTypes(
        fields: Map<String, ReaderField>,
        prefix: String,
        blockCount: Int
    ): List<String> {
        fields["$prefix.attention.sliding_window_pattern"]?.let { field ->
            runCatching { field.boolListValue() }.getOrNull()?.let { bools ->
                if (bools.size == blockCount) {
                    return bools.map { if (it) "sliding_attention" else "full_attention" }
                }
            }
        }
        val patternField = fields["$prefix.attention.layer_types"]
            ?: fields["$prefix.attention.layer_pattern"]
        if (patternField != null) {
            return try {
                patternField.stringListValue()
            } catch (_: Exception) {
                buildDefaultLayerTypes(blockCount)
            }
        }
        return buildDefaultLayerTypes(blockCount)
    }

    private fun extractStreamingLayerTypes(
        fields: Map<String, Any?>,
        prefix: String,
        blockCount: Int
    ): List<String> {
        // sliding_window_pattern is a boolean list: True = sliding_attention, False = full_attention.
        val slidingPattern = fields["$prefix.attention.sliding_window_pattern"]
        if (slidingPattern is List<*>) {
            val bools = slidingPattern.mapNotNull { it as? Boolean }
            if (bools.size == blockCount) {
                return bools.map { if (it) "sliding_attention" else "full_attention" }
            }
        }
        val value = fields["$prefix.attention.layer_types"]
            ?: fields["$prefix.attention.layer_pattern"]
        if (value is List<*>) {
            return value.mapNotNull { it as? String }
        }
        return buildDefaultLayerTypes(blockCount)
    }

    private fun buildDefaultLayerTypes(blockCount: Int): List<String> {
        return List(blockCount) { idx ->
            if (idx == blockCount - 1) "full_attention"
            else if ((idx + 1) % 6 == 0) "full_attention"
            else "sliding_attention"
        }
    }

    private fun validateMetadata(metadata: Gemma4ModelMetadata) {
        val validArchs = setOf("gemma4", "gemma", "llama", "unknown")
        require(metadata.architecture in validArchs || metadata.architecture.startsWith("gemma")) {
            "Unsupported architecture: ${metadata.architecture}. Expected gemma4 or compatible."
        }
        require(metadata.embeddingLength > 0) { "Invalid embedding length ${metadata.embeddingLength}" }
        require(metadata.blockCount > 0) { "Invalid block count ${metadata.blockCount}" }
        require(metadata.headCount > 0) { "Invalid head count ${metadata.headCount}" }
        require(metadata.contextLength > 0) { "Invalid context length ${metadata.contextLength}" }
        require(metadata.vocabSize > 0) { "Invalid vocab size ${metadata.vocabSize}" }
    }

    private fun requiredTensorNames(metadata: Gemma4ModelMetadata): List<String> {
        val names = mutableListOf<String>()
        names += Gemma4TensorNames.TOKEN_EMBEDDINGS
        names += Gemma4TensorNames.OUTPUT_NORM

        repeat(metadata.blockCount) { layer ->
            names += Gemma4TensorNames.inputLayernorm(layer)
            names += Gemma4TensorNames.attnQ(layer)
            names += Gemma4TensorNames.attnK(layer)
            names += Gemma4TensorNames.attnV(layer)
            names += Gemma4TensorNames.attnOut(layer)
            names += Gemma4TensorNames.postAttentionLayernorm(layer)
            names += Gemma4TensorNames.ffnGate(layer)
            names += Gemma4TensorNames.ffnDown(layer)
            names += Gemma4TensorNames.ffnUp(layer)
        }
        return names
    }

    private fun optionalTensorNames(metadata: Gemma4ModelMetadata): List<String> {
        val names = mutableListOf(
            Gemma4TensorNames.ROPE_FREQS_REAL,
            Gemma4TensorNames.ROPE_FREQS_IMAG,
            Gemma4TensorNames.PER_LAYER_TOKEN_EMBD,
            Gemma4TensorNames.PER_LAYER_MODEL_PROJ,
            Gemma4TensorNames.PER_LAYER_PROJ_NORM,
        )
        repeat(metadata.blockCount) { layer ->
            names += Gemma4TensorNames.perLayerInput(layer)
            names += Gemma4TensorNames.perLayerOutput(layer)
            names += Gemma4TensorNames.attnQNorm(layer)
            names += Gemma4TensorNames.attnKNorm(layer)
            names += Gemma4TensorNames.postAttentionNorm(layer)
            names += Gemma4TensorNames.postFfwNorm(layer)
            names += Gemma4TensorNames.layerOutputScale(layer)
            names += Gemma4TensorNames.pleInpGate(layer)
            names += Gemma4TensorNames.plePostNorm(layer)
            names += Gemma4TensorNames.pleProj(layer)
        }
        return names
    }

    // ============== Tensor conversion (sequential path) ==============

    /** Dequantize any GGUF tensor to a dense tensor of the requested dtype. */
    private fun <T : DType, V> readerTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: GGUFReader,
        rt: ReaderTensor,
    ): Tensor<T, V> {
        val shape = Shape(*rt.shape.map { it.toInt() }.toIntArray())
        val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
        val floats = when (rt.tensorType) {
            GGMLQuantizationType.F32 -> {
                @Suppress("UNCHECKED_CAST")
                (raw as List<Float>).toFloatArray()
            }
            GGMLQuantizationType.F16 -> DequantOps.dequantF16(raw)
            GGMLQuantizationType.BF16 -> DequantOps.dequantBF16(raw)
            else -> {
                val bytes = DequantOps.toByteArray(raw, rt.name)
                DequantOps.dequantFromBytes(bytes, rt.tensorType, rt.nElements)
            }
        }
        return createTensor(ctx, dtype, shape, floats)
    }

    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> createTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        originalShape: Shape,
        data: FloatArray
    ): Tensor<T, V> {
        return if (originalShape.rank == 2) {
            val rows = originalShape[0]
            val cols = originalShape[1]
            val transposed = DequantOps.transposeColumnMajorToRowMajor(data, rows, cols)
            val newShape = Shape(cols, rows)
            ctx.fromFloatArray<T, Float>(newShape, dtype, transposed) as Tensor<T, V>
        } else {
            ctx.fromFloatArray<T, Float>(originalShape, dtype, data) as Tensor<T, V>
        }
    }

    // ============== Helper methods ==============

    private fun inferEmbeddingFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer embedding length without token embeddings tensor")
        return token.shape.map { it.toInt() }.minOrNull()
            ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
    }

    private fun inferVocabFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer vocab size without token embeddings tensor")
        return token.shape.map { it.toInt() }.maxOrNull()
            ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
    }

    private fun inferEmbeddingFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val token = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
            ?: return 2304
        // GGUF stores shapes in column-major order; embedding is the smaller dimension
        return token.shape.map { it.toInt() }.minOrNull() ?: 2304
    }

    private fun inferVocabFromStreamingTensor(tensors: List<StreamingTensorInfo>): Int {
        val token = tensors.firstOrNull { it.name == Gemma4TensorNames.TOKEN_EMBEDDINGS }
            ?: return 262144
        // GGUF stores shapes in column-major order; vocab is the larger dimension
        return token.shape.map { it.toInt() }.maxOrNull() ?: 262144
    }

    private fun ReaderField.scalarInt(): Int {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        val value = (part as List<*>).firstOrNull()
            ?: error("Empty data part for field $name")
        return when (value) {
            is Int -> value
            is UInt -> value.toInt()
            is Long -> value.toInt()
            is ULong -> value.toInt()
            is Short -> value.toInt()
            is UShort -> value.toInt()
            is Byte -> value.toInt()
            is UByte -> value.toInt()
            else -> error("Unsupported scalar type ${value::class} for field $name")
        }
    }

    private fun ReaderField.scalarFloat(): Float {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        val value = (part as List<*>).firstOrNull()
            ?: error("Empty data part for field $name")
        return when (value) {
            is Float -> value
            is Double -> value.toFloat()
            is Number -> value.toFloat()
            else -> error("Unsupported scalar type ${value::class} for field $name")
        }
    }

    private fun ReaderField.stringValue(): String {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        @Suppress("UNCHECKED_CAST")
        val bytes = (part as List<Any>).mapNotNull {
            when (it) {
                is UByte -> it.toByte()
                is Byte -> it
                else -> null
            }
        }
        return bytes.toByteArray().decodeToString()
    }

    private fun ReaderField.stringListValue(): List<String> {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        @Suppress("UNCHECKED_CAST")
        return (part as List<*>).mapNotNull { it as? String }
    }

    private fun ReaderField.intListValue(): List<Int> {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        @Suppress("UNCHECKED_CAST")
        return (part as List<*>).mapNotNull { (it as? Number)?.toInt() }
    }

    private fun ReaderField.boolListValue(): List<Boolean> {
        val idx = data.firstOrNull() ?: 0
        val part = parts.getOrNull(idx) ?: error("Missing data part for field $name")
        return (part as List<*>).mapNotNull { it as? Boolean }
    }

    private fun Any?.toIntValue(): Int? = when (this) {
        is Int -> this
        is UInt -> this.toInt()
        is Long -> this.toInt()
        is ULong -> this.toInt()
        is Short -> this.toInt()
        is UShort -> this.toInt()
        is Byte -> this.toInt()
        is UByte -> this.toInt()
        else -> null
    }

    private fun Any?.toFloatValue(): Float? = when (this) {
        is Float -> this
        is Double -> this.toFloat()
        is Number -> this.toFloat()
        else -> null
    }
}
