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
public class GemmaWeightLoader private constructor(
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
        dtypePolicy: DTypePolicy = DTypePolicy.Any,
    ) : this(
        sourceProvider = sourceProvider,
        randomAccessProvider = null,
        loadTensorData = loadTensorData,
        dtypePolicy = dtypePolicy,
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
    ): GemmaModelMetadata {
        return loadFromGguf(ctx, dtype, onTensorLoaded)
    }

    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): GemmaModelMetadata = load(ctx, T::class, onTensorLoaded)

    public suspend fun <T : DType, V> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): GemmaWeights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val meta = loadFromGguf(ctx, dtype) { name, tensor -> byName[name] = tensor }
        return GemmaWeights(meta, byName)
    }

    public suspend inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext
    ): GemmaWeights<T, V> = loadToMap(ctx, T::class)

    // Streaming API

    public suspend fun <T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): GemmaModelMetadata {
        return loadFromStreamingGguf(ctx, dtype, onTensorLoaded)
    }

    public suspend inline fun <reified T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): GemmaModelMetadata = loadStreaming(ctx, T::class, onTensorLoaded)

    public suspend fun <T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): GemmaWeights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val meta = loadFromStreamingGguf(ctx, dtype) { name, tensor -> byName[name] = tensor }
        return GemmaWeights(meta, byName)
    }

    public suspend inline fun <reified T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext
    ): GemmaWeights<T, V> = loadToMapStreaming(ctx, T::class)

    // ============== Sequential loading (dequantize everything) ==============

    private fun <T : DType, V> loadFromGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit,
    ): GemmaModelMetadata {
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
            if (name == GemmaTensorNames.TOKEN_EMBEDDINGS) embedTensor = tensor
            onTensorLoaded(name, tensor)
        }

        // Output weight with weight tying fallback
        val outputRt = tensorByName[GemmaTensorNames.OUTPUT_WEIGHT]
        if (outputRt != null) {
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, outputRt)
            onTensorLoaded(GemmaTensorNames.OUTPUT_WEIGHT, tensor)
        } else {
            val embedRt = tensorByName[GemmaTensorNames.TOKEN_EMBEDDINGS]
                ?: error("Missing both output.weight and token_embd.weight")
            // Tied output: alias the SAME tensor (same BufferHandle) as
            // token_embd — no second read. The trace then sees one weight
            // and the export emits one global / one archive blob (#260).
            val tensor: Tensor<T, V> = embedTensor
                ?: readerTensorToTensor(ctx, dtype, reader, embedRt)
            onTensorLoaded(GemmaTensorNames.OUTPUT_WEIGHT, tensor)
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
    ): GemmaModelMetadata {
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
        val metadata: GemmaModelMetadata
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

            tiedEmbeddings = tensorByName[GemmaTensorNames.OUTPUT_WEIGHT] == null
            wanted = buildSet {
                addAll(required)
                add(GemmaTensorNames.OUTPUT_WEIGHT)
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
                    GemmaTensorNames.PER_LAYER_TOKEN_EMBD ->
                        WeightForm(shape = WeightShapeOrientation.OUT_IN)
                    else -> null
                }
            }
        )
        var tokenEmbedding: Tensor<T, V>? = null
        engineLoader.load(ctx, dtype) { name: String, tensor: Tensor<T, V> ->
            if (name !in wanted) return@load
            val delivered = when (name) {
                GemmaTensorNames.PER_LAYER_TOKEN_EMBD ->
                    wrapGemmaPleIfPacked(ctx, dtype, tensor, ggufTypeByName[name])
                // token_embd rides the default keep-packed form like every other
                // tensor, then is rewrapped as a row-dequant source (llama's
                // embeddingReady pattern): Embedding gathers rows via dequantRow,
                // and the *tied* output head still sees PackedBlockStorage for
                // the packed matmul chain instead of a ~2.4 GB dense FP32 table
                // running off the fast-kernel path every decode step.
                GemmaTensorNames.TOKEN_EMBEDDINGS ->
                    embeddingReady(ctx, dtype, tensor)
                else -> tensor
            }
            if (name == GemmaTensorNames.TOKEN_EMBEDDINGS) tokenEmbedding = delivered
            onTensorLoaded(name, delivered)
        }
        if (tiedEmbeddings) {
            // Tied output: alias the SAME tensor (same BufferHandle) as
            // token_embd — one weight in the trace, one export blob (#260).
            onTensorLoaded(
                GemmaTensorNames.OUTPUT_WEIGHT,
                tokenEmbedding
                    ?: error("Tied embeddings detected but ${GemmaTensorNames.TOKEN_EMBEDDINGS} was not delivered")
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
    ): GemmaModelMetadata = parseGemmaMetadata(
        object : GgufFieldAccess {
            override fun has(key: String): Boolean = key in fields
            override fun string(key: String): String? = fields[key]?.stringValue()
            override fun int(key: String): Int? = fields[key]?.scalarInt()
            override fun float(key: String): Float? = fields[key]?.scalarFloat()
            override fun intList(key: String): List<Int>? =
                fields[key]?.let { runCatching { it.intListValue() }.getOrNull() }
            override fun boolList(key: String): List<Boolean>? =
                fields[key]?.let { runCatching { it.boolListValue() }.getOrNull() }
            override fun stringList(key: String): List<String>? =
                fields[key]?.let { runCatching { it.stringListValue() }.getOrNull() }
            override val tensorShapes: List<Pair<String, List<Int>>> =
                tensors.map { t -> t.name to t.shape.map { d -> d.toInt() } }
        },
    )

    private fun metadataFromStreamingGguf(
        fields: Map<String, Any?>,
        tensors: List<StreamingTensorInfo>
    ): GemmaModelMetadata = parseGemmaMetadata(
        object : GgufFieldAccess {
            override fun has(key: String): Boolean = key in fields
            override fun string(key: String): String? = fields[key] as? String
            override fun int(key: String): Int? = fields[key]?.toIntValue()
            override fun float(key: String): Float? = fields[key]?.toFloatValue()
            override fun intList(key: String): List<Int>? =
                (fields[key] as? List<*>)?.mapNotNull { (it as? Number)?.toInt() }
            override fun boolList(key: String): List<Boolean>? =
                (fields[key] as? List<*>)?.mapNotNull { it as? Boolean }
            override fun stringList(key: String): List<String>? =
                (fields[key] as? List<*>)?.mapNotNull { it as? String }
            override val tensorShapes: List<Pair<String, List<Int>>> =
                tensors.map { it.name to it.shape.map { d -> d.toInt() } }
        },
    )

    /**
     * The GGUF field surface the metadata parse needs, abstracted over the two reader APIs
     * (sequential `ReaderField` vs streaming `Any?`) — the pre-#375 twin ~100-line parsers had
     * to be kept in sync by hand and had already drifted in their inference fallbacks.
     */
    private interface GgufFieldAccess {
        fun has(key: String): Boolean
        fun string(key: String): String?
        fun int(key: String): Int?
        fun float(key: String): Float?
        fun intList(key: String): List<Int>?
        fun boolList(key: String): List<Boolean>?
        fun stringList(key: String): List<String>?
        val tensorShapes: List<Pair<String, List<Int>>>
    }

    /** The single Gemma GGUF metadata parse — both reader lanes feed it via [GgufFieldAccess]. */
    private fun parseGemmaMetadata(f: GgufFieldAccess): GemmaModelMetadata {
        val arch = f.string("general.architecture") ?: "unknown"
        val prefix = listOf("gemma3", "gemma4", "gemma", "llama")
            .firstOrNull { f.has("$it.embedding_length") || f.has("$it.block_count") } ?: arch

        val tokenShape = f.tensorShapes.firstOrNull { it.first == GemmaTensorNames.TOKEN_EMBEDDINGS }?.second
        val embeddingLength = f.int("$prefix.embedding_length") ?: tokenShape?.minOrNull() ?: 2304
        val contextLength = f.int("$prefix.context_length") ?: 131072
        val blockCount = f.int("$prefix.block_count") ?: 34
        val headCount = f.int("$prefix.attention.head_count") ?: 8
        val kvHeadCount = f.int("$prefix.attention.head_count_kv") ?: 4
        // GGUF uses attention.key_length_swa for the sliding head dim, attention.key_length for global.
        val headDim = f.int("$prefix.attention.key_length_swa") ?: f.int("$prefix.attention.head_dim") ?: 256
        val globalHeadDim = f.int("$prefix.attention.key_length") ?: f.int("$prefix.attention.global_head_dim") ?: headDim
        val vocabSize = f.int("$prefix.vocab_size") ?: tokenShape?.maxOrNull() ?: 262144
        val perLayerIntermediateSize = f.intList("$prefix.feed_forward_length") ?: emptyList()
        val intermediateSize = f.int("$prefix.feed_forward_length")
            ?: perLayerIntermediateSize.firstOrNull()
            ?: (embeddingLength * 4)
        val slidingWindow = f.int("$prefix.attention.sliding_window") ?: GemmaModelMetadata.DEFAULT_SLIDING_WINDOW
        // KV-sharing only applies when the gguf explicitly declares it (gemma3n / gemma-4). A plain
        // gemma3 checkpoint (e.g. FunctionGemma-270M) omits the key and uses no KV-sharing —
        // default 0, NOT DEFAULT_KV_SHARED_LAYERS (20), which on an 18-layer model gives
        // firstSharedLayer = 18-20 = -2 and crashes GemmaNetworkDef.
        val kvSharedLayers = f.int("$prefix.attention.shared_kv_layers") ?: f.int("$prefix.kv_shared_layers") ?: 0
        val perLayerEmbeddingLength = f.int("$prefix.embedding_length_per_layer_input") ?: 0

        // sliding_window_pattern is a boolean list: true = sliding_attention, false = full_attention.
        val layerTypes = f.boolList("$prefix.attention.sliding_window_pattern")
            ?.takeIf { it.size == blockCount }
            ?.map { if (it) "sliding_attention" else "full_attention" }
            ?: f.stringList("$prefix.attention.layer_types")
            ?: f.stringList("$prefix.attention.layer_pattern")
            ?: buildDefaultLayerTypes(blockCount)

        val ropeBase = f.float("$prefix.rope.freq_base") ?: 1000000f
        // GGUF uses rope.freq_base_swa for the sliding-window RoPE base.
        val ropeBaseLocal = f.float("$prefix.rope.freq_base_swa") ?: f.float("$prefix.rope.freq_base_local") ?: 10000f
        val ropeFactor = f.float("$prefix.rope.factor") ?: 1.0f
        // GGUF does not carry partial_rotary_factor directly for Gemma 4, but it DOES carry
        // rope.dimension_count — the actual number of dims rotated for the global/full scheme.
        // Derive the factor from the real field when present; only fall back to the guessed
        // default when the checkpoint omits rope.dimension_count entirely (the old hardcoded
        // 0.25 silently rotated 128 of 512 dims on every global layer and corrupted decode).
        val partialRotaryDefault = if (arch.startsWith("gemma")) 0.25f else 1.0f
        val partialRotaryFactor = f.float("$prefix.rope.partial_rotary_factor")
            ?: f.int("$prefix.rope.dimension_count")?.takeIf { globalHeadDim > 0 }?.let { it.toFloat() / globalHeadDim }
            ?: partialRotaryDefault
        val finalLogitSoftcapping = f.float("$prefix.final_logit_softcapping") ?: 0f
        val rmsNormEps = f.float("$prefix.attention.layer_norm_rms_epsilon") ?: 1e-6f
        // Token ids: without these, downstream defaults prefill BOS=1 — which is Gemma's <eos> —
        // and generation collapses into turn-token spam (#325 arc).
        val bosTokenId = f.int("tokenizer.ggml.bos_token_id") ?: 2
        val eosTokenId = f.int("tokenizer.ggml.eos_token_id") ?: 1
        val padTokenId = f.int("tokenizer.ggml.padding_token_id") ?: 0

        return GemmaModelMetadata(
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
            ropeParametersFull = GemmaRopeConfig(
                base = ropeBase,
                ropeType = "proportional",
                factor = ropeFactor,
                partialRotaryFactor = partialRotaryFactor
            ),
            ropeParametersSliding = GemmaRopeConfig(
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





    private fun buildDefaultLayerTypes(blockCount: Int): List<String> {
        return List(blockCount) { idx ->
            if (idx == blockCount - 1) "full_attention"
            else if ((idx + 1) % 6 == 0) "full_attention"
            else "sliding_attention"
        }
    }

    private fun validateMetadata(metadata: GemmaModelMetadata) {
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

    private fun requiredTensorNames(metadata: GemmaModelMetadata): List<String> {
        val names = mutableListOf<String>()
        names += GemmaTensorNames.TOKEN_EMBEDDINGS
        names += GemmaTensorNames.OUTPUT_NORM

        repeat(metadata.blockCount) { layer ->
            names += GemmaTensorNames.inputLayernorm(layer)
            names += GemmaTensorNames.attnQ(layer)
            names += GemmaTensorNames.attnK(layer)
            names += GemmaTensorNames.attnV(layer)
            names += GemmaTensorNames.attnOut(layer)
            names += GemmaTensorNames.postAttentionLayernorm(layer)
            names += GemmaTensorNames.ffnGate(layer)
            names += GemmaTensorNames.ffnDown(layer)
            names += GemmaTensorNames.ffnUp(layer)
        }
        return names
    }

    private fun optionalTensorNames(metadata: GemmaModelMetadata): List<String> {
        val names = mutableListOf(
            GemmaTensorNames.ROPE_FREQS_REAL,
            GemmaTensorNames.ROPE_FREQS_IMAG,
            GemmaTensorNames.PER_LAYER_TOKEN_EMBD,
            GemmaTensorNames.PER_LAYER_MODEL_PROJ,
            GemmaTensorNames.PER_LAYER_PROJ_NORM,
        )
        repeat(metadata.blockCount) { layer ->
            names += GemmaTensorNames.perLayerInput(layer)
            names += GemmaTensorNames.perLayerOutput(layer)
            names += GemmaTensorNames.attnQNorm(layer)
            names += GemmaTensorNames.attnKNorm(layer)
            names += GemmaTensorNames.postAttentionNorm(layer)
            names += GemmaTensorNames.postFfwNorm(layer)
            names += GemmaTensorNames.layerOutputScale(layer)
            names += GemmaTensorNames.pleInpGate(layer)
            names += GemmaTensorNames.plePostNorm(layer)
            names += GemmaTensorNames.pleProj(layer)
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
