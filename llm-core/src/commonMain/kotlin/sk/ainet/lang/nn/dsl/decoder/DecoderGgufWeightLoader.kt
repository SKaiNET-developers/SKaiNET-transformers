package sk.ainet.lang.nn.dsl.decoder

import kotlinx.io.Source
import kotlinx.io.buffered
import sk.ainet.apps.llm.DTypePolicyValidation
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGMLQuantizationType
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.ReaderField
import sk.ainet.io.gguf.ReaderTensor
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.StreamingTensorInfo
import sk.ainet.io.gguf.StreamingGgufParametersLoader
import sk.ainet.io.gguf.dequant.DequantOps
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.nn.quant.PackedRowDequantTensorData
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.data.Bf16DenseTensorData
import sk.ainet.lang.tensor.data.Fp16DenseTensorData
import sk.ainet.lang.tensor.data.NarrowFloatTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import kotlin.reflect.KClass

public data class GgufDecoderMetadata(
    val architecture: String,
    override val embeddingLength: Int,
    override val contextLength: Int,
    override val blockCount: Int,
    override val headCount: Int,
    override val kvHeadCount: Int,
    override val feedForwardLength: Int,
    override val ropeDimensionCount: Int?,
    override val vocabSize: Int,
    override val ropeFreqBase: Float = 10_000f,
    override val rmsNormEps: Float = 1e-5f,
    override val bosTokenId: Int = 1,
    override val eosTokenId: Int = 2,
) : sk.ainet.lang.nn.dsl.decoder.DecoderModelMetadata

public data class DecoderGgufWeights<T : DType, V>(
    val metadata: GgufDecoderMetadata,
    val tensors: Map<String, Tensor<T, V>>
)

/**
 * [WeightForm] requesting every tensor fully dequantized to dense FP32 with
 * logical `[out, in]` shapes — the form the legacy eager runtime and
 * trace/export harnesses consume.
 */
@ExperimentalMemoryApi
public val DECODER_DEQUANTIZE_ALL: WeightForm = WeightForm(
    encoding = EncodingRequest.DequantizeTo(FP32),
    shape = WeightShapeOrientation.OUT_IN
)

public object DecoderTensorNames {
    const val TOKEN_EMBEDDINGS: String = "token_embd.weight"
    const val OUTPUT_NORM: String = "output_norm.weight"
    const val OUTPUT_WEIGHT: String = "output.weight"
    const val ROPE_FREQS_REAL: String = "rope.freq_cis_real"
    const val ROPE_FREQS_IMAG: String = "rope.freq_cis_imag"

    fun attnNorm(layer: Int): String = "blk.$layer.attn_norm.weight"
    fun attnQ(layer: Int): String = "blk.$layer.attn_q.weight"
    fun attnK(layer: Int): String = "blk.$layer.attn_k.weight"
    fun attnV(layer: Int): String = "blk.$layer.attn_v.weight"
    fun attnOut(layer: Int): String = "blk.$layer.attn_output.weight"
    fun ffnNorm(layer: Int): String = "blk.$layer.ffn_norm.weight"
    fun ffnGate(layer: Int): String = "blk.$layer.ffn_gate.weight"
    fun ffnDown(layer: Int): String = "blk.$layer.ffn_down.weight"
    fun ffnUp(layer: Int): String = "blk.$layer.ffn_up.weight"
    fun attnQNorm(layer: Int): String = "blk.$layer.attn_q_norm.weight"
    fun attnKNorm(layer: Int): String = "blk.$layer.attn_k_norm.weight"

    fun attnQBias(layer: Int): String = "blk.$layer.attn_q.bias"
    fun attnKBias(layer: Int): String = "blk.$layer.attn_k.bias"
    fun attnVBias(layer: Int): String = "blk.$layer.attn_v.bias"
    fun attnOutBias(layer: Int): String = "blk.$layer.attn_output.bias"
}

/**
 * Adapter that loads LLaMA weights from GGUF files and emits them in the canonical GGUF tensor
 * naming scheme. Validation covers metadata presence and basic shape consistency for the tensors
 * we materialize.
 */
@OptIn(ExperimentalMemoryApi::class)
public class DecoderGgufWeightLoader private constructor(
    private val sourceProvider: (() -> Source)?,
    private val randomAccessProvider: (() -> RandomAccessSource)?,
    private val acceptedArchitectures: Set<String> = setOf("llama"),
    private val dtypePolicy: DTypePolicy = DTypePolicy.Any,
    private val weightForm: WeightForm? = null,
) {
    /**
     * Keep `F16` source tensors in their on-disk 2-bytes-per-element layout instead of widening
     * them to FP32. Resolved from [dtypePolicy] exactly as the engine's
     * `StreamingGgufParametersLoader.keepsNative` does, so a policy carried down from
     * `LlamaNetworkLoader.withDtypePolicy` means the same thing on both sides.
     */
    private val keepF16Native: Boolean = DTypePolicyValidation.keepsNative(dtypePolicy, FP16)

    /** As [keepF16Native], for `BF16` sources. Resolved independently — see [keepsNarrowNative]. */
    private val keepBf16Native: Boolean = DTypePolicyValidation.keepsNative(dtypePolicy, BF16)

    /**
     * Primary constructor for sequential Source-based loading.
     * Loads entire file into memory - suitable for models under 2GB.
     * The sequential path always dequantizes to dense tensors.
     *
     * @param acceptedArchitectures GGUF architecture strings accepted by this loader.
     *   Defaults to `setOf("llama")`. Consumers loading compatible architectures
     *   (e.g. Qwen, Mistral) pass their own set — no changes needed here.
     * @param dtypePolicy narrow-float handling. Default [DTypePolicy.Any] widens F16/BF16
     *   sources to FP32; a policy naming BF16 or FP16 keeps that format packed.
     */
    public constructor(
        sourceProvider: () -> Source,
        acceptedArchitectures: Set<String> = setOf("llama"),
        dtypePolicy: DTypePolicy = DTypePolicy.Any,
    ) : this(
        sourceProvider = sourceProvider,
        randomAccessProvider = null,
        acceptedArchitectures = acceptedArchitectures,
        dtypePolicy = dtypePolicy,
    )

    /**
     * Secondary constructor for streaming RandomAccessSource-based loading.
     * Parses metadata only (~1MB memory) and streams tensors through the
     * engine's [StreamingGgufParametersLoader]. Suitable for models of any size.
     *
     * @param acceptedArchitectures GGUF architecture strings accepted by this loader.
     *   Defaults to `setOf("llama")`. Consumers loading compatible architectures
     *   (e.g. Qwen, Mistral) pass their own set — no changes needed here.
     * @param weightForm per-tensor materialization request. `null` (default) keeps
     *   quantized tensors in their stored block encoding as packed
     *   [sk.ainet.lang.tensor.storage.PackedBlockStorage] data with logical
     *   `[out, in]` shapes and `MAPPED` residency — servable encodings are
     *   served zero-copy from file-backed pages (heap-staged where the
     *   platform or encoding cannot map); pass
     *   `WeightForm(encoding = EncodingRequest.DequantizeTo(FP32), shape = WeightShapeOrientation.OUT_IN)`
     *   to fully dequantize (the legacy eager runtime path). A packed token
     *   embedding is rewrapped as a row-dequant source on delivery — see
     *   [embeddingReady].
     */
    public constructor(
        randomAccessProvider: () -> RandomAccessSource,
        acceptedArchitectures: Set<String> = setOf("llama"),
        dtypePolicy: DTypePolicy = DTypePolicy.Any,
        weightForm: WeightForm? = null,
    ) : this(
        sourceProvider = null,
        randomAccessProvider = randomAccessProvider,
        acceptedArchitectures = acceptedArchitectures,
        dtypePolicy = dtypePolicy,
        weightForm = weightForm,
    )

    /**
     * Load weights and invoke [onTensorLoaded] for each required tensor. Returns parsed metadata.
     */
    public suspend fun <T : DType, V> load(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): GgufDecoderMetadata {
        return loadFromGguf(ctx, dtype, onTensorLoaded)
    }

    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): GgufDecoderMetadata = load(ctx, T::class, onTensorLoaded)

    /** Convenience helper that collects tensors into a map alongside metadata. */
    public suspend fun <T : DType, V> loadToMap(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): DecoderGgufWeights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val meta = loadFromGguf(ctx, dtype) { name, tensor -> byName[name] = tensor }
        return DecoderGgufWeights(meta, byName)
    }

    public suspend inline fun <reified T : DType, V> loadToMap(
        ctx: ExecutionContext
    ): DecoderGgufWeights<T, V> = loadToMap(ctx, T::class)

    // ============== Streaming API (for large files >2GB) ==============

    /**
     * Load weights using streaming API - parses metadata only, loads tensors on-demand.
     * Requires [randomAccessProvider] constructor.
     */
    public suspend fun <T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): GgufDecoderMetadata {
        return loadFromStreamingGguf(ctx, dtype, onTensorLoaded)
    }

    public suspend inline fun <reified T : DType, V> loadStreaming(
        ctx: ExecutionContext,
        noinline onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): GgufDecoderMetadata = loadStreaming(ctx, T::class, onTensorLoaded)

    /**
     * Load weights to map using streaming API.
     * Requires [randomAccessProvider] constructor.
     */
    public suspend fun <T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext,
        dtype: KClass<T>
    ): DecoderGgufWeights<T, V> {
        val byName = linkedMapOf<String, Tensor<T, V>>()
        val meta = loadFromStreamingGguf(ctx, dtype) { name, tensor -> byName[name] = tensor }
        return DecoderGgufWeights(meta, byName)
    }

    public suspend inline fun <reified T : DType, V> loadToMapStreaming(
        ctx: ExecutionContext
    ): DecoderGgufWeights<T, V> = loadToMapStreaming(ctx, T::class)

    private fun <T : DType, V> loadFromGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): GgufDecoderMetadata {
        require(dtype == FP32::class || dtype == FP16::class) {
            "LLaMA GGUF loader supports FP32 and FP16 tensors (got ${dtype.simpleName})"
        }
        requireNotNull(sourceProvider) {
            "Sequential loading requires sourceProvider constructor. Use loadFromStreamingGguf for RandomAccessSource."
        }

        val reader = sourceProvider.invoke().buffered().use { src ->
            GGUFReader(src, loadTensorData = true)
        }

        val metadata = metadataFromGguf(reader.fields, reader.tensors)
        validateMetadata(metadata)

        val required = requiredTensorNames(metadata)
        val tensorByName = reader.tensors.associateBy { it.name }

        // Tied embeddings (Qwen2.5-0.5B/1.5B, Gemma, etc.): reuse token_embd.weight as output.weight
        val tiedEmbeddings = tensorByName[DecoderTensorNames.OUTPUT_WEIGHT] == null &&
            tensorByName[DecoderTensorNames.TOKEN_EMBEDDINGS] != null
        if (tiedEmbeddings) {
            println("Tied word embeddings: output.weight = token_embd.weight")
        }

        required.forEach { name ->
            val lookupName = if (name == DecoderTensorNames.OUTPUT_WEIGHT && tiedEmbeddings) {
                DecoderTensorNames.TOKEN_EMBEDDINGS
            } else {
                name
            }
            val rt = tensorByName[lookupName]
                ?: error("Missing required tensor in GGUF payload: $name")
            validateTensorShape(name, rt, metadata)
            val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt)
            onTensorLoaded(name, tensor)
        }

        // Optional tensors (e.g., precomputed RoPE tables, QK-norm) if present
        val optionalNames = mutableListOf(
            DecoderTensorNames.ROPE_FREQS_REAL,
            DecoderTensorNames.ROPE_FREQS_IMAG
        )
        repeat(metadata.blockCount) { layer ->
            optionalNames += DecoderTensorNames.attnQNorm(layer)
            optionalNames += DecoderTensorNames.attnKNorm(layer)
            optionalNames += DecoderTensorNames.attnQBias(layer)
            optionalNames += DecoderTensorNames.attnKBias(layer)
            optionalNames += DecoderTensorNames.attnVBias(layer)
            optionalNames += DecoderTensorNames.attnOutBias(layer)
        }
        optionalNames.forEach { name ->
            val rt = tensorByName[name]
            if (rt != null) {
                val tensor: Tensor<T, V> = readerTensorToTensor(ctx, dtype, reader, rt)
                onTensorLoaded(name, tensor)
            }
        }

        return metadata
    }

    /**
     * Load using the engine's [StreamingGgufParametersLoader] — parses metadata into memory,
     * then streams tensor payloads one at a time (the per-tensor callback keeps the load peak
     * at one tensor's transient buffers, which is what lets a 2 GB board load multi-GB models).
     *
     * Quantized tensors keep their stored block encoding as packed
     * [sk.ainet.lang.tensor.storage.PackedBlockStorage] data with logical `[out, in]` shapes
     * unless [weightForm] requests otherwise. The token embedding is always dequantized to a
     * dense tensor (`Embedding.gather()` needs element access); with tied embeddings it also
     * serves as `output.weight`.
     */
    private suspend fun <T : DType, V> loadFromStreamingGguf(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        onTensorLoaded: (String, Tensor<T, V>) -> Unit
    ): GgufDecoderMetadata {
        require(dtype == FP32::class) {
            "Engine-backed streaming GGUF loading delivers FP32-typed tensors (got ${dtype.simpleName})"
        }
        requireNotNull(randomAccessProvider) {
            "Streaming loading requires randomAccessProvider constructor. Use loadFromGguf for Source."
        }

        // Header pass: metadata, shape validation, tied-embedding detection.
        val metadata: GgufDecoderMetadata
        val wanted: Set<String>
        val tiedEmbeddings: Boolean
        StreamingGGUFReader.open(randomAccessProvider.invoke()).use { reader ->
            metadata = decoderMetadataFromGguf(reader.fields, reader.tensors)
            validateMetadata(metadata)

            val required = requiredTensorNames(metadata)
            val tensorByName = reader.tensors.associateBy { it.name }

            // Tied embeddings: small models (Qwen2.5-0.5B/1.5B, etc.) omit output.weight
            // and reuse token_embd.weight as the LM head. Detect and alias.
            tiedEmbeddings = tensorByName[DecoderTensorNames.OUTPUT_WEIGHT] == null &&
                tensorByName[DecoderTensorNames.TOKEN_EMBEDDINGS] != null
            if (tiedEmbeddings) {
                println("Tied word embeddings: output.weight = token_embd.weight")
            }

            required.forEach { name ->
                val lookupName = if (name == DecoderTensorNames.OUTPUT_WEIGHT && tiedEmbeddings) {
                    DecoderTensorNames.TOKEN_EMBEDDINGS
                } else {
                    name
                }
                val st = tensorByName[lookupName]
                    ?: error("Missing required tensor in GGUF payload: $name")
                // Shape validation uses the logical name (e.g., OUTPUT_WEIGHT) even when
                // the physical tensor is TOKEN_EMBEDDINGS — both must have [vocab, dim] shape.
                validateStreamingTensorShape(name, st, metadata)
            }

            val optionalNames = mutableListOf(
                DecoderTensorNames.ROPE_FREQS_REAL,
                DecoderTensorNames.ROPE_FREQS_IMAG
            )
            repeat(metadata.blockCount) { layer ->
                optionalNames += DecoderTensorNames.attnQNorm(layer)
                optionalNames += DecoderTensorNames.attnKNorm(layer)
                optionalNames += DecoderTensorNames.attnQBias(layer)
                optionalNames += DecoderTensorNames.attnKBias(layer)
                optionalNames += DecoderTensorNames.attnVBias(layer)
                optionalNames += DecoderTensorNames.attnOutBias(layer)
            }
            wanted = buildSet {
                addAll(required)
                add(DecoderTensorNames.TOKEN_EMBEDDINGS)
                optionalNames.forEach { if (it in tensorByName) add(it) }
            }
        }

        // Payload pass through the engine loader. MAPPED residency is the
        // default (#342 arc, P5): servable encodings stay in file-backed
        // pages and the row-major kernel pack reads them zero-copy; the
        // engine heap-stages anything it cannot serve from the mapping.
        val defaultForm = weightForm ?: WeightForm(
            shape = WeightShapeOrientation.OUT_IN,
            residency = WeightResidency.MAPPED,
        )
        val engineLoader = StreamingGgufParametersLoader(
            sourceProvider = randomAccessProvider,
            keepF16Native = keepF16Native && dtype == FP32::class,
            keepBf16Native = keepBf16Native && dtype == FP32::class,
            weightForm = defaultForm,
        )
        var tokenEmbedding: Tensor<T, V>? = null
        engineLoader.load(ctx, dtype) { name: String, tensor: Tensor<T, V> ->
            // The token embedding rides the default form like every other
            // tensor (mapped keep-packed), then gets rewrapped as a
            // row-dequant source so `Embedding` reads it row-by-row — a
            // 152k-vocab table stays at its packed footprint instead of a
            // ~1 GB dense FP32 heap array, and the *tied* output head still
            // sees PackedBlockStorage for the packed matmul chain. Never
            // deliver a bare packed table to Embedding: packed `get()`
            // returns raw quantization codes, not values.
            val delivered =
                if (name == DecoderTensorNames.TOKEN_EMBEDDINGS) embeddingReady(ctx, dtype, tensor)
                else tensor
            if (name == DecoderTensorNames.TOKEN_EMBEDDINGS) tokenEmbedding = delivered
            if (name in wanted) onTensorLoaded(name, delivered)
        }
        if (tiedEmbeddings) {
            onTensorLoaded(
                DecoderTensorNames.OUTPUT_WEIGHT,
                tokenEmbedding ?: error("Tied embeddings detected but ${DecoderTensorNames.TOKEN_EMBEDDINGS} was not delivered")
            )
        }

        return metadata
    }

    /**
     * The token-embedding tensor in a form `Embedding.gather` can read:
     * packed deliveries are rewrapped as a [PackedRowDequantTensorData]
     * row-dequant source; a packed table whose rows do not lie on block
     * boundaries (no canonical GGUF quantization produces one, but the type
     * system allows it) is dequantized outright rather than left where
     * element access would return raw quantization codes. Dense deliveries
     * pass through untouched.
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






    private fun validateStreamingTensorShape(name: String, tensor: StreamingTensorInfo, metadata: GgufDecoderMetadata) {
        val dims = tensor.shape.map { it.toInt() }
        when (name) {
            DecoderTensorNames.TOKEN_EMBEDDINGS, DecoderTensorNames.OUTPUT_WEIGHT -> {
                require(dims.size == 2 && dims.contains(metadata.embeddingLength)) {
                    "Tensor $name must be [vocab, dim] shaped; got $dims"
                }
            }

            DecoderTensorNames.OUTPUT_NORM -> {
                require(dims.size == 1 && dims[0] == metadata.embeddingLength) {
                    "Tensor $name must be [${metadata.embeddingLength}] shaped; got $dims"
                }
            }

            DecoderTensorNames.ROPE_FREQS_REAL, DecoderTensorNames.ROPE_FREQS_IMAG -> {
                val headSize = metadata.embeddingLength / metadata.headCount
                require(dims.size == 2 && dims[0] == metadata.contextLength && dims[1] == headSize / 2) {
                    val expectedShape = "[${metadata.contextLength}, ${headSize / 2}]"
                    "Tensor $name must be [seqLen, headSize/2]=$expectedShape shaped; got $dims"
                }
            }

            else -> {
                when {
                    name.contains("attn_norm") || name.contains("ffn_norm") -> {
                        require(dims.size == 1 && dims[0] == metadata.embeddingLength) {
                            "Tensor $name must be [${metadata.embeddingLength}] shaped; got $dims"
                        }
                    }

                    name.contains("attn_q") || name.contains("attn_output") -> {
                        // Q: [q_dim, dim] where q_dim = nHeads * headDim (headDim may differ from dim/nHeads)
                        // O: [dim, q_dim]
                        // Accept any 2D tensor where one dimension is dim (the other is q_dim)
                        require(dims.size == 2 && dims.any { it == metadata.embeddingLength }) {
                            "Tensor $name must be 2D with one dim=${metadata.embeddingLength}; got $dims"
                        }
                    }

                    name.contains("attn_k") || name.contains("attn_v") -> {
                        // K/V: [kv_dim, dim] where kv_dim = kvHeadCount * headDim
                        // Accept any 2D tensor where one dimension is dim
                        require(dims.size == 2 && dims.any { it == metadata.embeddingLength }) {
                            "Tensor $name must be 2D with one dim=${metadata.embeddingLength}; got $dims"
                        }
                    }

                    name.contains("ffn_gate") || name.contains("ffn_up") -> {
                        val expected = metadata.feedForwardLength * metadata.embeddingLength
                        require(dims.size == 2 && dims.product() == expected) {
                            "Tensor $name must have product $expected; got $dims"
                        }
                    }

                    name.contains("ffn_down") -> {
                        val expected = metadata.embeddingLength * metadata.feedForwardLength
                        require(dims.size == 2 && dims.product() == expected) {
                            "Tensor $name must have product $expected; got $dims"
                        }
                    }
                }
            }
        }
    }

    private fun metadataFromGguf(
        fields: Map<String, ReaderField>,
        tensors: List<ReaderTensor>
    ): GgufDecoderMetadata {
        val arch = fields["general.architecture"]?.stringValue() ?: "unknown"
        val prefix = arch

        val embeddingLength = fields["$prefix.embedding_length"]?.scalarInt()
            ?: inferEmbeddingFromTensor(tensors)
        val contextLength = fields["$prefix.context_length"]?.scalarInt() ?: 0
        val blockCount = fields["$prefix.block_count"]?.scalarInt() ?: 0
        val headCount = fields["$prefix.attention.head_count"]?.scalarInt() ?: 0
        val kvHeadCount = fields["$prefix.attention.head_count_kv"]?.scalarInt() ?: headCount
        val feedForwardLength = fields["$prefix.feed_forward_length"]?.scalarInt() ?: 0
        var ropeDim = fields["$prefix.rope.dimension_count"]?.scalarInt()
        val vocabSize = fields["$prefix.vocab_size"]?.scalarInt()
            ?: inferVocabFromTensor(tensors)
        val ropeFreqBase = fields["$prefix.rope.freq_base"]?.scalarFloat() ?: 10_000f
        val rmsNormEps = fields["$prefix.attention.layer_norm_rms_epsilon"]?.scalarFloat() ?: 1e-5f
        val bosTokenId = fields["tokenizer.ggml.bos_token_id"]?.scalarInt() ?: 1
        val eosTokenId = fields["tokenizer.ggml.eos_token_id"]?.scalarInt() ?: 2

        // Infer head_dim from Q weight shape when rope dimension not set
        if (ropeDim == null && headCount > 0) {
            val qTensor = tensors.firstOrNull { it.name == "blk.0.attn_q.weight" }
            if (qTensor != null && qTensor.shape.size == 2) {
                val embUInt = embeddingLength.toUInt()
                val qDim = (qTensor.shape.firstOrNull { it != embUInt } ?: qTensor.shape[0]).toInt()
                val inferredHeadDim = qDim / headCount
                if (inferredHeadDim > 0 && inferredHeadDim * headCount == qDim) {
                    ropeDim = inferredHeadDim
                }
            }
        }

        return GgufDecoderMetadata(
            architecture = arch,
            embeddingLength = embeddingLength,
            contextLength = contextLength,
            blockCount = blockCount,
            headCount = headCount,
            kvHeadCount = kvHeadCount,
            feedForwardLength = feedForwardLength,
            ropeDimensionCount = ropeDim,
            vocabSize = vocabSize,
            ropeFreqBase = ropeFreqBase,
            rmsNormEps = rmsNormEps,
            bosTokenId = bosTokenId,
            eosTokenId = eosTokenId
        )
    }

    private fun validateMetadata(metadata: GgufDecoderMetadata) {
        require(metadata.architecture in acceptedArchitectures) {
            "Unsupported architecture: ${metadata.architecture}. Accepted: $acceptedArchitectures"
        }
        require(metadata.embeddingLength > 0) { "Invalid embedding length ${metadata.embeddingLength}" }
        require(metadata.blockCount > 0) { "Invalid block count ${metadata.blockCount}" }
        require(metadata.headCount > 0) { "Invalid head count ${metadata.headCount}" }
        require(metadata.contextLength > 0) { "Invalid context length ${metadata.contextLength}" }
        require(metadata.vocabSize > 0) { "Invalid vocab size ${metadata.vocabSize}" }
    }

    private fun requiredTensorNames(metadata: GgufDecoderMetadata): List<String> {
        val names = mutableListOf<String>()
        names += DecoderTensorNames.TOKEN_EMBEDDINGS
        names += DecoderTensorNames.OUTPUT_NORM
        names += DecoderTensorNames.OUTPUT_WEIGHT

        repeat(metadata.blockCount) { layer ->
            names += DecoderTensorNames.attnNorm(layer)
            names += DecoderTensorNames.attnQ(layer)
            names += DecoderTensorNames.attnK(layer)
            names += DecoderTensorNames.attnV(layer)
            names += DecoderTensorNames.attnOut(layer)
            names += DecoderTensorNames.ffnNorm(layer)
            names += DecoderTensorNames.ffnGate(layer)
            names += DecoderTensorNames.ffnDown(layer)
            names += DecoderTensorNames.ffnUp(layer)
        }
        return names
    }

    private fun validateTensorShape(name: String, tensor: ReaderTensor, metadata: GgufDecoderMetadata) {
        val dims = tensor.shape.map { it.toInt() }
        when (name) {
            DecoderTensorNames.TOKEN_EMBEDDINGS, DecoderTensorNames.OUTPUT_WEIGHT -> {
                require(dims.size == 2 && dims.contains(metadata.embeddingLength)) {
                    "Tensor $name must be [vocab, dim] shaped; got $dims"
                }
            }

            DecoderTensorNames.OUTPUT_NORM -> {
                require(dims.size == 1 && dims[0] == metadata.embeddingLength) {
                    "Tensor $name must be [${
                        metadata.embeddingLength
                    }] shaped; got $dims"
                }
            }

            DecoderTensorNames.ROPE_FREQS_REAL, DecoderTensorNames.ROPE_FREQS_IMAG -> {
                val headSize = metadata.embeddingLength / metadata.headCount
                require(dims.size == 2 && dims[0] == metadata.contextLength && dims[1] == headSize / 2) {
                    val expectedShape = "[${metadata.contextLength}, ${headSize / 2}]"
                    "Tensor $name must be [seqLen, headSize/2]=$expectedShape shaped; got $dims"
                }
            }

            else -> {
                when {
                    name.contains("attn_norm") || name.contains("ffn_norm") -> {
                        require(dims.size == 1 && dims[0] == metadata.embeddingLength) {
                            "Tensor $name must be [${metadata.embeddingLength}] shaped; got $dims"
                        }
                    }

                    name.contains("attn_q") || name.contains("attn_output") -> {
                        require(dims.size == 2 && dims.any { it == metadata.embeddingLength }) {
                            "Tensor $name must be 2D with one dim=${metadata.embeddingLength}; got $dims"
                        }
                    }

                    name.contains("attn_k") || name.contains("attn_v") -> {
                        require(dims.size == 2 && dims.any { it == metadata.embeddingLength }) {
                            "Tensor $name must be 2D with one dim=${metadata.embeddingLength}; got $dims"
                        }
                    }

                    name.contains("ffn_gate") || name.contains("ffn_up") -> {
                        val expected = metadata.feedForwardLength * metadata.embeddingLength
                        require(dims.size == 2 && dims.product() == expected) {
                            "Tensor $name must have product $expected; got $dims"
                        }
                    }

                    name.contains("ffn_down") -> {
                        val expected = metadata.embeddingLength * metadata.feedForwardLength
                        require(dims.size == 2 && dims.product() == expected) {
                            "Tensor $name must have product $expected; got $dims"
                        }
                    }
                }
            }
        }
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
            is Int -> value.toFloat()
            is UInt -> value.toFloat()
            is Long -> value.toFloat()
            is ULong -> value.toFloat()
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

    private fun inferEmbeddingFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == DecoderTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer embedding length without token embeddings tensor")
        // For most LLMs, embedding_length < vocab_size, so we take the min
        return token.shape.map { it.toInt() }.minOrNull()
            ?: error("Cannot infer embedding length from tensor shape ${token.shape}")
    }

    private fun inferVocabFromTensor(tensors: List<ReaderTensor>): Int {
        val token = tensors.firstOrNull { it.name == DecoderTensorNames.TOKEN_EMBEDDINGS }
            ?: error("Cannot infer vocab size without token embeddings tensor")
        // For most LLMs, vocab_size > embedding_length, so we take the max
        return token.shape.map { it.toInt() }.maxOrNull()
            ?: error("Cannot infer vocab size from tensor shape ${token.shape}")
    }

    private fun List<Int>.product(): Int = fold(1) { acc, v -> acc * v }

    /**
     * Create a tensor from float data, transposing 2D tensors from column-major to row-major.
     * GGUF stores 2D tensors in column-major order, so we transpose them at load time.
     * The dtype parameter determines the GPU storage format (FP32 or FP16).
     */
    @Suppress("UNCHECKED_CAST")
    private fun <T : DType, V> createTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        originalShape: Shape,
        data: FloatArray
    ): Tensor<T, V> {
        return if (originalShape.rank == 2) {
            // Transpose 2D tensors from column-major to row-major
            val rows = originalShape[0]
            val cols = originalShape[1]
            val transposed = DequantOps.transposeColumnMajorToRowMajor(data, rows, cols)
            // Shape is now [cols, rows] after transpose
            val newShape = Shape(cols, rows)
            ctx.fromFloatArray<T, Float>(newShape, dtype, transposed) as Tensor<T, V>
        } else {
            ctx.fromFloatArray<T, Float>(originalShape, dtype, data) as Tensor<T, V>
        }
    }

    /**
     * Whether a GGUF tensor of [tensorType] should keep its on-disk 16-bit bytes.
     *
     * KEEP_NATIVE is restricted to `dtype == FP32` — that is the declared dtype the packed
     * tensor presents to consumers (`get` decodes to `Float`), matching the SafeTensors path.
     * An explicit `FP16::class` request is a storage-format ask for the FP32-array path and is
     * left on the widening route rather than silently reinterpreted.
     */
    internal fun <T : DType> keepsNarrowNative(tensorType: GGMLQuantizationType, dtype: KClass<T>): Boolean =
        dtype == FP32::class && when (tensorType) {
            GGMLQuantizationType.F16 -> keepF16Native
            GGMLQuantizationType.BF16 -> keepBf16Native
            else -> false
        }

    /**
     * Wrap packed 16-bit GGUF bytes as a narrow-float tensor — the KEEP_NATIVE counterpart of
     * [createTensor], and it must mirror that function's layout handling exactly.
     *
     * For rank 2 that means swapping the shape to `[cols, rows]` and **moving no bytes**.
     * GGUF's header dims are reversed relative to the logical row-major shape, so the
     * column-major → row-major step is a reinterpretation, not a permutation — which is why
     * `DequantOps.transposeColumnMajorToRowMajor` returns its input untouched and
     * [createTensor] only rebuilds the `Shape`. Doing an actual element transpose here would
     * hand the matmul kernel a silently transposed weight matrix.
     *
     * The result is genuinely zero-copy: the on-disk buffer becomes the tensor's storage.
     */
    @Suppress("UNCHECKED_CAST")
    internal fun <T : DType, V> createNarrowTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        originalShape: Shape,
        bytes: ByteArray,
        tensorType: GGMLQuantizationType,
    ): Tensor<T, V> {
        val shape = if (originalShape.rank == 2) {
            Shape(originalShape[1], originalShape[0])
        } else {
            originalShape
        }

        val required = shape.volume * NarrowFloatTensorData.BYTES_PER_ELEMENT
        require(bytes.size >= required) {
            "Narrow-float buffer of ${bytes.size} bytes is short of the $required bytes needed " +
                "for a ${shape.dimensions.toList()} $tensorType tensor"
        }

        val data = when (tensorType) {
            GGMLQuantizationType.F16 -> Fp16DenseTensorData.fromRawBytes(shape, bytes)
            GGMLQuantizationType.BF16 -> Bf16DenseTensorData.fromRawBytes(shape, bytes)
            else -> error("createNarrowTensor called with non-narrow type $tensorType")
        }
        return ctx.fromData(data as TensorData<T, Float>, dtype) as Tensor<T, V>
    }

    private fun <T : DType, V> readerTensorToTensor(
        ctx: ExecutionContext,
        dtype: KClass<T>,
        reader: GGUFReader,
        rt: ReaderTensor
    ): Tensor<T, V> {
        val shape = Shape(*rt.shape.map { it.toInt() }.toIntArray())
        return when (rt.tensorType) {
            GGMLQuantizationType.F32 -> {
                @Suppress("UNCHECKED_CAST")
                val floats = (if (rt.data.isEmpty()) reader.materialize(rt) else rt.data) as List<Float>
                createTensor(ctx, dtype, shape, floats.toFloatArray())
            }

            GGMLQuantizationType.F16,
            GGMLQuantizationType.BF16 -> {
                val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                if (keepsNarrowNative(rt.tensorType, dtype)) {
                    createNarrowTensor(
                        ctx, dtype, shape, DequantOps.toByteArray(raw, rt.name), rt.tensorType,
                    )
                } else {
                    val floats = when (rt.tensorType) {
                        GGMLQuantizationType.F16 -> DequantOps.dequantF16(raw)
                        GGMLQuantizationType.BF16 -> DequantOps.dequantBF16(raw)
                        else -> error("Unsupported native type ${rt.tensorType}")
                    }
                    createTensor(ctx, dtype, shape, floats)
                }
            }

            GGMLQuantizationType.I8,
            GGMLQuantizationType.I16,
            GGMLQuantizationType.I32 -> error("Native type ${rt.tensorType} not yet supported in LLaMA loader")

            GGMLQuantizationType.UNKNOWN -> error(
                "Tensor '${rt.name}' has unknown quantization type (raw value: ${rt.rawTypeValue})"
            )

            else -> {
                val raw = if (rt.data.isEmpty()) reader.materialize(rt) else rt.data
                val bytes = DequantOps.toByteArray(raw, rt.name)
                val floats = DequantOps.dequantFromBytes(bytes, rt.tensorType, rt.nElements)
                createTensor(ctx, dtype, shape, floats)
            }
        }
    }
}
