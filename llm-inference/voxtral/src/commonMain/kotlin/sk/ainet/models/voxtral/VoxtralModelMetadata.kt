package sk.ainet.models.voxtral

import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata

/**
 * Metadata for the Voxtral TTS model (mistralai/Voxtral-4B-TTS-2603).
 *
 * Voxtral is a hybrid TTS model with three components:
 * 1. **Text backbone** — a 26-layer Ministral-3B transformer that generates semantic tokens
 *    autoregressively. Uses standard LLaMA architecture (GQA + SwiGLU + RoPE + RMSNorm).
 * 2. **Acoustic transformer** — a 3-layer flow-matching transformer that generates 36
 *    acoustic codebooks in parallel from the backbone's hidden states.
 * 3. **Voxtral Codec** — a convolutional + transformer decoder that converts semantic +
 *    acoustic tokens to 24kHz audio waveform.
 *
 * The text backbone and acoustic transformer are both LLaMA-compatible architectures,
 * so they share [GgufDecoderMetadata] for their transformer configuration.
 */
public data class VoxtralModelMetadata(
    /** Metadata for the main text transformer backbone. */
    val backbone: GgufDecoderMetadata,
    /** Metadata for the acoustic flow-matching transformer. */
    val acousticModel: GgufDecoderMetadata,
    /** Codec configuration for audio tokenization/detokenization. */
    val codec: VoxtralCodecMetadata,
    /** Audio-specific configuration. */
    val audio: VoxtralAudioConfig
)

/**
 * Configuration for the Voxtral audio codec.
 *
 * The codec uses a hybrid VQ-FSQ (Vector Quantization + Finite Scalar Quantization)
 * scheme with a convolutional + transformer architecture.
 */
public data class VoxtralCodecMetadata(
    val channels: Int = 1,
    val samplingRate: Int = 24000,
    val pretransformPatchSize: Int = 240,
    val patchProjKernelSize: Int = 7,
    val semanticCodebookSize: Int = 8192,
    val semanticDim: Int = 256,
    val acousticCodebookSize: Int = 21,
    val acousticDim: Int = 36,
    /** Codec input dim: semanticDim + acousticDim (256 + 36 = 292). */
    val inputDim: Int = 292,
    val dim: Int = 1024,
    val hiddenDim: Int = 4096,
    val nHeads: Int = 8,
    val nKVHeads: Int = 8,
    val headDim: Int = 128,
    val causal: Boolean = true,
    val qkNorm: Boolean = true,
    val qkNormEps: Float = 1e-6f,
    val normEps: Float = 0.01f,
    val layerScaleInit: Float = 0.01f,
    /** Whether convolutions use weight normalization (g/v decomposition). */
    val convWeightNorm: Boolean = true,
    /** Transformer layers per decoder stage: [2, 2, 2, 2] = 8 total. */
    val decoderTransformerLengths: List<Int> = listOf(2, 2, 2, 2),
    /** Convolution kernel sizes per decoder stage. */
    val decoderConvsKernels: List<Int> = listOf(3, 4, 4, 4),
    /** Convolution strides per decoder stage (total upsampling = product of strides). */
    val decoderConvsStrides: List<Int> = listOf(1, 2, 2, 2),
    /**
     * Sliding window sizes per transformer stage (blocks 1,3,5,7).
     * Window halves with each downsampling level (half_attn_window_upon_downsampling).
     */
    val decoderWindowSizes: List<Int> = listOf(2, 4, 8, 16)
)

/**
 * Audio-specific token and codebook configuration for Voxtral.
 */
public data class VoxtralAudioConfig(
    val semanticCodebookSize: Int = 8192,
    val acousticCodebookSize: Int = 21,
    val nAcousticCodebooks: Int = 36,
    val numCodebooks: Int = 37,
    val samplingRate: Int = 24000,
    val frameRate: Float = 12.5f,
    val codebookPattern: String = "parallel",
    val interleaveAudioTokensPerSegment: Int = 8192,
    val interleaveTextTokensPerSegment: Int = 8192,
    val inputEmbeddingConcatType: String = "sum",
    val bosTokenId: Int = 1,
    val audioTokenId: Int = 24,
    val beginAudioTokenId: Int = 25,
    val conditionDroppedTokenId: Int = 42
) {
    /** Total number of codebooks: 1 semantic + N acoustic. */
    val totalCodebooks: Int get() = 1 + nAcousticCodebooks
}

/**
 * Default Voxtral-4B configuration matching mistralai/Voxtral-4B-TTS-2603.
 */
public object VoxtralDefaults {

    public val BACKBONE: GgufDecoderMetadata = GgufDecoderMetadata(
        architecture = "voxtral_tts",
        embeddingLength = 3072,
        contextLength = 65536,
        blockCount = 26,
        headCount = 32,
        kvHeadCount = 8,
        feedForwardLength = 9216,
        ropeDimensionCount = 128,
        vocabSize = 131072
    )

    public val ACOUSTIC_MODEL: GgufDecoderMetadata = GgufDecoderMetadata(
        architecture = "voxtral_tts_acoustic",
        embeddingLength = 3072,
        contextLength = 65536,
        blockCount = 3,
        headCount = 32,
        kvHeadCount = 8,
        feedForwardLength = 9216,
        ropeDimensionCount = 128,
        vocabSize = 131072
    )

    public val CODEC: VoxtralCodecMetadata = VoxtralCodecMetadata()

    public val AUDIO: VoxtralAudioConfig = VoxtralAudioConfig()

    public val DEFAULT: VoxtralModelMetadata = VoxtralModelMetadata(
        backbone = BACKBONE,
        acousticModel = ACOUSTIC_MODEL,
        codec = CODEC,
        audio = AUDIO
    )
}
