package sk.ainet.models.gemma

import kotlinx.io.Source
import sk.ainet.apps.llm.DTypePolicyValidation
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.weights.MappingConfig
import sk.ainet.io.weights.WeightMapper
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy

/**
 * End-to-end loader that builds a [gemmaNetwork] module and populates it
 * with weights from GGUF or SafeTensors files. Phase 5a of
 * PLAN-unified-pipeline.md — feeds the unified DSL → ComputeGraph → CPU-on-JVM
 * execution path.
 *
 * Uses [LlamaGGUFNameResolver] because Gemma 4 GGUF tensor names follow the
 * standard LLaMA block layout (`blk.N.attn_*`, `blk.N.ffn_*`, etc.) and the
 * DSL module parameter names (`gate_proj`, `up_proj`, `down_proj`, `q_proj`,
 * `k_proj`, `v_proj`, `o_proj`, `attn_norm`, `ffn_norm`, `output_norm`,
 * `output`, `token_embd`) resolve to those canonical names unchanged.
 *
 * Usage:
 * ```kotlin
 * val model = GemmaNetworkLoader.fromGguf(randomAccessProvider = { ras })
 *     .load<FP32, Float>(ctx)
 * ```
 */
@OptIn(ExperimentalMemoryApi::class)
public class GemmaNetworkLoader @PublishedApi internal constructor(
    @PublishedApi internal val weightsProvider: WeightsProvider,
    @PublishedApi internal val debug: Boolean = false
) {
    /** See [sk.ainet.models.llama.LlamaNetworkLoader.dtypePolicy]. */
    public var dtypePolicy: DTypePolicy = DTypePolicy.Any
        private set

    /** See [sk.ainet.models.llama.LlamaNetworkLoader.withDtypePolicy]. */
    public fun withDtypePolicy(policy: DTypePolicy): GemmaNetworkLoader {
        // The GGUF lane honors narrow-float keep-native exactly as llama's shared loader does
        // (GemmaWeightLoader resolves keepF16Native/keepBf16Native from the policy) — the old
        // `keepNative = emptySet()` claim predated that and silently rejected `Require(BF16)`
        // on a loader that supports it (#375). The SafeTensors lane now rides the engine's
        // sharded loader (SKaiNET#1246) and keeps native under the same set.
        DTypePolicyValidation.validate(
            policy, "GemmaNetworkLoader.withDtypePolicy",
            keepNative = sk.ainet.lang.nn.dsl.decoder.DECODER_NARROW_KEEP_NATIVE,
        )
        this.dtypePolicy = policy
        return this
    }

    @PublishedApi
    internal sealed interface WeightsProvider {
        data class GgufSource(
            val sourceProvider: () -> Source
        ) : WeightsProvider

        data class GgufRandomAccess(
            val randomAccessProvider: () -> RandomAccessSource,
            val weightForm: WeightForm?
        ) : WeightsProvider

        data class SafeTensorsIndex(
            val indexPath: String
        ) : WeightsProvider

        data class Preloaded<T : DType, V>(
            val weights: GemmaWeights<T, V>
        ) : WeightsProvider
    }

    public companion object {
        /** Load from a GGUF file via sequential Source (models under 2GB, dequantized to dense FP32). */
        public fun fromGguf(
            sourceProvider: () -> Source,
            debug: Boolean = false
        ): GemmaNetworkLoader = GemmaNetworkLoader(
            WeightsProvider.GgufSource(sourceProvider), debug
        )

        /**
         * Load from a GGUF file via streaming RandomAccessSource (any size)
         * through the engine loader. Default form keeps quantized tensors
         * packed with logical `[out, in]` shapes; pass [GEMMA_DEQUANTIZE_ALL]
         * as [weightForm] for a dense FP32 load (the export/tracing path).
         */
        @kotlin.jvm.JvmName("fromGgufRandomAccess")
        public fun fromGguf(
            randomAccessProvider: () -> RandomAccessSource,
            weightForm: WeightForm? = null,
            debug: Boolean = false
        ): GemmaNetworkLoader = GemmaNetworkLoader(
            WeightsProvider.GgufRandomAccess(randomAccessProvider, weightForm), debug
        )

        /** Load from HuggingFace SafeTensors with index file. */
        public fun fromSafeTensors(
            indexPath: String,
            debug: Boolean = false
        ): GemmaNetworkLoader = GemmaNetworkLoader(
            WeightsProvider.SafeTensorsIndex(indexPath), debug
        )

        /** Build from pre-loaded [GemmaWeights]. */
        public inline fun <reified T : DType, V> fromWeights(
            ctx: ExecutionContext,
            weights: GemmaWeights<T, V>,
            debug: Boolean = false
        ): Module<T, V> = fromWeights(ctx, weights, T::class, debug)

        /**
         * Non-reified variant — explicit [dtype] so non-reified callers
         * (e.g. `GemmaIngestion<T>`) can build the DSL network without
         * propagating reification through their public API.
         */
        public fun <T : DType, V> fromWeights(
            ctx: ExecutionContext,
            weights: GemmaWeights<T, V>,
            dtype: kotlin.reflect.KClass<T>,
            debug: Boolean = false
        ): Module<T, V> = applyWeightsToNetworkNonReified(ctx, weights, dtype, debug)
    }

    /**
     * Load weights and build a fully initialized DSL network.
     */
    public suspend inline fun <reified T : DType, V> load(
        ctx: ExecutionContext,
        maxInferenceLen: Int? = null,
    ): Module<T, V> {
        val weights: GemmaWeights<T, V> = when (val wp = weightsProvider) {
            is WeightsProvider.GgufSource -> {
                val loader = GemmaWeightLoader(wp.sourceProvider, dtypePolicy = dtypePolicy)
                loader.loadToMap<T, V>(ctx)
            }
            is WeightsProvider.GgufRandomAccess -> {
                val loader = GemmaWeightLoader(
                    wp.randomAccessProvider, weightForm = wp.weightForm, dtypePolicy = dtypePolicy,
                )
                loader.loadToMapStreaming<T, V>(ctx)
            }
            is WeightsProvider.SafeTensorsIndex -> {
                // The SafeTensors lane rides the engine's ShardedSafeTensorsParametersLoader
                // (SKaiNET#1246), which keeps BF16/FP16 native under Require() exactly as the
                // GGUF lane does — the same keep-native set applies.
                DTypePolicyValidation.validate(
                    dtypePolicy, "GemmaNetworkLoader(SafeTensors)",
                    keepNative = sk.ainet.lang.nn.dsl.decoder.DECODER_NARROW_KEEP_NATIVE,
                )
                val loader = GemmaSafeTensorsLoader(wp.indexPath, dtypePolicy = dtypePolicy)
                @Suppress("UNCHECKED_CAST")
                loader.loadToMap(ctx, T::class) as GemmaWeights<T, V>
            }
            is WeightsProvider.Preloaded<*, *> -> {
                @Suppress("UNCHECKED_CAST")
                wp.weights as GemmaWeights<T, V>
            }
        }

        return applyWeightsToNetwork(ctx, weights, maxInferenceLen)
    }

    @PublishedApi
    internal inline fun <reified T : DType, V> applyWeightsToNetwork(
        ctx: ExecutionContext,
        weights: GemmaWeights<T, V>,
        maxInferenceLen: Int? = null,
    ): Module<T, V> = applyWeightsToNetworkNonReified(ctx, weights, T::class, debug, maxInferenceLen)
}

/** Shared non-reified impl used by both the inline-reified companion helpers
 *  and the DSL-ingestion entry points that only have a runtime `KClass<T>`. */
@PublishedApi
internal fun <T : DType, V> applyWeightsToNetworkNonReified(
    ctx: ExecutionContext,
    weights: GemmaWeights<T, V>,
    dtype: kotlin.reflect.KClass<T>,
    debug: Boolean,
    maxInferenceLen: Int? = null,
): Module<T, V> {
    // Enable optional Gemma 4 features iff the checkpoint actually carries
    // their weights. Real Gemma 4 GGUFs do; synthetic toy-model tests do not,
    // and forcing them on would make the WeightMapper strict-check fail.
    val hasQKNorm = weights.tensors.keys.any { it.endsWith(".attn_q_norm.weight") }
    val hasSandwichNorms = weights.tensors.keys.any { it.endsWith(".post_attention_norm.weight") }
    // Phase 5f.4 + 5f.5: auto-detect layer_output_scale and PLE on weight presence.
    // Authoritative HF `Gemma4TextDecoderLayer.forward` (transformers 5.6.0) is
    // pinned in the sibling `gemma4-research/findings/` scratch dir — the PLE
    // math in [PerLayerEmbedding] + [PerLayerInputBlockHook] and the scalar
    // multiply in [LayerScalarMul] match the reference. The older diagnostic
    // that gated these features off was taken before `final_logit_softcapping`
    // was wired into [GemmaModel]; unbounded post-PLE logits were responsible
    // for the "Hi pełni" degenerate output, not PLE itself.
    val hasLayerOutputScale = weights.tensors.keys.any { it.endsWith(".layer_output_scale.weight") }
    val hasPle = weights.tensors.keys.any { it == "per_layer_token_embd.weight" }
    val model = gemmaNetwork<T, V>(
        weights.metadata,
        dtype,
        maxInferenceLen = maxInferenceLen ?: minOf(weights.metadata.contextLength, 4096),
        qkNorm = hasQKNorm,
        sandwichNorms = hasSandwichNorms,
        layerOutputScale = hasLayerOutputScale,
        ple = hasPle
    )

    val weightTensors = weights.tensors.map { (name, tensor) ->
        WeightTensor(
            name = name,
            shape = tensor.shape.dimensions.toList(),
            tensor = tensor
        )
    }

    val config = MappingConfig(
        usePathBasedMatching = false,
        fallbackToShapeMatching = false,
        debug = debug,
        nameResolver = GemmaGGUFNameResolver()
    )

    val result = WeightMapper.applyWeights(model, weightTensors, config)

    // Gemma has no bias tensors. Allow unmapped bias params from dense().
    val unmappedNonBias = result.missingParams.filter { !it.contains(".bias") }
    require(unmappedNonBias.isEmpty()) {
        buildString {
            appendLine("Failed to map ${unmappedNonBias.size} weight parameters:")
            unmappedNonBias.forEach { appendLine("  - $it") }
            if (result.unusedTensors.isNotEmpty()) {
                appendLine("Unused tensors (${result.unusedTensors.size}):")
                result.unusedTensors.take(10).forEach { appendLine("  - $it") }
            }
        }.trim()
    }

    return model
}
