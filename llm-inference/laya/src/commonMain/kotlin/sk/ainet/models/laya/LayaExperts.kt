package sk.ainet.models.laya

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonObject
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP32

/**
 * `expert.json` beside an expert head's `expert.safetensors`: the name the router uses, the domains
 * it was trained for, and the calibration temperatures fitted for this head (same keys and buckets
 * as `rl_agent_config.json`). Unknown keys are kept out of the way; `train` is reported as written.
 */
@Serializable
public data class LayaExpertMeta(
    val expert: String,
    val domains: List<String> = emptyList(),
    @SerialName("base_model") val baseModel: String? = null,
    val temperature: List<Double> = listOf(1.0, 1.0, 1.0),
    @SerialName("temperature_by_options") val temperatureByOptions: Map<String, Double> = emptyMap(),
    val train: JsonObject? = null,
) {
    init {
        require(expert.isNotBlank()) { "expert.json needs a non-empty expert name" }
        require(temperature.size == 3) { "temperature must hold one value per question type, got ${temperature.size}" }
    }

    public companion object {
        public fun parse(json: String): LayaExpertMeta = layaJson.decodeFromString(serializer(), json)
    }
}

/**
 * A decision head trained for a group of domains on top of a frozen base encoder, with its own
 * calibration: [head] runs on the base model's encoder output ([LayaDecisionModel.logitsFrom]),
 * [decoder] carries the expert's temperatures over the base agent's budgets.
 */
public class LayaExpert<T : DType>(
    public val meta: LayaExpertMeta,
    public val head: LayaDecisionHead<T>,
    public val decoder: LayaDecoder,
) {
    public val name: String get() = meta.expert
    public val domains: List<String> get() = meta.domains
}

/** Loads an expert head (`expert.json` + `expert.safetensors`) for a base [LayaAgent]. */
public object LayaExpertLoader {

    /** The base agent's config with the expert's temperatures: budgets and layout stay the base's. */
    public fun decoderFor(meta: LayaExpertMeta, base: LayaAgentConfig): LayaDecoder =
        LayaDecoder(base.copy(temperature = meta.temperature, temperatureByOptions = meta.temperatureByOptions))

    /** [load] from tensors already read, for callers with their own weight source. */
    public fun <T : DType> fromTensors(
        meta: LayaExpertMeta,
        raw: List<WeightTensor<T, Float>>,
        base: LayaAgent<T>,
        ctx: ExecutionContext,
        dtype: kotlin.reflect.KClass<T>,
        exactGelu: Boolean = true,
    ): LayaExpert<T> {
        val hidden = base.model.hiddenSize
        val layers = base.preprocessor.agentConfig.headLayers
        val prepared = LayaWeightPreparation.prepareHead(raw, hidden, layers, ctx, dtype)
        val head = LayaDecisionModel.buildHead(prepared, hidden, layers, dtype, exactGelu)
        return LayaExpert(meta, head, decoderFor(meta, base.preprocessor.agentConfig))
    }

    /**
     * Reads `expert.safetensors` through [weights] (the engine's SafeTensors loader; [weightForm] /
     * [weightFormFor] as in [LayaModelLoader], so [LayaWeightForms.q8_0For] packs the head's
     * projections like the base model's) and binds the head for [base]. The head must match the base
     * model's hidden size and layer count; a head trained on another base fails at binding.
     */
    public suspend fun load(
        metaJson: String,
        weights: () -> RandomAccessSource,
        base: LayaAgent<FP32>,
        ctx: ExecutionContext,
        exactGelu: Boolean = true,
        weightForm: WeightForm? = null,
        weightFormFor: ((String) -> WeightForm?)? = null,
    ): LayaExpert<FP32> {
        val meta = LayaExpertMeta.parse(metaJson)
        val raw = ArrayList<WeightTensor<FP32, Float>>()
        SafeTensorsParametersLoader(sourceProvider = weights, weightForm = weightForm, weightFormFor = weightFormFor)
            .load<FP32, Float>(ctx, FP32::class) { name, tensor ->
                raw += WeightTensor(name, tensor.shape.dimensions.toList(), tensor)
            }
        return fromTensors(meta, raw, base, ctx, FP32::class, exactGelu)
    }
}
