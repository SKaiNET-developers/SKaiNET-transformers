package sk.ainet.transformers.gemma.iree

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * The FunctionGemma compiled-pipeline contract, as emitted by
 * `sk.ainet.models.functiongemma.FunctionGemmaContract.manifestJson` (`:llm-inference:functiongemma`,
 * `manifest.json` alongside the exported vmfbs). Plain-JSON parsing ONLY — this module does NOT
 * depend on `:llm-inference:functiongemma` (gemma-iree's target set — `jvm/linuxX64/linuxArm64/
 * macosArm64` — must not pick up functiongemma's JVM-only export-tooling deps), so the contract is
 * duplicated here as data, not shared as a type. [GemmaKvDecoder]'s manifest-reading constructor
 * (D3) consumes the arg/result orders and architecture constants mechanically instead of
 * hardcoding them; the historical hardcoded values remain the class defaults for callers that
 * don't have a manifest yet.
 */
@Serializable
public data class GemmaManifest(
    val contractVersion: Int,
    val nLayers: Int,
    val headDim: Int,
    val nKvHeads: Int,
    val seq: Int,
    val eot: Int,
    val slidingRopeBase: Float,
    val globalRopeBase: Float,
    val globalLayerPeriod: Int,
    val kFirstInOutput: Boolean,
    val parameterScope: String,
    val parameters: Parameters,
    val prefillArgs: List<String>,
    val prefillOutputs: List<String>,
    val withPastArgs: List<String>,
    val withPastOutputs: List<String>,
    val toolMap: Map<String, String?> = emptyMap(),
) {
    @Serializable
    public data class Parameters(val redecode: String, val prefill: String, val withPast: String)

    public companion object {
        private val json = Json { ignoreUnknownKeys = true }

        /** Parse `manifest.json` text into a [GemmaManifest]. */
        public fun parse(text: String): GemmaManifest = json.decodeFromString(serializer(), text)
    }
}
