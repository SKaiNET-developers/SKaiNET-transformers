package sk.ainet.models.laya

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import java.nio.file.Files
import java.nio.file.Path
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonPrimitive

/**
 * Golden fixtures of the Laya JVM port (Apache-2.0, see `laya-fixtures/NOTICE.md`) and the
 * checkpoint directory the gated tests need. `LAYA_MULTILINGUAL_PATH` points at a directory with
 * `rl_agent_config.json`, `encoder/config.json`, `tokenizer/tokenizer.json` and
 * `tokenizer/tokenizer_config.json`; the weights are not needed for the preprocessing tests.
 */
internal object LayaFixtures {
    const val CHECKPOINT_ENV = "LAYA_MULTILINGUAL_PATH"

    fun resource(name: String): JsonObject {
        val stream = LayaFixtures::class.java.getResourceAsStream("/laya-fixtures/$name")
            ?: error("fixture $name missing from test resources")
        return Json.parseToJsonElement(stream.use { it.readBytes().decodeToString() }).jsonObject
    }

    fun checkpointDir(): Path? {
        val raw = System.getenv(CHECKPOINT_ENV)?.trim().orEmpty()
        if (raw.isEmpty()) {
            println("[skip] $CHECKPOINT_ENV not set; the multilingual tokenizer files are needed for this test")
            return null
        }
        val dir = Path.of(raw)
        if (!Files.exists(dir.resolve("tokenizer").resolve("tokenizer.json"))) {
            println("[skip] $dir has no tokenizer/tokenizer.json")
            return null
        }
        return dir
    }

    fun preprocessor(): LayaPreprocessor? = checkpointDir()?.let { LayaPreprocessor.fromFiles(LayaCheckpointFiles.read(it)) }

    fun ints(element: JsonElement?): List<Int> = (element as? kotlinx.serialization.json.JsonArray)?.map { it.jsonObjectOrPrimitiveInt() } ?: emptyList()

    private fun JsonElement.jsonObjectOrPrimitiveInt(): Int = (this as kotlinx.serialization.json.JsonPrimitive).content.toInt()

    /** The public request shape (`type`, `instructions`, `criteria`, `labels`), normalised as `Agent._to_internal` does. */
    fun questionOf(q: JsonObject): LayaQuestion {
        val insElement = q["instructions"]!!
        val ins = if (insElement is JsonPrimitive && insElement.isString) insElement.content else PythonJson.dumps(insElement)
        val crit = q["criteria"]
        return when (q["type"]!!.jsonPrimitive.content) {
            "choice" -> when (crit) {
                is JsonArray -> LayaQuestion.Choice(ins, crit.map { it.jsonPrimitive.content to null })
                is JsonObject -> LayaQuestion.Choice(ins, crit.entries.map { (k, v) -> k to v.takeIf { it !is JsonNull } })
                else -> error("choice without criteria")
            }
            "score" -> LayaQuestion.Score(ins, (crit as JsonArray).toList())
            "noul" -> {
                val c = crit as? JsonObject
                val labels = (q["labels"] as? JsonObject)?.let { it["false"]!!.jsonPrimitive.content to it["true"]!!.jsonPrimitive.content }
                fun get(key: String): JsonElement? = c?.entries?.firstOrNull { it.key.lowercase() == key }?.value?.takeIf { it !is JsonNull }
                LayaQuestion.Noul(ins, get("false"), get("true"), labels)
            }
            else -> error("unknown type ${q["type"]}")
        }
    }
}
