package sk.ainet.models.laya

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject
import java.nio.file.Files
import java.nio.file.Path

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
}
