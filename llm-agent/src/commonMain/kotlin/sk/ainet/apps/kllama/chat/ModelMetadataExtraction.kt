package sk.ainet.apps.kllama.chat

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonPrimitive

/**
 * Best-effort extraction of [ModelMetadata] from loader-side sources (#37).
 *
 * The functions here are pure and format-agnostic at the type level: callers
 * hand in whatever they already have (a GGUF key/value field map, the text of
 * a HuggingFace `tokenizer_config.json`), and get back a [ModelMetadata]
 * suitable for [ToolCallingSupportResolver] auto-detection.
 *
 * Extraction is **best-effort only** — missing or malformed input degrades to
 * empty/partial metadata, never to an exception. Metadata is a hint for
 * template/provider selection, not authoritative proof of tool-calling
 * capability (see #35).
 */
public object ModelMetadataExtraction {

    /**
     * Special-token markers that hint at tool-calling capability when they
     * appear in a chat template or the tokenizer vocabulary.
     */
    private val TOOL_HINT_MARKERS: List<String> = listOf(
        "<tool_call>",
        "<tool_response>",
        "<|tool_call|>",
        "<|python_tag|>",
        "[AVAILABLE_TOOLS]",
        "<start_of_turn>",
        "<|im_start|>",
        "<|start_header_id|>"
    )

    /**
     * Build a [ModelMetadata] from a GGUF key/value field map
     * (e.g. `StreamingGGUFReader.fields`) without loading any tensor data.
     *
     * Reads (all optional):
     * - `general.architecture` → [ModelMetadata.architecture] and the derived [ModelMetadata.family]
     * - `tokenizer.chat_template` → [ModelMetadata.chatTemplate]
     * - `tokenizer.ggml.tokens` (if present as a string collection) → scanned for
     *   known tool/chat special tokens to populate [ModelMetadata.tokenizerHints]
     */
    public fun fromGgufFields(fields: Map<String, Any?>): ModelMetadata {
        val arch = fields["general.architecture"] as? String
        val chatTemplate = fields["tokenizer.chat_template"] as? String
        val tokens = (fields["tokenizer.ggml.tokens"] as? Collection<*>)
            ?.filterIsInstance<String>()
        return ModelMetadata(
            family = familyFromArchitecture(arch),
            architecture = arch ?: "unknown",
            chatTemplate = chatTemplate,
            tokenizerHints = tokenizerHints(chatTemplate, tokens),
            sourceFormat = "gguf"
        )
    }

    /**
     * Build a [ModelMetadata] from HuggingFace-side config files that sit next
     * to safetensors checkpoints (#38). All inputs are optional file *contents*
     * (not paths — the caller reads whatever exists); malformed JSON in any of
     * them is ignored rather than thrown.
     *
     * - [tokenizerConfigJson] — `tokenizer_config.json`; reads `chat_template`
     *   (string, or HF's list-of-named-templates form where the `"default"`
     *   entry wins, else the first) and `additional_special_tokens` (strings or
     *   `{"content": ...}` objects) for hint scanning.
     * - [chatTemplateJson] — `chat_template.json` (shipped by some repos,
     *   notably VLM processors); used when `tokenizer_config.json` has no
     *   template. Accepts `{"chat_template": "..."}` or a bare JSON string.
     * - [modelConfigJson] — `config.json`; reads `model_type` (e.g. "qwen2",
     *   "llama", "gemma3") as the architecture, mapped to a family via
     *   [familyFromArchitecture].
     */
    public fun fromHuggingFaceConfig(
        tokenizerConfigJson: String? = null,
        chatTemplateJson: String? = null,
        modelConfigJson: String? = null
    ): ModelMetadata {
        val tokenizerConfig = tokenizerConfigJson?.let(::parseObjectOrNull)
        val chatTemplate = tokenizerConfig?.let(::chatTemplateFromTokenizerConfig)
            ?: chatTemplateJson?.let(::chatTemplateFromChatTemplateJson)
        val specialTokens = tokenizerConfig?.let(::additionalSpecialTokens)
        val arch = modelConfigJson?.let(::parseObjectOrNull)
            ?.get("model_type")?.stringOrNull()
        return ModelMetadata(
            family = familyFromArchitecture(arch),
            architecture = arch,
            chatTemplate = chatTemplate,
            tokenizerHints = tokenizerHints(chatTemplate, specialTokens),
            sourceFormat = "hf"
        )
    }

    /**
     * Map a GGUF `general.architecture` string to a coarse model-family
     * identifier understood by the built-in [ToolCallingSupport] providers.
     *
     * Unknown architectures pass through unchanged so custom providers can
     * still match on them; `null` stays `null`.
     */
    public fun familyFromArchitecture(architecture: String?): String? {
        val arch = architecture?.lowercase() ?: return null
        return when {
            arch.startsWith("qwen") -> "qwen"
            arch.startsWith("gemma") -> "gemma"
            arch == "llama" -> "llama"
            else -> arch
        }
    }

    /**
     * Collect tool/chat special-token hints from a chat template and/or the
     * tokenizer vocabulary. Order is stable (marker declaration order),
     * duplicates removed.
     */
    public fun tokenizerHints(
        chatTemplate: String?,
        vocabTokens: Collection<String>? = null
    ): List<String> {
        if (chatTemplate == null && vocabTokens.isNullOrEmpty()) return emptyList()
        val vocab: Set<String> = vocabTokens?.let { tokens ->
            // Only special-token-shaped entries matter; avoids holding a
            // 100k+ vocab set just to probe eight markers.
            tokens.filterTo(mutableSetOf()) { it.startsWith('<') || it.startsWith('[') }
        } ?: emptySet()
        return TOOL_HINT_MARKERS.filter { marker ->
            (chatTemplate?.contains(marker) == true) || marker in vocab
        }
    }

    // -----------------------------------------------------------------------
    // HF-side JSON helpers (all best-effort, never throw)
    // -----------------------------------------------------------------------

    private val json = Json { ignoreUnknownKeys = true }

    private fun parseObjectOrNull(text: String): JsonObject? = try {
        json.parseToJsonElement(text) as? JsonObject
    } catch (_: Exception) {
        null
    }

    private fun JsonElement.stringOrNull(): String? =
        (this as? JsonPrimitive)?.takeIf { it.isString }?.content

    /**
     * `chat_template` in `tokenizer_config.json` is either a plain string or a
     * list of `{"name": ..., "template": ...}` entries (multi-template models).
     * For the list form the `"default"` entry wins, else the first.
     */
    private fun chatTemplateFromTokenizerConfig(config: JsonObject): String? {
        return when (val tpl = config["chat_template"]) {
            is JsonPrimitive -> tpl.stringOrNull()
            is JsonArray -> {
                val entries = tpl.mapNotNull { it as? JsonObject }
                val chosen = entries.firstOrNull { it["name"]?.stringOrNull() == "default" }
                    ?: entries.firstOrNull()
                chosen?.get("template")?.stringOrNull()
            }
            else -> null
        }
    }

    /** `chat_template.json` is `{"chat_template": "..."}` or a bare JSON string. */
    private fun chatTemplateFromChatTemplateJson(text: String): String? = try {
        when (val element = json.parseToJsonElement(text)) {
            is JsonObject -> element["chat_template"]?.stringOrNull()
            is JsonPrimitive -> element.stringOrNull()
            else -> null
        }
    } catch (_: Exception) {
        null
    }

    /**
     * `additional_special_tokens` entries are plain strings or
     * `{"content": ...}` objects depending on the exporter.
     */
    private fun additionalSpecialTokens(config: JsonObject): List<String>? {
        val arr = config["additional_special_tokens"] as? JsonArray ?: return null
        return arr.mapNotNull { entry ->
            when (entry) {
                is JsonPrimitive -> entry.stringOrNull()
                is JsonObject -> entry["content"]?.stringOrNull()
                else -> null
            }
        }
    }
}
