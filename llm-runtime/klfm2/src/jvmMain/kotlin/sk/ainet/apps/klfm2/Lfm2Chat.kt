package sk.ainet.apps.klfm2

import kotlinx.serialization.json.*
import sk.ainet.models.lfm2.Lfm2Config
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.frontend.tokenizer.TextFrontEnds
import java.io.File
import java.security.MessageDigest

/** Supported checkpoint-template input: text messages, with optional assistant thinking. */
data class Lfm2ChatMessage(val role: String, val content: String, val thinking: String? = null)

/** Distinct skip-special policies observed in the checkpoint and pinned Transformers 5 reference. */
enum class Lfm2DecodingPolicy { CHECKPOINT, HUGGING_FACE_5_COMPAT }

/**
 * Runtime frontend for the pinned LFM2.5-350M checkpoint's text-conversation subset.
 * This is a checked Kotlin implementation of that template, not a general Jinja engine.
 * Tools, multimodal content and arbitrary template revisions require separate support.
 */
class Lfm2TextFrontend internal constructor(
    private val tokenizer: Tokenizer,
    private val bosToken: String,
    val bosTokenId: Int,
    val stopTokenIds: Set<Int>,
    val vocabularySize: Int,
    val metadataHashes: Map<String, String>,
    val effectiveTokenizerSha256: String,
    private val assignedTokenIds: Set<Int>,
    private val checkpointSpecialTokenIds: Set<Int>,
    private val hf5CompatSpecialTokenIds: Set<Int>,
) {
    val tokenizerCompatibility: String = "jvm_unicode_split_character_classes"
    val templateImplementation: String = "pinned_lfm25_text_subset_v1"
    /** Assigned tokenizer IDs can occupy fewer rows than the model's padded output tensor. */
    val tokenizerVocabularySize: Int = assignedTokenIds.size
    val maximumAssignedTokenId: Int = assignedTokenIds.maxOrNull() ?: -1
    val checkpointSpecialTokenCount: Int = checkpointSpecialTokenIds.size
    val hf5CompatSpecialTokenCount: Int = hf5CompatSpecialTokenIds.size
    val hf5PromotedSpecialTokenIds: Set<Int> = hf5CompatSpecialTokenIds - checkpointSpecialTokenIds

    fun render(
        messages: List<Lfm2ChatMessage>,
        addGenerationPrompt: Boolean = true,
        preserveThinking: Boolean = false,
    ): String = renderLfm2TextChat(messages, bosToken, addGenerationPrompt, preserveThinking)

    /** Template output already contains BOS and message delimiters; never add another BOS/EOS. */
    fun encodeMessages(
        messages: List<Lfm2ChatMessage>,
        addGenerationPrompt: Boolean = true,
        preserveThinking: Boolean = false,
    ): IntArray = encode(render(messages, addGenerationPrompt, preserveThinking))

    fun encode(text: String): IntArray = tokenizer.encode(text, addSpecialTokens = false).also { ids ->
        require(ids.all { it in 0 until vocabularySize }) { "Tokenizer produced an ID outside the model vocabulary" }
        require(ids.all { it in assignedTokenIds }) { "Tokenizer produced an ID absent from its declared inventory" }
    }

    /**
     * CHECKPOINT preserves added tokens whose source special flag is false. The pinned HF 5
     * constructor re-registers every added token as special, including thinking/tool markers
     * and the ordinary pieces "Mathias" and "python"; HUGGING_FACE_5_COMPAT reproduces that
     * observed policy explicitly. Neither policy treats the ordinary piece "null" as special.
     */
    fun decode(
        ids: IntArray,
        skipSpecialTokens: Boolean = true,
        policy: Lfm2DecodingPolicy = Lfm2DecodingPolicy.CHECKPOINT,
    ): String {
        require(ids.all { it in 0 until vocabularySize }) { "Cannot decode an ID outside the model vocabulary" }
        require(ids.all { it in assignedTokenIds }) { "Cannot decode an unassigned tokenizer ID or padded model output row" }
        val special = when (policy) {
            Lfm2DecodingPolicy.CHECKPOINT -> checkpointSpecialTokenIds
            Lfm2DecodingPolicy.HUGGING_FACE_5_COMPAT -> hf5CompatSpecialTokenIds
        }
        val kept = if (skipSpecialTokens) ids.filterNot { it in special }.toIntArray() else ids
        // Filter whole token arrays before decoding, so remaining UTF-8 byte pieces join correctly.
        return if (kept.isEmpty()) "" else tokenizer.decode(kept, skipSpecialTokens = false)
    }
}

private data class MetadataPin(val size: Long, val sha256: String)

// LiquidAI/LFM2.5-350M @ 9e6c6ccf47cd318696e137d381a7ded8fe4df09f.
private val textMetadataPins = linkedMapOf(
    "config.json" to MetadataPin(1281, "720b43d6ddc2ed25be23eed355aefcf342434a176dedad23dbe0a5e3ac24bbb8"),
    "generation_config.json" to MetadataPin(134, "0ec7d228c4de7e3453e98bf6910c7eb7841e87fede63ea11c0f75a5c64e1c3c3"),
    "tokenizer_config.json" to MetadataPin(595, "3701f370c70034d06e28947ff51c1063983393319f429e3ce58f06a45fae67cc"),
    "chat_template.jinja" to MetadataPin(5487, "ba551d58630afa3190b1be3602e28301f3d2e9bbac978dfc49d6d825171648b6"),
    "tokenizer.json" to MetadataPin(4733389, "df1d8d5ec5d091b460562ffd545e4a5e91d17d4a0db7ebe733be34ed374377bd"),
    "LICENSE" to MetadataPin(10574, "4d28ca14dedc0b3d0fcc2b3339f0e79931faa33874f3d24f522183a8fc70068c"),
)

internal fun requirePinnedLfm2TextMetadata(name: String, bytes: ByteArray): String {
    val pin = requireNotNull(textMetadataPins[name]) { "Unknown LFM2 text metadata file: $name" }
    require(bytes.size.toLong() == pin.size && textSha256(bytes) == pin.sha256) {
        "Unsupported or changed LFM2 text metadata: $name; this renderer requires the pinned checkpoint"
    }
    return pin.sha256
}

/** Verify original metadata bytes before creating the Unicode-compatible in-memory tokenizer. */
fun loadLfm2TextFrontend(directory: File): Lfm2TextFrontend {
    require(directory.isDirectory) { "Model metadata directory does not exist" }
    val originals = linkedMapOf<String, String>()
    val hashes = linkedMapOf<String, String>()
    for ((name, pin) in textMetadataPins) {
        val file = File(directory, name)
        require(file.isFile && file.length() == pin.size) { "Missing or incorrectly sized LFM2 metadata: $name" }
        val bytes = file.readBytes()
        hashes[name] = requirePinnedLfm2TextMetadata(name, bytes)
        originals[name] = bytes.toString(Charsets.UTF_8)
    }
    val config = Lfm2Config.parse(originals.getValue("config.json"))
    val special = resolveLfm2GenerationTokens(config.bosTokenId, config.eosTokenIds,
        originals.getValue("generation_config.json"), config.vocabularySize)
    val tokenizerConfig = Json.parseToJsonElement(originals.getValue("tokenizer_config.json")).jsonObject
    val bosToken = tokenizerConfig.getValue("bos_token").jsonPrimitive.content
    val effective = unicodeLfm2TokenizerSpecification(originals.getValue("tokenizer.json"))
    val tokenizer = TextFrontEnds.fromTokenizerJson(effective, originals.getValue("tokenizer_config.json"))
    val inventory = lfm2TokenizerInventory(originals.getValue("tokenizer.json"), config.vocabularySize)
    val decoding = lfm2DecodingTokenSets(originals.getValue("tokenizer.json"))
    require(tokenizer.vocabSize == inventory.size) { "Tokenizer size differs from its declared unique ID inventory" }
    for ((id, piece) in inventory) {
        require(tokenizer.vocabulary.idToToken(id) == piece && tokenizer.vocabulary.tokenToId(piece) == id) {
            "Tokenizer token mapping differs from checkpoint metadata at ID $id"
        }
    }
    for (id in 0 until config.vocabularySize) {
        if (id !in inventory) require(tokenizer.vocabulary.idToToken(id) == null) { "Tokenizer exposes undeclared ID $id" }
    }
    require(special.bos in inventory && special.eos.all { it in inventory }) { "Generation BOS/EOS must be assigned tokenizer IDs" }
    require(tokenizer.encode(bosToken, addSpecialTokens = false).contentEquals(intArrayOf(special.bos))) {
        "Generation BOS ID differs from the checkpoint template token"
    }
    val eosToken = tokenizerConfig.getValue("eos_token").jsonPrimitive.content
    val eosIds = tokenizer.encode(eosToken, addSpecialTokens = false)
    require(eosIds.size == 1 && eosIds[0] in special.eos) { "Checkpoint EOS token is missing from generation stop IDs" }
    return Lfm2TextFrontend(tokenizer, bosToken, special.bos, special.eos.toSet(),
        config.vocabularySize, hashes.toMap(), textSha256(effective.toByteArray(Charsets.UTF_8)), inventory.keys.toSet(),
        decoding.checkpoint, decoding.hf5Compat)
}

internal data class Lfm2DecodingTokenSets(val checkpoint: Set<Int>, val hf5Compat: Set<Int>)

/** Applicable only with this checkpoint's byte-verified tokenizer and pinned HF reference policy. */
internal fun lfm2DecodingTokenSets(original: String): Lfm2DecodingTokenSets {
    val added = Json.parseToJsonElement(original).jsonObject.getValue("added_tokens").jsonArray
    val checkpoint = linkedSetOf<Int>()
    val hf5Compat = linkedSetOf<Int>()
    for (element in added) {
        val token = element.jsonObject
        val id = token.getValue("id").jsonPrimitive.int
        val special = token.getValue("special").jsonPrimitive.boolean
        hf5Compat.add(id)
        if (special) checkpoint.add(id)
    }
    return Lfm2DecodingTokenSets(checkpoint.toSet(), hf5Compat.toSet())
}

/** Merge base and added-token mappings by identity; the model may have unused padded rows. */
internal fun lfm2TokenizerInventory(original: String, modelVocabularySize: Int): Map<Int, String> {
    require(modelVocabularySize > 0)
    val root = Json.parseToJsonElement(original).jsonObject
    val model = root.getValue("model").jsonObject
    require(model.getValue("type").jsonPrimitive.content == "BPE") { "Expected pinned BPE tokenizer model" }
    val byId = linkedMapOf<Int, String>()
    val byPiece = mutableMapOf<String, Int>()
    fun register(id: Int, piece: String, allowExisting: Boolean) {
        require(id in 0 until modelVocabularySize) { "Tokenizer ID $id is outside the model vocabulary" }
        require(piece.isNotEmpty()) { "Empty tokenizer piece is unsupported" }
        require(id !in byId || (allowExisting && byId[id] == piece)) { "Conflicting tokenizer pieces at ID $id" }
        require(piece !in byPiece || (allowExisting && byPiece[piece] == id)) { "Conflicting tokenizer IDs for piece $piece" }
        byId[id] = piece; byPiece[piece] = id
    }
    for ((piece, value) in model.getValue("vocab").jsonObject) {
        require(!value.jsonPrimitive.isString) { "Tokenizer IDs must be integers" }
        register(requireNotNull(value.jsonPrimitive.intOrNull) { "Invalid tokenizer ID" }, piece, false)
    }
    for (element in root.getValue("added_tokens").jsonArray) {
        val added = element.jsonObject
        val value = added.getValue("id").jsonPrimitive
        val piece = added.getValue("content").jsonPrimitive
        require(!value.isString && piece.isString) { "Added token requires integer ID and string content" }
        register(requireNotNull(value.intOrNull) { "Invalid added tokenizer ID" }, piece.content, true)
    }
    require(byId.isNotEmpty()) { "Tokenizer inventory is empty" }
    return byId.toMap()
}

internal data class Lfm2GenerationTokens(val bos: Int, val eos: Set<Int>)

/** A present generation value overrides model configuration; an explicit null is unsupported. */
internal fun resolveLfm2GenerationTokens(
    modelBos: Int?, modelEos: Set<Int>, generationJson: String, vocabulary: Int,
): Lfm2GenerationTokens {
    require(vocabulary > 0)
    val generation = Json.parseToJsonElement(generationJson).jsonObject
    val bos = if ("bos_token_id" in generation) {
        requireNotNull(generation.getValue("bos_token_id").jsonPrimitive.intOrNull) { "Invalid generation bos_token_id" }
    } else requireNotNull(modelBos) { "A BOS token is required by this chat template" }
    val eos = if ("eos_token_id" in generation) {
        when (val value = generation.getValue("eos_token_id")) {
            is JsonArray -> value.map { requireNotNull(it.jsonPrimitive.intOrNull) { "Invalid generation eos_token_id" } }.toSet()
            else -> setOf(requireNotNull(value.jsonPrimitive.intOrNull) { "Invalid generation eos_token_id" })
        }
    } else modelEos
    require(bos in 0 until vocabulary && eos.isNotEmpty() && eos.all { it in 0 until vocabulary }) {
        "Generation BOS/EOS IDs must be inside the model vocabulary, with at least one EOS"
    }
    return Lfm2GenerationTokens(bos, eos.toSet())
}

internal fun renderLfm2TextChat(
    messages: List<Lfm2ChatMessage>, bosToken: String,
    addGenerationPrompt: Boolean = true, preserveThinking: Boolean = false,
): String {
    require(messages.isNotEmpty()) { "At least one chat message is required" }
    messages.forEachIndexed { index, message ->
        require(message.role in setOf("system", "user", "assistant")) { "Unsupported text chat role: ${message.role}" }
        require(message.role != "system" || index == 0) { "Only a leading system message is supported" }
        require(message.role == "assistant" || message.thinking == null) { "Thinking is only supported on assistant messages" }
    }
    val remaining = if (messages.first().role == "system") messages.drop(1) else messages
    val lastAssistant = remaining.indexOfLast { it.role == "assistant" }
    return buildString {
        append(bosToken)
        if (messages.first().role == "system" && messages.first().content.isNotEmpty()) {
            append("<|im_start|>system\n"); append(messages.first().content); append("<|im_end|>\n")
        }
        remaining.forEachIndexed { index, message ->
            append("<|im_start|>"); append(message.role); append('\n')
            if (message.role == "assistant") {
                val keepThinking = preserveThinking || index == lastAssistant
                if (keepThinking && !message.thinking.isNullOrEmpty()) {
                    append("<think>"); append(message.thinking); append("</think>")
                }
                val content = if (!keepThinking && "</think>" in message.content) {
                    message.content.substringAfterLast("</think>").trim(::pythonWhitespace)
                } else message.content
                // The template's CONTINUE_FINAL_MESSAGE_TAG remove/reappend is an identity
                // for text-only messages (tool calls are outside this implementation).
                append(content)
            } else append(message.content)
            append("<|im_end|>\n")
        }
        if (addGenerationPrompt) append("<|im_start|>assistant\n")
    }
}

/** Python str.strip/Jinja trim whitespace, including NEL and the four ASCII separators. */
private fun pythonWhitespace(c: Char): Boolean = c in '\u0009'..'\u000d' || c in '\u001c'..'\u0020' ||
    c == '\u0085' || c == '\u00a0' || c == '\u1680' || c in '\u2000'..'\u200a' ||
    c == '\u2028' || c == '\u2029' || c == '\u202f' || c == '\u205f' || c == '\u3000'

/** Rust regex character classes are Unicode-aware; Java requires explicit U semantics. */
internal fun unicodeLfm2TokenizerSpecification(original: String): String {
    val root = Json.parseToJsonElement(original).jsonObject
    val pre = root.getValue("pre_tokenizer").jsonObject
    require(pre.getValue("type").jsonPrimitive.content == "Sequence") { "Expected pinned Sequence pre-tokenizer" }
    var changes = 0
    val elements = pre.getValue("pretokenizers").jsonArray.map { element ->
        val item = element.jsonObject
        if (item.getValue("type").jsonPrimitive.content != "Split") element else {
            val pattern = item.getValue("pattern").jsonObject
            val regex = pattern.getValue("Regex").jsonPrimitive.content
            changes++
            JsonObject(item + ("pattern" to JsonObject(pattern + ("Regex" to JsonPrimitive("(?U)$regex")))))
        }
    }
    require(changes == 1) { "Expected exactly one Split regex in the pinned tokenizer" }
    return JsonObject(root + ("pre_tokenizer" to JsonObject(pre + ("pretokenizers" to JsonArray(elements))))).toString()
}

private fun textSha256(bytes: ByteArray): String = MessageDigest.getInstance("SHA-256").digest(bytes).joinToString("") { "%02x".format(it) }
