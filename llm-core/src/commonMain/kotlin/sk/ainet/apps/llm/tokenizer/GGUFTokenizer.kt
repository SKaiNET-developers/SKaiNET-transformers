package sk.ainet.apps.llm.tokenizer

import kotlinx.io.Source
import kotlinx.io.buffered
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.llm.TokenizerStrategy
import sk.ainet.apps.llm.TokenizerType
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.GGUFReader
import sk.ainet.io.gguf.ReaderField
import sk.ainet.io.gguf.StreamingGGUFReader

/**
 * Tokenizer that extracts vocabulary from GGUF file metadata.
 * Supports decoding (token ID -> string) and basic BPE encoding (string -> token IDs).
 *
 * Automatically detects tokenizer type (SentencePiece, BPE, WordPiece) from GGUF
 * metadata and uses the appropriate preprocessing strategy.
 */
class GGUFTokenizer private constructor(
    private val vocab: List<String>,
    private val scores: FloatArray,
    private val _bosTokenId: Int,
    private val _eosTokenId: Int,
    private val unkTokenId: Int,
    private val strategy: TokenizerStrategy,
    private val tokenTypes: IntArray? = null,
    /**
     * Whether SentencePiece's leading word-boundary marker (`▁`) is prepended
     * to the input text on `encode`. Per GGUF metadata key
     * `tokenizer.ggml.add_space_prefix`. Llama-family checkpoints set this to
     * `true` (the historical default). Gemma 4 sets it to `false` — when
     * unset, our default of `true` would tokenise "Hi" as `[18428]` (= `▁Hi`)
     * instead of the correct `[10979]` (= `Hi`).
     */
    private val addSpacePrefix: Boolean = true,
) : Tokenizer {

    companion object {
        private const val DEFAULT_BOS_TOKEN_ID = 1
        private const val DEFAULT_EOS_TOKEN_ID = 2
        private const val DEFAULT_UNK_TOKEN_ID = 0
        private const val DEFAULT_ADD_SPACE_PREFIX = true

        // GGUF token_type values (per llama.cpp convention).
        // CONTROL marks atomic special tokens like <|begin_of_text|>, <|eot_id|>, <|im_start|>.
        // USER_DEFINED marks visible-but-atomic tokens; e.g. Gemma 4's
        // `<|tool_call>` / `<tool_call|>` / `<|tool_response>` / `<tool_response|>`
        // are USER_DEFINED specifically so the chat parser can read them
        // (see convert_hf_to_gguf.py Gemma4Model.set_vocab). Both must be
        // emitted as a single token id on encode — otherwise multi-turn
        // prompts that include tool responses byte-fragment, and the model
        // sees a different prompt structure than what it was trained on.
        private const val TOKEN_TYPE_CONTROL = 3
        private const val TOKEN_TYPE_USER_DEFINED = 4

        /** Test-only factory; do not use from production code. */
        internal fun forTesting(
            vocab: List<String>,
            scores: FloatArray,
            bosTokenId: Int,
            eosTokenId: Int,
            unkTokenId: Int,
            strategy: TokenizerStrategy,
            tokenTypes: IntArray? = null
        ): GGUFTokenizer = GGUFTokenizer(vocab, scores, bosTokenId, eosTokenId, unkTokenId, strategy, tokenTypes)

        /**
         * Create a tokenizer from a HuggingFace tokenizer.json string.
         *
         * Parses the "model.vocab" and "model.merges" sections to build
         * a BPE tokenizer compatible with LLaMA 3 models.
         * Also reads "added_tokens" for BOS/EOS token IDs.
         */
        fun fromTokenizerJson(json: String, debug: Boolean = false): GGUFTokenizer {
            // --- Parse vocab: {"token": id, ...} → List<String> indexed by id ---
            val vocabStart = json.indexOf("\"vocab\"")
            if (vocabStart < 0) error("tokenizer.json: no \"vocab\" section found")
            val vocabBraceStart = json.indexOf('{', vocabStart + 7)
            if (vocabBraceStart < 0) error("tokenizer.json: malformed vocab section")
            val vocabBraceEnd = findMatchingBrace(json, vocabBraceStart)

            val vocabMap = mutableMapOf<String, Int>()
            val vocabContent = json.substring(vocabBraceStart + 1, vocabBraceEnd)
            val vocabPattern = Regex("\"((?:[^\"\\\\]|\\\\.)*)\"\\s*:\\s*(\\d+)")
            for (match in vocabPattern.findAll(vocabContent)) {
                val token = unescapeJsonString(match.groupValues[1])
                val id = match.groupValues[2].toInt()
                vocabMap[token] = id
            }

            if (debug) println("DEBUG: Parsed ${vocabMap.size} vocab entries")

            // Build vocab list indexed by id
            val maxId = vocabMap.values.maxOrNull() ?: 0

            // --- Parse added_tokens for special tokens and to extend vocab ---
            var bosTokenId = DEFAULT_BOS_TOKEN_ID
            var eosTokenId = DEFAULT_EOS_TOKEN_ID
            var unkTokenId = DEFAULT_UNK_TOKEN_ID

            // Track added-token IDs so we can mark them as CONTROL in
            // tokenTypes — that's what flips the atomic-special-token branch
            // in `encode()` on. Without this, `<bos>`, `<|turn>`, `<turn|>`
            // etc. get split into per-character BPE pieces and the model
            // sees a mangled prompt instead of the chat-template control
            // grammar it was trained on.
            val addedTokenIds = mutableListOf<Int>()
            val addedTokensStart = json.indexOf("\"added_tokens\"")
            if (addedTokensStart >= 0) {
                val arrStart = json.indexOf('[', addedTokensStart)
                if (arrStart >= 0) {
                    val arrEnd = findMatchingBracket(json, arrStart)
                    val addedContent = json.substring(arrStart, arrEnd + 1)
                    val idPattern = Regex("\"id\"\\s*:\\s*(\\d+)")
                    val contentPattern = Regex("\"content\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"")
                    // Parse each added token object
                    val objPattern = Regex("\\{[^}]+\\}")
                    for (objMatch in objPattern.findAll(addedContent)) {
                        val obj = objMatch.value
                        val idMatch = idPattern.find(obj)
                        val contentMatch = contentPattern.find(obj)
                        if (idMatch != null && contentMatch != null) {
                            val id = idMatch.groupValues[1].toInt()
                            val content = unescapeJsonString(contentMatch.groupValues[1])
                            vocabMap[content] = id
                            addedTokenIds += id
                            when {
                                content.contains("begin_of_text") || content == "<s>" -> bosTokenId = id
                                content.contains("end_of_text") || content == "</s>" -> eosTokenId = id
                                content == "<bos>" -> bosTokenId = id
                                content == "<eos>" -> eosTokenId = id
                                content == "<unk>" -> unkTokenId = id
                            }
                        }
                    }
                }
            }

            val totalVocabSize = maxOf(maxId + 1, (vocabMap.values.maxOrNull() ?: 0) + 1)
            val vocab = MutableList(totalVocabSize) { "<unk>" }
            for ((token, id) in vocabMap) {
                if (id < vocab.size) vocab[id] = token
            }

            // Mark every added token (BOS/EOS/PAD plus chat-template specials
            // like `<|turn>`, tool delimiters, etc.) as TOKEN_TYPE_CONTROL so
            // the encoder treats them atomically. The HF `tokenizer.json`
            // doesn't carry the GGUF `tokenizer.ggml.token_type` array; this
            // reconstructs the equivalent from `added_tokens`.
            val tokenTypes: IntArray? = if (addedTokenIds.isEmpty()) null else {
                IntArray(totalVocabSize).also { arr ->
                    for (id in addedTokenIds) {
                        if (id in arr.indices) arr[id] = TOKEN_TYPE_CONTROL
                    }
                }
            }

            if (debug) println("DEBUG: Total vocab size = ${vocab.size}, BOS=$bosTokenId, EOS=$eosTokenId")

            // --- Parse merges: ["tok1 tok2", ...] → scores (earlier merge = higher score) ---
            val scores = FloatArray(vocab.size) { 0f }
            val mergesStart = json.indexOf("\"merges\"")
            if (mergesStart >= 0) {
                val mergesArrStart = json.indexOf('[', mergesStart)
                if (mergesArrStart >= 0) {
                    val mergesArrEnd = findMatchingBracket(json, mergesArrStart)
                    val mergesContent = json.substring(mergesArrStart + 1, mergesArrEnd)
                    val mergePattern = Regex("\"((?:[^\"\\\\]|\\\\.)*)\"")
                    var mergeRank = 0
                    for (match in mergePattern.findAll(mergesContent)) {
                        val merge = unescapeJsonString(match.groupValues[1])
                        val parts = merge.split(' ', limit = 2)
                        if (parts.size == 2) {
                            val merged = parts[0] + parts[1]
                            val mergedId = vocabMap[merged]
                            if (mergedId != null && mergedId < scores.size) {
                                // Higher score = higher merge priority (earlier in list)
                                scores[mergedId] = (1_000_000 - mergeRank).toFloat()
                            }
                            mergeRank++
                        }
                    }
                    if (debug) println("DEBUG: Parsed $mergeRank merges")
                }
            }

            val strategy = BPEStrategy
            println("Tokenizer: BPE (from tokenizer.json, vocab=${vocab.size}, special=${addedTokenIds.size})")

            return GGUFTokenizer(vocab, scores, bosTokenId, eosTokenId, unkTokenId, strategy, tokenTypes)
        }

        private fun findMatchingBrace(s: String, start: Int): Int {
            var depth = 0
            var inStr = false
            var i = start
            while (i < s.length) {
                val c = s[i]
                when {
                    inStr -> { if (c == '"') inStr = false; if (c == '\\') i++ }
                    c == '"' -> inStr = true
                    c == '{' -> depth++
                    c == '}' -> { depth--; if (depth == 0) return i }
                }
                i++
            }
            return s.length - 1
        }

        private fun findMatchingBracket(s: String, start: Int): Int {
            var depth = 0
            var inStr = false
            var i = start
            while (i < s.length) {
                val c = s[i]
                when {
                    inStr -> { if (c == '"') inStr = false; if (c == '\\') i++ }
                    c == '"' -> inStr = true
                    c == '[' -> depth++
                    c == ']' -> { depth--; if (depth == 0) return i }
                }
                i++
            }
            return s.length - 1
        }

        private fun unescapeJsonString(s: String): String {
            val sb = StringBuilder()
            var i = 0
            while (i < s.length) {
                if (s[i] == '\\' && i + 1 < s.length) {
                    when (s[i + 1]) {
                        '"' -> { sb.append('"'); i += 2 }
                        '\\' -> { sb.append('\\'); i += 2 }
                        '/' -> { sb.append('/'); i += 2 }
                        'n' -> { sb.append('\n'); i += 2 }
                        'r' -> { sb.append('\r'); i += 2 }
                        't' -> { sb.append('\t'); i += 2 }
                        'u' -> {
                            if (i + 5 < s.length) {
                                val cp = s.substring(i + 2, i + 6).toInt(16)
                                sb.append(cp.toChar())
                                i += 6
                            } else { sb.append(s[i]); i++ }
                        }
                        else -> { sb.append(s[i]); i++ }
                    }
                } else { sb.append(s[i]); i++ }
            }
            return sb.toString()
        }

        /**
         * Create a tokenizer by reading GGUF metadata from a source.
         * Only reads metadata (not tensor data) for efficiency.
         */
        fun fromSource(source: Source, debug: Boolean = false): GGUFTokenizer {
            val reader = source.buffered().use { src ->
                GGUFReader(src, loadTensorData = false)
            }
            return fromGGUF(reader, debug)
        }

        /**
         * Create a tokenizer from GGUF reader fields.
         */
        fun fromGGUF(reader: GGUFReader, debug: Boolean = false): GGUFTokenizer {
            val fields = reader.fields

            // Extract vocabulary tokens
            val tokensField = fields["tokenizer.ggml.tokens"]
                ?: error("GGUF file missing tokenizer.ggml.tokens field")
            val vocab = extractStringArray(tokensField)

            if (debug) {
                println("DEBUG: Vocab size = ${vocab.size}")
                println("DEBUG: First 10 tokens:")
                vocab.take(10).forEachIndexed { idx, token ->
                    val bytes = token.encodeToByteArray()
                    val hexStr = bytes.joinToString(" ") { b ->
                        val hex = (b.toInt() and 0xFF).toString(16).uppercase()
                        if (hex.length == 1) "0$hex" else hex
                    }
                    println("  [$idx] = '$token' (bytes: $hexStr)")
                }
                println("DEBUG: Tokens around index 1000:")
                vocab.drop(1000).take(5).forEachIndexed { idx, token ->
                    println("  [${1000 + idx}] = '$token'")
                }
            }

            // Extract BPE scores (used for merge priority during encoding)
            val scoresField = fields["tokenizer.ggml.scores"]
            val scores = if (scoresField != null) {
                extractFloatArray(scoresField)
            } else {
                // Default scores if not present
                FloatArray(vocab.size) { 0f }
            }

            // Extract special token IDs
            val bosTokenId = fields["tokenizer.ggml.bos_token_id"]?.scalarInt() ?: DEFAULT_BOS_TOKEN_ID
            val eosTokenId = fields["tokenizer.ggml.eos_token_id"]?.scalarInt() ?: DEFAULT_EOS_TOKEN_ID
            val unkTokenId = fields["tokenizer.ggml.unknown_token_id"]?.scalarInt() ?: DEFAULT_UNK_TOKEN_ID
            val addSpacePrefix = fields["tokenizer.ggml.add_space_prefix"]?.scalarBool() ?: DEFAULT_ADD_SPACE_PREFIX

            // Per-token classification (CONTROL = atomic special tokens that must not be BPE-split).
            val tokenTypes = fields["tokenizer.ggml.token_type"]?.let { extractIntArray(it) }

            // Detect tokenizer type from metadata
            val modelType = fields["tokenizer.ggml.model"]?.scalarString()
            val strategy = detectStrategy(modelType, vocab, debug)

            // Always log the tokenizer strategy
            println("Tokenizer: ${strategy.type} (model=${modelType ?: "auto-detected"})")

            if (debug) {
                println("DEBUG: BOS=$bosTokenId, EOS=$eosTokenId, UNK=$unkTokenId addSpacePrefix=$addSpacePrefix")
                println("DEBUG: Tokenizer model type from metadata: ${modelType ?: "(not specified)"}")
                println("DEBUG: Using tokenizer strategy: ${strategy.type}")
            }

            return GGUFTokenizer(vocab, scores, bosTokenId, eosTokenId, unkTokenId, strategy, tokenTypes, addSpacePrefix)
        }

        /**
         * Create a tokenizer using streaming API.
         * Parses metadata only (~1MB memory), suitable for large models.
         * The source is closed after reading metadata.
         */
        fun fromRandomAccessSource(source: RandomAccessSource, debug: Boolean = false): GGUFTokenizer {
            return StreamingGGUFReader.open(source).use { reader ->
                fromStreamingFields(reader.fields, debug)
            }
        }

        /**
         * Create a tokenizer from StreamingGGUFReader fields.
         * StreamingGGUFReader.fields returns direct values (Map<String, Any?>),
         * not ReaderField objects.
         */
        private fun fromStreamingFields(fields: Map<String, Any?>, debug: Boolean = false): GGUFTokenizer {
            // Extract vocabulary tokens (stored as List<String> in streaming reader)
            val tokensValue = fields["tokenizer.ggml.tokens"]
                ?: error("GGUF file missing tokenizer.ggml.tokens field")
            val vocab = extractStringList(tokensValue)

            if (debug) {
                println("DEBUG: Vocab size = ${vocab.size}")
                println("DEBUG: First 10 tokens:")
                vocab.take(10).forEachIndexed { idx, token ->
                    val bytes = token.encodeToByteArray()
                    val hexStr = bytes.joinToString(" ") { b ->
                        val hex = (b.toInt() and 0xFF).toString(16).uppercase()
                        if (hex.length == 1) "0$hex" else hex
                    }
                    println("  [$idx] = '$token' (bytes: $hexStr)")
                }
                println("DEBUG: Tokens around index 1000:")
                vocab.drop(1000).take(5).forEachIndexed { idx, token ->
                    println("  [${1000 + idx}] = '$token'")
                }
            }

            // Extract BPE scores
            val scoresValue = fields["tokenizer.ggml.scores"]
            val scores = if (scoresValue != null) {
                extractFloatList(scoresValue)
            } else {
                FloatArray(vocab.size) { 0f }
            }

            // Extract special token IDs
            val bosTokenId = fields["tokenizer.ggml.bos_token_id"]?.toIntValue() ?: DEFAULT_BOS_TOKEN_ID
            val eosTokenId = fields["tokenizer.ggml.eos_token_id"]?.toIntValue() ?: DEFAULT_EOS_TOKEN_ID
            val unkTokenId = fields["tokenizer.ggml.unknown_token_id"]?.toIntValue() ?: DEFAULT_UNK_TOKEN_ID
            val addSpacePrefix = (fields["tokenizer.ggml.add_space_prefix"] as? Boolean) ?: DEFAULT_ADD_SPACE_PREFIX

            // Per-token classification (CONTROL = atomic special tokens that must not be BPE-split).
            val tokenTypes = fields["tokenizer.ggml.token_type"]?.let { extractIntList(it) }

            // Detect tokenizer type from metadata
            val modelType = fields["tokenizer.ggml.model"]?.toString()
            val strategy = detectStrategy(modelType, vocab, debug)

            // Always log the tokenizer strategy
            println("Tokenizer: ${strategy.type} (model=${modelType ?: "auto-detected"})")

            if (debug) {
                println("DEBUG: BOS=$bosTokenId, EOS=$eosTokenId, UNK=$unkTokenId addSpacePrefix=$addSpacePrefix")
                println("DEBUG: Tokenizer model type from metadata: ${modelType ?: "(not specified)"}")
                println("DEBUG: Using tokenizer strategy: ${strategy.type}")
            }

            return GGUFTokenizer(vocab, scores, bosTokenId, eosTokenId, unkTokenId, strategy, tokenTypes, addSpacePrefix)
        }

        /**
         * Detect the tokenizer strategy based on GGUF metadata and vocabulary inspection.
         */
        private fun detectStrategy(
            modelType: String?,
            vocab: List<String>,
            debug: Boolean
        ): TokenizerStrategy {
            // First, try to detect from explicit model type in metadata
            val fromMetadata = when (modelType?.lowercase()) {
                "llama", "sentencepiece" -> SentencePieceStrategy
                "gpt2", "bpe" -> BPEStrategy
                "bert", "wordpiece" -> WordPieceStrategy
                else -> null
            }

            if (fromMetadata != null) {
                if (debug) {
                    println("DEBUG: Detected tokenizer type from metadata: ${fromMetadata.type}")
                }
                return fromMetadata
            }

            // Fallback: inspect vocabulary for characteristic markers
            val fromVocab = detectFromVocab(vocab)
            if (debug) {
                println("DEBUG: Detected tokenizer type from vocab inspection: ${fromVocab.type}")
            }
            return fromVocab
        }

        /**
         * Detect tokenizer type by inspecting vocabulary for characteristic markers.
         */
        private fun detectFromVocab(vocab: List<String>): TokenizerStrategy {
            val sentencePieceMarker = "\u2581" // ▁
            val bpeMarker = "\u0120" // Ġ
            val wordPieceMarker = "##"

            var sentencePieceCount = 0
            var bpeCount = 0
            var wordPieceCount = 0

            // Sample first 1000 tokens (or all if less)
            val sampleSize = minOf(vocab.size, 1000)
            for (i in 0 until sampleSize) {
                val token = vocab[i]
                when {
                    token.contains(sentencePieceMarker) -> sentencePieceCount++
                    token.contains(bpeMarker) -> bpeCount++
                    token.startsWith(wordPieceMarker) -> wordPieceCount++
                }
            }

            // Return strategy based on which marker is most prevalent
            return when {
                sentencePieceCount >= bpeCount && sentencePieceCount >= wordPieceCount && sentencePieceCount > 0 ->
                    SentencePieceStrategy
                bpeCount > sentencePieceCount && bpeCount >= wordPieceCount ->
                    BPEStrategy
                wordPieceCount > sentencePieceCount && wordPieceCount > bpeCount ->
                    WordPieceStrategy
                else ->
                    // Default to SentencePiece/Unknown since most GGUF models use it
                    UnknownStrategy
            }
        }

        /**
         * Extract a list of strings from streaming field value.
         */
        @Suppress("UNCHECKED_CAST")
        private fun extractStringList(value: Any): List<String> {
            return when (value) {
                is List<*> -> value.filterIsInstance<String>()
                else -> error("Expected List<String> for tokens field, got ${value::class.simpleName}")
            }
        }

        /**
         * Extract float array from streaming field value.
         */
        @Suppress("UNCHECKED_CAST")
        private fun extractFloatList(value: Any): FloatArray {
            return when (value) {
                is List<*> -> {
                    val floats = mutableListOf<Float>()
                    for (item in value) {
                        when (item) {
                            is Float -> floats.add(item)
                            is Double -> floats.add(item.toFloat())
                            is Number -> floats.add(item.toFloat())
                        }
                    }
                    floats.toFloatArray()
                }
                else -> error("Expected List<Number> for scores field, got ${value::class.simpleName}")
            }
        }

        /**
         * Convert streaming field value to Int.
         */
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

        private fun extractStringArray(field: ReaderField): List<String> {
            val strings = mutableListOf<String>()
            // For array fields, data contains indexes to string parts
            for (idx in field.data) {
                if (idx < 0 || idx >= field.parts.size) continue
                val part = field.parts[idx]
                // Handle all numeric types that could represent bytes
                val bytes = part.mapNotNull { value ->
                    when (value) {
                        is UByte -> value.toByte()
                        is Byte -> value
                        is Number -> value.toInt().toByte()
                        else -> null
                    }
                }
                strings.add(bytes.toByteArray().decodeToString())
            }
            return strings
        }

        private fun extractFloatArray(field: ReaderField): FloatArray {
            val floats = mutableListOf<Float>()
            for (idx in field.data) {
                if (idx < 0 || idx >= field.parts.size) continue
                val part = field.parts[idx]
                for (value in part) {
                    when (value) {
                        is Float -> floats.add(value)
                        is Double -> floats.add(value.toFloat())
                        is Number -> floats.add(value.toFloat())
                    }
                }
            }
            return floats.toFloatArray()
        }

        private fun extractIntArray(field: ReaderField): IntArray {
            val ints = mutableListOf<Int>()
            for (idx in field.data) {
                if (idx < 0 || idx >= field.parts.size) continue
                val part = field.parts[idx]
                for (value in part) {
                    when (value) {
                        is Int -> ints.add(value)
                        is UInt -> ints.add(value.toInt())
                        is Long -> ints.add(value.toInt())
                        is ULong -> ints.add(value.toInt())
                        is Number -> ints.add(value.toInt())
                    }
                }
            }
            return ints.toIntArray()
        }

        private fun extractIntList(value: Any): IntArray {
            return when (value) {
                is List<*> -> {
                    val ints = mutableListOf<Int>()
                    for (item in value) {
                        when (item) {
                            is Int -> ints.add(item)
                            is UInt -> ints.add(item.toInt())
                            is Long -> ints.add(item.toInt())
                            is ULong -> ints.add(item.toInt())
                            is Number -> ints.add(item.toInt())
                        }
                    }
                    ints.toIntArray()
                }
                else -> error("Expected List<Number> for token_type field, got ${value::class.simpleName}")
            }
        }

        private fun ReaderField.scalarInt(): Int {
            val idx = data.firstOrNull() ?: 0
            val part = parts.getOrNull(idx) ?: return 0
            val value = (part as? List<*>)?.firstOrNull() ?: return 0
            return when (value) {
                is Int -> value
                is UInt -> value.toInt()
                is Long -> value.toInt()
                is ULong -> value.toInt()
                is Number -> value.toInt()
                else -> 0
            }
        }

        private fun ReaderField.scalarString(): String? {
            val idx = data.firstOrNull() ?: return null
            val part = parts.getOrNull(idx) ?: return null
            // Handle bytes to string conversion
            val bytes = (part as? List<*>)?.mapNotNull { value ->
                when (value) {
                    is UByte -> value.toByte()
                    is Byte -> value
                    is Number -> value.toInt().toByte()
                    else -> null
                }
            } ?: return null
            return bytes.toByteArray().decodeToString()
        }

        private fun ReaderField.scalarBool(): Boolean? {
            val idx = data.firstOrNull() ?: return null
            val part = parts.getOrNull(idx) ?: return null
            val value = (part as? List<*>)?.firstOrNull() ?: return null
            return when (value) {
                is Boolean -> value
                is Number -> value.toInt() != 0
                else -> null
            }
        }
    }

    /** The detected tokenizer type/strategy in use */
    val tokenizerType: TokenizerType get() = strategy.type

    override val bosTokenId: Int get() = _bosTokenId
    override val eosTokenId: Int get() = _eosTokenId
    override val vocabSize: Int get() = vocab.size

    @Deprecated("Use eosTokenId", replaceWith = ReplaceWith("eosTokenId"))
    val eosId: Int get() = _eosTokenId

    @Deprecated("Use bosTokenId", replaceWith = ReplaceWith("bosTokenId"))
    val bosId: Int get() = _bosTokenId

    // Build reverse lookup for encoding
    private val tokenToId: Map<String, Int> by lazy {
        vocab.mapIndexed { idx, token -> token to idx }.toMap()
    }

    // Build sorted vocab by score for BPE merging
    private val sortedVocabByScore: List<Pair<String, Int>> by lazy {
        vocab.mapIndexed { idx, token -> token to idx }
            .sortedByDescending { (_, idx) -> scores.getOrElse(idx) { 0f } }
    }

    // Atomic special tokens (GGUF token_type ∈ {CONTROL, USER_DEFINED}), longest-first
    // for greedy matching. These must be emitted as a single ID even though greedy
    // bottom-up BPE could never assemble their multi-character form from single-char
    // starting tokens. USER_DEFINED is included because Gemma 4's tool-call /
    // tool-response markers are USER_DEFINED-by-design — they need atomic emission
    // exactly the same way as CONTROL tokens to round-trip through chat templates.
    private val specialTokensByLength: List<Pair<String, Int>> by lazy {
        if (tokenTypes == null) emptyList()
        else vocab.asSequence()
            .mapIndexedNotNull { id, str ->
                val isAtomic = id < tokenTypes.size &&
                    (tokenTypes[id] == TOKEN_TYPE_CONTROL || tokenTypes[id] == TOKEN_TYPE_USER_DEFINED)
                if (isAtomic && str.isNotEmpty()) str to id else null
            }
            .sortedByDescending { it.first.length }
            .toList()
    }

    override fun encode(text: String): IntArray {
        if (text.isEmpty()) return intArrayOf()

        // Handle WordPiece differently - it splits on whitespace first
        if (strategy.type == TokenizerType.WORDPIECE) {
            return encodeWordPiece(text)
        }

        val specials = specialTokensByLength
        if (specials.isEmpty()) {
            // No CONTROL tokens registered — preserve legacy behavior.
            return encodeBPE(preprocessText(text))
        }

        // Two-pass: walk the raw text, emit atomic IDs for special-token matches,
        // BPE-encode the plain-text gaps in between.
        val out = mutableListOf<Int>()
        val gap = StringBuilder()
        var i = 0
        while (i < text.length) {
            var match: Pair<String, Int>? = null
            for (s in specials) {
                if (text.regionMatches(i, s.first, 0, s.first.length)) {
                    match = s
                    break
                }
            }
            if (match != null) {
                if (gap.isNotEmpty()) {
                    for (id in encodeBPE(preprocessText(gap.toString()))) out.add(id)
                    gap.clear()
                }
                out.add(match.second)
                i += match.first.length
            } else {
                gap.append(text[i])
                i++
            }
        }
        if (gap.isNotEmpty()) {
            for (id in encodeBPE(preprocessText(gap.toString()))) out.add(id)
        }
        return out.toIntArray()
    }

    /**
     * Strategy-specific preprocessing, with the GGUF
     * `tokenizer.ggml.add_space_prefix` flag honoured for SentencePiece.
     * When `addSpacePrefix=false` (e.g. Gemma 4), the leading word-boundary
     * marker that `SentencePieceStrategy.preprocess` always prepends is
     * stripped — otherwise `encode("Hi")` would return `[18428]` (= `▁Hi`)
     * instead of `[10979]` (= `Hi`).
     */
    private fun preprocessText(text: String): String {
        val out = strategy.preprocess(text)
        if (!addSpacePrefix && strategy.type == TokenizerType.SENTENCEPIECE) {
            val marker = strategy.spaceMarker
            if (marker.isNotEmpty() && out.startsWith(marker)) {
                return out.substring(marker.length)
            }
        }
        return out
    }

    /**
     * Standard BPE encoding used by SentencePiece and GPT-2 style tokenizers.
     */
    private fun encodeBPE(preprocessed: String): IntArray {
        // Convert text to a list of single-char tokens
        val tokens = mutableListOf<String>()
        for (char in preprocessed) {
            tokens.add(char.toString())
        }

        // Greedy BPE merging
        var changed = true
        while (changed && tokens.size > 1) {
            changed = false
            var bestIdx = -1
            var bestScore = Float.NEGATIVE_INFINITY
            var bestMerge = ""

            // Find the best merge
            for (i in 0 until tokens.size - 1) {
                val merge = tokens[i] + tokens[i + 1]
                val tokenId = tokenToId[merge]
                if (tokenId != null) {
                    val score = scores.getOrElse(tokenId) { 0f }
                    if (score > bestScore) {
                        bestScore = score
                        bestIdx = i
                        bestMerge = merge
                    }
                }
            }

            // Apply best merge
            if (bestIdx >= 0) {
                tokens[bestIdx] = bestMerge
                tokens.removeAt(bestIdx + 1)
                changed = true
            }
        }

        // Convert tokens to IDs
        return tokens.map { token ->
            tokenToId[token] ?: findFallbackToken(token)
        }.toIntArray()
    }

    /**
     * WordPiece encoding - splits on whitespace first, then applies subword tokenization.
     */
    private fun encodeWordPiece(text: String): IntArray {
        val result = mutableListOf<Int>()
        val words = text.split(Regex("\\s+")).filter { it.isNotEmpty() }

        for ((wordIndex, word) in words.withIndex()) {
            // Add space token between words (if not first word)
            if (wordIndex > 0) {
                tokenToId[" "]?.let { result.add(it) }
            }

            // Try to find the word in vocab
            val wordId = tokenToId[word]
            if (wordId != null) {
                result.add(wordId)
                continue
            }

            // Break into subwords
            var start = 0
            var foundAny = false
            while (start < word.length) {
                var end = word.length
                var found = false

                while (start < end) {
                    val substr = if (start == 0) {
                        word.substring(start, end)
                    } else {
                        "##" + word.substring(start, end)
                    }

                    val id = tokenToId[substr]
                    if (id != null) {
                        result.add(id)
                        start = end
                        found = true
                        foundAny = true
                        break
                    }
                    end--
                }

                if (!found) {
                    // Character not found, use UNK or byte fallback
                    if (start < word.length) {
                        result.add(findFallbackToken(word[start].toString()))
                        start++
                    }
                }
            }

            if (!foundAny && word.isNotEmpty()) {
                result.add(unkTokenId)
            }
        }

        return result.toIntArray()
    }

    private fun findFallbackToken(token: String): Int {
        // Try byte fallback tokens (common in LLaMA tokenizers)
        if (token.length == 1) {
            val byte = token[0].code
            // Try <0xXX> format
            val hexToken = "<0x${byte.toString(16).uppercase().padStart(2, '0')}>"
            tokenToId[hexToken]?.let { return it }
            // Try raw byte token
            val byteToken = byteArrayOf(byte.toByte()).decodeToString()
            tokenToId[byteToken]?.let { return it }
        }
        // Fall back to UNK token
        return unkTokenId
    }

    override fun decode(tokens: IntArray): String {
        // Accumulate byte tokens and decode them together as UTF-8
        val result = StringBuilder()
        val byteBuffer = mutableListOf<Byte>()

        for (tokenId in tokens) {
            if (tokenId < 0 || tokenId >= vocab.size) continue
            val token = vocab[tokenId]

            val byteValue = extractByteToken(token)
            if (byteValue != null) {
                byteBuffer.add(byteValue)
            } else {
                // Flush accumulated bytes as UTF-8
                if (byteBuffer.isNotEmpty()) {
                    result.append(byteBuffer.toByteArray().decodeToString())
                    byteBuffer.clear()
                }
                result.append(decodeToken(token))
            }
        }

        // Flush remaining bytes
        if (byteBuffer.isNotEmpty()) {
            result.append(byteBuffer.toByteArray().decodeToString())
        }

        return result.toString()
    }

    override fun decode(token: Int): String {
        if (token < 0 || token >= vocab.size) return ""
        val text = vocab[token]
        // Handle special byte tokens like <0xXX>
        return decodeToken(text)
    }

    /**
     * Extract byte value from <0xXX> format token.
     * Returns null if token is not a byte token.
     */
    private fun extractByteToken(token: String): Byte? {
        if (token.startsWith("<0x") && token.endsWith(">") && token.length == 6) {
            val hex = token.substring(3, 5)
            val value = hex.toIntOrNull(16)
            if (value != null) {
                return value.toByte()
            }
        }
        return null
    }

    private fun decodeToken(token: String): String {
        // Handle byte tokens in <0xXX> format
        val byteValue = extractByteToken(token)
        if (byteValue != null) {
            // For single-token decode, convert byte to string
            // Note: This may not handle multi-byte UTF-8 correctly in streaming mode,
            // but it's the best we can do for single-token decoding
            return byteArrayOf(byteValue).decodeToString()
        }

        // Handle common special tokens
        return when (token) {
            "<s>" -> "" // BOS
            "</s>" -> "" // EOS
            "<unk>" -> "" // Unknown
            "<pad>" -> "" // Padding
            strategy.spaceMarker -> " "
            else -> strategy.postprocess(token)
        }
    }
}
