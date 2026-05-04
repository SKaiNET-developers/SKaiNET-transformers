package sk.ainet.apps.llm.tokenizer

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.io.tokenizer.SentencePieceTokenizer

/**
 * Downstream model-specific decorator that adds atomic special-token
 * splitting on top of upstream [SentencePieceTokenizer].
 *
 * Why this lives here, not upstream
 * ---------------------------------
 * Upstream `sk.ainet.io.tokenizer.SentencePieceTokenizer.encode()` does pure
 * SentencePiece — it never treats any vocab entry as atomic / non-splittable.
 * That works for vanilla LLaMA but breaks chat-template models like Gemma 4
 * whose `<bos>`, `<|turn>`, `<turn|>` and similar markers must encode as a
 * single id rather than fragmenting into per-character byte-fallback pieces.
 *
 * Until upstream extracts a shared `SpecialTokenSplitter` decorator
 * (proposed follow-up — see issue tracker), this class fills the gap as a
 * downstream model-specific tokenizer handler. It also patches two other
 * upstream gaps for the HuggingFace `tokenizer.json` path:
 *
 *  - Resolves `bosTokenId` / `eosTokenId` from the `added_tokens` array
 *    (upstream's HF factory reads neither).
 *  - Allows overriding `addSpacePrefix` (upstream's HF factory hard-codes
 *    `true`, but Gemma 4 needs `false` to match HuggingFace reference IDs).
 *
 * Algorithm
 * ---------
 *  - **encode(text)**: walk left-to-right; at each position try the
 *    longest registered special-token string. On a match, emit its id and
 *    skip past it. Otherwise extend the current non-special segment until
 *    the next special boundary (or end-of-text), then `base.encode(segment)`.
 *  - **decode(ids)**: scan ids; collect contiguous non-special ids into
 *    a buffer, flushing via `base.decode(buffer)` when we hit a special
 *    id, then emit the special's string form. The byte-level UTF-8
 *    spanning that SentencePiece does inside `decode(IntArray)` is
 *    preserved within each non-special run, because special-token
 *    boundaries always sit on UTF-8 boundaries (specials are literal
 *    strings).
 */
public class SentencePieceSpecialTokens(
    private val base: SentencePieceTokenizer,
    private val specialTokens: Map<String, Int>,
    bosTokenId: Int? = null,
    eosTokenId: Int? = null,
) : Tokenizer {

    private val specialIdToString: Map<Int, String> =
        specialTokens.entries.associate { (k, v) -> v to k }

    /** Longest-first ordering so e.g. `<|im_start|>` wins over `<|im`. */
    private val specialsByLengthDesc: List<String> =
        specialTokens.keys.sortedByDescending { it.length }

    override val vocabSize: Int = base.vocabSize
    override val bosTokenId: Int = (bosTokenId ?: base.bosTokenId) ?: -1
    override val eosTokenId: Int = (eosTokenId ?: base.eosTokenId) ?: -1

    override fun encode(text: String): IntArray {
        val out = ArrayList<Int>(text.length)
        var i = 0
        while (i < text.length) {
            val matched = matchSpecialAt(text, i)
            if (matched != null) {
                out.add(specialTokens.getValue(matched))
                i += matched.length
                continue
            }
            val nextSpecial = nextSpecialStart(text, i)
            val segment = text.substring(i, nextSpecial)
            for (id in base.encode(segment)) out.add(id)
            i = nextSpecial
        }
        return IntArray(out.size) { out[it] }
    }

    override fun decode(tokens: IntArray): String {
        if (tokens.isEmpty()) return ""
        val sb = StringBuilder()
        val buffer = ArrayList<Int>()
        for (id in tokens) {
            val special = specialIdToString[id]
            if (special != null) {
                if (buffer.isNotEmpty()) {
                    sb.append(base.decode(buffer.toIntArray()))
                    buffer.clear()
                }
                sb.append(special)
            } else {
                buffer.add(id)
            }
        }
        if (buffer.isNotEmpty()) {
            sb.append(base.decode(buffer.toIntArray()))
        }
        return sb.toString()
    }

    override fun decode(token: Int): String {
        val special = specialIdToString[token]
        if (special != null) return special
        return base.decode(intArrayOf(token))
    }

    private fun matchSpecialAt(text: String, from: Int): String? {
        for (tok in specialsByLengthDesc) {
            if (tok.isNotEmpty() && text.regionMatches(from, tok, 0, tok.length)) return tok
        }
        return null
    }

    private fun nextSpecialStart(text: String, from: Int): Int {
        var earliest = text.length
        for (tok in specialTokens.keys) {
            if (tok.isEmpty()) continue
            val idx = text.indexOf(tok, startIndex = from + 1)
            if (idx in 0 until earliest) earliest = idx
        }
        return earliest
    }

    public companion object {
        private const val TOKEN_TYPE_CONTROL = 3
        private const val TOKEN_TYPE_USER_DEFINED = 4

        private val JSON: Json = Json { ignoreUnknownKeys = true; isLenient = true }

        /**
         * Wrap an upstream SentencePiece tokenizer constructed from GGUF
         * metadata, layering atomic special-token splitting on top.
         *
         * Special tokens are identified by `tokenizer.ggml.token_type`
         * entries with code `TOKEN_TYPE_CONTROL` (3) or
         * `TOKEN_TYPE_USER_DEFINED` (4). Both are treated as atomic — the
         * local fork's behavior, preserved here so chat-template markers
         * like `<bos>` (CONTROL) and `<|tool_call>` (often USER_DEFINED)
         * encode as a single id.
         */
        public fun fromGgufFields(fields: Map<String, Any?>): Tokenizer {
            val base = SentencePieceTokenizer.fromGgufFields(fields)

            @Suppress("UNCHECKED_CAST")
            val tokens = (fields["tokenizer.ggml.tokens"] as? List<*>)
                ?.filterIsInstance<String>().orEmpty()
            val tokenTypes = (fields["tokenizer.ggml.token_type"] as? List<*>)
                ?.mapNotNull { (it as? Number)?.toInt() }.orEmpty()

            val specials = HashMap<String, Int>()
            val limit = minOf(tokens.size, tokenTypes.size)
            for (i in 0 until limit) {
                val type = tokenTypes[i]
                if (type == TOKEN_TYPE_CONTROL || type == TOKEN_TYPE_USER_DEFINED) {
                    val tok = tokens[i]
                    if (tok.isNotEmpty()) specials[tok] = i
                }
            }

            // If GGUF didn't carry token_type at all (some older files), the
            // SentencePiece encoder still works — just no atomic splitting.
            return SentencePieceSpecialTokens(base, specials)
        }

        /**
         * Wrap an upstream SentencePiece tokenizer constructed from a
         * HuggingFace `tokenizer.json` string, layering atomic special-
         * token splitting and patching upstream's HF-path gaps.
         *
         * Reads:
         *  - `added_tokens[]` for the special-token registry, BOS, and EOS.
         *    Entries with `"special": true` (or missing the field) are
         *    treated as special.
         *  - Optional `tokenizer_config.json` (passed as [configJson]) for
         *    `add_space_prefix` — Gemma 4 sets this to `false`, which
         *    upstream's HF factory does not honor.
         */
        public fun fromTokenizerJson(json: String, configJson: String? = null): Tokenizer {
            val root = JSON.parseToJsonElement(json).jsonObject
            val configRoot = configJson?.let { JSON.parseToJsonElement(it).jsonObject }

            val addSpacePrefix = configRoot?.get("add_space_prefix")?.jsonPrimitive?.boolean
                ?: detectAddSpacePrefixFromNormalizer(root)
                ?: true

            val base = buildBaseFromTokenizerJson(root, addSpacePrefix)

            val (specials, bosId, eosId) = extractSpecialsFromAddedTokens(root)
            return SentencePieceSpecialTokens(
                base = base,
                specialTokens = specials,
                bosTokenId = bosId,
                eosTokenId = eosId,
            )
        }

        /**
         * Build an upstream SentencePieceTokenizer with a custom
         * [addSpacePrefix]. Upstream's `SentencePieceTokenizer.fromTokenizerJson`
         * hard-codes `addSpacePrefix=true`; Gemma 4 needs `false`. This
         * mirrors the upstream parsing of `model.vocab` (an array of
         * `[token, score]` pairs) and `model.unk_id`, then rebuilds the
         * tokenizer with the desired flag.
         */
        private fun buildBaseFromTokenizerJson(root: JsonObject, addSpacePrefix: Boolean): SentencePieceTokenizer {
            val model = root["model"]?.jsonObject ?: error("tokenizer.json missing 'model'")
            val vocabArr = model["vocab"]?.jsonArray ?: error("tokenizer.json missing 'model.vocab'")
            val tokens = ArrayList<String>(vocabArr.size)
            val scores = ArrayList<Float>(vocabArr.size)
            for (entry in vocabArr) {
                val pair = entry.jsonArray
                tokens.add(pair[0].jsonPrimitive.content)
                val raw = pair[1].jsonPrimitive
                scores.add(raw.content.toFloatOrNull() ?: 0f)
            }
            val unknownId = model["unk_id"]?.jsonPrimitive?.int
            return SentencePieceTokenizer(
                tokens = tokens,
                scores = scores,
                unknownTokenId = unknownId,
                addSpacePrefix = addSpacePrefix,
            )
        }

        /**
         * Walk `tokenizer.json#normalizer` looking for a `Prepend` step
         * with content `▁` — that's HF's encoding of `add_dummy_prefix=true`.
         * Returns null if the normalizer doesn't say either way.
         */
        private fun detectAddSpacePrefixFromNormalizer(root: JsonObject): Boolean? {
            val normalizer = root["normalizer"] as? JsonObject ?: return null
            // Sequence of normalizers
            val seq = normalizer["normalizers"]?.jsonArray
            val candidates = seq ?: listOf(normalizer)
            for (n in candidates) {
                val obj = n as? JsonObject ?: continue
                val type = obj["type"]?.jsonPrimitive?.content ?: continue
                if (type == "Prepend") {
                    val prepend = obj["prepend"]?.jsonPrimitive?.content
                    if (prepend == "▁") return true
                }
            }
            return false
        }

        private data class AddedTokens(
            val specials: Map<String, Int>,
            val bosId: Int?,
            val eosId: Int?,
        )

        private fun extractSpecialsFromAddedTokens(root: JsonObject): AddedTokens {
            val added = root["added_tokens"]?.jsonArray
            if (added == null) return AddedTokens(emptyMap(), null, null)
            val specials = HashMap<String, Int>(added.size)
            var bosId: Int? = null
            var eosId: Int? = null
            for (entry in added) {
                val obj = entry.jsonObject
                val content = obj["content"]?.jsonPrimitive?.content ?: continue
                val id = obj["id"]?.jsonPrimitive?.int ?: continue
                val isSpecial = obj["special"]?.jsonPrimitive?.boolean ?: true
                if (isSpecial) specials[content] = id
                when (content) {
                    "<bos>", "<|begin_of_text|>", "<s>" -> bosId = bosId ?: id
                    "<eos>", "<|end_of_text|>", "</s>" -> eosId = eosId ?: id
                }
            }
            return AddedTokens(specials, bosId, eosId)
        }
    }
}
