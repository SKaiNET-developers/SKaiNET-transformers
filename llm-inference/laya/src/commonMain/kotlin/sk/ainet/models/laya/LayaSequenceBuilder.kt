package sk.ainet.models.laya

import sk.ainet.frontend.text.Tokenizer

/** The four roles `build_sequence` needs, resolved against the checkpoint's own tokenizer. */
public data class LayaSpecialIds(
    val cls: Int,
    val sep: Int,
    val mask: Int,
    val pad: Int,
    /** The literal mask string; its occurrences in user text are replaced by a space before tokenising. */
    val maskToken: String,
) {
    public companion object {
        /** Looks every role up in the tokenizer's vocabulary; a missing one is a checkpoint error, not a default. */
        public fun resolve(tokenizer: Tokenizer, config: LayaTokenizerConfig): LayaSpecialIds {
            fun id(role: String, token: String): Int = tokenizer.vocabulary.tokenToId(token)
                ?: throw UnsupportedLayaCheckpointException("$role '$token' from tokenizer_config.json is not in the tokenizer vocabulary")
            return LayaSpecialIds(
                cls = id("cls_token", config.clsToken),
                sep = id("sep_token", config.sepToken),
                mask = id("mask_token", config.maskToken),
                pad = id("pad_token", config.padToken),
                maskToken = config.maskToken,
            )
        }
    }
}

/** What the head budget did to the options (`stats` of the reference). */
public data class LayaOptionStats(
    val options: Int,
    /** Options that still have a token span of their own after the cut; fewer than [options] means two collided. */
    val optionsDistinct: Int,
    /** The cap applied to every option, or null when none was. */
    val tokensPerOption: Int?,
)

/** What the state budget did to the state (`truncation` of the reference). */
public data class LayaTruncation(
    val stateTokens: Int,
    val stateTokensUsed: Int,
    val stateTokensDropped: Int,
) {
    val truncated: Boolean get() = stateTokensUsed < stateTokens
}

/** The question half of a sequence: `[CLS] head [SEP] ([MASK] option)… [SEP]`. */
public class LayaHead(
    public val ids: IntArray,
    /** Position of each option's `[MASK]`, in option (slot) order. */
    public val markers: IntArray,
    public val stats: LayaOptionStats,
)

/** A packed sequence ready for the encoder. */
public class LayaSequence(
    public val ids: IntArray,
    public val markers: IntArray,
    public val stats: LayaOptionStats,
    public val truncation: LayaTruncation,
    /** Slot `s` holds option `optionOrder[s]`; identity when the caller gave no permutation. */
    public val optionOrder: IntArray,
) {
    public val length: Int get() = ids.size
}

/** The question cannot be packed as asked: fewer markers than options, or an invalid permutation. */
public class LayaSequenceException(message: String) : IllegalArgumentException(message)

/**
 * Port of the reference `build_head` / `build_sequence` (REQ-PRE-002, REQ-PRE-003).
 *
 * Layout: `[CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] state [SEP]`.
 * Every text is tokenised without automatic special tokens. The budgets are the checkpoint's
 * `max_len` and `head_max_len`; the state takes only the room left after the head, and the head
 * is cut in a fixed order: option texts are capped at [OPTION_TOKEN_CAP] tokens each, then, if
 * less than [MIN_OPTION_BUDGET] tokens remain for the instruction, every option (mask included)
 * is cut to `max(MIN_OPTION_TOKENS, (head_max_len − 16) / k)`, and the instruction keeps
 * `max(MIN_HEAD_TOKENS, remaining budget)` tokens.
 */
public class LayaSequenceBuilder(
    public val tokenizer: Tokenizer,
    public val specialIds: LayaSpecialIds,
    public val maxLen: Int,
    public val headMaxLen: Int,
) {
    public constructor(tokenizer: Tokenizer, specialIds: LayaSpecialIds, config: LayaAgentConfig) :
        this(tokenizer, specialIds, config.maxLen, config.headMaxLen)

    init {
        require(maxLen > 0 && headMaxLen > 0) { "maxLen and headMaxLen must be positive" }
    }

    /** Tokenise user-facing text the way the reference does: mask literal removed, no special tokens. */
    public fun encodeText(text: String): IntArray = tokenizer.encode(text.replace(specialIds.maskToken, " "), addSpecialTokens = false)

    /** The state's token ids, computed once; callers asking several questions about one state reuse them. */
    public fun encodeState(state: LayaState): IntArray = encodeText(state.serialize())

    public fun buildHead(question: LayaQuestion, optionOrder: IntArray? = null): LayaHead {
        val options = question.renderOptions()
        val order = resolveOrder(optionOrder, options.size)
        var headIds = encodeText("${question.type.wire} question: ${question.instructions}")
        var optionIds: List<IntArray> = order.map { i ->
            val body = encodeText(" " + options[i])
            intArrayOf(specialIds.mask) + if (body.size > OPTION_TOKEN_CAP) body.copyOf(OPTION_TOKEN_CAP) else body
        }
        var optionBudget = headMaxLen - optionIds.sumOf { it.size }
        var perOption: Int? = null
        if (optionBudget < MIN_OPTION_BUDGET) {
            val per = maxOf(MIN_OPTION_TOKENS, (headMaxLen - MIN_OPTION_BUDGET).floorDiv(maxOf(1, optionIds.size)))
            perOption = per
            optionIds = optionIds.map { if (it.size > per) it.copyOf(per) else it }
            optionBudget = headMaxLen - optionIds.sumOf { it.size }
        }
        val keep = maxOf(MIN_HEAD_TOKENS, optionBudget)
        if (headIds.size > keep) headIds = headIds.copyOf(keep)

        val ids = ArrayList<Int>(2 + headIds.size + optionIds.sumOf { it.size } + 1)
        ids.add(specialIds.cls)
        headIds.forEach(ids::add)
        ids.add(specialIds.sep)
        val markers = IntArray(optionIds.size)
        optionIds.forEachIndexed { slot, option ->
            markers[slot] = ids.size
            option.forEach(ids::add)
        }
        ids.add(specialIds.sep)
        val distinct = optionIds.map { it.toList() }.toSet().size
        return LayaHead(ids.toIntArray(), markers, LayaOptionStats(optionIds.size, distinct, perOption))
    }

    /**
     * The full sequence. [truncateLeft] defaults to the state's own rule (a list state keeps its
     * end). [stateIds] lets a caller pass [encodeState] once for many questions.
     */
    public fun build(
        state: LayaState,
        question: LayaQuestion,
        optionOrder: IntArray? = null,
        truncateLeft: Boolean = state.truncatesLeft,
        stateIds: IntArray? = null,
    ): LayaSequence {
        val head = buildHead(question, optionOrder)
        val stateTokens = stateIds ?: encodeState(state)
        val room = maxOf(0, maxLen - head.ids.size - 1)
        val used = minOf(room, stateTokens.size)
        val from = if (truncateLeft) stateTokens.size - used else 0
        val ids = IntArray(minOf(maxLen, head.ids.size + used + 1))
        var n = 0
        for (id in head.ids) { if (n < ids.size) ids[n++] = id }
        for (k in 0 until used) { if (n < ids.size) ids[n++] = stateTokens[from + k] }
        if (n < ids.size) ids[n++] = specialIds.sep
        val markers = head.markers.filter { it < maxLen }.toIntArray()
        return LayaSequence(
            ids = ids,
            markers = markers,
            stats = head.stats,
            truncation = LayaTruncation(stateTokens.size, used, stateTokens.size - used),
            optionOrder = resolveOrder(optionOrder, question.optionCount),
        )
    }

    private fun resolveOrder(optionOrder: IntArray?, count: Int): IntArray {
        if (optionOrder == null) return IntArray(count) { it }
        if (optionOrder.size != count || optionOrder.sorted() != (0 until count).toList()) {
            throw LayaSequenceException("option_order must be a permutation of 0 until $count, got ${optionOrder.toList()}")
        }
        return optionOrder.copyOf()
    }

    public companion object {
        /** Tokens kept of every rendered option (`truncation=True, max_length=48` at the tokenizer). */
        public const val OPTION_TOKEN_CAP: Int = 48

        /** Below this many tokens left for the instruction, every option is cut to a per-option floor. */
        public const val MIN_OPTION_BUDGET: Int = 16

        /** The per-option floor, mask token included. */
        public const val MIN_OPTION_TOKENS: Int = 4

        /** The instruction keeps at least this many tokens. */
        public const val MIN_HEAD_TOKENS: Int = 8
    }
}
