package sk.ainet.asr.domain

/** Greedy decode options (greedy only — beam search is out of scope, PRD N5 determinism). */
public data class DecodingOptions(
    val maxTokens: Int = 224,
    /** ISO 639-1 language for multilingual models (F2 `--language`); ignored by English-only models. */
    val language: String = "en",
    /** Translate-to-English task instead of transcribe (F2 `--task translate`). */
    val translate: Boolean = false,
    /** Emit timestamp tokens (F2 `--timestamps`, segment-level). */
    val withTimestamps: Boolean = false,
    /**
     * Whisper-reference logit filtering: suppress non-speech tokens, EOT/blank at the first
     * step, and timestamp tokens when timestamps are off. Disable only for parity runs
     * against decoders without filtering (the P0 baseline anchor).
     */
    val suppressNonSpeech: Boolean = true,
    /** Stop if the last N generated ids are identical (token-level looping). 0 = off. */
    val repetitionGuardN: Int = 6,
    /** Stop before emitting an n-gram that already occurred (phrase-level looping). 0 = off. */
    val noRepeatNgram: Int = 4,
)
