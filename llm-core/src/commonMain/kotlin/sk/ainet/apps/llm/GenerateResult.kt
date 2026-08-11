package sk.ainet.apps.llm

/**
 * Result of EOS-aware generation.
 *
 * @param tokens The generated token IDs (excluding the prompt).
 * @param text The decoded text (if a tokenizer was provided).
 * @param stoppedByEos True if generation stopped because EOS was emitted.
 */
public data class GenerateResult(
    val tokens: List<Int>,
    val text: String,
    val stoppedByEos: Boolean
)
