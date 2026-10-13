package sk.ainet.apps.kllama

/**
 * Backwards-compatibility alias. GGUFTokenizer has moved to llm-core.
 * @see sk.ainet.apps.llm.tokenizer.GGUFTokenizer
 */
@Suppress("unused", "DEPRECATION")
@Deprecated("Legacy tokenizer; use sk.ainet.apps.llm.tokenizer.TokenizerFactory (SKEEP-006). Removed in 0.60.")
public typealias GGUFTokenizer = sk.ainet.apps.llm.tokenizer.GGUFTokenizer
