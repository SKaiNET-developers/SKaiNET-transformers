@file:Suppress("DEPRECATION")

package sk.ainet.apps.kgemma

/** Pre-#374 versioned names, kept one release for source compatibility. */
@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaIngestion"))
public typealias Gemma4Ingestion<T> = GemmaIngestion<T>

@Deprecated("Renamed (transformers#374).", ReplaceWith("GemmaStopTokens"))
public typealias Gemma4StopTokens = GemmaStopTokens
