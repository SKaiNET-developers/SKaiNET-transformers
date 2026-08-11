@file:Suppress("unused")

package sk.ainet.apps.kllama.agent

/**
 * Re-export for backward compatibility.
 * The canonical definition is now in [sk.ainet.apps.llm.GenerateResult]
 * (promoted to `llm-core` in issue #49 Phase 1 so any runner can use
 * stop-token-aware generation without the agent layer).
 */
public typealias GenerateResult = sk.ainet.apps.llm.GenerateResult
