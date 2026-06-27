package sk.ainet.models.llama

// JVM GC reclaims under allocation pressure on its own — no explicit collect needed.
internal actual fun gcCollectHint() {}
