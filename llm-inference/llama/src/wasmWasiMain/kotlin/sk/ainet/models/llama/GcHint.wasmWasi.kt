package sk.ainet.models.llama

// JS / Wasm GC is automatic and these targets don't run the board load path — no-op.
internal actual fun gcCollectHint() {}
