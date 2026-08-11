package sk.ainet.apps.kllama

// Browser Wasm has no process environment — the inference spike always skips.
actual fun readEnv(name: String): String? = null
