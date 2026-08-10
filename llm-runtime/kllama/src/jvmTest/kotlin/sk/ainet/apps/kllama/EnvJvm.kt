package sk.ainet.apps.kllama

actual fun readEnv(name: String): String? = System.getenv(name)
