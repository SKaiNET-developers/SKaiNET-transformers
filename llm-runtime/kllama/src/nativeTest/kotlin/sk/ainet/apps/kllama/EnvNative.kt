package sk.ainet.apps.kllama

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import platform.posix.getenv

@OptIn(ExperimentalForeignApi::class)
actual fun readEnv(name: String): String? = getenv(name)?.toKString()
