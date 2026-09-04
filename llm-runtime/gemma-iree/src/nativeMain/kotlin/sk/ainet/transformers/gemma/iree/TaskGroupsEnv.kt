package sk.ainet.transformers.gemma.iree

import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import platform.posix.getenv

/**
 * The run-time core knob for IREE task topology (SKaiNET SKEEP-005 phase 2): `SKAINET_TASK_GROUPS`,
 * shared with the Android runtime's `IreeTaskTopology`; `GEMMA_TASK_GROUPS` stays as a
 * deprecated alias. Default 2 (the SL2610's two A55 cores); `0` or empty drops the flag.
 */
@OptIn(ExperimentalForeignApi::class)
internal object TaskGroupsEnv {
    const val ENV = "SKAINET_TASK_GROUPS"
    const val DEPRECATED_ENV = "GEMMA_TASK_GROUPS"
    const val DEFAULT = 2

    fun read(): Int? {
        val raw = getenv(ENV)?.toKString()
            ?: getenv(DEPRECATED_ENV)?.toKString()?.also {
                println("[gemma-iree] $DEPRECATED_ENV is deprecated; use $ENV")
            }
        return (raw?.trim()?.toIntOrNull() ?: DEFAULT).takeIf { it > 0 }
    }
}
