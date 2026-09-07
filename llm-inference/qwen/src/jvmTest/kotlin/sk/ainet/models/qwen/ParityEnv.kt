package sk.ainet.models.qwen

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.schedule.Schedule
import sk.ainet.exec.schedule.CoroutineSchedule
import sk.ainet.lang.nn.dsl.decoder.DecoderKVCacheKind

/**
 * Environment switches for the golden-token parity gates (SKaiNET SKEEP-005, transformers#412/#413).
 *
 * - `SKAINET_ATTN_SCHEDULE` = `sequential` | `parallel` (alias `hardware`). Unset keeps the
 *   engine's platform default, which on the JVM is already [CoroutineSchedule.hardware].
 * - `SKAINET_KV_CACHE` = `append` (default) | `positional` (copy-free in-place K/V views).
 *
 * Whatever the combination, the gates assert the same oracle text: a schedule changes where
 * work runs, never what it computes.
 */
internal object ParityEnv {
    val scheduleName: String = System.getenv("SKAINET_ATTN_SCHEDULE")?.trim()?.lowercase().orEmpty()

    val kvCacheKind: DecoderKVCacheKind = when (System.getenv("SKAINET_KV_CACHE")?.trim()?.lowercase()) {
        "positional" -> DecoderKVCacheKind.POSITIONAL
        else -> DecoderKVCacheKind.APPEND
    }

    fun context(): DirectCpuExecutionContext = when (scheduleName) {
        "sequential" -> DirectCpuExecutionContext(schedule = Schedule.Sequential)
        "parallel", "hardware" -> DirectCpuExecutionContext(schedule = CoroutineSchedule.hardware())
        else -> DirectCpuExecutionContext()
    }

    fun describe(): String = "schedule=${scheduleName.ifEmpty { "default" }} kvCache=$kvCacheKind"
}
