package sk.ainet.transformers.iree.android

/**
 * The one run-time knob for how many cores an IREE local-task device uses (SKaiNET SKEEP-005
 * phase 2, "structure at compile time, cores at run time"). A `.vmfb` carries no core count;
 * the same number that drives the engine's eager `Schedule.parallelism` becomes the device's
 * task-topology group count when the device is created.
 *
 * This module depends on nothing from the engine, so the mapping takes a plain `Int`: pass
 * `ctx.schedule.parallelism` (`Schedule.Sequential.parallelism == 1` → one group).
 */
public object IreeTaskTopology {
    /** Environment knob shared with `gemma-iree`; `GEMMA_TASK_GROUPS` is its deprecated alias there. */
    public const val ENV: String = "SKAINET_TASK_GROUPS"

    /**
     * Group count from [ENV]: `null` when unset, blank, non-numeric or `0` — "let IREE detect
     * the topology" — otherwise the positive count.
     */
    public fun fromEnv(read: (String) -> String? = System::getenv): Int? = parse(read(ENV))

    /** [fromEnv] on a raw value. */
    public fun parse(raw: String?): Int? = raw?.trim()?.toIntOrNull()?.takeIf { it > 0 }

    /** Task groups for an engine schedule's `parallelism`: at least one group. */
    public fun groupCountFor(parallelism: Int): Int = parallelism.coerceAtLeast(1)
}
