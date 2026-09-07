package sk.ainet.lang.nn.transformer.schedule

/**
 * How a [sk.ainet.lang.nn.transformer.MultiHeadAttention] layer splits its heads across the
 * tasks of the context's `Schedule` (SKEEP-005). A policy is a deployment choice — it never
 * changes the arithmetic of a head, so every policy produces the sequential result bit for bit.
 */
public sealed interface AttentionSchedulePolicy {

    /** Today's behaviour: every head on the calling thread, whatever the context's schedule. */
    public object Sequential : AttentionSchedulePolicy

    /** One unit per query head, `ceil(nHeads / parallelism)` heads per task. */
    public data class PerHead(val minSeqKV: Int = DEFAULT_MIN_SEQ_KV) : AttentionSchedulePolicy

    /**
     * One unit per KV head: the `nHeads / nKVHeads` query heads sharing a KV group run on one
     * task, so each K/V slice is streamed once per core.
     */
    public data class PerKVGroup(val minSeqKV: Int = DEFAULT_MIN_SEQ_KV) : AttentionSchedulePolicy

    /** [PerKVGroup] when there are at least as many KV groups as workers, else [PerHead]. The default. */
    public data class Auto(val minSeqKV: Int = DEFAULT_MIN_SEQ_KV) : AttentionSchedulePolicy

    public companion object {
        /** Below this many cached positions a region costs more than it saves. */
        public const val DEFAULT_MIN_SEQ_KV: Int = 64
    }
}

/**
 * The resolved split for one forward: [units] work items of [headsPerUnit] consecutive query
 * heads, handed to `Schedule.forRange(units, grain)`. `tasks` bounds the scratch slots a
 * coordinator must pre-allocate.
 */
public class HeadPlan(public val units: Int, public val headsPerUnit: Int, public val grain: Int) {
    public val tasks: Int get() = (units + grain - 1) / grain
    override fun toString(): String = "HeadPlan(units=$units, headsPerUnit=$headsPerUnit, grain=$grain, tasks=$tasks)"
}

/** `null` means "run on the calling thread". */
public fun AttentionSchedulePolicy.plan(nHeads: Int, nKVHeads: Int, seqKV: Int, parallelism: Int): HeadPlan? {
    if (parallelism <= 1 || nHeads <= 1) return null
    val nRep = nHeads / nKVHeads
    fun ceilDiv(a: Int, b: Int) = (a + b - 1) / b
    fun perHead() = HeadPlan(units = nHeads, headsPerUnit = 1, grain = ceilDiv(nHeads, parallelism))
    fun perKVGroup() = HeadPlan(units = nKVHeads, headsPerUnit = nRep, grain = ceilDiv(nKVHeads, parallelism))
    return when (this) {
        AttentionSchedulePolicy.Sequential -> null
        is AttentionSchedulePolicy.PerHead -> if (seqKV < minSeqKV) null else perHead()
        is AttentionSchedulePolicy.PerKVGroup -> if (seqKV < minSeqKV) null else perKVGroup()
        is AttentionSchedulePolicy.Auto -> when {
            seqKV < minSeqKV -> null
            nRep > 1 && nKVHeads >= parallelism -> perKVGroup()
            else -> perHead()
        }
    }
}
