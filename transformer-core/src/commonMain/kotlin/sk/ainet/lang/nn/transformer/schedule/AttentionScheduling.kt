package sk.ainet.lang.nn.transformer.schedule

import sk.ainet.context.schedule.Schedule
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.types.DType

/**
 * Set the schedule policy and/or an explicit schedule on every [MultiHeadAttention] in a module
 * tree (SKEEP-005) — the deployment-side knob for a model that was defined once. `null` leaves
 * the respective setting untouched.
 */
public fun <T : DType, V> Module<T, V>.configureAttention(
    policy: AttentionSchedulePolicy? = null,
    schedule: Schedule? = null,
): Module<T, V> {
    val queue = ArrayDeque<Module<T, V>>()
    queue.addLast(this)
    while (queue.isNotEmpty()) {
        val m = queue.removeFirst()
        if (m is MultiHeadAttention<T, V>) {
            policy?.let { m.schedulePolicy = it }
            if (schedule != null) m.schedule = schedule
        }
        queue.addAll(m.modules)
    }
    return this
}
