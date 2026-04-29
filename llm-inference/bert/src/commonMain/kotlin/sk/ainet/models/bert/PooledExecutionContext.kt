package sk.ainet.models.bert

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.scratch.ScratchPool
import sk.ainet.lang.tensor.scratch.SizeClassedScratchPool

/**
 * Wraps an [ExecutionContext] with a [SizeClassedScratchPool] so that
 * upstream SIMD kernels and per-forward intermediates are pooled across
 * encoder calls.
 *
 * Use this when you intend to compute many embeddings from the same model:
 *
 * ```kotlin
 * val baseCtx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)
 * val pooledCtx = PooledExecutionContext(baseCtx)
 *
 * val runtime = BertRuntime(pooledCtx, weights, FP32::class)
 *
 * // Each forward acquires + releases scratch buffers in a per-call scope.
 * val v1 = runtime.encode(tokens1)
 * val v2 = runtime.encode(tokens2)   // reuses pooled buffers
 * ```
 *
 * For one-shot use the default `NoopScratchPool` on a plain
 * `DirectCpuExecutionContext` is fine — pooling has no benefit when the
 * pool is never reused.
 *
 * **Threading:** `SizeClassedScratchPool` is single-threaded by intent.
 * Concurrent encoder calls must each have their own pooled context.
 */
public class PooledExecutionContext(
    private val delegate: ExecutionContext,
    override val scratch: ScratchPool = SizeClassedScratchPool(),
) : ExecutionContext by delegate
