package sk.ainet.apps.kgemma

import org.junit.jupiter.api.Assumptions.assumeTrue
import java.io.File

/**
 * Shared preconditions for the real-checkpoint FunctionGemma-270M integration tests.
 *
 * Both gates are JUnit 5 assumptions — kgemma's `kotlin.test.Test` compiles to
 * `org.junit.jupiter.api.Test`, so an unmet assumption aborts the test and Jupiter records it
 * as SKIPPED. That matters: the previous `println(...) + return` guards reported an *absent*
 * checkpoint as a green PASS, while an under-provisioned heap killed the worker and surfaced
 * as a mystery SKIPPED. Both signals were inverted.
 *
 * Do NOT swap these for `org.junit.Assume` (JUnit 4): its `AssumptionViolatedException` is not
 * an `org.opentest4j.TestAbortedException`, so the Jupiter engine would record a FAILURE
 * instead of a skip. (JUnit 4 is not on kgemma's test classpath, so it would not compile here.)
 */
internal object FunctionGemmaFixture {

    /** FunctionGemma-270M Q5_K_M checkpoint; override with `GEMMA_GGUF`. */
    val gguf: String = System.getenv("GEMMA_GGUF")
        ?: "/home/miso/projects/coral/SKaiNET-embedded/sl2610-function-calling/models/functiongemma-physical-ai-v10-Q5_K_M.gguf"

    /**
     * Measured peak *live* heap for one export/trace pass is ~4.3 GiB: three near-simultaneous
     * FP32 copies of the 436,111,680-element weight set (loader tensors,
     * `TraceToGraphBuilder.extractFloatArray`'s `buffer.copyOf()`, and the `BufferHandle.Owned`
     * external-parameter ByteArrays). 4g OOMs; the module default of 12g passes.
     *
     * Gate at 7 GiB, compared in BYTES. An `-Xmx8g` JVM reports >= 7.7 GiB even under
     * ParallelGC's survivor-space shave, while `-Xmx4g` reports <= 4 GiB. Deliberately not the
     * `maxMemory() / 1024³ < 8` idiom used elsewhere in the repo — integer-GiB truncation turns
     * that 7.7 into 7 and would falsely skip a perfectly adequate run.
     */
    private const val MIN_HEAP_BYTES = 7L * 1024 * 1024 * 1024

    /** Checkpoint first, heap second — so CI (which has no model) skips for the honest reason. */
    fun assumeRealCheckpointRunnable() {
        assumeTrue(
            File(gguf).exists(),
            "FunctionGemma GGUF not present at $gguf — set GEMMA_GGUF to run this test. Skipping.",
        )
        val max = Runtime.getRuntime().maxMemory()
        assumeTrue(
            max >= MIN_HEAP_BYTES,
            "Heap too small: maxMemory=${max / (1024 * 1024)} MiB < 7168 MiB. " +
                "The Q5_K -> FP32 trace peaks at ~4.3 GiB live. " +
                "Rerun with -PkgemmaTestMaxHeap=12g (Gradle) or -Xmx12g (IDE run config). Skipping.",
        )
    }
}
