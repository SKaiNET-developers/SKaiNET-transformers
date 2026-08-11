package sk.ainet.models.functiongemma

import org.junit.jupiter.api.Assumptions.assumeTrue
import java.io.File

/**
 * Shared preconditions for the real-checkpoint FunctionGemma-270M integration tests in this
 * module — same gates as `:llm-runtime:kgemma`'s `FunctionGemmaFixture` (the checkpoint this
 * module's export harness was moved FROM), kept in lock-step so both modules skip/run together
 * during the kgemma-shim transition period.
 *
 * Both gates are JUnit 5 assumptions — `kotlin.test.Test` compiles to `org.junit.jupiter.api.Test`,
 * so an unmet assumption aborts the test and Jupiter records it as SKIPPED (not a false PASS on an
 * absent checkpoint, and not a mystery SKIPPED on an OOM kill).
 */
internal object FunctionGemmaFixture {

    /** FunctionGemma-270M Q5_K_M checkpoint; override with `GEMMA_GGUF`. */
    val gguf: String = System.getenv("GEMMA_GGUF")
        ?: "/home/miso/projects/coral/SKaiNET-embedded/sl2610-function-calling/models/functiongemma-physical-ai-v10-Q5_K_M.gguf"

    /**
     * Measured peak *live* heap for one export/trace pass is ~4.3 GiB (see kgemma's
     * FunctionGemmaFixture for the full accounting). Gate at 7 GiB, compared in BYTES — an
     * `-Xmx8g` JVM reports >= 7.7 GiB even under ParallelGC's survivor-space shave, while
     * `-Xmx4g` reports <= 4 GiB.
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
                "Rerun with -PfunctiongemmaTestMaxHeap=12g (Gradle) or -Xmx12g (IDE run config). Skipping.",
        )
    }
}
