package sk.ainet.apps.kllama

import kotlinx.coroutines.test.runTest
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.generate
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.createRandomAccessSource
import sk.ainet.lang.nn.dsl.decoder.DECODER_DEQUANTIZE_ALL
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaNetworkLoader
import kotlin.test.Test
import kotlin.time.Duration.Companion.minutes
import kotlin.time.measureTime
import kotlin.time.measureTimedValue

/**
 * Reads an environment variable. Cross-target: JVM `System.getenv`,
 * Kotlin/Native `platform.posix.getenv`; the browser-only JS/Wasm targets
 * have no process environment and return null, so the spike skips there.
 */
expect fun readEnv(name: String): String?

/**
 * Cross-target decode-throughput spike for SmolLM2-135M-Instruct (Q8_0) —
 * the reproducer half of transformers#272. Identical commonMain code runs on
 * JVM, Kotlin/Native (linuxX64 on a Linux host, iosSimulatorArm64 on a Mac),
 * so a real tok/s number can be compared across targets from one source.
 *
 * Uses the DSL path — `LlamaNetworkLoader.fromGguf(...).load()` +
 * `OptimizedLLMRuntime(DIRECT)` — which is the only Llama loader that honors
 * the engine loader's keep-packed default (packed block
 * tensor data since engine 0.49.0). The tokenizer is read from the GGUF's own
 * metadata (`GGUFTokenizer.fromSource`), so only the model path is needed.
 *
 * Gated: skips cleanly unless `SMOLLM2_MODEL` points at an existing `.gguf`.
 * Knobs: `SMOLLM2_QUANT` = `native` (default) | `dequant`; `SMOLLM2_STEPS`
 * (default 44 — one sentence, matching the field report's 44-token dare).
 *
 * ```
 * export SMOLLM2_MODEL=/abs/path/SmolLM2-135M-Instruct-Q8_0.gguf
 * ./gradlew :llm-runtime:kllama:jvmTest --tests '*SmolLm2InferenceSpike*'
 * ./gradlew :llm-runtime:kllama:linuxX64Test                 # native, on Linux
 * ./gradlew :llm-runtime:kllama:iosSimulatorArm64Test        # native, on macOS
 * ```
 */
class SmolLm2InferenceSpike {

    // runTest, not runBlocking: runBlocking does not exist on the JS/Wasm
    // targets, and this class compiles for every target of the module. The
    // generous timeout covers scalar Kotlin/Native decode (~0.6 tok/s).
    @Test
    fun measure_decode_tokens_per_second() = runTest(timeout = 30.minutes) {
        val modelPath = readEnv("SMOLLM2_MODEL")?.trim().orEmpty()
        if (modelPath.isEmpty()) {
            println("[skip] SMOLLM2_MODEL not set — skipping SmolLM2 inference spike.")
            return@runTest
        }
        if (!modelPath.endsWith(".gguf") || !SystemFileSystem.exists(Path(modelPath))) {
            println("[skip] SMOLLM2_MODEL=$modelPath is not an existing .gguf — skipping.")
            return@runTest
        }
        // "native" (default) keeps quantized tensors packed via the engine loader;
        // "dequant" requests dense FP32.
        val quantForm = when (readEnv("SMOLLM2_QUANT")?.trim()?.lowercase()) {
            "dequant" -> DECODER_DEQUANTIZE_ALL
            else -> null
        }
        val steps = readEnv("SMOLLM2_STEPS")?.trim()?.toIntOrNull() ?: 44

        // Without this, every Kotlin/Native target in this spike runs
        // packed-quant matmul scalar: DirectCpuExecutionContext registers
        // only the scalar (+ Accelerate on Apple) provider by default, and
        // K/N has no ServiceLoader to pick up the native-cinterop one (#300).
        installPlatformNativeKernels()
        val ctx = DirectCpuExecutionContext()

        val tokenizer = GGUFTokenizer.fromSource(
            SystemFileSystem.source(Path(modelPath)).buffered()
        )

        // Use the RandomAccessSource (streaming) overload: it keeps packed
        // quant weights packed under NATIVE_OPTIMIZED, whereas the
        // sequential Source path densifies them and trips on Q8_0's 34/32
        // byte ratio. `createRandomAccessSource` is cross-platform — JVM
        // FileChannel, native posix pread, Android positional reads (#922).
        val racProvider: () -> RandomAccessSource = {
            createRandomAccessSource(modelPath)
                ?: error("createRandomAccessSource returned null for $modelPath on this platform")
        }
        val (model, loadElapsed) = measureTimedValue {
            LlamaNetworkLoader.fromGguf(racProvider, weightForm = quantForm).load<FP32, Float>(ctx)
        }
        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
            bos = 1,
        )

        val prompt = tokenizer.encode("The capital of France is")
        val out = StringBuilder()
        var generated = 0
        val decodeElapsed = measureTime {
            runtime.generate(prompt = prompt, steps = steps, temperature = 0f) { tokenId ->
                generated++
                out.append(tokenizer.decode(tokenId))
            }
        }

        val tokPerSec = if (decodeElapsed.inWholeMilliseconds > 0) {
            generated.toDouble() / decodeElapsed.inWholeMilliseconds * 1000.0
        } else {
            Double.POSITIVE_INFINITY
        }
        println("=== SmolLM2-135M inference spike (${if (quantForm == null) "keep-packed" else "dequant"}) ===")
        println("load: ${loadElapsed.inWholeMilliseconds} ms")
        println("decode: $generated tokens in ${decodeElapsed.inWholeMilliseconds} ms")
        println("tok/s: ${(tokPerSec * 100).toLong() / 100.0}")
        println("output: ${out.toString().take(120)}")
    }
}
