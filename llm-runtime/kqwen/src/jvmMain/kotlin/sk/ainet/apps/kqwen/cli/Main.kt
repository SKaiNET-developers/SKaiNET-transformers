package sk.ainet.apps.kqwen.cli

import sk.ainet.apps.kqwen.QwenIngestion
import sk.ainet.apps.kqwen.QwenLoadConfig
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.MemSegWeightConverter
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import java.lang.foreign.Arena
import java.nio.file.Path
import kotlinx.coroutines.runBlocking
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.system.exitProcess
import kotlin.time.measureTime

private data class CliArgs(
    val modelPath: Path,
    val prompt: String,
    val steps: Int,
    val temperature: Float
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: kqwen <model> <prompt> [steps] [temperature]")
    println("  model        Path to .gguf model (required)")
    println("  prompt       Prompt text (required)")
    println("  steps        Generation steps (default: 32)")
    println("  temperature  Sampling temperature (default: 0.8)")
    println()
    println("Example:")
    println("  kqwen Qwen3-1.7B-Q8_0.gguf \"The capital of France is\" 32 0.0")
    exitProcess(if (errorMessage == null) 0 else 1)
}

private fun parseArgs(args: Array<String>): CliArgs {
    if (args.isEmpty()) usage("Missing arguments.")
    if (args[0] == "-h" || args[0] == "--help") usage()

    val modelPath = Path.of(args[0])
    val prompt = args.getOrElse(1) { usage("Prompt is required.") }
    val steps = args.getOrElse(2) { "32" }.toIntOrNull() ?: usage("Invalid steps value '${args[2]}'.")
    val temperature = args.getOrElse(3) { "0.8" }.toFloatOrNull() ?: usage("Invalid temperature '${args[3]}'.")

    return CliArgs(modelPath, prompt, steps, temperature)
}

fun main(args: Array<String>) {
    runBlocking {
        val cliArgs = parseArgs(args)
        val modelPath = cliArgs.modelPath

        if (!modelPath.exists()) error("Model not found: $modelPath")
        require(modelPath.extension.lowercase() == "gguf") { "Only .gguf models are supported" }

        val quantArena = Arena.ofShared()
        val memSegFactory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

        Runtime.getRuntime().addShutdownHook(Thread {
            quantArena.close()
            memSegFactory.close()
        })

        val ingestion = QwenIngestion<FP32>(
            ctx = ctx,
            dtype = FP32::class,
            config = QwenLoadConfig(
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
                allowQuantized = true
            )
        )

        println("Loading Qwen GGUF model from $modelPath (streaming mode)...")
        val rawWeights = ingestion.loadStreaming {
            JvmRandomAccessSource.open(modelPath.toString())
        }

        val runtimeWeights = if (rawWeights.quantTypes.isNotEmpty()) {
            println("Converting ${rawWeights.quantTypes.size} quantized tensors to MemorySegment-backed SIMD format...")
            MemSegWeightConverter.convert(rawWeights, ctx, quantArena)
        } else {
            rawWeights
        }

        val backend = CpuAttentionBackend<FP32>(
            ctx, runtimeWeights, FP32::class,
            ropeFreqBase = runtimeWeights.metadata.ropeFreqBase
        )
        val runtime = LlamaRuntime<FP32>(
            ctx, runtimeWeights, backend, FP32::class,
            eps = runtimeWeights.metadata.rmsNormEps
        )

        val tokenizer: Tokenizer = run {
            println("Loading embedded GGUF tokenizer...")
            JvmRandomAccessSource.open(modelPath.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }
        }

        val promptTokens = tokenizer.encode(cliArgs.prompt)

        println("Generating ${cliArgs.steps} tokens with temperature=${cliArgs.temperature}...")
        println("---")
        print(cliArgs.prompt)

        val elapsed = measureTime {
            runtime.generate(prompt = promptTokens, steps = cliArgs.steps, temperature = cliArgs.temperature) { id ->
                print(tokenizer.decode(id))
            }
        }.inWholeMilliseconds

        val tokPerSec = cliArgs.steps / elapsed.toDouble() * 1000
        println("\n---")
        println("tok/s: $tokPerSec")
    }
}
