package sk.ainet.apps.kapertus.cli

import sk.ainet.apps.kapertus.ApertusIngestion
import sk.ainet.apps.kapertus.ApertusLoadConfig
import sk.ainet.models.apertus.ApertusRuntimeWeights
import sk.ainet.models.apertus.ApertusRuntime
import sk.ainet.models.apertus.ApertusCpuAttentionBackend
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import kotlinx.coroutines.runBlocking
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.io.path.isDirectory
import kotlin.system.exitProcess
import kotlin.time.measureTime

private enum class ModelFormat { GGUF, SAFETENSORS }

private data class CliArgs(
    val modelPath: Path,
    val prompt: String?,
    val steps: Int,
    val temperature: Float
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: kapertus -m <model> [-s <steps>] [-k <temperature>] <prompt>")
    println("  -m, --model         Path to .gguf, .safetensors model, or HuggingFace directory (required)")
    println("  -s, --steps         Generation steps (default: 64)")
    println("  -k, --temperature   Sampling temperature (default: 0.8)")
    println("  -h, --help          Show this help")
    println()
    println("Example:")
    println("  kapertus -m model.gguf -s 96 -k 0.7 \"Hello, how are you?\"")
    exitProcess(if (errorMessage == null) 0 else 1)
}

private fun parseArgs(args: Array<String>): CliArgs {
    if (args.isEmpty()) usage("Missing arguments.")

    var model: String? = null
    var steps = 64
    var temperature = 0.8f
    var prompt: String? = null

    var idx = 0
    while (idx < args.size) {
        val arg = args[idx]

        fun nextValue(option: String): String {
            if (idx + 1 >= args.size) usage("Missing value for $option.")
            return args[++idx]
        }

        when {
            arg == "-h" || arg == "--help" -> usage()
            arg == "-m" || arg == "--model" -> model = nextValue(arg)
            arg.startsWith("--model=") -> model = arg.substringAfter("=")
            arg == "-s" || arg == "--steps" -> {
                val value = nextValue(arg)
                steps = value.toIntOrNull() ?: usage("Invalid steps value '$value'. Expected integer.")
            }
            arg.startsWith("--steps=") -> {
                val value = arg.substringAfter("=")
                steps = value.toIntOrNull() ?: usage("Invalid steps value '$value'. Expected integer.")
            }
            arg == "-k" || arg == "--temperature" -> {
                val value = nextValue(arg)
                temperature = value.toFloatOrNull() ?: usage("Invalid temperature '$value'. Expected float.")
            }
            arg.startsWith("--temperature=") -> {
                val value = arg.substringAfter("=")
                temperature = value.toFloatOrNull() ?: usage("Invalid temperature '$value'. Expected float.")
            }
            arg.startsWith("-") -> usage("Unknown option: $arg")
            else -> prompt = arg
        }
        idx++
    }

    if (model == null) usage("--model is required.")
    val modelPath = Path.of(model)
    if (!modelPath.exists()) usage("Model path does not exist: $model")

    return CliArgs(modelPath, prompt, steps, temperature)
}

private fun detectFormat(path: Path): ModelFormat {
    if (path.isDirectory()) {
        val indexFile = path.resolve("model.safetensors.index.json")
        val singleFile = path.resolve("model.safetensors")
        return when {
            indexFile.exists() || singleFile.exists() -> ModelFormat.SAFETENSORS
            else -> usage("Cannot detect model format in directory: $path")
        }
    }
    return when (path.extension.lowercase()) {
        "gguf" -> ModelFormat.GGUF
        "safetensors" -> ModelFormat.SAFETENSORS
        else -> usage("Unsupported file extension: ${path.extension}")
    }
}

fun main(args: Array<String>) {
    val cliArgs = parseArgs(args)
    val format = detectFormat(cliArgs.modelPath)
    val prompt = cliArgs.prompt ?: "Hello"

    println("Apertus-8B Inference (Kotlin)")
    println("Model: ${cliArgs.modelPath}")
    println("Format: $format")
    println("Steps: ${cliArgs.steps}, Temperature: ${cliArgs.temperature}")
    println()

    val ctx = DirectCpuExecutionContext()
    val ingestion = ApertusIngestion(
        ctx = ctx,
        dtype = FP32::class,
        config = ApertusLoadConfig(quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32)
    )

    println("Loading weights...")
    lateinit var runtimeWeights: ApertusRuntimeWeights<FP32>
    val loadTime = measureTime {
        runtimeWeights = runBlocking {
            when (format) {
                ModelFormat.GGUF -> ingestion.loadStreaming {
                    JvmRandomAccessSource.open(cliArgs.modelPath.toString())
                }
                ModelFormat.SAFETENSORS -> {
                    val indexPath = if (cliArgs.modelPath.isDirectory()) {
                        val index = cliArgs.modelPath.resolve("model.safetensors.index.json")
                        if (index.exists()) index.toString()
                        else cliArgs.modelPath.resolve("model.safetensors").toString()
                    } else {
                        cliArgs.modelPath.toString()
                    }
                    ingestion.loadSafeTensors(indexPath)
                }
            }
        }
    }
    println("Weights loaded in $loadTime")

    val metadata = runtimeWeights.metadata
    println("Model: ${metadata.architecture}, dim=${metadata.embeddingLength}, layers=${metadata.blockCount}, " +
        "heads=${metadata.headCount}, kv_heads=${metadata.kvHeadCount}, vocab=${metadata.vocabSize}")

    val backend = ApertusCpuAttentionBackend<FP32>(
        ctx = ctx,
        weights = runtimeWeights,
        dtype = FP32::class,
        ropeFreqBase = metadata.ropeTheta
    )

    val runtime = ApertusRuntime(
        ctx = ctx,
        weights = runtimeWeights,
        attentionBackend = backend,
        dtype = FP32::class
    )

    println()
    println("Prompt: $prompt")
    println("---")

    // Tokenizer integration TBD — forward pass is wired and ready
    println("[Tokenizer integration pending — forward pass is ready]")
}
