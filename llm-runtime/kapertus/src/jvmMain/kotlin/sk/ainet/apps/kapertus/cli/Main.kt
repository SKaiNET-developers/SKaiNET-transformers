package sk.ainet.apps.kapertus.cli

import sk.ainet.apps.kapertus.ApertusIngestion
import sk.ainet.apps.kapertus.ApertusLoadConfig
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.models.apertus.ApertusRuntimeWeights
import sk.ainet.models.apertus.ApertusRuntime
import sk.ainet.models.apertus.ApertusCpuAttentionBackend
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import kotlinx.coroutines.runBlocking
import java.nio.file.Path
import java.nio.file.Files
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.io.path.isDirectory
import kotlin.io.path.readText
import kotlin.system.exitProcess
import kotlin.time.measureTime

private enum class ModelFormat { GGUF, SAFETENSORS }

private data class CliArgs(
    val modelPath: Path,
    val tokenizerPath: Path?,
    val prompt: String?,
    val steps: Int,
    val temperature: Float,
    val contextLength: Int?
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: kapertus -m <model> [-t <tokenizer>] [-s <steps>] [-k <temperature>] [-c <context>] <prompt>")
    println("  -m, --model         Path to .gguf, .safetensors model, or HuggingFace directory (required)")
    println("  -t, --tokenizer     Path to tokenizer.json (auto-detected for .gguf and .safetensors)")
    println("  -s, --steps         Generation steps (default: 64)")
    println("  -k, --temperature   Sampling temperature (default: 0.8)")
    println("  -c, --context       Max context length (default: min(model, 2048))")
    println("  -h, --help          Show this help")
    println()
    println("Example:")
    println("  kapertus -m model.gguf -s 96 -k 0.7 \"Hello, how are you?\"")
    println("  kapertus -m ./apertus-hf/ \"Tell me about Kotlin\"")
    exitProcess(if (errorMessage == null) 0 else 1)
}

private fun parseArgs(args: Array<String>): CliArgs {
    if (args.isEmpty()) usage("Missing arguments.")

    var model: String? = null
    var tokenizer: String? = null
    var steps = 64
    var temperature = 0.8f
    var contextLength: Int? = null
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
            arg == "-t" || arg == "--tokenizer" -> tokenizer = nextValue(arg)
            arg.startsWith("--tokenizer=") -> tokenizer = arg.substringAfter("=")
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
            arg == "-c" || arg == "--context" -> {
                val value = nextValue(arg)
                contextLength = value.toIntOrNull() ?: usage("Invalid context length '$value'. Expected integer.")
            }
            arg.startsWith("--context=") -> {
                val value = arg.substringAfter("=")
                contextLength = value.toIntOrNull() ?: usage("Invalid context length '$value'. Expected integer.")
            }
            arg.startsWith("-") -> usage("Unknown option: $arg")
            else -> prompt = arg
        }
        idx++
    }

    if (model == null) usage("--model is required.")
    val modelPath = Path.of(model)
    if (!modelPath.exists()) usage("Model path does not exist: $model")

    return CliArgs(modelPath, tokenizer?.let(Path::of), prompt, steps, temperature, contextLength)
}

private fun detectFormat(path: Path): ModelFormat {
    if (path.isDirectory()) {
        val indexFile = path.resolve("model.safetensors.index.json")
        val singleFile = path.resolve("model.safetensors")
        return when {
            indexFile.exists() || singleFile.exists() -> ModelFormat.SAFETENSORS
            else -> {
                val gguf = path.toFile().listFiles()?.firstOrNull { it.extension == "gguf" }
                if (gguf != null) ModelFormat.GGUF
                else usage("Cannot detect model format in directory: $path")
            }
        }
    }
    return when (path.extension.lowercase()) {
        "gguf" -> ModelFormat.GGUF
        "safetensors" -> ModelFormat.SAFETENSORS
        else -> usage("Unsupported file extension: ${path.extension}")
    }
}

private fun resolveModelDir(path: Path): Path =
    if (path.isDirectory()) path else path.parent ?: path

fun main(args: Array<String>) = runBlocking {
    val cliArgs = parseArgs(args)
    val format = detectFormat(cliArgs.modelPath)
    val prompt = cliArgs.prompt ?: "Hello"

    println("Apertus Inference (Kotlin)")
    println("Model: ${cliArgs.modelPath}")
    println("Format: $format")
    println("Steps: ${cliArgs.steps}, Temperature: ${cliArgs.temperature}")
    println()

    val ctx = DirectCpuExecutionContext()
    val ingestion = ApertusIngestion(
        ctx = ctx,
        dtype = FP32::class,
        config = ApertusLoadConfig(
            quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
        )
    )

    // ---- Load weights ----
    println("Loading weights...")
    lateinit var runtimeWeights: ApertusRuntimeWeights<FP32>
    val loadTime = measureTime {
        runtimeWeights = when (format) {
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
    println("Weights loaded in $loadTime")

    // Limit context length to avoid OOM (KV cache allocation)
    val maxCtx = cliArgs.contextLength ?: minOf(runtimeWeights.metadata.contextLength, 2048)
    if (maxCtx < runtimeWeights.metadata.contextLength) {
        runtimeWeights = runtimeWeights.copy(
            metadata = runtimeWeights.metadata.copy(contextLength = maxCtx)
        )
    }

    val metadata = runtimeWeights.metadata
    println("  arch=${metadata.architecture}, dim=${metadata.embeddingLength}, layers=${metadata.blockCount}, " +
        "heads=${metadata.headCount}, kv_heads=${metadata.kvHeadCount}, vocab=${metadata.vocabSize}, " +
        "ctx=${metadata.contextLength}")

    // ---- Load tokenizer ----
    val tokenizerPath: Path? = when {
        cliArgs.tokenizerPath != null -> cliArgs.tokenizerPath
        format == ModelFormat.SAFETENSORS -> {
            val modelDir = resolveModelDir(cliArgs.modelPath)
            val autoTokenizer = modelDir.resolve("tokenizer.json")
            if (autoTokenizer.exists()) autoTokenizer else null
        }
        else -> null
    }

    val tokenizer: Tokenizer = when {
        format == ModelFormat.SAFETENSORS -> {
            val tPath = tokenizerPath ?: error("tokenizer.json not found for SafeTensors model")
            if (!tPath.exists()) error("Tokenizer not found: $tPath")
            println("Loading tokenizer from $tPath...")
            GGUFTokenizer.fromTokenizerJson(tPath.readText())
        }
        format == ModelFormat.GGUF && tokenizerPath == null -> {
            println("Loading embedded GGUF tokenizer...")
            JvmRandomAccessSource.open(cliArgs.modelPath.toString()).use { source ->
                GGUFTokenizer.fromRandomAccessSource(source)
            }
        }
        else -> {
            val tPath = tokenizerPath ?: error("Tokenizer path is required. Use -t/--tokenizer.")
            if (!tPath.exists()) error("Tokenizer not found: $tPath")
            println("Loading tokenizer from $tPath...")
            GGUFTokenizer.fromTokenizerJson(tPath.readText())
        }
    }

    // ---- Build runtime ----
    val backend = ApertusCpuAttentionBackend<FP32>(
        ctx = ctx,
        weights = runtimeWeights,
        dtype = FP32::class
    )

    val runtime = ApertusRuntime(
        ctx = ctx,
        weights = runtimeWeights,
        attentionBackend = backend,
        dtype = FP32::class
    )

    // ---- Generate ----
    val promptTokens = tokenizer.encode(prompt)
    println()
    println("Prompt: $prompt (${promptTokens.size} tokens)")
    println("Generating ${cliArgs.steps} tokens with temperature=${cliArgs.temperature}...")
    println("---")
    print(prompt)

    var generated = 0
    val elapsed = measureTime {
        runtime.generate(prompt = promptTokens, steps = cliArgs.steps, temperature = cliArgs.temperature) { id ->
            print(tokenizer.decode(id))
            generated++
        }
    }.inWholeMilliseconds

    val tokPerSec = if (elapsed > 0) generated / elapsed.toDouble() * 1000 else 0.0
    println("\n---")
    println("Generated $generated tokens in ${elapsed}ms (%.2f tok/s)".format(tokPerSec))
}
