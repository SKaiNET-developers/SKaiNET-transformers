package sk.ainet.apps.kgemma.cli

import sk.ainet.apps.kgemma.Gemma3nIngestion
import sk.ainet.apps.kgemma.Gemma3nLoadConfig
import sk.ainet.apps.kgemma.Gemma4Ingestion
import sk.ainet.apps.kgemma.Gemma4LoadConfig
import sk.ainet.apps.kgemma.loadDslRuntimeNative
import sk.ainet.apps.kgemma.loadDslRuntimeNativeStreaming
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.chat.ChatSession
import sk.ainet.apps.kllama.chat.ModelMetadata
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.llm.generate
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
import kotlin.io.path.isDirectory
import kotlin.io.path.readText
import kotlin.system.exitProcess
import kotlin.time.measureTime

private enum class ModelFormat { GGUF, SAFETENSORS }

private enum class GemmaVariant { GEMMA3N, GEMMA4 }

/**
 * Detect Gemma model variant from config.json or GGUF metadata.
 * For SafeTensors directories: reads model_type from config.json.
 * For GGUF files: peeks at general.architecture metadata field.
 * Falls back to GEMMA3N if detection fails.
 */
private fun detectGemmaVariant(modelPath: Path, format: ModelFormat): GemmaVariant {
    // Try config.json in model directory
    val modelDir = if (modelPath.isDirectory()) modelPath else modelPath.parent ?: modelPath
    val configFile = modelDir.resolve("config.json")
    if (configFile.exists()) {
        val configText = configFile.readText()
        val modelTypeRegex = """"model_type"\s*:\s*"(\w+)"""".toRegex()
        val match = modelTypeRegex.find(configText)
        if (match != null) {
            val modelType = match.groupValues[1]
            if (modelType == "gemma4") return GemmaVariant.GEMMA4
            if (modelType.startsWith("gemma3") || modelType == "gemma3n") return GemmaVariant.GEMMA3N
        }
    }

    // For GGUF: peek at architecture from metadata
    if (format == ModelFormat.GGUF && modelPath.exists()) {
        try {
            JvmRandomAccessSource.open(modelPath.toString()).use { source ->
                val reader = sk.ainet.io.gguf.StreamingGGUFReader.open(source)
                val arch = reader.fields["general.architecture"]
                if (arch is String && arch.contains("gemma4", ignoreCase = true)) {
                    return GemmaVariant.GEMMA4
                }
                // Also check filename as last resort
                val filename = modelPath.fileName.toString().lowercase()
                if (filename.contains("gemma-4") || filename.contains("gemma4")) {
                    return GemmaVariant.GEMMA4
                }
            }
        } catch (_: Exception) {
            // Fall through to default
        }
    }

    // Filename heuristic for non-GGUF too
    val filename = modelPath.fileName?.toString()?.lowercase() ?: ""
    if (filename.contains("gemma-4") || filename.contains("gemma4")) {
        return GemmaVariant.GEMMA4
    }

    return GemmaVariant.GEMMA3N
}

private data class CliArgs(
    val modelPath: Path,
    val prompt: String,
    val steps: Int,
    val temperature: Float,
    val agent: Boolean = false,
    val toolsSpec: String? = null
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: kgemma <model> <prompt> [steps] [temperature] [--agent]")
    println("  model               Path to .gguf model or SafeTensors directory (required)")
    println("  prompt              Prompt text (required)")
    println("  steps               Generation steps (default: 32)")
    println("  temperature         Sampling temperature (default: 0.8)")
    println("  --agent             Route through ChatSession with the model-appropriate")
    println("                      chat template and an agent tool registry. Default off")
    println("                      (raw runtime.generate).")
    println("  --tools=LIST        Comma-separated tool names to register when --agent is set.")
    println("                      Available: ${DefaultTools.names.joinToString(", ")}.")
    println("                      Default: calculator.")
    println()
    println("Runtime: gemmaNetwork() + OptimizedLLMRuntime (DSL, NATIVE_OPTIMIZED Q4_0/Q8_0 +")
    println("         dequant K-series to FP32 until a K-kernel lands).")
    println()
    println("Example:")
    println("  kgemma models/gemma-3-270m-it-Q8_0.gguf \"Hello, how are you?\" 32 0.8")
    println("  kgemma model.gguf \"What is 3+4?\" 64 0.0 --agent")
    println("  kgemma model.gguf \"List /tmp\" 64 0.0 --agent --tools=list_files")
    exitProcess(if (errorMessage == null) 0 else 1)
}

private fun parseArgs(args: Array<String>): CliArgs {
    if (args.isEmpty()) usage("Missing arguments.")
    if (args[0] == "-h" || args[0] == "--help") usage()

    // Split positional args from flags so flags can appear in any position.
    val flags = args.filter { it.startsWith("--") }
    val positional = args.filter { !it.startsWith("--") }

    val modelPath = Path.of(positional.getOrElse(0) { usage("Model path is required.") })
    val prompt = positional.getOrElse(1) { usage("Prompt is required.") }
    val steps = positional.getOrElse(2) { "32" }.toIntOrNull() ?: usage("Invalid steps value '${positional[2]}'.")
    val temperature = positional.getOrElse(3) { "0.8" }.toFloatOrNull() ?: usage("Invalid temperature '${positional[3]}'.")

    var agent = false
    var toolsSpec: String? = null
    for (flag in flags) {
        when {
            flag == "--agent" -> agent = true
            flag.startsWith("--tools=") -> toolsSpec = flag.substringAfter("=")
            flag == "--help" || flag == "-h" -> usage()
            else -> usage("Unknown flag '$flag'.")
        }
    }

    if (toolsSpec != null && !agent) {
        usage("--tools has no effect without --agent.")
    }

    return CliArgs(modelPath, prompt, steps, temperature, agent, toolsSpec)
}

private fun detectFormat(path: Path): ModelFormat {
    if (path.isDirectory()) {
        val st = path.resolve("model.safetensors")
        val stIndex = path.resolve("model.safetensors.index.json")
        if (st.exists() || stIndex.exists()) return ModelFormat.SAFETENSORS
        error("Directory $path does not contain model.safetensors or model.safetensors.index.json")
    }
    return when (path.extension.lowercase()) {
        "gguf" -> ModelFormat.GGUF
        "safetensors" -> ModelFormat.SAFETENSORS
        else -> error("Unsupported model format: ${path.extension}. Use .gguf or .safetensors")
    }
}

fun main(args: Array<String>) {
    runBlocking {
        val cliArgs = parseArgs(args)
        val modelPath = cliArgs.modelPath

        if (!modelPath.exists()) error("Model not found: $modelPath")

        val format = detectFormat(modelPath)

        val memSegFactory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)
        // Arena owns all Q4/Q8 MemorySegment tensor buffers produced by the
        // DSL+NATIVE_OPTIMIZED path. Closed on JVM exit alongside the factory.
        val quantArena = Arena.ofShared()

        Runtime.getRuntime().addShutdownHook(Thread {
            quantArena.close()
            memSegFactory.close()
        })

        val variant = detectGemmaVariant(modelPath, format)
        println("Detected model variant: $variant")

        val runtime: InferenceRuntime<FP32> = when (variant) {
            GemmaVariant.GEMMA4 -> {
                val ingestion = Gemma4Ingestion<FP32>(
                    ctx = ctx,
                    dtype = FP32::class,
                    config = Gemma4LoadConfig(
                        quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                        allowQuantized = true
                    )
                )
                val runtimeLabel = "gemmaNetwork() + OptimizedLLMRuntime (DSL, NATIVE_OPTIMIZED Q4_0/Q8_0 + dequant K-series)"
                when (format) {
                    ModelFormat.GGUF -> {
                        println("Loading Gemma 4 GGUF model from $modelPath via $runtimeLabel (streaming)...")
                        ingestion.loadDslRuntimeNativeStreaming(
                            randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
                            ctx = ctx,
                            dtype = FP32::class,
                            arena = quantArena
                        )
                    }
                    ModelFormat.SAFETENSORS -> {
                        val modelDir = if (modelPath.isDirectory()) modelPath else modelPath.parent ?: modelPath
                        val indexPath = modelDir.resolve("model.safetensors.index.json")
                        val safetensorsPath = if (indexPath.exists()) indexPath.toString()
                            else modelDir.resolve("model.safetensors").toString()
                        println("Loading Gemma 4 SafeTensors model from $safetensorsPath via $runtimeLabel...")
                        ingestion.loadDslRuntimeFromSafeTensors(safetensorsPath)
                    }
                }
            }
            GemmaVariant.GEMMA3N -> {
                val ingestion = Gemma3nIngestion<FP32>(
                    ctx = ctx,
                    dtype = FP32::class,
                    config = Gemma3nLoadConfig(
                        quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                        allowQuantized = true
                    )
                )
                when (format) {
                    ModelFormat.GGUF -> {
                        println("Loading Gemma 3n GGUF model from $modelPath (streaming mode)...")
                        ingestion.loadRuntimeStreaming {
                            JvmRandomAccessSource.open(modelPath.toString())
                        }
                    }
                    ModelFormat.SAFETENSORS -> {
                        val modelDir = if (modelPath.isDirectory()) modelPath else modelPath.parent ?: modelPath
                        val indexPath = modelDir.resolve("model.safetensors.index.json")
                        val safetensorsPath = if (indexPath.exists()) indexPath.toString()
                            else modelDir.resolve("model.safetensors").toString()
                        println("Loading Gemma 3n SafeTensors model from $safetensorsPath...")
                        ingestion.loadRuntimeFromSafeTensors(safetensorsPath)
                    }
                }
            }
        }

        // Load tokenizer from GGUF or from tokenizer.json in model directory
        val tokenizer: Tokenizer = when (format) {
            ModelFormat.GGUF -> {
                println("Loading embedded GGUF tokenizer...")
                JvmRandomAccessSource.open(modelPath.toString()).use { source ->
                    GGUFTokenizer.fromRandomAccessSource(source)
                }
            }
            ModelFormat.SAFETENSORS -> {
                val modelDir = if (modelPath.isDirectory()) modelPath else modelPath.parent ?: modelPath
                val tokenizerFile = modelDir.resolve("tokenizer.json")
                if (!tokenizerFile.exists()) error("tokenizer.json not found in $modelDir")
                println("Loading tokenizer from $tokenizerFile...")
                GGUFTokenizer.fromTokenizerJson(tokenizerFile.readText())
            }
        }

        if (cliArgs.agent) {
            // ChatSession routes the prompt through the model-appropriate
            // chat template and runs the agent loop (single turn, up to 5
            // tool rounds by default). Architecture string drives template
            // selection via ToolCallingSupportResolver — Gemma 4 should land
            // on Gemma4ChatTemplate via Gemma4ToolCallingSupport (added in
            // Phase 6b).
            val metadata = ModelMetadata(
                family = "gemma",
                architecture = when (variant) {
                    GemmaVariant.GEMMA4 -> "gemma4"
                    GemmaVariant.GEMMA3N -> "gemma3n"
                },
                sourceFormat = when (format) {
                    ModelFormat.GGUF -> "gguf"
                    ModelFormat.SAFETENSORS -> "hf"
                }
            )
            val session = ChatSession(runtime, tokenizer, metadata)

            // Resolve --tools=… spec. Default (no flag) registers the
            // calculator — enough to exercise the full template + parse +
            // dispatch loop on Gemma 4 and matches skainet-cli's default.
            val tools = if (cliArgs.toolsSpec.isNullOrBlank()) {
                listOf(sk.ainet.apps.kllama.cli.CalculatorTool())
            } else {
                DefaultTools.parse(cliArgs.toolsSpec) { name ->
                    System.err.println("Error: unknown tool '$name'. Available: ${DefaultTools.names.joinToString(", ")}.")
                } ?: exitProcess(1)
            }

            println("Agent mode: chat template = ${session.chatTemplate::class.simpleName}, " +
                "tool-calling provider = ${session.providerFamily}, " +
                "tools = [${tools.joinToString(", ") { it.definition.name }}]")
            println("Generating up to ${cliArgs.steps} tokens/round at temperature=${cliArgs.temperature}...")
            println("---")
            print(cliArgs.prompt)
            println()
            val elapsed = measureTime {
                val response = session.runSingleTurn(
                    prompt = cliArgs.prompt,
                    tools = tools,
                    maxTokens = cliArgs.steps,
                    temperature = cliArgs.temperature
                )
                println(response)
            }.inWholeMilliseconds
            println("---")
            println("agent round elapsed: ${elapsed}ms")
        } else {
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
}
