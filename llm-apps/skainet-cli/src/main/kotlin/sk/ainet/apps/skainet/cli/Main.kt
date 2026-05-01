package sk.ainet.apps.skainet.cli

import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.cli.AgentCli
import sk.ainet.apps.kllama.cli.ToolCallingDemo
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.ModelFamily
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.llm.UnifiedModelLoader
import sk.ainet.apps.llm.generate
import sk.ainet.models.apertus.ApertusNetworkLoader
import sk.ainet.models.gemma.Gemma4WeightLoader
import sk.ainet.models.gemma.Gemma4Weights
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.models.gemma.convertGemmaWeightsToMemSeg
import sk.ainet.apps.llm.backend.BackendRegistry
import sk.ainet.apps.llm.backend.bestAvailable
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.apps.kllama.chat.ModelMetadata
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaWeightLoader
import sk.ainet.models.llama.LlamaWeightMapper
import sk.ainet.models.llama.MemSegWeightConverter
import java.lang.foreign.Arena
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.system.exitProcess
import kotlinx.coroutines.runBlocking
import kotlin.time.measureTime

private data class CliArgs(
    val modelPath: Path,
    val steps: Int,
    val temperature: Float,
    val prompt: String?,
    val chatMode: Boolean,
    val agentMode: Boolean,
    val demoMode: Boolean,
    val templateName: String?,
    val contextLength: Int?
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: skainet -m <model.gguf> [-s <steps>] [-k <temperature>] [--chat] [--agent] [--demo] [--template=NAME] <prompt>")
    println()
    println("  -m, --model         Path to .gguf model (required)")
    println("  -s, --steps         Generation steps (default: 64)")
    println("  -k, --temperature   Sampling temperature (default: 0.8)")
    println("  --chat              Interactive chat mode")
    println("  --agent             Interactive agent mode with tool calling")
    println("  --demo              Tool calling demo with file listing and calculator")
    println("  --template=NAME     Chat template: llama3, chatml, qwen, gemma (auto-detected if omitted)")
    println("  --context=N         Cap context length to N tokens")
    println("  -h, --help          Show this help")
    println()
    println("Supported architectures (auto-detected from GGUF metadata):")
    println("  LLaMA, Mistral, Qwen2, Qwen3, Gemma, Apertus")
    println()
    println("Examples:")
    println("  skainet -m model.gguf \"The capital of France is\"")
    println("  skainet -m model.gguf --chat")
    println("  skainet -m model.gguf --demo \"What is 2 + 2?\"")
    exitProcess(if (errorMessage == null) 0 else 1)
}

private fun parseArgs(args: Array<String>): CliArgs {
    if (args.isEmpty()) usage("Missing arguments.")

    var model: String? = null
    var steps = 64
    var temperature = 0.8f
    var prompt: String? = null
    var chatMode = false
    var agentMode = false
    var demoMode = false
    var templateName: String? = null
    var contextLength: Int? = null

    var idx = 0
    fun nextValue(flag: String): String {
        if (idx + 1 >= args.size) usage("$flag requires a value.")
        return args[++idx]
    }

    while (idx < args.size) {
        val arg = args[idx]
        when {
            arg == "-h" || arg == "--help" -> usage()
            arg == "-m" || arg == "--model" -> model = nextValue(arg)
            arg.startsWith("--model=") -> model = arg.substringAfter("=")
            arg == "-s" || arg == "--steps" -> {
                val value = nextValue(arg)
                steps = value.toIntOrNull() ?: usage("Invalid steps '$value'.")
            }
            arg == "-k" || arg == "--temperature" -> {
                val value = nextValue(arg)
                temperature = value.toFloatOrNull() ?: usage("Invalid temperature '$value'.")
            }
            arg == "--chat" -> chatMode = true
            arg == "--agent" -> agentMode = true
            arg == "--demo" -> demoMode = true
            arg.startsWith("--template=") -> templateName = arg.substringAfter("=")
            arg.startsWith("--context=") -> {
                val value = arg.substringAfter("=")
                contextLength = value.toIntOrNull() ?: usage("Invalid context length '$value'.")
            }
            arg.startsWith("-") -> usage("Unknown option '$arg'.")
            else -> {
                if (prompt != null) usage("Multiple prompts provided.")
                prompt = arg
            }
        }
        idx++
    }

    val modelPath = model?.let { Path.of(it) } ?: usage("Model is required (-m/--model).")

    if (!chatMode && !agentMode && !demoMode && prompt == null) {
        usage("Prompt is required (or use --chat/--agent/--demo mode).")
    }

    return CliArgs(modelPath, steps, temperature, prompt, chatMode, agentMode, demoMode, templateName, contextLength)
}

fun main(args: Array<String>) {
    runBlocking {
        val cliArgs = parseArgs(args)
        val modelPath = cliArgs.modelPath

        if (!modelPath.exists()) error("Model not found: $modelPath")
        if (modelPath.extension.lowercase() != "gguf") {
            error("Only GGUF models are supported by the unified CLI. Use model-specific CLIs for other formats.")
        }

        // Auto-detect architecture
        val modelInfo = UnifiedModelLoader.peek { JvmRandomAccessSource.open(modelPath.toString()) }
        println("Architecture: ${modelInfo.architecture}, Family: ${modelInfo.family.displayName}")
        println("Dimensions: ${modelInfo.embeddingLength}d, ${modelInfo.blockCount} layers, vocab=${modelInfo.vocabSize}")

        // Select backend
        val provider = BackendRegistry.bestAvailable()
        println("Backend: ${provider.displayName}")

        // Set up execution context
        val quantArena = Arena.ofShared()
        val memSegFactory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

        Runtime.getRuntime().addShutdownHook(Thread {
            quantArena.close()
            memSegFactory.close()
        })

        // Load model based on detected family. Gemma and Apertus route
        // through the DSL pipeline (their respective network() builder +
        // OptimizedLLMRuntime); everything else (LLaMA, Qwen, ...) takes
        // the LlamaRuntime path which supports NATIVE_OPTIMIZED quant
        // tensors for low-RAM loads. Apertus had previously fallen
        // through to the LlamaRuntime branch — that runtime doesn't
        // implement Apertus's xIELU activation, QK-Norm, or ungated FFN,
        // so logits silently diverged from the checkpoint's intent. See
        // APERTUS_ROLLOUT.md (PR 1) for the rollout context.
        val runtime: InferenceRuntime<FP32> = if (modelInfo.family == ModelFamily.GEMMA) {
            println("Loading Gemma GGUF model from $modelPath via gemmaNetwork() + OptimizedLLMRuntime (NATIVE_OPTIMIZED)...")
            if (cliArgs.contextLength != null) {
                println("  --context flag currently ignored on the Gemma path; uses model default capped to 4096.")
            }
            val rawWeights = Gemma4WeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED
            ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
            @Suppress("UNCHECKED_CAST")
            val converted = convertGemmaWeightsToMemSeg(rawWeights, ctx, quantArena) as Gemma4Weights<FP32, Float>
            val model = GemmaNetworkLoader.fromWeights(ctx, converted, FP32::class)
            OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class)
        } else if (modelInfo.family == ModelFamily.APERTUS) {
            println("Loading Apertus GGUF model from $modelPath via apertusNetwork() + OptimizedLLMRuntime (NATIVE_OPTIMIZED)...")
            if (cliArgs.contextLength != null) {
                println("  --context flag currently ignored on the Apertus path; uses model default.")
            }
            val model = ApertusNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED
            ).load<FP32, Float>(ctx)
            OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class)
        } else {
            val acceptedArchitectures = modelInfo.family.architectures + setOf(modelInfo.architecture)
            val loader = LlamaWeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
                acceptedArchitectures = acceptedArchitectures
            )

            println("Loading GGUF model from $modelPath (${modelInfo.family.displayName}, streaming)...")
            val loaded = loader.loadToMapStreaming<FP32, Float>(ctx, FP32::class)
            val rawWeights = LlamaWeightMapper.map(loaded)

            val runtimeWeights = if (rawWeights.quantTypes.isNotEmpty()) {
                println("Converting ${rawWeights.quantTypes.size} quantized tensors to SIMD format...")
                MemSegWeightConverter.convert(rawWeights, ctx, quantArena)
            } else {
                rawWeights
            }

            if (cliArgs.contextLength != null) {
                println("Context length capped to ${cliArgs.contextLength} (model default: ${runtimeWeights.metadata.contextLength})")
            }

            val backend = CpuAttentionBackend<FP32>(
                ctx, runtimeWeights, FP32::class,
                ropeFreqBase = runtimeWeights.metadata.ropeFreqBase,
                maxContextLength = cliArgs.contextLength
            )

            @Suppress("DEPRECATION")
            LlamaRuntime<FP32>(
                ctx, runtimeWeights, backend, FP32::class,
                eps = runtimeWeights.metadata.rmsNormEps
            )
        }

        // Load tokenizer from GGUF
        println("Loading embedded GGUF tokenizer...")
        val tokenizer: Tokenizer = JvmRandomAccessSource.open(modelPath.toString()).use { source ->
            TokenizerFactory.fromGGUF(source)
        }

        // Build model metadata for chat template auto-detection
        val metadata = ModelMetadata(
            family = modelInfo.family.id,
            architecture = modelInfo.architecture,
            chatTemplate = modelInfo.fields["tokenizer.chat_template"] as? String
        )

        // Dispatch
        if (cliArgs.chatMode || cliArgs.agentMode || cliArgs.demoMode) {
            when {
                cliArgs.demoMode -> {
                    val demo = ToolCallingDemo(runtime, tokenizer, cliArgs.templateName, metadata)
                    if (cliArgs.prompt != null) {
                        demo.runSingleShot(cliArgs.prompt, maxTokens = cliArgs.steps, temperature = cliArgs.temperature)
                    } else {
                        demo.run(maxTokens = cliArgs.steps, temperature = cliArgs.temperature)
                    }
                }
                cliArgs.agentMode -> {
                    val agentCli = AgentCli(runtime, tokenizer, cliArgs.templateName, metadata)
                    agentCli.runAgent(maxTokens = cliArgs.steps, temperature = cliArgs.temperature)
                }
                else -> {
                    val agentCli = AgentCli(runtime, tokenizer, cliArgs.templateName, metadata)
                    agentCli.runChat(maxTokens = cliArgs.steps, temperature = cliArgs.temperature)
                }
            }
            return@runBlocking
        }

        // Standard generation mode
        val promptText = cliArgs.prompt ?: error("Prompt is required for standard generation mode.")
        // Tokenize and prepend BOS — Gemma 4's GGUF sets
        // `tokenizer.ggml.add_bos_token = true` and the model is trained to
        // expect BOS at position 0. The Tokenizer interface intentionally
        // doesn't auto-prepend special tokens (so other callers can handle
        // chat templates / system prefixes themselves), so the CLI does it
        // here. Most other modern decoder GGUFs also want this; the only
        // ones that don't would set add_bos_token=false. (We currently don't
        // surface that flag — until a non-BOS GGUF comes up, prepending
        // unconditionally is the right default.)
        val rawTokens = tokenizer.encode(promptText)
        val promptTokens = intArrayOf(tokenizer.bosTokenId) + rawTokens

        println("Generating ${cliArgs.steps} tokens with temperature=${cliArgs.temperature}...")
        println("---")
        print(promptText)

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
