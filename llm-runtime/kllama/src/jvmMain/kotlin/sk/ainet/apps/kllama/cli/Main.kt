package sk.ainet.apps.kllama.cli

import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.LlamaIngestion
import sk.ainet.apps.kllama.LlamaLoadConfig
import sk.ainet.models.llama.LlamaConfigParser
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.models.llama.DecoderGgufMemSegConverter
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.MemSegWeightConverter
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.Llama2DotCWeightLoader
import sk.ainet.models.qwen.QwenNetworkLoader
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.apps.llm.backend.BackendRegistry
import sk.ainet.apps.llm.backend.availableNames
import sk.ainet.apps.llm.backend.bestAvailable
import sk.ainet.apps.llm.backend.find
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import kotlinx.io.buffered
import kotlinx.io.asSource
import kotlin.io.path.inputStream
import java.lang.foreign.Arena
import java.nio.file.Path
import java.nio.file.Files
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.io.path.isDirectory
import kotlin.io.path.readText
import kotlin.system.exitProcess
import kotlin.time.measureTime
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.chat.ModelMetadata
import sk.ainet.apps.kllama.chat.ModelMetadataExtraction
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.generate
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.models.llama.DecoderGgufWeightLoader

private enum class ModelFormat { GGUF, SAFETENSORS, BIN }

private data class CliArgs(
    val modelPath: Path,
    val tokenizerPath: Path?,
    val prompt: String?,
    val systemPrompt: String?,
    val steps: Int,
    val temperature: Float,
    val chatMode: Boolean,
    val agentMode: Boolean,
    val demoMode: Boolean,
    val templateName: String?,
    val backend: String?,
    val contextLength: Int?
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: kllama -m <model> [-t <tokenizer>] [-s <steps>] [-k <temperature>] [-p <systemprompt>] [--chat] [--agent] [--demo] [--template=NAME] <prompt>")
    println("  -m, --model         Path to .gguf, .safetensors, .bin model, or HuggingFace directory (required)")
    println("  -t, --tokenizer     Path to tokenizer (auto-detected for .gguf and .safetensors)")
    println("  -s, --steps         Generation steps (default: 64)")
    println("  -k, --temperature   Sampling temperature (default: 0.8)")
    println("  -p, --systemprompt  Optional system prompt prepended to user prompt")
    println("  --chat              Interactive chat mode")
    println("  --agent             Interactive agent mode with tool calling")
    println("  --demo              Tool calling demo with file listing and calculator")
    println("  --template=NAME     Chat template: llama3, chatml, qwen, gemma, smollm (auto-detected if omitted)")
    println("  --context=N         Cap context length to N tokens (reduces memory usage)")
    println("  --backend=NAME      Compute backend: auto-selects best available (see --list-backends)")
    println("  --list-backends     List available compute backends and exit")
    println("  -h, --help          Show this help")
    println()
    println("Example:")
    println("  kllama -m model.gguf -s 96 -k 0.7 -p \"You are concise\" \"Hallo\"")
    println("  kllama -m model.gguf --chat")
    println("  kllama -m model.gguf --agent --template=chatml")
    println("  kllama -m model.gguf --demo")
    exitProcess(if (errorMessage == null) 0 else 1)
}

private fun parseArgs(args: Array<String>): CliArgs {
    if (args.isEmpty()) usage("Missing arguments.")

    var model: String? = null
    var tokenizer: String? = null
    var steps = 64
    var temperature = 0.8f
    var systemPrompt: String? = null
    var prompt: String? = null
    var chatMode = false
    var agentMode = false
    var demoMode = false
    var templateName: String? = null
    var backend: String? = null
    var contextLength: Int? = null

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
            arg == "-p" || arg == "--systemprompt" -> systemPrompt = nextValue(arg)
            arg.startsWith("--systemprompt=") -> systemPrompt = arg.substringAfter("=")
            arg == "--chat" -> chatMode = true
            arg == "--agent" -> agentMode = true
            arg == "--demo" -> demoMode = true
            arg.startsWith("--template=") -> templateName = arg.substringAfter("=")
            arg.startsWith("--context=") -> {
                val value = arg.substringAfter("=")
                contextLength = value.toIntOrNull() ?: usage("Invalid context length '$value'. Expected integer.")
            }
            arg == "--backend" -> backend = nextValue(arg)
            arg.startsWith("--backend=") -> backend = arg.substringAfter("=")
            arg == "--list-backends" -> {
                val available = BackendRegistry.providers()
                println("Available backends:")
                for (p in available) {
                    val status = if (p.isAvailable()) "available" else "unavailable"
                    println("  ${p.name.padEnd(12)} ${p.displayName} (priority=${p.priority}, $status)")
                }
                exitProcess(0)
            }
            arg.startsWith("-") -> usage("Unknown option '$arg'.")
            else -> {
                if (prompt != null) usage("Multiple prompts provided. Prompt must be a single positional argument.")
                prompt = arg
            }
        }

        idx++
    }

    val modelPath = model?.let(Path::of) ?: usage("Model is required (-m/--model).")
    val tokenizerPath = tokenizer?.let(Path::of)

    // In chat/agent/demo mode, prompt is optional
    if (!chatMode && !agentMode && !demoMode && prompt == null) {
        usage("Prompt is required as a positional argument (or use --chat/--agent/--demo mode).")
    }

    return CliArgs(
        modelPath = modelPath,
        tokenizerPath = tokenizerPath,
        prompt = prompt,
        systemPrompt = systemPrompt,
        steps = steps,
        temperature = temperature,
        chatMode = chatMode,
        agentMode = agentMode,
        demoMode = demoMode,
        templateName = templateName,
        backend = backend,
        contextLength = contextLength
    )
}

private const val DEFAULT_SYSTEM_PROMPT = "You are a helpful assistant. Answer concisely."

private fun buildEffectivePrompt(prompt: String, systemPrompt: String?): String {
    val resolvedSystemPrompt = systemPrompt?.trim().takeUnless { it.isNullOrEmpty() } ?: DEFAULT_SYSTEM_PROMPT
    val userPrompt = prompt.trim()

    return buildString {
        append("<|system|>\n")
        append(resolvedSystemPrompt)
        append('\n')
        append("<|user|>\n")
        append(userPrompt)
        append('\n')
        append("<|assistant|>")
    }
}

private fun resolveModelPath(candidate: Path): Path {
    if (!candidate.exists()) error("Model not found: $candidate")
    if (!Files.isDirectory(candidate)) return candidate

    // If directory contains model.safetensors, return directory for SafeTensors loading
    val safetensorsModel = candidate.resolve("model.safetensors")
    if (safetensorsModel.exists()) return candidate

    val modelCandidates = mutableListOf<Path>()
    Files.list(candidate).use { stream ->
        stream.forEach { entry ->
            if (!Files.isRegularFile(entry)) return@forEach
            val ext = entry.extension.lowercase()
            if (ext == "gguf" || ext == "bin") {
                modelCandidates.add(entry)
            }
        }
    }

    when {
        modelCandidates.isEmpty() -> {
            error("No .gguf, .safetensors, or .bin model found in directory: $candidate")
        }
        modelCandidates.size > 1 -> {
            val choices = modelCandidates.sortedBy { it.fileName.toString() }.joinToString(", ")
            error("Multiple model files found in directory. Use -m with an exact file path: $choices")
        }
        else -> {
            val resolved = modelCandidates.single()
            println("Resolved model file: $resolved")
            return resolved
        }
    }
}

/**
 * Detect model format from the given path.
 * Supports: .gguf, .safetensors, .bin files, and directories containing model.safetensors.
 */
private fun detectFormat(path: Path): ModelFormat {
    if (path.isDirectory()) {
        // Check for safetensors model in directory
        val st = path.resolve("model.safetensors")
        if (st.exists()) return ModelFormat.SAFETENSORS
        val gguf = path.toFile().listFiles()?.firstOrNull { it.extension == "gguf" }
        if (gguf != null) return ModelFormat.GGUF
        error("Directory $path does not contain model.safetensors or .gguf file")
    }
    return when (path.extension.lowercase()) {
        "gguf" -> ModelFormat.GGUF
        "safetensors" -> ModelFormat.SAFETENSORS
        else -> ModelFormat.BIN
    }
}

/**
 * Resolve model directory: if path is a file, return its parent; if directory, return it.
 */
private fun resolveModelDir(path: Path): Path =
    if (path.isDirectory()) path else path.parent ?: path

/**
 * Peek at the architecture and chat template fields from a GGUF file without loading weights.
 * Returns a [ModelMetadata] for capability detection.
 */
private fun peekGgufMetadata(modelPath: Path): ModelMetadata {
    return JvmRandomAccessSource.open(modelPath.toString()).use { source ->
        StreamingGGUFReader.open(source).use { reader ->
            ModelMetadataExtraction.fromGgufFields(reader.fields)
        }
    }
}

/** Peek at the EOS token ID from GGUF metadata. */
private fun peekEosTokenId(modelPath: Path): Int {
    return JvmRandomAccessSource.open(modelPath.toString()).use { source ->
        StreamingGGUFReader.open(source).use { reader ->
            (reader.fields["tokenizer.ggml.eos_token_id"] as? Number)?.toInt() ?: 2
        }
    }
}

/** Set of GGUF architecture strings that are compatible with the Llama runtime. */
private val LLAMA_COMPATIBLE_ARCHITECTURES: Set<String> = setOf(
    "llama", "qwen2", "qwen3", "qwen35", "mistral"
)

fun main(args: Array<String>) {
    runBlocking {
        val cliArgs = parseArgs(args)
        val modelPath = resolveModelPath(cliArgs.modelPath)
        val format = detectFormat(modelPath)

        // Resolve tokenizer: use explicit -t if provided, auto-discover for SafeTensors
        val tokenizerPath: Path? = when {
            cliArgs.tokenizerPath != null -> cliArgs.tokenizerPath
            format == ModelFormat.SAFETENSORS -> {
                val modelDir = resolveModelDir(modelPath)
                val autoTokenizer = modelDir.resolve("tokenizer.json")
                if (autoTokenizer.exists()) autoTokenizer else null
            }
            else -> null
        }
        if (format == ModelFormat.BIN && tokenizerPath == null) {
            error("Tokenizer path required for .bin models. Use -t/--tokenizer.")
        }

        if (!modelPath.exists()) error("Model not found: $modelPath")

        // Select compute backend
        val provider = cliArgs.backend?.let { name ->
            BackendRegistry.find(name) ?: run {
                System.err.println("Warning: Backend '$name' not found. Available: ${BackendRegistry.availableNames()}")
                BackendRegistry.bestAvailable()
            }
        } ?: BackendRegistry.bestAvailable()
        println("Backend: ${provider.displayName}")

        val quantArena = Arena.ofShared()
        val memSegFactory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

        Runtime.getRuntime().addShutdownHook(Thread {
            quantArena.close()
            memSegFactory.close()
        })

        // Peek GGUF metadata for architecture-aware loading and auto-detection
        val ggufMetadata: ModelMetadata? = if (format == ModelFormat.GGUF) {
            peekGgufMetadata(modelPath).also {
                println("GGUF architecture: ${it.architecture}, family: ${it.family}")
            }
        } else null

        val isQwen = ggufMetadata?.family == "qwen"

        // Build the inference runtime — Qwen uses the modern DSL path,
        // Llama/others use the legacy LlamaRuntime path.
        val runtime: InferenceRuntime<FP32>
        var eosTokenId: Int = 2
        var binVocabSize: Int = 0

        if (format == ModelFormat.GGUF && isQwen) {
            // --- Qwen: DSL path. QwenNetworkLoader builds a `qwenNetwork()`
            // module (RoPE NEOX, QK-norm, metadata-driven eps) populated
            // from the GGUF; DecoderGgufMemSegConverter wraps Q4_0/Q8_0
            // tensors as packed MemorySegment data for the SIMD quant
            // matmul kernels. OptimizedLLMRuntime DIRECT mode runs the
            // module tree forward.
            //
            // Bit-for-bit parity with the legacy LlamaRuntime path on
            // identical weights is pinned by QwenDslLegacyParityTest
            // (#120, closes #114).
            val qwenArchitectures = setOf("qwen2", "qwen3", "qwen35")
            val loader = DecoderGgufWeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
                quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
                acceptedArchitectures = qwenArchitectures,
            )
            println("Loading GGUF model from $modelPath (Qwen, DSL streaming mode)...")
            val rawWeights = loader.loadToMapStreaming<FP32, Float>(ctx)

            val convertedWeights = if (rawWeights.quantTypes.isNotEmpty()) {
                println("Converting ${rawWeights.quantTypes.size} quantized tensors to MemorySegment-backed SIMD format...")
                DecoderGgufMemSegConverter.convert(rawWeights, ctx, quantArena)
            } else {
                rawWeights
            }

            if (cliArgs.contextLength != null) {
                println("Context length capped to ${cliArgs.contextLength} (model default: ${convertedWeights.metadata.contextLength})")
            }
            val qwenModel = QwenNetworkLoader.fromWeights(convertedWeights)
            runtime = OptimizedLLMRuntime(
                model = qwenModel,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class,
                bos = convertedWeights.metadata.bosTokenId,
            )
            eosTokenId = convertedWeights.metadata.eosTokenId
        } else {
            // --- Llama / SafeTensors / BIN: legacy LlamaRuntime path.
            // The DSL path is functionally correct but ~8x slower for Q8/Q4
            // GGUFs because every linearProject forward calls ops.transpose
            // on packed quant weights through a generic dispatch (the DSL
            // doesn't yet have first-class Q4/Q8 DTypes). Until that lands,
            // run Llama through the legacy LlamaRuntime + CpuAttentionBackend
            // + MemSegWeightConverter path that previously hit ~2 t/s.
            // Qwen GGUF stays on the DSL branch above.
            val runtimeWeights = when (format) {
                ModelFormat.GGUF -> {
                    val ingestion = LlamaIngestion<FP32>(
                        ctx = ctx,
                        dtype = FP32::class,
                        config = LlamaLoadConfig(
                            quantPolicy = QuantPolicy.NATIVE_OPTIMIZED,
                            allowQuantized = true,
                            acceptedArchitectures = LLAMA_COMPATIBLE_ARCHITECTURES,
                        ),
                    )
                    println("Loading GGUF model from $modelPath (Llama, eager streaming mode)...")
                    val rawWeights = ingestion.loadStreaming {
                        JvmRandomAccessSource.open(modelPath.toString())
                    }
                    if (rawWeights.quantTypes.isNotEmpty()) {
                        println("Converting ${rawWeights.quantTypes.size} quantized tensors to MemorySegment-backed SIMD format...")
                        MemSegWeightConverter.convert(rawWeights, ctx, quantArena)
                    } else {
                        rawWeights
                    }
                }
                ModelFormat.SAFETENSORS -> {
                    val modelDir = resolveModelDir(modelPath)
                    val safetensorsFile = if (modelPath.isDirectory()) {
                        modelDir.resolve("model.safetensors")
                    } else {
                        modelPath
                    }
                    val configFile = modelDir.resolve("config.json")
                    if (!configFile.exists()) error("config.json not found in $modelDir")

                    println("Loading SafeTensors model from $safetensorsFile...")
                    val configJson = configFile.readText()
                    val safeMetadata = LlamaConfigParser.parse(configJson)
                    val tiedEmbeddings = LlamaConfigParser.isTiedEmbeddings(configJson)
                    println("  Architecture: ${safeMetadata.architecture}, layers=${safeMetadata.blockCount}, " +
                        "dim=${safeMetadata.embeddingLength}, heads=${safeMetadata.headCount}, " +
                        "kvHeads=${safeMetadata.kvHeadCount}, vocab=${safeMetadata.vocabSize}")
                    if (tiedEmbeddings) println("  Tied word embeddings: output.weight = embed_tokens.weight")

                    val ingestion = LlamaIngestion<FP32>(ctx = ctx, dtype = FP32::class)
                    ingestion.loadSafeTensors(
                        randomAccessProvider = { JvmRandomAccessSource.open(safetensorsFile.toString()) },
                        metadata = safeMetadata,
                        tiedEmbeddings = tiedEmbeddings,
                    )
                }
                ModelFormat.BIN -> {
                    println("Loading Karpathy .bin model from $modelPath...")
                    modelPath.inputStream().use { input ->
                        Llama2DotCWeightLoader.load(ctx, input.asSource().buffered())
                    }
                }
            }

            if (cliArgs.contextLength != null) {
                println("Context length capped to ${cliArgs.contextLength} (model default: ${runtimeWeights.metadata.contextLength})")
            }
            val backend = CpuAttentionBackend<FP32>(
                ctx, runtimeWeights, FP32::class,
                ropeFreqBase = runtimeWeights.metadata.ropeFreqBase,
                maxContextLength = cliArgs.contextLength,
            )
            @Suppress("DEPRECATION")
            runtime = LlamaRuntime<FP32>(
                ctx, runtimeWeights, backend, FP32::class,
                eps = runtimeWeights.metadata.rmsNormEps,
            )
            eosTokenId = runtimeWeights.metadata.eosTokenId
            binVocabSize = runtimeWeights.metadata.vocabSize
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
                JvmRandomAccessSource.open(modelPath.toString()).use { source ->
                    TokenizerFactory.fromGgufSource(source)
                }
            }
            else -> {
                val tPath = tokenizerPath ?: error("Tokenizer path required for .bin models")
                if (!tPath.exists()) error("Tokenizer not found: $tPath")
                println("Loading tokenizer from $tPath...")
                tPath.inputStream().use { input ->
                    TokenizerUtils.buildTokenizer(input.asSource().buffered(), binVocabSize)
                }
            }
        }

        // Dispatch to chat/agent/demo mode — works with any Tokenizer, not just GGUFTokenizer
        if (cliArgs.chatMode || cliArgs.agentMode || cliArgs.demoMode) {
            val metadata = ggufMetadata ?: ModelMetadata()
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
                    if (cliArgs.prompt != null) {
                        agentCli.runChatOnce(cliArgs.prompt, maxTokens = cliArgs.steps, temperature = cliArgs.temperature)
                    } else {
                        agentCli.runChat(maxTokens = cliArgs.steps, temperature = cliArgs.temperature)
                    }
                }
            }
            return@runBlocking
        }

        // Standard generation mode — raw prompt, no chat template wrapping
        val promptText = cliArgs.prompt ?: error("Prompt is required for standard generation mode.")
        val promptTokens = tokenizer.encode(promptText)

        if (!cliArgs.systemPrompt.isNullOrBlank()) {
            println("Using system prompt.")
        }
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
