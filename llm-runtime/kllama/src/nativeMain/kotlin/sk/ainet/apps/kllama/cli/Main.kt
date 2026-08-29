package sk.ainet.apps.kllama.cli

import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlin.reflect.KClass
import kotlin.time.measureTime
import sk.ainet.apps.kllama.CpuAttentionBackend
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.kllama.Llama2DotCWeightLoader
import sk.ainet.apps.kllama.TokenizerUtils
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.llm.backend.BackendRegistry
import sk.ainet.apps.llm.backend.availableNames
import sk.ainet.apps.llm.backend.bestAvailable
import sk.ainet.apps.llm.backend.find
import sk.ainet.apps.llm.generate
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeightLoader
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.models.llama.LlamaRuntime
import sk.ainet.models.llama.LlamaRuntimeWeights

private fun usage(): Nothing {
    println("Usage: kllama <model> [tokenizer] <prompt> [steps=64] [temperature=0.8] [--backend=cpu] [--dtype=fp16|fp32]")
    println("  <model>         Path to .gguf or .bin model")
    println("  <tokenizer>     Path to tokenizer.bin (required for .bin, optional for .gguf)")
    println("  <prompt>        Text prompt")
    println("  --backend=NAME  Execution backend (default: ${BackendRegistry.bestAvailable().name})")
    println("  --dtype=TYPE    Tensor dtype: fp16 or fp32 (default: fp32)")
    println("  --list-backends List available backends and exit")
    println("Available backends: ${BackendRegistry.availableNames().joinToString(", ")}")
    throw IllegalArgumentException("Invalid arguments")
}

fun main(args: Array<String>) = runBlocking {
    registerPlatformBackends()

    var backendName: String? = null
    var dtypeStr = "fp32"
    val filteredArgs = args.filter { arg ->
        when {
            arg.startsWith("--backend=") -> { backendName = arg.substringAfter("="); false }
            arg.startsWith("--dtype=") -> { dtypeStr = arg.substringAfter("=").lowercase(); false }
            arg == "--list-backends" -> {
                val providers = BackendRegistry.providers()
                println("Available backends:")
                for (p in providers) {
                    val status = if (p.isAvailable()) "available" else "unavailable"
                    println("  ${p.name.padEnd(12)} ${p.displayName} (priority=${p.priority}, $status)")
                }
                return@runBlocking
            }
            else -> true
        }
    }.toTypedArray()

    if (filteredArgs.size < 2) usage()

    val provider = backendName?.let { name ->
        BackendRegistry.find(name) ?: run {
            println("Warning: Backend '$name' not found. Available: ${BackendRegistry.availableNames()}")
            BackendRegistry.bestAvailable()
        }
    } ?: BackendRegistry.bestAvailable()
    println("Backend: ${provider.displayName}")

    val firstArgStr = filteredArgs[0]
    val isGguf = firstArgStr.endsWith(".gguf", ignoreCase = true)

    val (modelPathStr, tokenizerPathStr, promptIdx) = if (isGguf && filteredArgs.size >= 2) {
        val secondArg = filteredArgs[1]
        val secondPath = Path(secondArg)
        if (SystemFileSystem.exists(secondPath) && secondArg.contains(".")) {
            Triple(firstArgStr, secondArg, 2)
        } else {
            Triple(firstArgStr, null, 1)
        }
    } else if (filteredArgs.size >= 3) {
        Triple(firstArgStr, filteredArgs[1], 2)
    } else {
        usage()
    }

    val prompt = filteredArgs.getOrNull(promptIdx) ?: usage()
    val steps = filteredArgs.getOrNull(promptIdx + 1)?.toIntOrNull() ?: 64
    val temperature = filteredArgs.getOrNull(promptIdx + 2)?.toFloatOrNull() ?: 0.8f

    val modelPath = Path(modelPathStr)
    if (!SystemFileSystem.exists(modelPath)) {
        error("Model not found: $modelPathStr")
    }

    val ctx = provider.createContext()

    when (dtypeStr) {
        "fp16" -> runInference<FP16>(ctx, FP16::class, isGguf, modelPathStr, modelPath, tokenizerPathStr, prompt, steps, temperature)
        "fp32" -> runInference<FP32>(ctx, FP32::class, isGguf, modelPathStr, modelPath, tokenizerPathStr, prompt, steps, temperature)
        else -> error("Unsupported dtype: $dtypeStr. Use fp16 or fp32.")
    }
}

// Reified so we can call `LlamaNetworkLoader.fromWeights<T, V>` and
// `DecoderGgufWeightLoader.loadToMap<T, V>` (both `inline reified T`).
// The legacy `LlamaRuntime<T>` ctor doesn't need reification — only the
// DSL path does.
@Suppress("DuplicatedCode")
private suspend inline fun <reified T : DType> runInference(
    ctx: ExecutionContext,
    dtype: KClass<T>,
    isGguf: Boolean,
    modelPathStr: String,
    modelPath: Path,
    tokenizerPathStr: String?,
    prompt: String,
    steps: Int,
    temperature: Float,
) {
    val runtime: InferenceRuntime<T>
    val vocabSize: Int

    if (isGguf) {
        // DSL path; the sequential Source loader dequantizes everything to dense tensors.
        println("Loading GGUF model from $modelPathStr (Llama, DSL streaming, dtype=${dtype.simpleName})...")
        val weights = DecoderGgufWeightLoader(
            sourceProvider = { SystemFileSystem.source(modelPath).buffered() },
        ).loadToMap<T, Float>(ctx)
        val model = LlamaNetworkLoader.fromWeights(weights)
        runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = dtype,
            bos = weights.metadata.bosTokenId,
        )
        vocabSize = weights.metadata.vocabSize
    } else {
        // BIN (Karpathy llama2.c format) — kept on legacy LlamaRuntime; the
        // .bin loader returns LlamaRuntimeWeights directly. Migrating .bin
        // to the DSL path requires a converter and isn't in scope here.
        println("Loading Karpathy .bin model from $modelPathStr...")
        @Suppress("UNCHECKED_CAST")
        val runtimeWeights = Llama2DotCWeightLoader.load(ctx, SystemFileSystem.source(modelPath).buffered())
            as LlamaRuntimeWeights<T>
        val cpuBackend = CpuAttentionBackend<T>(ctx, runtimeWeights, dtype)
        @Suppress("DEPRECATION")
        runtime = LlamaRuntime<T>(ctx, runtimeWeights, cpuBackend, dtype)
        vocabSize = runtimeWeights.metadata.vocabSize
    }

    val tokenizer: Tokenizer = if (isGguf && tokenizerPathStr == null) {
        println("Loading embedded GGUF tokenizer...")
        GGUFTokenizer.fromSource(SystemFileSystem.source(modelPath).buffered())
    } else {
        val tPathStr = tokenizerPathStr ?: error("Tokenizer path required for .bin models")
        val tPath = Path(tPathStr)
        if (!SystemFileSystem.exists(tPath)) error("Tokenizer not found: $tPathStr")
        println("Loading tokenizer from $tPathStr...")
        TokenizerUtils.buildTokenizer(SystemFileSystem.source(tPath).buffered(), vocabSize)
    }

    val promptTokens = tokenizer.encode(prompt)

    println("Generating $steps tokens with temperature=$temperature...")
    println("---")

    val elapsed = measureTime {
        runtime.generate(prompt = promptTokens, steps = steps, temperature = temperature) { id ->
            print(tokenizer.decode(id))
        }
    }.inWholeMilliseconds

    val tokPerSec = steps / elapsed.toDouble() * 1000
    println("\n---")
    println("tok/s: $tokPerSec")
}
