package sk.ainet.apps.voxtral.cli

import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.models.voxtral.VoxtralAcousticRuntime
import sk.ainet.models.voxtral.VoxtralBackboneRuntime
import sk.ainet.models.voxtral.VoxtralCodecRuntime
import sk.ainet.models.voxtral.VoxtralConfigParser
import sk.ainet.models.voxtral.VoxtralDefaults
import sk.ainet.models.voxtral.VoxtralNetworkLoader
import sk.ainet.models.voxtral.VoxtralSafeTensorsLoader
import sk.ainet.models.voxtral.TekkenTokenizerAdapter
import kotlinx.coroutines.runBlocking
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.extension
import kotlin.io.path.isDirectory
import kotlin.io.path.readText
import kotlin.system.exitProcess
import kotlin.time.measureTime
import kotlin.time.measureTimedValue

private enum class ModelFormat { GGUF, SAFETENSORS }

private data class CliArgs(
    val modelPath: Path,
    val tokenizerPath: Path?,
    val text: String,
    val outputPath: Path,
    val steps: Int,
    val temperature: Float,
    val flowSteps: Int,
    val flowMethod: String
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: kvoxtral -m <model> -o <output.wav> [-t <tokenizer>] [-s <steps>] [-k <temperature>] [--flow-steps N] [--flow-method euler|midpoint] <text>")
    println()
    println("  -m, --model         Path to .gguf or .safetensors model, or HuggingFace directory (required)")
    println("  -o, --output        Output WAV file path (required)")
    println("  -t, --tokenizer     Path to tokenizer.json or tekken.json (auto-detected)")
    println("  -s, --steps         Max audio tokens to generate (default: 512)")
    println("  -k, --temperature   Sampling temperature (default: 0.7)")
    println("  --flow-steps        ODE solver steps for acoustic model (default: 16)")
    println("  --flow-method       ODE solver: euler or midpoint (default: euler)")
    println("  -h, --help          Show this help")
    println()
    println("Example:")
    println("  kvoxtral -m Voxtral-4B-TTS-2603/ -o hello.wav \"Hello, how are you today?\"")
    println("  kvoxtral -m voxtral.gguf -o speech.wav -s 1024 --flow-steps 32 \"The quick brown fox\"")
    exitProcess(if (errorMessage == null) 0 else 1)
}

private fun parseArgs(args: Array<String>): CliArgs {
    if (args.isEmpty()) usage("Missing arguments.")

    var model: String? = null
    var tokenizer: String? = null
    var output: String? = null
    var steps = 512
    var temperature = 0.7f
    var flowSteps = 16
    var flowMethod = "euler"
    var text: String? = null

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
            arg == "-o" || arg == "--output" -> output = nextValue(arg)
            arg.startsWith("--output=") -> output = arg.substringAfter("=")
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
            arg == "--flow-steps" -> {
                val value = nextValue(arg)
                flowSteps = value.toIntOrNull() ?: usage("Invalid flow-steps value '$value'. Expected integer.")
            }
            arg.startsWith("--flow-steps=") -> {
                val value = arg.substringAfter("=")
                flowSteps = value.toIntOrNull() ?: usage("Invalid flow-steps value '$value'. Expected integer.")
            }
            arg == "--flow-method" -> flowMethod = nextValue(arg)
            arg.startsWith("--flow-method=") -> flowMethod = arg.substringAfter("=")
            arg.startsWith("-") -> usage("Unknown option: $arg")
            else -> {
                if (text != null) usage("Multiple text inputs provided. Text must be a single positional argument.")
                text = arg
            }
        }
        idx++
    }

    val modelPath = model?.let(Path::of) ?: usage("--model is required.")
    val outputPath = output?.let(Path::of) ?: usage("--output is required.")

    if (!outputPath.toString().endsWith(".wav", ignoreCase = true)) {
        usage("Output file must have .wav extension: $outputPath")
    }

    if (text == null) usage("Text input is required as a positional argument.")

    if (flowMethod !in listOf("euler", "midpoint")) {
        usage("Invalid flow method '$flowMethod'. Expected: euler or midpoint.")
    }

    return CliArgs(modelPath, tokenizer?.let(Path::of), text, outputPath, steps, temperature, flowSteps, flowMethod)
}

private fun detectFormat(path: Path): ModelFormat {
    if (path.isDirectory()) {
        val consolidatedSt = path.resolve("consolidated.safetensors")
        val modelSt = path.resolve("model.safetensors")
        return when {
            consolidatedSt.exists() || modelSt.exists() -> ModelFormat.SAFETENSORS
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

private fun resolveSafeTensorsFile(modelPath: Path): Path {
    if (!modelPath.isDirectory()) return modelPath
    val consolidated = modelPath.resolve("consolidated.safetensors")
    if (consolidated.exists()) return consolidated
    val model = modelPath.resolve("model.safetensors")
    if (model.exists()) return model
    error("No .safetensors file found in $modelPath")
}

private fun resolveConfigFile(modelDir: Path): Path? {
    val params = modelDir.resolve("params.json")
    if (params.exists()) return params
    val config = modelDir.resolve("config.json")
    if (config.exists()) return config
    return null
}

private fun resolveTokenizerFile(modelDir: Path, explicit: Path?): Path {
    if (explicit != null) {
        if (!explicit.exists()) error("Tokenizer not found: $explicit")
        return explicit
    }
    val tekken = modelDir.resolve("tekken.json")
    if (tekken.exists()) return tekken
    val tokenizer = modelDir.resolve("tokenizer.json")
    if (tokenizer.exists()) return tokenizer
    error("No tokenizer found in $modelDir. Provide one with -t/--tokenizer.")
}

fun main(args: Array<String>) = runBlocking {
    val cliArgs = parseArgs(args)

    if (!cliArgs.modelPath.exists()) {
        error("Model not found: ${cliArgs.modelPath}")
    }

    val format = detectFormat(cliArgs.modelPath)
    val modelDir = resolveModelDir(cliArgs.modelPath)
    val audioConfig = VoxtralDefaults.AUDIO

    println("Voxtral TTS (Kotlin)")
    println("Model: ${cliArgs.modelPath}")
    println("Format: $format")
    println("Text: \"${cliArgs.text}\"")
    println("Output: ${cliArgs.outputPath}")
    println()

    val ctx = DirectCpuExecutionContext()

    // ---- Load model (all components) ----
    println("Loading model...")
    var allTensors: Map<String, Tensor<FP32, Float>> = emptyMap()

    val backboneModel = measureTimedValue {
        when (format) {
            ModelFormat.GGUF -> {
                val ggufPath = if (cliArgs.modelPath.isDirectory()) {
                    cliArgs.modelPath.toFile().listFiles()
                        ?.firstOrNull { it.extension == "gguf" }
                        ?.toPath() ?: error("No .gguf file found in ${cliArgs.modelPath}")
                } else {
                    cliArgs.modelPath
                }
                println("  Loading GGUF model from $ggufPath...")
                val loader = VoxtralNetworkLoader.fromGguf(
                    randomAccessProvider = { JvmRandomAccessSource.open(ggufPath.toString()) },
                    quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32
                )
                loader.loadBackbone<FP32, Float>(ctx)
            }
            ModelFormat.SAFETENSORS -> {
                val stFile = resolveSafeTensorsFile(cliArgs.modelPath)
                val configFile = resolveConfigFile(modelDir)

                val metadata = if (configFile != null) {
                    println("  Config: $configFile")
                    VoxtralConfigParser.parse(configFile.readText()).backbone
                } else {
                    println("  Using Voxtral-4B defaults")
                    VoxtralDefaults.BACKBONE
                }

                val tiedEmbeddings = if (configFile != null) {
                    VoxtralConfigParser.isTiedEmbeddings(configFile.readText())
                } else {
                    true
                }

                println("  SafeTensors: $stFile")
                println("  layers=${metadata.blockCount}, dim=${metadata.embeddingLength}, " +
                    "heads=${metadata.headCount}, kvHeads=${metadata.kvHeadCount}, vocab=${metadata.vocabSize}")
                if (tiedEmbeddings) println("  Tied embeddings: output.weight = token_embd.weight")

                // Use VoxtralSafeTensorsLoader to capture ALL tensors (backbone + acoustic + codec)
                val stLoader = VoxtralSafeTensorsLoader<FP32>(ctx, FP32::class, metadata, tiedEmbeddings)
                val voxtralWeights = stLoader.loadAll { JvmRandomAccessSource.open(stFile.toString()) }
                allTensors = voxtralWeights.allTensors

                val acousticCount = allTensors.keys.count { it.startsWith("acoustic.") }
                val codecCount = allTensors.keys.count { it.startsWith("codec.") }
                println("  Loaded: ${allTensors.size} tensors (backbone=${allTensors.size - acousticCount - codecCount}, acoustic=$acousticCount, codec=$codecCount)")

                // Build backbone from the backbone weights
                VoxtralNetworkLoader.backboneFromWeights(voxtralWeights.backbone)
            }
        }
    }

    println("  Model loaded in ${backboneModel.duration}")
    val model = backboneModel.value

    // ---- Load tokenizer ----
    val tokenizerFile = resolveTokenizerFile(modelDir, cliArgs.tokenizerPath)
    println("Loading tokenizer from $tokenizerFile...")
    val isTekken = tokenizerFile.toString().endsWith("tekken.json", ignoreCase = true)
    val tokenizer: Tokenizer = if (format == ModelFormat.GGUF && cliArgs.tokenizerPath == null) {
        val ggufPath = if (cliArgs.modelPath.isDirectory()) {
            cliArgs.modelPath.toFile().listFiles()?.first { it.extension == "gguf" }!!.toPath()
        } else {
            cliArgs.modelPath
        }
        println("  Using embedded GGUF tokenizer")
        GGUFTokenizer.fromRandomAccessSource(
            JvmRandomAccessSource.open(ggufPath.toString())
        )
    } else if (isTekken) {
        println("  Using Tekken tokenizer (Mistral format)")
        TekkenTokenizerAdapter.fromJson(tokenizerFile.readText())
    } else {
        println("  Using HuggingFace tokenizer.json")
        GGUFTokenizer.fromTokenizerJson(tokenizerFile.readText())
    }

    // ---- Build backbone runtime (captures hidden states) ----
    val backboneRuntime = VoxtralBackboneRuntime(
        model = model,
        ctx = ctx,
        dtype = FP32::class,
        bos = audioConfig.bosTokenId
    )

    // ---- Step 1: Generate semantic tokens + capture hidden states ----
    val promptTokens = tokenizer.encode(cliArgs.text)
    println()
    println("Input: ${promptTokens.size} tokens")
    println("Generating up to ${cliArgs.steps} semantic tokens (temperature=${cliArgs.temperature})...")

    val generatedTokens = mutableListOf<Int>()
    val genTime = measureTime {
        backboneRuntime.generate(
            prompt = promptTokens,
            steps = cliArgs.steps,
            temperature = cliArgs.temperature
        ) { tokenId ->
            generatedTokens.add(tokenId)
            // Print progress every 50 tokens
            if (generatedTokens.size % 50 == 0) {
                print("\r  ${generatedTokens.size} tokens...")
                System.out.flush()
            }
        }
    }

    val tokPerSec = if (genTime.inWholeMilliseconds > 0) {
        generatedTokens.size / genTime.inWholeMilliseconds.toDouble() * 1000
    } else 0.0
    println("\r  ${generatedTokens.size} tokens in $genTime (%.2f tok/s)".format(tokPerSec))

    // ---- Step 2: Get hidden states for acoustic model ----
    val hiddenStates = backboneRuntime.lastHiddenStates()
    if (hiddenStates != null) {
        println("Hidden states: ${hiddenStates.shape} (${hiddenStates.shape[0]} frames x ${hiddenStates.shape[1]} dim)")
    }

    // ---- Step 3: Generate acoustic codes via flow matching ----
    // This step is only meaningful when acoustic model weights are loaded.
    // For now, we create an acoustic runtime with zero-initialized projections
    // to demonstrate the pipeline flow. With real weights, this produces actual
    // acoustic codes that the codec can decode to audio.
    val nCodebooks = audioConfig.nAcousticCodebooks
    val codebookLevels = audioConfig.acousticCodebookSize
    var acousticCodes: IntArray? = null

    if (hiddenStates != null) {
        val hasAcousticWeights = allTensors.keys.any { it.startsWith("acoustic.") }
        println("Running acoustic flow matching (${cliArgs.flowSteps} ${cliArgs.flowMethod} steps, " +
            "$nCodebooks codebooks x $codebookLevels levels" +
            if (hasAcousticWeights) ", with model weights" else ", zero-initialized" +
            ")...")
        val acousticTime = measureTime {
            val dim = hiddenStates.shape[1]
            val acousticDim = nCodebooks * codebookLevels
            val acousticMeta = VoxtralDefaults.ACOUSTIC_MODEL

            // Use loaded weights if available, otherwise zero-initialized
            val inputProj = allTensors[sk.ainet.models.voxtral.VoxtralTensorNames.ACOUSTIC_INPUT_PROJ]
                ?: createZeroTensor(ctx, dim, acousticDim)
            val outputProj = allTensors[sk.ainet.models.voxtral.VoxtralTensorNames.ACOUSTIC_OUTPUT_PROJ]
                ?: createZeroTensor(ctx, acousticDim, dim)
            val inputProjBias = allTensors[sk.ainet.models.voxtral.VoxtralTensorNames.ACOUSTIC_INPUT_PROJ_BIAS]
            val outputProjBias = allTensors[sk.ainet.models.voxtral.VoxtralTensorNames.ACOUSTIC_OUTPUT_PROJ_BIAS]

            val acousticRuntime = VoxtralAcousticRuntime(
                acousticTransformer = sk.ainet.models.voxtral.voxtralAcousticNetwork<FP32, Float>(acousticMeta),
                inputProj = inputProj,
                outputProj = outputProj,
                inputProjBias = inputProjBias,
                outputProjBias = outputProjBias,
                ctx = ctx,
                dtype = FP32::class,
                nCodebooks = nCodebooks,
                codebookLevels = codebookLevels,
                dim = dim
            )

            acousticCodes = acousticRuntime.generate(
                backboneHidden = hiddenStates,
                numSteps = cliArgs.flowSteps,
                method = cliArgs.flowMethod
            )
        }
        println("  ${acousticCodes!!.size} acoustic codes in $acousticTime " +
            "(${acousticCodes!!.size / nCodebooks} frames)")
    }

    // ---- Step 4: Decode to audio via codec (or tone-map fallback) ----
    var usedCodec = false
    val audioSamples: FloatArray = if (acousticCodes != null) {
        val codecWeights = allTensors.filterKeys { it.startsWith("codec.") }
        val hasCodecWeights = codecWeights.isNotEmpty()
        println("Running codec decoder${if (hasCodecWeights) " (${codecWeights.size} weights)" else " (no weights)"}...")
        try {
            val codec = VoxtralCodecRuntime<FP32>(
                weights = codecWeights,
                metadata = VoxtralDefaults.CODEC,
                ctx = ctx,
                dtype = FP32::class
            )
            var decoded: FloatArray? = null
            val codecTime = measureTime {
                decoded = codec.decode(
                    semanticCodes = generatedTokens.toIntArray(),
                    acousticCodes = acousticCodes!!
                )
            }
            usedCodec = true
            println("  Codec decoded ${decoded!!.size} samples in $codecTime")
            decoded!!
        } catch (e: Exception) {
            println("  Codec unavailable (${e.message}), using tone-map fallback")
            toneMapTokens(generatedTokens, audioConfig)
        }
    } else {
        toneMapTokens(generatedTokens, audioConfig)
    }

    // ---- Write WAV ----
    println()
    println("Writing WAV: ${cliArgs.outputPath} (${audioSamples.size} samples, ${audioConfig.samplingRate}Hz, mono)")
    WavWriter.write(
        path = cliArgs.outputPath,
        samples = audioSamples,
        sampleRate = audioConfig.samplingRate,
        channels = 1
    )

    println("Done.")
    println("Output: ${cliArgs.outputPath}")
    println()
    println("Pipeline: text -> ${promptTokens.size} prompt tokens -> ${generatedTokens.size} semantic tokens")
    if (acousticCodes != null) {
        println("          -> ${acousticCodes!!.size} acoustic codes (${acousticCodes!!.size / nCodebooks} frames)")
    }
    println("          -> ${audioSamples.size} audio samples -> WAV")
    if (!usedCodec) {
        println()
        println("Note: Audio is tone-mapped (codec weights not loaded).")
        println("Load codec weights for actual speech synthesis.")
    }
}

/**
 * Fallback audio: map token IDs to sine bursts for audible (non-speech) output.
 */
private fun toneMapTokens(
    tokens: List<Int>,
    audioConfig: sk.ainet.models.voxtral.VoxtralAudioConfig
): FloatArray {
    val samplesPerToken = (audioConfig.samplingRate / audioConfig.frameRate).toInt()
    val totalSamples = tokens.size * samplesPerToken
    val samples = FloatArray(totalSamples)

    for ((i, tokenId) in tokens.withIndex()) {
        val freq = 200.0 + (tokenId % 1800)
        val offset = i * samplesPerToken
        for (s in 0 until samplesPerToken) {
            val t = s.toDouble() / audioConfig.samplingRate
            val envelope = if (s < 20) s / 20.0f
                else if (s > samplesPerToken - 20) (samplesPerToken - s) / 20.0f
                else 1.0f
            samples[offset + s] = (Math.sin(2.0 * Math.PI * freq * t) * 0.3 * envelope).toFloat()
        }
    }
    return samples
}

@Suppress("UNCHECKED_CAST")
private fun createZeroTensor(
    ctx: sk.ainet.context.ExecutionContext,
    rows: Int,
    cols: Int
): sk.ainet.lang.tensor.Tensor<FP32, Float> {
    val data = FloatArray(rows * cols)
    return ctx.fromFloatArray<FP32, Float>(
        sk.ainet.lang.tensor.Shape(rows, cols), FP32::class, data
    ) as sk.ainet.lang.tensor.Tensor<FP32, Float>
}

