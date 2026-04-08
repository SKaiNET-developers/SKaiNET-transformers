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
import sk.ainet.models.voxtral.VoxtralVoiceLoader
import sk.ainet.models.voxtral.VoxtralVoices
import sk.ainet.models.voxtral.TekkenTokenizerAdapter
import sk.ainet.apps.llm.validation.PipelineShapeValidator
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
    val flowMethod: String,
    val voice: String?,
    val beep: Boolean,
    val testCodec: Boolean
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
    println("  --voice             Voice preset name (default: auto-detect, or 'none')")
    println("  --list-voices       List available voice presets and exit")
    println("  --beep              Also write a beep-encoded WAV (token → frequency mapping)")
    println("  --test-codec        Skip backbone, use random tokens to test acoustic+codec pipeline")
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
    var voice: String? = null
    var beep = false
    var testCodec = false
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
            arg == "--voice" -> voice = nextValue(arg)
            arg.startsWith("--voice=") -> voice = arg.substringAfter("=")
            arg == "--beep" -> beep = true
            arg == "--test-codec" -> testCodec = true
            arg == "--list-voices" -> {
                println("Available voice presets:")
                VoxtralVoices.PRESETS.forEach { (name, idx) ->
                    println("  ${name.padEnd(20)} (index $idx)")
                }
                exitProcess(0)
            }
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

    if (text == null && !testCodec) usage("Text input is required as a positional argument.")

    if (flowMethod !in listOf("euler", "midpoint")) {
        usage("Invalid flow method '$flowMethod'. Expected: euler or midpoint.")
    }

    return CliArgs(modelPath, tokenizer?.let(Path::of), text ?: "", outputPath, steps, temperature, flowSteps, flowMethod, voice, beep, testCodec)
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
    println("Loading tokenizer...")
    val tokenizer: Tokenizer = run {
        // Try explicit tokenizer file first
        if (cliArgs.tokenizerPath != null) {
            val path = cliArgs.tokenizerPath
            if (!path.exists()) error("Tokenizer not found: $path")
            val isTekken = path.toString().endsWith("tekken.json", ignoreCase = true)
            return@run if (isTekken) {
                println("  Using Tekken tokenizer from $path")
                TekkenTokenizerAdapter.fromJson(path.readText())
            } else {
                println("  Using HuggingFace tokenizer from $path")
                GGUFTokenizer.fromTokenizerJson(path.readText())
            }
        }

        // Try auto-detected file tokenizer (tekken.json or tokenizer.json in model dir)
        val tekken = modelDir.resolve("tekken.json")
        val tokenizerJson = modelDir.resolve("tokenizer.json")
        when {
            tekken.exists() -> {
                println("  Using Tekken tokenizer from $tekken")
                TekkenTokenizerAdapter.fromJson(tekken.readText())
            }
            tokenizerJson.exists() -> {
                println("  Using HuggingFace tokenizer from $tokenizerJson")
                GGUFTokenizer.fromTokenizerJson(tokenizerJson.readText())
            }
            format == ModelFormat.GGUF -> {
                // Try embedded GGUF tokenizer as last resort
                val ggufPath = if (cliArgs.modelPath.isDirectory()) {
                    cliArgs.modelPath.toFile().listFiles()?.first { it.extension == "gguf" }!!.toPath()
                } else {
                    cliArgs.modelPath
                }
                try {
                    println("  Using embedded GGUF tokenizer")
                    GGUFTokenizer.fromRandomAccessSource(
                        JvmRandomAccessSource.open(ggufPath.toString())
                    )
                } catch (e: Exception) {
                    error("No tokenizer found. GGUF has no embedded tokenizer. " +
                        "Provide tekken.json with -t: kvoxtral -m model.gguf -t tekken.json ...")
                }
            }
            else -> {
                error("No tokenizer found in $modelDir. Provide one with -t/--tokenizer.")
            }
        }
    }

    // ---- Build backbone runtime (captures hidden states) ----
    val backboneRuntime = VoxtralBackboneRuntime(
        model = model,
        ctx = ctx,
        dtype = FP32::class,
        bos = audioConfig.bosTokenId
    )

    // ---- Load voice (optional) ----
    // Search for voice files in model dir and tokenizer dir (may differ for standalone GGUF)
    val voiceDirs = listOfNotNull(
        modelDir,
        cliArgs.tokenizerPath?.parent,
    ).distinct()

    fun findVoice(voiceName: String): sk.ainet.models.voxtral.VoxtralVoice? {
        for (dir in voiceDirs) {
            val voice = VoxtralVoiceLoader.loadFromDir(dir, voiceName)
            if (voice != null) return voice
        }
        return null
    }

    fun listAllVoices(): List<String> {
        return voiceDirs.flatMap { VoxtralVoiceLoader.listAvailable(it) }.distinct().sorted()
    }

    val voxtralVoice = if (cliArgs.voice != null && cliArgs.voice != "none") {
        val voiceName = cliArgs.voice
        println("Loading voice '$voiceName'...")
        val voice = findVoice(voiceName)
        if (voice != null) {
            println("  Loaded: ${voice.numFrames} frames x ${voice.dim} dim")
        } else {
            val available = listAllVoices()
            if (available.isEmpty()) {
                println("  No voice .pt files found in ${voiceDirs.joinToString(", ")}")
            } else {
                println("  Voice '$voiceName' not found. Available: ${available.joinToString(", ")}")
            }
        }
        voice
    } else {
        // Auto-detect: try to load default voice if available
        val available = listAllVoices()
        if (available.isNotEmpty() && cliArgs.voice != "none") {
            val defaultName = if (VoxtralVoices.DEFAULT in available) VoxtralVoices.DEFAULT else available.first()
            val voice = findVoice(defaultName)
            if (voice != null) {
                println("Auto-loaded voice '${voice.name}' (${voice.numFrames} frames)")
            }
            voice
        } else {
            null
        }
    }

    // ---- Step 1: Generate semantic tokens + capture hidden states ----
    val generatedTokens = mutableListOf<Int>()
    // allSemanticTokens = prompt + generated (matches hiddenStates frame count)
    val allSemanticTokens = mutableListOf<Int>()
    val hiddenStates: Tensor<FP32, Float>?

    if (cliArgs.testCodec) {
        // --test-codec: skip backbone, generate random semantic tokens
        val rng = kotlin.random.Random(42)
        val nTokens = cliArgs.steps.coerceAtMost(64)
        println()
        println("TEST-CODEC mode: generating $nTokens random semantic tokens (skipping backbone)")
        for (i in 0 until nTokens) {
            val tok = rng.nextInt(audioConfig.semanticCodebookSize)
            generatedTokens.add(tok)
            allSemanticTokens.add(tok)
        }
        hiddenStates = null // will generate random hidden states below
    } else {
        val promptTokens = tokenizer.encode(cliArgs.text)
        println()
        println("Input: ${promptTokens.size} tokens" +
            if (voxtralVoice != null) " + ${voxtralVoice.numFrames} voice frames" else "")
        println("Generating up to ${cliArgs.steps} semantic tokens (temperature=${cliArgs.temperature})...")

        // Phase 1: Voice conditioning (if applicable)
        if (voxtralVoice != null) {
            print("  Voice conditioning: 0/${voxtralVoice.numFrames} frames...")
            System.out.flush()
            val voiceTime = measureTime {
                backboneRuntime.reset()
                for (frame in 0 until voxtralVoice.numFrames) {
                    backboneRuntime.forwardEmbedding(voxtralVoice.frameEmbedding(frame))
                    if ((frame + 1) % 10 == 0 || frame == voxtralVoice.numFrames - 1) {
                        print("\r  Voice conditioning: ${frame + 1}/${voxtralVoice.numFrames} frames...")
                        System.out.flush()
                    }
                }
            }
            println("\r  Voice conditioning: ${voxtralVoice.numFrames} frames in $voiceTime")
        } else {
            backboneRuntime.reset()
        }

        // Phase 2: Prompt prefill
        print("  Prefill: 0/${promptTokens.size} tokens...")
        System.out.flush()
        val prefillTime = measureTime {
            for (i in 0 until promptTokens.size - 1) {
                backboneRuntime.forward(promptTokens[i])
                if ((i + 1) % 5 == 0) {
                    print("\r  Prefill: ${i + 1}/${promptTokens.size} tokens...")
                    System.out.flush()
                }
            }
        }
        println("\r  Prefill: ${promptTokens.size} tokens in $prefillTime")

        // Phase 3: Autoregressive generation
        // Track all tokens (prompt + generated) to match hidden states frame count
        allSemanticTokens.addAll(promptTokens.toList())

        println("  Generating up to ${cliArgs.steps} tokens...")
        val genTime = measureTime {
            var nextToken = if (promptTokens.isNotEmpty()) {
                val logits = backboneRuntime.forward(promptTokens.last())
                backboneRuntime.sample(logits, cliArgs.temperature)
            } else {
                VoxtralDefaults.AUDIO.bosTokenId
            }

            for (step in 0 until cliArgs.steps) {
                generatedTokens.add(nextToken)
                allSemanticTokens.add(nextToken)
                print("\r  Generated: ${step + 1}/${cliArgs.steps} tokens...")
                System.out.flush()
                val logits = backboneRuntime.forward(nextToken)
                nextToken = backboneRuntime.sample(logits, cliArgs.temperature)
            }
        }

        val tokPerSec = if (genTime.inWholeMilliseconds > 0) {
            generatedTokens.size / genTime.inWholeMilliseconds.toDouble() * 1000
        } else 0.0
        println("\r  Generated: ${generatedTokens.size} tokens in $genTime (%.2f tok/s)".format(tokPerSec))

        hiddenStates = backboneRuntime.lastHiddenStates()
    }

    // ---- Step 2: Get hidden states for acoustic model ----
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

    if (cliArgs.testCodec) {
        // In test-codec mode, generate random acoustic codes
        val rng = kotlin.random.Random(123)
        val nFrames = generatedTokens.size
        acousticCodes = IntArray(nFrames * nCodebooks) { rng.nextInt(codebookLevels) }
        println("TEST-CODEC: generated ${acousticCodes!!.size} random acoustic codes ($nFrames frames x $nCodebooks codebooks)")
    }

    val acousticTensors = allTensors.filterKeys { it.startsWith("acoustic.") }
    val hasAcousticWeights = acousticTensors.isNotEmpty()
    if (hiddenStates != null && hasAcousticWeights && acousticCodes == null) {
        val inputProjShape = allTensors[sk.ainet.models.voxtral.VoxtralTensorNames.ACOUSTIC_INPUT_PROJ]?.shape
        val outputProjShape = allTensors[sk.ainet.models.voxtral.VoxtralTensorNames.ACOUSTIC_OUTPUT_PROJ]?.shape
        println("Running acoustic flow matching (${cliArgs.flowSteps} ${cliArgs.flowMethod} steps, " +
            "$nCodebooks codebooks x $codebookLevels levels)")
        println("  inputProj: $inputProjShape, outputProj: $outputProjShape")

        val acousticTime = measureTime {
            val acousticRuntime = VoxtralAcousticRuntime<FP32>(
                weights = acousticTensors,
                ctx = ctx,
                dtype = FP32::class,
                nCodebooks = nCodebooks,
                codebookLevels = codebookLevels,
                dim = hiddenStates.shape[1]
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

    // ---- Pipeline shape validation ----
    if (acousticCodes != null) {
        val pv = PipelineShapeValidator()
        pv.matchCount(
            "semanticTokens", allSemanticTokens.size,
            "acousticFrames", acousticCodes!!.size / nCodebooks
        )
        val pvResult = pv.validate()
        if (!pvResult.isValid) {
            pvResult.printSummary(prefix = "  ")
            System.err.println("WARNING: Pipeline shape mismatch detected (see above)")
        }
    }

    // ---- Step 4: Decode to audio via codec (or tone-map fallback) ----
    var usedCodec = false
    val audioSamples: FloatArray = if (acousticCodes != null) {
        val codecWeights = allTensors.filterKeys { it.startsWith("codec.") }
        val hasCodecWeights = codecWeights.isNotEmpty()
        if (hasCodecWeights) {
            val convCount = codecWeights.keys.count { it.contains(".conv.") }
            val transformerCount = codecWeights.keys.count { it.contains(".layers.") }
            println("Running codec decoder (${codecWeights.size} weights: $convCount conv, $transformerCount transformer)...")
        } else {
            println("Running codec decoder (no weights)...")
        }
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
                    semanticCodes = allSemanticTokens.toIntArray(),
                    acousticCodes = acousticCodes!!
                )
            }
            usedCodec = true
            println("  Codec decoded ${decoded!!.size} samples in $codecTime")
            decoded!!
        } catch (e: Exception) {
            println("  Codec failed (${e.message}), using tone-map fallback")
            e.printStackTrace()
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

    // ---- Write beep WAV (optional) ----
    if (cliArgs.beep) {
        val beepPath = Path.of(cliArgs.outputPath.toString().replace(".wav", ".beep.wav"))
        val beepSamples = toneMapTokens(generatedTokens, audioConfig)
        println("Writing beep WAV: $beepPath (${beepSamples.size} samples)")
        WavWriter.write(
            path = beepPath,
            samples = beepSamples,
            sampleRate = audioConfig.samplingRate,
            channels = 1
        )
    }

    println()
    println("Done.")
    println("Output: ${cliArgs.outputPath}")
    if (cliArgs.beep) {
        println("Beep:   ${cliArgs.outputPath.toString().replace(".wav", ".beep.wav")}")
    }
    println()
    println("Pipeline: text -> ${generatedTokens.size} semantic tokens")
    if (acousticCodes != null) {
        println("          -> ${acousticCodes!!.size} acoustic codes (${acousticCodes!!.size / nCodebooks} frames)")
    }
    println("          -> ${audioSamples.size} audio samples -> WAV")
    if (!usedCodec) {
        println()
        println("Note: Audio is tone-mapped (codec not available).")
        if (format == ModelFormat.GGUF) {
            println("GGUF only contains backbone weights. Use SafeTensors model for speech synthesis.")
        } else {
            println("Load full SafeTensors model with codec weights for actual speech synthesis.")
        }
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

