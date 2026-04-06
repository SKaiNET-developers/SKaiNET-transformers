package sk.ainet.apps.voxtral.cli

import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.types.FP32
import sk.ainet.models.voxtral.VoxtralConfigParser
import sk.ainet.models.voxtral.VoxtralDefaults
import sk.ainet.models.voxtral.VoxtralNetworkLoader
import kotlinx.coroutines.runBlocking
import java.nio.file.Files
import java.nio.file.Path
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
    val text: String,
    val outputPath: Path,
    val steps: Int,
    val temperature: Float
)

private fun usage(errorMessage: String? = null): Nothing {
    if (errorMessage != null) {
        System.err.println("Error: $errorMessage")
        System.err.println()
    }

    println("Usage: kvoxtral -m <model> -o <output.wav> [-t <tokenizer>] [-s <steps>] [-k <temperature>] <text>")
    println()
    println("  -m, --model         Path to .gguf or .safetensors model, or HuggingFace directory (required)")
    println("  -o, --output        Output WAV file path (required)")
    println("  -t, --tokenizer     Path to tokenizer.json or tekken.json (auto-detected)")
    println("  -s, --steps         Generation steps / max audio tokens (default: 512)")
    println("  -k, --temperature   Sampling temperature (default: 0.7)")
    println("  -h, --help          Show this help")
    println()
    println("Example:")
    println("  kvoxtral -m Voxtral-4B-TTS-2603/ -o hello.wav \"Hello, how are you today?\"")
    println("  kvoxtral -m voxtral.gguf -o speech.wav -s 1024 \"The quick brown fox\"")
    exitProcess(if (errorMessage == null) 0 else 1)
}

private fun parseArgs(args: Array<String>): CliArgs {
    if (args.isEmpty()) usage("Missing arguments.")

    var model: String? = null
    var tokenizer: String? = null
    var output: String? = null
    var steps = 512
    var temperature = 0.7f
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

    return CliArgs(
        modelPath = modelPath,
        tokenizerPath = tokenizer?.let(Path::of),
        text = text,
        outputPath = outputPath,
        steps = steps,
        temperature = temperature
    )
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

/**
 * Resolve the SafeTensors file path. Voxtral uses `consolidated.safetensors` (Mistral format).
 */
private fun resolveSafeTensorsFile(modelPath: Path): Path {
    if (!modelPath.isDirectory()) return modelPath
    val consolidated = modelPath.resolve("consolidated.safetensors")
    if (consolidated.exists()) return consolidated
    val model = modelPath.resolve("model.safetensors")
    if (model.exists()) return model
    error("No .safetensors file found in $modelPath")
}

/**
 * Resolve config file. Voxtral uses `params.json` (Mistral format), falling back to `config.json`.
 */
private fun resolveConfigFile(modelDir: Path): Path? {
    val params = modelDir.resolve("params.json")
    if (params.exists()) return params
    val config = modelDir.resolve("config.json")
    if (config.exists()) return config
    return null
}

/**
 * Resolve tokenizer file. Voxtral uses `tekken.json` (Mistral tokenizer), falling back to `tokenizer.json`.
 */
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

    println("Voxtral TTS (Kotlin)")
    println("Model: ${cliArgs.modelPath}")
    println("Format: $format")
    println("Text: \"${cliArgs.text}\"")
    println("Output: ${cliArgs.outputPath}")
    println()

    val ctx = DirectCpuExecutionContext()

    // ---- Load model ----
    println("Loading model...")
    val loadTime = measureTime {
        val model = when (format) {
            ModelFormat.GGUF -> {
                val ggufPath = if (cliArgs.modelPath.isDirectory()) {
                    val gguf = cliArgs.modelPath.toFile().listFiles()
                        ?.firstOrNull { it.extension == "gguf" }
                        ?: error("No .gguf file found in ${cliArgs.modelPath}")
                    gguf.toPath()
                } else {
                    cliArgs.modelPath
                }
                println("Loading GGUF model from $ggufPath...")
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
                    println("Loading config from $configFile...")
                    val configJson = configFile.readText()
                    VoxtralConfigParser.parse(configJson).backbone
                } else {
                    println("No config file found, using Voxtral-4B defaults...")
                    VoxtralDefaults.BACKBONE
                }

                val tiedEmbeddings = if (configFile != null) {
                    VoxtralConfigParser.isTiedEmbeddings(configFile.readText())
                } else {
                    true
                }

                println("Loading SafeTensors model from $stFile...")
                println("  Architecture: ${metadata.architecture}, layers=${metadata.blockCount}, " +
                    "dim=${metadata.embeddingLength}, heads=${metadata.headCount}, " +
                    "kvHeads=${metadata.kvHeadCount}, vocab=${metadata.vocabSize}")
                if (tiedEmbeddings) println("  Tied word embeddings: output.weight = token_embd.weight")

                val loader = VoxtralNetworkLoader.fromSafeTensors(
                    metadata = metadata,
                    randomAccessProvider = { JvmRandomAccessSource.open(stFile.toString()) },
                    tiedEmbeddings = tiedEmbeddings
                )
                loader.loadBackbone<FP32, Float>(ctx)
            }
        }

        // ---- Load tokenizer ----
        val tokenizerFile = resolveTokenizerFile(modelDir, cliArgs.tokenizerPath)
        println("Loading tokenizer from $tokenizerFile...")
        val tokenizer: Tokenizer = if (format == ModelFormat.GGUF && cliArgs.tokenizerPath == null) {
            val ggufPath = if (cliArgs.modelPath.isDirectory()) {
                cliArgs.modelPath.toFile().listFiles()?.first { it.extension == "gguf" }!!.toPath()
            } else {
                cliArgs.modelPath
            }
            GGUFTokenizer.fromRandomAccessSource(
                JvmRandomAccessSource.open(ggufPath.toString())
            )
        } else {
            GGUFTokenizer.fromTokenizerJson(tokenizerFile.readText())
        }

        // ---- Build runtime ----
        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
            bos = VoxtralDefaults.AUDIO.bosTokenId
        )

        // ---- Encode text and generate semantic tokens ----
        val promptTokens = tokenizer.encode(cliArgs.text)
        println("Input: ${promptTokens.size} tokens")
        println("Generating up to ${cliArgs.steps} audio tokens with temperature=${cliArgs.temperature}...")
        println()

        val generatedTokens = mutableListOf<Int>()
        val genTime = measureTime {
            runtime.generate(
                prompt = promptTokens,
                steps = cliArgs.steps,
                temperature = cliArgs.temperature
            ) { tokenId ->
                generatedTokens.add(tokenId)
            }
        }

        val tokPerSec = if (genTime.inWholeMilliseconds > 0) {
            generatedTokens.size / genTime.inWholeMilliseconds.toDouble() * 1000
        } else 0.0
        println("Generated ${generatedTokens.size} tokens in $genTime (%.2f tok/s)".format(tokPerSec))

        // ---- Convert tokens to audio and write WAV ----
        // The backbone generates semantic token IDs. Full audio synthesis requires
        // the acoustic model (flow matching) and codec (convolutional decoder).
        // For now, we write generated tokens as a simple tone-mapped WAV to validate
        // the pipeline. Each token ID maps to a short sine burst at a frequency
        // derived from the token ID, producing an audible (if not speech-like) output.
        val audioConfig = VoxtralDefaults.AUDIO
        val samplesPerToken = (audioConfig.samplingRate / audioConfig.frameRate).toInt()
        val totalSamples = generatedTokens.size * samplesPerToken
        val audioSamples = FloatArray(totalSamples)

        for ((i, tokenId) in generatedTokens.withIndex()) {
            // Map token ID to a frequency (200-2000 Hz range)
            val freq = 200.0 + (tokenId % 1800)
            val offset = i * samplesPerToken
            for (s in 0 until samplesPerToken) {
                val t = s.toDouble() / audioConfig.samplingRate
                val envelope = if (s < 20) s / 20.0f else if (s > samplesPerToken - 20) (samplesPerToken - s) / 20.0f else 1.0f
                audioSamples[offset + s] = (Math.sin(2.0 * Math.PI * freq * t) * 0.3 * envelope).toFloat()
            }
        }

        println("Writing WAV: ${cliArgs.outputPath} (${totalSamples} samples, ${audioConfig.samplingRate}Hz, mono)")
        WavWriter.write(
            path = cliArgs.outputPath,
            samples = audioSamples,
            sampleRate = audioConfig.samplingRate,
            channels = 1
        )
    }

    println()
    println("Done in $loadTime")
    println("Output: ${cliArgs.outputPath}")
    println()
    println("Note: Full speech synthesis requires the acoustic model and codec decoder.")
    println("Currently, the backbone generates semantic tokens which are tone-mapped to audio.")
}
