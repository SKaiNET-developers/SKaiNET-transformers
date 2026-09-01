package sk.ainet.apps.skainet.cli

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
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.apps.llm.backend.BackendRegistry
import sk.ainet.apps.llm.backend.bestAvailable
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.apps.kllama.chat.ModelMetadata
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.I2sGgufLayout
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.io.gguf.planInput
import sk.ainet.lang.memory.plan.Budget
import sk.ainet.lang.memory.plan.MemoryPlans
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightResidency
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.types.FP32
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.exec.kernel.NativeTernaryLmheadKernel
import sk.ainet.models.bitnet.BitNetNetworkLoader
import sk.ainet.models.bitnet.BitNetWeightLoader
import sk.ainet.models.bitnet.BitNetTwoStageDecode
import sk.ainet.models.bitnet.bitnetPlanesHead
import sk.ainet.models.bitnet.generateTwoStage
import sk.ainet.lang.tensor.data.BitNetPlanesTensorData
import sk.ainet.models.qwen.QwenNetworkLoader
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
    val contextLength: Int?,
    val explainLoad: Boolean,
    val i2sLayout: String?,
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
    println("  --explain-load      Print per-weight placement decisions (mapped/heap and why) before loading")
    println("  --i2s-layout=L      BitNet I2_S converter flavor: group128 (default), group64, sequential")
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
    var explainLoad = false
    var i2sLayout: String? = null

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
            arg == "--explain-load" -> explainLoad = true
            arg.startsWith("--i2s-layout=") -> i2sLayout = arg.substringAfter("=")
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

    return CliArgs(modelPath, steps, temperature, prompt, chatMode, agentMode, demoMode, templateName, contextLength, explainLoad, i2sLayout)
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

        // 0.52 self-healing dispatch: providers, the row-major packs AND the ternary packs
        // (SKaiNET#1240) are ServiceLoader-discovered — one call replaces the per-pack
        // bootstrap this file used to carry. Explicit here (rather than lazy-on-first-matmul)
        // so the packs are in before the loaders' mapped-staging decisions.
        @OptIn(sk.ainet.lang.memory.ExperimentalMemoryApi::class)
        run {
            KernelDispatch.ensureInstalled()
        }

        // Memory plan before anything is allocated: header-only arithmetic
        // priced with the same form the loaders below request for every
        // tensor — MAPPED keep-packed; even the token embedding stays at its
        // packed footprint (rewrapped as a row-dequant source, not inflated
        // to dense FP32) — against the JVM heap cap. Mapped weights page
        // against device RAM, not this budget (#1189) — the whole point of
        // the MAPPED default.
        @OptIn(sk.ainet.lang.memory.ExperimentalMemoryApi::class)
        run {
            val mappedDefault = WeightForm(
                shape = WeightShapeOrientation.OUT_IN,
                residency = WeightResidency.MAPPED,
            )
            val planInput = JvmRandomAccessSource.open(modelPath.toString()).use { source ->
                StreamingGGUFReader.open(source).planInput(
                    ctx = cliArgs.contextLength,
                    formFor = { mappedDefault },
                )
            }
            val plan = MemoryPlans.plan(
                planInput,
                Budget(Runtime.getRuntime().maxMemory(), "JVM max heap (-Xmx)"),
            )
            println(plan.render())
            if (cliArgs.explainLoad) {
                // One line per weight: where it lands (mmap page cache vs heap)
                // and why — the same AllocationResolver the plan itself uses.
                println("Placements (--explain-load):")
                for (line in planInput.weights.map {
                    sk.ainet.lang.memory.plan.AllocationResolver.explain(it, sk.ainet.lang.memory.plan.PlannerProfile.DESKTOP)
                }) {
                    println("  $line")
                }
            }
            if (plan.fits == false) {
                System.err.println("WARNING: planned heap use exceeds the JVM heap cap — the load may OOM. See suggestions above.")
            }
        }

        // Set up execution context. Weights no longer go through a hand-managed
        // MemorySegment Arena — the engine loader owns residency (mapped pages
        // or heap arrays) per the requested WeightForm, and activations live on
        // the default heap factory.
        val ctx = DirectCpuExecutionContext()

        // Load model based on detected family. All families route through
        // the DSL pipeline (per-family network() builder +
        // OptimizedLLMRuntime). The legacy LlamaRuntime path was retired
        // for the kllama CLI in #121 / #122; this CLI follows in this PR.
        // Numerical equivalence with the legacy path on identical weights
        // is pinned by `QwenDslLegacyParityTest` (#120).
        //
        // Apertus had previously fallen through to the LlamaRuntime
        // branch — that runtime doesn't implement Apertus's xIELU
        // activation, QK-Norm, or ungated FFN, so logits silently
        // diverged from the checkpoint's intent. The DSL path is correct
        // for Apertus too. See APERTUS_ROLLOUT.md (PR 1).
        // Set on the BitNet path when the lm_head loaded as BITNET_PLANES — gates the
        // two-stage decode loop at the generation site (transformers#358).
        var bitnetTwoStageHead: BitNetPlanesTensorData? = null

        val runtime: InferenceRuntime<FP32> = if (modelInfo.family == ModelFamily.GEMMA) {
            println("Loading Gemma GGUF model from $modelPath via gemmaNetwork() + OptimizedLLMRuntime (engine loader, keep-packed, mapped)...")
            if (cliArgs.contextLength != null) {
                println("  --context flag currently ignored on the Gemma path; uses model default capped to 4096.")
            }
            val model = GemmaNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) }
            ).load<FP32, Float>(ctx)
            OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class)
        } else if (modelInfo.family == ModelFamily.APERTUS) {
            println("Loading Apertus GGUF model from $modelPath via apertusNetwork() + OptimizedLLMRuntime (engine loader, keep-packed, mapped)...")
            if (cliArgs.contextLength != null) {
                println("  --context flag currently ignored on the Apertus path; uses model default.")
            }
            val model = ApertusNetworkLoader.fromGguf(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) }
            ).load<FP32, Float>(ctx)
            OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class)
        } else if (modelInfo.family == ModelFamily.BITNET) {
            // BitNet b1.58: the packed I2_S path (transformers#337). Ternary projections load as
            // 2-bit BITNET_B1_58 tensors (0.25 B/weight) through the SKaiNET engine loader, and
            // dispatch runs them on the vendored NeoGPU NEON kernels installed below — the exact
            // f32 path, no requantization error. The lm_head is served as the fused BITNET_PLANES
            // format either way (#357): a file's own output.weight requantizes losslessly; a
            // tied-embeddings file (2B4T) gets the head materialized from token_embd — a bounded
            // 8-plane requantization, the NeoGPU lm_head design.
            val i2sLayout = when (cliArgs.i2sLayout?.lowercase()) {
                null, "group128" -> I2sGgufLayout.GROUP_128
                "group64" -> I2sGgufLayout.GROUP_64
                "sequential" -> I2sGgufLayout.SEQUENTIAL
                else -> usage("Invalid --i2s-layout '${cliArgs.i2sLayout}' (group128 | group64 | sequential).")
            }
            println("Loading BitNet GGUF model from $modelPath via BitNetWeightLoader (packed I2_S, ${i2sLayout.name} flavor)...")
            // Ternary kernels arrive via the self-healing SPI (SKaiNET#1240, in this BOM) —
            // the explicit installs this branch used to carry are gone.
            val weights = BitNetWeightLoader.loadRuntimeWeights(
                ctx,
                sourceProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
                i2sLayout = i2sLayout,
            )
            // --context caps the KV/RoPE allocation through bitnetNetwork's
            // maxInferenceLen = min(contextLength, 4096) default.
            val metadata = cliArgs.contextLength
                ?.let { weights.metadata.copy(contextLength = minOf(it, weights.metadata.contextLength)) }
                ?: weights.metadata
            if (cliArgs.contextLength != null) {
                println("Context length capped to ${metadata.contextLength} (model default: ${weights.metadata.contextLength})")
            }
            val loaded = BitNetWeightLoader.Loaded(
                sk.ainet.models.bitnet.BitNetRuntimeWeights(metadata, weights.tensors).toModule(),
                metadata,
            )
            bitnetTwoStageHead = bitnetPlanesHead(loaded.model)
            if (bitnetTwoStageHead != null) {
                println(
                    "Two-stage lm_head decode: enabled (BITNET_PLANES head, " +
                        "top-${BitNetTwoStageDecode.DEFAULT_CANDIDATES} exact rescoring).",
                )
            }
            OptimizedLLMRuntime(
                model = loaded.model,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class,
                bos = loaded.metadata.bosTokenId,
            )
        } else {
            // LLaMA / Qwen / Mistral DSL path. DecoderGgufWeightLoader
            // streams the GGUF through the engine loader, keeping quantized
            // tensors in their stored block encoding as packed tensor data,
            // then the per-family network loader builds the right module:
            //   - Qwen → qwenNetwork() (QK-norm + NEOX RoPE)
            //   - BitNet → bitnetNetwork() (relu² FFN + sub-norms)
            //   - else → llamaNetwork() (LLaMA / Mistral default)
            val acceptedArchitectures = modelInfo.family.architectures + setOf(modelInfo.architecture)
            val loader = DecoderGgufWeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
                acceptedArchitectures = acceptedArchitectures,
            )

            println("Loading GGUF model from $modelPath (${modelInfo.family.displayName}, DSL streaming, keep-packed, mapped)...")
            val convertedWeights = loader.loadToMapStreaming<FP32, Float>(ctx)

            if (cliArgs.contextLength != null) {
                println("Context length capped to ${cliArgs.contextLength} (model default: ${convertedWeights.metadata.contextLength})")
            }

            val model = when (modelInfo.family) {
                ModelFamily.QWEN -> QwenNetworkLoader.fromWeights(convertedWeights)
                ModelFamily.BITNET -> BitNetNetworkLoader.fromWeights(convertedWeights)
                else -> LlamaNetworkLoader.fromWeights(convertedWeights)
            }
            OptimizedLLMRuntime(
                model = model,
                ctx = ctx,
                mode = OptimizedLLMMode.DIRECT,
                dtype = FP32::class,
                bos = convertedWeights.metadata.bosTokenId,
            )
        }

        // Load tokenizer from already-parsed GGUF metadata. Routes to the
        // upstream sk.ainet.io.tokenizer impl (correct byte-level BPE for
        // Qwen/GPT-2 — see issue #52). The legacy fromGGUF(source) path
        // uses the local forked impl with broken byte-BPE.
        println("Loading embedded GGUF tokenizer...")
        val tokenizer: Tokenizer = TokenizerFactory.fromGgufFields(modelInfo.fields)

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

        val twoStageHead = bitnetTwoStageHead
        @Suppress("UNCHECKED_CAST")
        val twoStageRuntime = if (twoStageHead != null) runtime as? OptimizedLLMRuntime<FP32> else null
        val elapsed = measureTime {
            if (twoStageHead != null && twoStageRuntime != null) {
                // transformers#358: trunk-only forwards + fused stage-1 scan + exact top-200
                // rescoring — the full-vocab lm_head matmul never runs, prefill included.
                twoStageRuntime.generateTwoStage(
                    prompt = promptTokens,
                    steps = cliArgs.steps,
                    temperature = cliArgs.temperature,
                    head = twoStageHead,
                    native = if (NativeTernaryLmheadKernel.isAvailable()) {
                        sk.ainet.models.bitnet.BitNetStage1Kernel(NativeTernaryLmheadKernel::lmheadStage1)
                    } else {
                        null
                    },
                ) { id ->
                    print(tokenizer.decode(id))
                }
            } else {
                runtime.generate(prompt = promptTokens, steps = cliArgs.steps, temperature = cliArgs.temperature) { id ->
                    print(tokenizer.decode(id))
                }
            }
        }.inWholeMilliseconds

        val tokPerSec = cliArgs.steps / elapsed.toDouble() * 1000
        println("\n---")
        println("tok/s: $tokPerSec")
    }
}
