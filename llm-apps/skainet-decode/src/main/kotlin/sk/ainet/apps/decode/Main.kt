package sk.ainet.apps.decode

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.ModelFamily
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.UnifiedModelLoader
import sk.ainet.apps.llm.sampleFromTensor
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.memory.trace.GenerationMetrics
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.memory.trace.decodeStep
import sk.ainet.lang.memory.trace.prefill
import sk.ainet.lang.memory.trace.sample
import sk.ainet.lang.types.FP32
import sk.ainet.models.bitnet.BitNetWeightLoader
import sk.ainet.models.llama.DecoderGgufWeightLoader
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.models.qwen.QwenNetworkLoader
import kotlin.system.exitProcess

/**
 * `skainet-decode` (SKaiNET#1129): load a GGUF, decode, and report [GenerationMetrics] — TTFT,
 * prefill and decode tok/s, **effective memory bandwidth**, kernel/adapter shares, page-fault
 * rate. These are the numbers the SKEEP-003 memory work exists to move, measured on a real
 * model instead of a microbench.
 *
 * Two deliberate properties, straight from the issue:
 *
 * - **Weight forms are resolved, not configured.** The family loaders' defaults do the deciding
 *   (keep-packed, `MAPPED` where the file can be served zero-copy); this sample carries no
 *   policy flags at all.
 * - **A traced run is a reportable run.** The generation loop opens the `prefill` /
 *   `decode` / `sample` spans on a [RecordingTraceSink]; kernel runs, adapter insertions and
 *   byte counters arrive through `KernelDispatch.defaultSink`; [GenerationMetrics.from] reads
 *   the stream back.
 *
 * Families: BitNet (packed I2_S path) and the shared decoder families (Llama / Mistral / Qwen).
 */
@OptIn(ExperimentalMemoryApi::class)
fun main(args: Array<String>) {
    var model: String? = null
    var prompt = "The capital of France is"
    var steps = 32
    var temperature = 0f
    var i = 0
    while (i < args.size) {
        when (val a = args[i]) {
            "-m" -> model = args.getOrNull(++i)
            "-p" -> prompt = args.getOrNull(++i) ?: prompt
            "-s" -> steps = args.getOrNull(++i)?.toIntOrNull() ?: steps
            "-k" -> temperature = args.getOrNull(++i)?.toFloatOrNull() ?: temperature
            else -> { System.err.println("unknown arg '$a'"); exitProcess(1) }
        }
        i++
    }
    val modelPath = model ?: run {
        System.err.println("usage: skainet-decode -m <model.gguf> [-p prompt] [-s steps] [-k temperature]")
        exitProcess(1)
    }

    // One sink sees everything: the loop's phase spans (via ctx.traceSink) and — through the
    // 0.52 diagnostic hook — every kernel run, adapter insertion and byte counter the
    // dispatcher emits. That is what lights up the bandwidth / kernel-share rows below.
    val sink = RecordingTraceSink(capacity = 1 shl 20)
    KernelDispatch.defaultSink = sink
    val ctx: ExecutionContext = object : ExecutionContext by DirectCpuExecutionContext() {
        override val traceSink: TraceSink get() = sink
    }
    // Self-healing dispatch (ternary packs included, SKaiNET#1240) — no per-pack bootstrap.
    KernelDispatch.ensureInstalled()

    runBlocking {
        val modelInfo = UnifiedModelLoader.peek { JvmRandomAccessSource.open(modelPath) }
        println("Model: $modelPath (${modelInfo.family.displayName}, ${modelInfo.blockCount} layers, vocab=${modelInfo.vocabSize})")
        val tokenizer = TokenizerFactory.fromGgufFields(modelInfo.fields)

        val (module, bos) = when (modelInfo.family) {
            ModelFamily.BITNET -> {
                val loaded = BitNetWeightLoader.loadWithMetadata(
                    ctx, { JvmRandomAccessSource.open(modelPath) },
                )
                loaded.model to loaded.metadata.bosTokenId
            }
            else -> {
                val weights = DecoderGgufWeightLoader(
                    randomAccessProvider = { JvmRandomAccessSource.open(modelPath) },
                    acceptedArchitectures = modelInfo.family.architectures + setOf(modelInfo.architecture),
                ).loadToMapStreaming<FP32, Float>(ctx)
                val m = when (modelInfo.family) {
                    ModelFamily.QWEN -> QwenNetworkLoader.fromWeights(weights)
                    else -> LlamaNetworkLoader.fromWeights(weights)
                }
                m to weights.metadata.bosTokenId
            }
        }
        val runtime = OptimizedLLMRuntime(module, ctx, OptimizedLLMMode.DIRECT, FP32::class, bos = bos)

        val raw = tokenizer.encode(prompt)
        val promptTokens = if (raw.isNotEmpty() && raw[0] == tokenizer.bosTokenId) raw else intArrayOf(tokenizer.bosTokenId) + raw

        println("Generating $steps tokens (temperature=$temperature)...")
        print(prompt)

        // Prompt ingestion under one prefill span — logits of all but the last prompt token are
        // discarded, so their forwards are pure ingestion.
        sink.prefill(tokens = promptTokens.size - 1) {
            for (p in 0 until promptTokens.size - 1) runtime.forward(promptTokens[p])
        }

        var token = promptTokens.last()
        repeat(steps) { step ->
            val logits = sink.decodeStep(step) { runtime.forward(token) }
            val next = sink.sample(step) { sampleFromTensor(logits, temperature) }
            print(tokenizer.decode(next))
            token = next
        }
        println()
        println()

        val metrics = GenerationMetrics.from(sink.events())
        println(metrics.render())
        if (sink.dropped > 0) {
            println("(trace ring dropped ${sink.dropped} events — early prefill spans may be undercounted)")
        }
    }
}
