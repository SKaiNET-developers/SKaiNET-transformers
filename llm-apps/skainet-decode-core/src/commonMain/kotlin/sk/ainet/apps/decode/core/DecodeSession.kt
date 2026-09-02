package sk.ainet.apps.decode.core

import sk.ainet.apps.llm.GGUFModelInfo
import sk.ainet.apps.llm.ModelFamily
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.UnifiedModelLoader
import sk.ainet.apps.llm.sampleFromTensor
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.MemoryProbe
import sk.ainet.lang.memory.sample
import sk.ainet.lang.memory.trace.CompositeTraceSink
import sk.ainet.lang.memory.trace.GenerationMetrics
import sk.ainet.lang.memory.trace.RecordingTraceSink
import sk.ainet.lang.memory.trace.TraceSink
import sk.ainet.lang.memory.trace.decodeStep
import sk.ainet.lang.memory.trace.prefill
import sk.ainet.lang.memory.trace.sample
import sk.ainet.lang.types.FP32
import sk.ainet.models.bitnet.BitNetWeightLoader
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader
import sk.ainet.models.llama.LlamaNetworkLoader
import sk.ainet.models.qwen.QwenNetworkLoader

/** What one [DecodeSession.run] measured: the metrics, the generated text, and trace-ring health. */
public data class DecodeReport(
    val metrics: GenerationMetrics,
    /** The decoded continuation (prompt not included). */
    val text: String,
    /** Events the recording ring dropped; > 0 means early prefill spans may be undercounted. */
    val droppedTraceEvents: Long,
)

/**
 * The shared `skainet-decode` flow (SKaiNET#1129/#1244): load a GGUF, decode, and report
 * [GenerationMetrics] — TTFT, prefill and decode tok/s, effective memory bandwidth,
 * kernel/adapter shares, page-fault rate. The JVM CLI and the Android activity are both thin
 * callers of [run], so the two legs cannot diverge.
 *
 * Two deliberate properties, straight from the issue:
 *
 * - **Weight forms are resolved, not configured.** The family loaders' defaults do the deciding
 *   (keep-packed, `MAPPED` where the file can be served zero-copy); this flow carries no policy
 *   flags at all.
 * - **A traced run is a reportable run.** The generation loop opens the `prefill` / `decode` /
 *   `sample` spans on a [RecordingTraceSink]; kernel runs, adapter insertions and byte counters
 *   arrive through `KernelDispatch.defaultSink`; [GenerationMetrics.from] reads the stream back.
 *   [MemoryProbe] is sampled inside every decode span, so the RSS / page-fault rows light up on
 *   any platform with a `/proc` answer (Android, Linux) and render "—" elsewhere.
 *
 * Families: BitNet (packed I2_S path) and the shared decoder families (Llama / Mistral / Qwen).
 *
 * Threading: [RecordingTraceSink] is not thread-safe — construct the session and call [run] on
 * one thread, and only hand the returned [DecodeReport] across threads.
 *
 * @param extraSink an additional sink composed beside the recording one (e.g. an
 *   `AndroidTraceSink` for Perfetto spans); metrics always come from the recording sink.
 */
public class DecodeSession(
    capacity: Int = 1 shl 20,
    extraSink: TraceSink? = null,
) {
    /** The metrics source; exposed so callers can inspect or export the raw events. */
    public val recording: RecordingTraceSink = RecordingTraceSink(capacity = capacity)

    private val sink: TraceSink =
        if (extraSink != null) CompositeTraceSink(listOf(recording, extraSink)) else recording

    /** The execution context the whole session runs under; its trace sink is [sink]. */
    public val ctx: ExecutionContext

    init {
        // One sink sees everything: the loop's phase spans (via ctx.traceSink) and — through the
        // 0.52 diagnostic hook — every kernel run, adapter insertion and byte counter the
        // dispatcher emits. That is what lights up the bandwidth / kernel-share rows.
        KernelDispatch.defaultSink = sink
        ctx = object : ExecutionContext by DirectCpuExecutionContext() {
            override val traceSink: TraceSink get() = sink
        }
        // Self-healing dispatch (ternary packs included, SKaiNET#1240) — no per-pack bootstrap.
        KernelDispatch.ensureInstalled()
    }

    /**
     * Load the model behind [sourceProvider], ingest [prompt], decode [steps] tokens, and return
     * the [DecodeReport].
     *
     * @param onModelInfo fires after the GGUF header peek, before weights load.
     * @param onGenerationStart fires once the runtime is built, right before the prefill span.
     * @param onToken fires per decoded token with its text.
     */
    public suspend fun run(
        sourceProvider: () -> RandomAccessSource,
        prompt: String,
        steps: Int,
        temperature: Float = 0f,
        peakBytesPerSecond: Long? = null,
        onModelInfo: (GGUFModelInfo) -> Unit = {},
        onGenerationStart: () -> Unit = {},
        onToken: (String) -> Unit = {},
    ): DecodeReport {
        val modelInfo = UnifiedModelLoader.peek(sourceProvider)
        onModelInfo(modelInfo)
        val tokenizer = TokenizerFactory.fromGgufFields(modelInfo.fields)

        val (module, bos) = when (modelInfo.family) {
            ModelFamily.BITNET -> {
                val loaded = BitNetWeightLoader.loadWithMetadata(ctx, sourceProvider)
                loaded.model to loaded.metadata.bosTokenId
            }
            else -> {
                val weights = DecoderGgufWeightLoader(
                    randomAccessProvider = sourceProvider,
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
        val promptTokens =
            if (raw.isNotEmpty() && raw[0] == tokenizer.bosTokenId) raw else intArrayOf(tokenizer.bosTokenId) + raw

        onGenerationStart()

        // Prompt ingestion under one prefill span — logits of all but the last prompt token are
        // discarded, so their forwards are pure ingestion.
        sink.prefill(tokens = promptTokens.size - 1) {
            for (p in 0 until promptTokens.size - 1) runtime.forward(promptTokens[p])
        }

        val text = StringBuilder()
        var token = promptTokens.last()
        repeat(steps) { step ->
            val logits = sink.decodeStep(step) {
                val l = runtime.forward(token)
                // Sampled inside the open decode span: GenerationMetrics derives the page-fault
                // rate from the first-vs-last counter values it sees between decode boundaries.
                MemoryProbe.sample().emitTo(sink)
                l
            }
            val next = sink.sample(step) { sampleFromTensor(logits, temperature) }
            val piece = tokenizer.decode(next)
            text.append(piece)
            onToken(piece)
            token = next
        }

        return DecodeReport(
            metrics = GenerationMetrics.from(recording.events(), peakBytesPerSecond),
            text = text.toString(),
            droppedTraceEvents = recording.dropped,
        )
    }
}
