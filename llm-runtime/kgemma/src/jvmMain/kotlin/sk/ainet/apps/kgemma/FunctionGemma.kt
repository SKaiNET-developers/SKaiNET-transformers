package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.generate
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL
import sk.ainet.models.gemma.Gemma4WeightLoader
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.transformers.gemma.iree.CompactCodec
import sk.ainet.transformers.gemma.iree.FunctionGemmaChatTemplate
import sk.ainet.transformers.gemma.iree.ToolCall
import kotlin.random.Random

/**
 * FunctionGemma — one-line function-calling from the SKaiNET DSL, in BOTH execution modes.
 *
 * ```
 * val fg = FunctionGemma.fromGguf("…functiongemma…Q5_K_M.gguf")
 * fg.call("turn the light on")     // EAGER (DirectCpu, no iree) -> ToolCall(set_lights, {state=on})
 * fg.exportCompiled("build/mlir")  // COMPILED (edge) -> gemma-gen.mlir + bf16 gemma.safetensors
 * ```
 *
 * One `gemmaNetwork()` DSL definition, HW selected by the SKaiNET seams: the **execution
 * context** (eager — runs anywhere on CPU) or the **compile target** (compiled — the SL2610
 * edge vmfb, via [FunctionGemmaExport]). Eager dequantizes Q5_K → FP32 ("works, moderate
 * speed"); the compiled path is the edge-perf artifact.
 */
public class FunctionGemma private constructor(
    private val ggufPath: String,
    private val runtime: InferenceRuntime<FP32>,
    private val tokenizer: GGUFTokenizer,
    private val bos: Int,
    private val eot: Int,
    private val eos: Int,
) {
    public data class Turn(val text: String, val calls: List<ToolCall>)

    private val chatTemplate = FunctionGemmaChatTemplate()

    /**
     * EAGER function-calling: apply the Octopus-v2 chat template ([FunctionGemmaChatTemplate] —
     * the exact `<start_of_turn>user\n…<end_of_turn>\n<start_of_turn>model\n` string this method
     * used to hardcode), greedily generate the compact tool call, and parse it — entirely
     * in-process (no iree-compile, no board).
     */
    public fun call(userText: String, maxTokens: Int = 24): Turn {
        val prompt = chatTemplate.apply(listOf(ChatMessage(ChatRole.USER, userText)))
        val ptoks = tokenizer.encode(prompt) // generate() prepends bos
        val gen = ArrayList<Int>(maxTokens)
        var stopped = false
        runtime.generate(prompt = ptoks, steps = maxTokens, temperature = 0f, bosToken = bos) { id ->
            if (!stopped) {
                if (id == eot || id == eos) stopped = true else gen.add(id)
            }
        }
        val text = tokenizer.decode(gen.toIntArray())
        return Turn(text, CompactCodec.parse(text))
    }

    /**
     * COMPILED export for the edge: trace `gemmaNetwork()` (DSL argMax tail) → StableHLO with bf16
     * external params. Produces `gemma-gen.mlir` + `gemma.safetensors`; `compile-gemma.sh` turns
     * them into `gemma-gen.vmfb` + `gemma-gen.irpa`.
     */
    public fun exportCompiled(outDir: String, seq: Int = 24, bf16: Boolean = true): FunctionGemmaExport.Result =
        FunctionGemmaExport.export(gguf = ggufPath, outDir = outDir, seq = seq, bf16 = bf16)

    public companion object {
        /**
         * Load FunctionGemma from a GGUF checkpoint for EAGER use. Tokenizer + weights come from the
         * GGUF; global-layer RoPE is forced to full rotary ([partialRotary] = 1.0 — the gemma3
         * convention the gguf omits, which the loader would otherwise default to 0.25 and mis-rotate).
         */
        public fun fromGguf(gguf: String, partialRotary: Float = 1.0f): FunctionGemma = runBlocking {
            val tok = GGUFTokenizer.fromRandomAccessSource(JvmRandomAccessSource.open(gguf))
            val ctx = DirectCpuExecutionContext.create()
            val weights = Gemma4WeightLoader(
                randomAccessProvider = { JvmRandomAccessSource.open(gguf) },
                weightForm = GEMMA_DEQUANTIZE_ALL,
            ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
            val patched = weights.copy(
                metadata = weights.metadata.copy(
                    ropeParametersFull = weights.metadata.ropeParametersFull.copy(partialRotaryFactor = partialRotary),
                ),
            )
            val model = GemmaNetworkLoader.fromWeights(ctx, patched, FP32::class)
            val runtime = OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class, random = Random.Default)
            FunctionGemma(
                ggufPath = gguf,
                runtime = runtime,
                tokenizer = tok,
                bos = tok.bosTokenId,
                eot = tok.encode("<end_of_turn>").single(),
                eos = tok.eosTokenId,
            )
        }
    }
}
