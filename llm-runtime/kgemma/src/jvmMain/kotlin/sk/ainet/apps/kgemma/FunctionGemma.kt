package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.GGUFTokenizer
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.generateUntilStop
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GEMMA_DEQUANTIZE_ALL
import sk.ainet.models.gemma.Gemma4WeightLoader
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ChatTemplate
import sk.ainet.apps.kllama.chat.ToolDefinition
import sk.ainet.transformers.gemma.iree.CompactCodec
import sk.ainet.transformers.gemma.iree.FunctionGemmaChatTemplate
import sk.ainet.transformers.gemma.iree.FunctionGemmaOfficialChatTemplate
import sk.ainet.transformers.gemma.iree.FunctionGemmaOfficialToolCallParserStrategy
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
 *
 * Two prompt/output dialects exist behind [Style]:
 *  - [Style.OCTOPUS_V2] — the physical-ai fine-tune's `<tool_N>(args)<end>` compact format
 *    (the original consumer of this class; templates baked into the checkpoint's vocab).
 *  - [Style.OFFICIAL] — Google's released `google/functiongemma-270m-it`:
 *    `<start_function_declaration>` tool blocks in the prompt,
 *    `<start_function_call>call:name{…}<end_function_call>` in the output. Tools are
 *    declared per call via [callWithTools]. Recommended sampling per the model card is
 *    temperature 1.0 / topK 64 / topP 0.95 — pass those to [callWithTools] for production;
 *    tests keep temperature 0 for determinism.
 */
public class FunctionGemma private constructor(
    private val ggufPath: String,
    private val runtime: InferenceRuntime<FP32>,
    private val tokenizer: GGUFTokenizer,
    private val style: Style,
    private val bos: Int,
    private val eot: Int,
    private val eos: Int,
) {
    public enum class Style { OCTOPUS_V2, OFFICIAL }

    public data class Turn(val text: String, val calls: List<ToolCall>)

    private val chatTemplate: ChatTemplate = when (style) {
        Style.OCTOPUS_V2 -> FunctionGemmaChatTemplate()
        Style.OFFICIAL -> FunctionGemmaOfficialChatTemplate()
    }
    private val officialParser = FunctionGemmaOfficialToolCallParserStrategy()

    /**
     * EAGER function-calling in the checkpoint's dialect. For [Style.OCTOPUS_V2] this is the
     * historical behavior (tool vocabulary baked into the checkpoint, no declarations needed).
     * For [Style.OFFICIAL] prefer [callWithTools]; calling this without tools renders a
     * plain chat turn.
     */
    public fun call(userText: String, maxTokens: Int = 24): Turn =
        callWithTools(userText, tools = emptyList(), maxTokens = maxTokens)

    /**
     * EAGER function-calling with in-prompt tool declarations ([Style.OFFICIAL]) or the
     * baked-in vocabulary ([Style.OCTOPUS_V2], where [tools] is ignored by the template).
     *
     * @param temperature 0 = greedy (deterministic — the test default). The official model
     *   card recommends `temperature = 1.0f, topK = 64, topP = 0.95f`.
     */
    public fun callWithTools(
        userText: String,
        tools: List<ToolDefinition>,
        maxTokens: Int = 64,
        temperature: Float = 0f,
        topK: Int = 0,
        topP: Float = 1f,
        random: Random = Random.Default,
    ): Turn {
        val prompt = chatTemplate.apply(listOf(ChatMessage(ChatRole.USER, userText)), tools = tools)
        val ptoks = tokenizer.encode(prompt)
        val full = if (ptoks.isEmpty() || ptoks[0] != bos) intArrayOf(bos) + ptoks else ptoks
        // Each call is a fresh single turn — without this, the KV cache still
        // holds the previous call's turn and the new prompt is prefilled on top
        // of it (observed as answers bleeding content across calls).
        runtime.reset()
        val result = runtime.generateUntilStop(
            prompt = full,
            maxTokens = maxTokens,
            eosTokenIds = setOf(eot, eos),
            temperature = temperature,
            topK = topK,
            topP = topP,
            random = random,
        )
        val text = tokenizer.decode(result.tokens.toIntArray())
        val calls = when (style) {
            Style.OCTOPUS_V2 -> CompactCodec.parse(text)
            Style.OFFICIAL -> officialParser.parseCompact(text)
        }
        return Turn(text, calls)
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
         *
         * @param style prompt/output dialect — [Style.OCTOPUS_V2] (default, the physical-ai
         *   fine-tune) or [Style.OFFICIAL] (google/functiongemma-270m-it).
         */
        public fun fromGguf(
            gguf: String,
            partialRotary: Float = 1.0f,
            style: Style = Style.OCTOPUS_V2,
        ): FunctionGemma = runBlocking {
            KgemmaKernels.ensureInstalled()
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
                style = style,
                bos = tok.bosTokenId,
                eot = tok.tokenId("<end_of_turn>") ?: tok.encode("<end_of_turn>").single(),
                eos = tok.eosTokenId,
            )
        }
    }
}
