package sk.ainet.models.gemma

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.apps.kllama.chat.AgentListener
import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.apps.kllama.chat.ChatSession
import sk.ainet.apps.kllama.chat.Gemma4ChatTemplate
import sk.ainet.apps.kllama.chat.ModelMetadata
import sk.ainet.apps.kllama.chat.Tool
import sk.ainet.apps.kllama.chat.ToolCall
import sk.ainet.apps.kllama.chat.ToolDefinition
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.Tokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.types.FP32

/**
 * End-to-end sibling of `ChatSessionAgentIntegrationTest` that runs the agent
 * loop over a **real** DSL-based Gemma 4 runtime (PLE + layer_output_scale +
 * sandwich norms + QK-Norm + softcapping, all active) instead of a plain
 * mock.
 *
 * The real Gemma 4 E2B checkpoint would be the ideal source of truth, but
 * we can't load it here: it's ~3 GB on disk (and needs Q4_K lazy-transpose
 * support that the currently composite-built SKaiNET 0.20-SNAPSHOT has
 * regressed). Synthetic weights give us "the full DSL forward pass runs
 * cleanly" but not "the model generates a parseable tool call on its own".
 *
 * Approach: run the real DSL forward for every decode step (exercising PLE,
 * softcap, etc. — proves nothing crashes mid-decode), then ignore the
 * resulting logits and return a one-hot logits tensor whose argmax is the
 * next byte in a scripted Gemma-4-formatted tool call. The agent loop sees
 * a coherent `<|tool_call>{…}<tool_call|>` emission and should parse it
 * via [Gemma4ChatTemplate] and dispatch to our calculator tool.
 *
 * What this validates that the mock-runtime test does NOT:
 * - `GemmaNetworkLoader.fromWeights` + `OptimizedLLMRuntime.forward` on
 *   Gemma-4-shaped weights (PLE active) produce finite, non-NaN logits at
 *   every decode step during an agent round.
 * - `ChatSession.runSingleTurn` does not drop or corrupt the runtime's
 *   forward calls when handed a real (non-mock) `InferenceRuntime`.
 * - The `CalculatorTool` + `Gemma4ChatTemplate` round-trip works end-to-end
 *   over a real runtime's token stream.
 *
 * What it deliberately does NOT validate:
 * - That a trained Gemma 4 E2B spontaneously emits a `<|tool_call>` block
 *   on a prompt like "what is 3+4?". That's the manual E2E smoke test
 *   that runs outside CI — it needs a real checkpoint.
 */
class GemmaDslToolCallIntegrationTest {

    private val dim = 8
    private val ffDim = 16
    private val vocabSize = 1024
    private val perLayerDim = 4
    private val numLayers = 2
    private val nHeads = 2
    private val kvHeads = 2
    private val headDim = dim / nHeads
    // Needs to comfortably hold the Gemma 4 chat template (system prompt +
    // tool defs + `<|turn>` boilerplate) plus the scripted tool-call reply.
    // Measured at ~700 tokens; 2048 gives plenty of headroom.
    private val seqLen = 2048

    private val ctx = DirectCpuExecutionContext()

    private fun randn(shape: Shape, seed: Int): Tensor<FP32, Float> {
        val rng = kotlin.random.Random(seed)
        val values = FloatArray(shape.volume) { (rng.nextFloat() - 0.5f) * 0.1f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    private fun ones(shape: Shape): Tensor<FP32, Float> {
        val values = FloatArray(shape.volume) { 1.0f }
        return ctx.fromFloatArray(shape, FP32::class, values)
    }

    /**
     * Synthetic Gemma 4 metadata big enough that every optional Phase 5f
     * feature wires up. `perLayerEmbeddingLength` triggers PLE through
     * `GemmaNetworkLoader.fromWeights`.
     */
    private val metadata = Gemma4ModelMetadata(
        architecture = "gemma4",
        embeddingLength = dim,
        contextLength = seqLen,
        blockCount = numLayers,
        headCount = nHeads,
        kvHeadCount = kvHeads,
        intermediateSize = ffDim,
        headDim = headDim,
        globalHeadDim = headDim,
        vocabSize = vocabSize,
        slidingWindow = seqLen,
        kvSharedLayers = 0,
        layerTypes = List(numLayers) { "full_attention" },
        ropeParametersFull = Gemma4RopeConfig(base = 10000f),
        ropeParametersSliding = Gemma4RopeConfig(base = 10000f),
        maxPositionEmbeddings = seqLen,
        perLayerEmbeddingLength = perLayerDim
    )

    private fun buildWeights(): Gemma4Weights<FP32, Float> {
        val tensors = linkedMapOf<String, Tensor<FP32, Float>>()
        tensors[Gemma4TensorNames.TOKEN_EMBEDDINGS] = randn(Shape(vocabSize, dim), seed = 10)
        tensors[Gemma4TensorNames.OUTPUT_NORM] = ones(Shape(dim))
        tensors[Gemma4TensorNames.OUTPUT_WEIGHT] = randn(Shape(vocabSize, dim), seed = 11)
        tensors["per_layer_token_embd.weight"] =
            randn(Shape(vocabSize, numLayers * perLayerDim), seed = 30)
        tensors["per_layer_model_proj.weight"] =
            randn(Shape(numLayers * perLayerDim, dim), seed = 31)
        tensors["per_layer_proj_norm.weight"] = ones(Shape(perLayerDim))
        for (layer in 0 until numLayers) {
            tensors[Gemma4TensorNames.inputLayernorm(layer)] = ones(Shape(dim))
            tensors[Gemma4TensorNames.attnQ(layer)] = randn(Shape(dim, dim), seed = 100 + layer * 10)
            tensors[Gemma4TensorNames.attnK(layer)] = randn(Shape(dim, dim), seed = 101 + layer * 10)
            tensors[Gemma4TensorNames.attnV(layer)] = randn(Shape(dim, dim), seed = 102 + layer * 10)
            tensors[Gemma4TensorNames.attnOut(layer)] = randn(Shape(dim, dim), seed = 103 + layer * 10)
            tensors[Gemma4TensorNames.postAttentionLayernorm(layer)] = ones(Shape(dim))
            tensors[Gemma4TensorNames.ffnGate(layer)] = randn(Shape(ffDim, dim), seed = 104 + layer * 10)
            tensors[Gemma4TensorNames.ffnDown(layer)] = randn(Shape(dim, ffDim), seed = 105 + layer * 10)
            tensors[Gemma4TensorNames.ffnUp(layer)] = randn(Shape(ffDim, dim), seed = 106 + layer * 10)
            tensors["blk.$layer.inp_gate.weight"] = randn(Shape(perLayerDim, dim), seed = 200 + layer * 10)
            tensors["blk.$layer.proj.weight"] = randn(Shape(dim, perLayerDim), seed = 201 + layer * 10)
            tensors["blk.$layer.post_norm.weight"] = ones(Shape(dim))
        }
        return Gemma4Weights(metadata = metadata, tensors = tensors)
    }

    private fun oneHotLogits(argmax: Int): Tensor<FP32, Float> {
        val buf = FloatArray(vocabSize)
        buf[argmax] = 100f
        val data = DenseFloatArrayTensorData<FP32>(Shape(vocabSize), buf)
        return VoidOpsTensor(data, FP32::class)
    }

    /** Byte-level tokenizer — each char is one token. */
    private inner class ByteTokenizer : Tokenizer {
        override val eosTokenId: Int = 0
        override val bosTokenId: Int = 1
        override val vocabSize: Int = this@GemmaDslToolCallIntegrationTest.vocabSize
        override fun encode(text: String): IntArray =
            text.map { it.code + 2 }.toIntArray()
        override fun decode(tokens: IntArray): String =
            tokens.joinToString("") { decode(it) }
        override fun decode(token: Int): String = when (token) {
            eosTokenId, bosTokenId -> ""
            else -> (token - 2).toChar().toString()
        }
    }

    /**
     * Decorator that runs every [forward] through the underlying DSL
     * runtime (exercising the full Gemma 4 forward pass including PLE,
     * softcap, etc.) then discards the result and returns a one-hot logits
     * tensor pointing at the next byte in [scriptedOutput]. The underlying
     * runtime's forward is validated for finiteness — if the DSL crashes
     * mid-decode on synthetic Gemma 4 weights this test fails.
     *
     * [promptLen] is the number of tokens in the agent-loop prompt. The
     * decorator uses the same cursor semantics as the mock-runtime test:
     * prompt tokens drive filler logits; the `promptLen`-th forward
     * returns the first scripted argmax.
     */
    private inner class ScriptedDslRuntime(
        private val inner: OptimizedLLMRuntime<FP32>,
        private val promptLen: Int,
        private val scriptedOutput: IntArray
    ) : InferenceRuntime<FP32> {
        private var cursor = 0
        override fun reset() {
            cursor = 0
            inner.reset()
        }
        override fun forward(tokenId: Int): Tensor<FP32, Float> {
            val real = inner.forward(tokenId)
            val buf = real.data.copyToFloatArray()
            for ((i, v) in buf.withIndex()) {
                assertTrue(
                    v.isFinite(),
                    "DSL runtime produced non-finite logit[$i]=$v at cursor=$cursor " +
                        "(PLE/softcap path is NaN-ing out on synthetic weights)"
                )
            }
            val argmax = when {
                cursor < promptLen - 1 -> 0  // EOS-equivalent during prompt feed
                cursor - promptLen + 1 < scriptedOutput.size ->
                    scriptedOutput[cursor - promptLen + 1]
                else -> 0
            }
            cursor++
            return oneHotLogits(argmax)
        }
    }

    private class FixedCalculatorTool(
        private val expectedExpression: String,
        private val result: String
    ) : Tool {
        var invoked = false
            private set
        var receivedExpression: String? = null
            private set
        override val definition = ToolDefinition(
            name = "calculator",
            description = "Compute a simple arithmetic expression",
            parameters = JsonObject(emptyMap())
        )
        override fun execute(arguments: JsonObject): String {
            invoked = true
            receivedExpression = arguments["expression"]?.jsonPrimitive?.content
            return result
        }
    }

    @Test
    fun `ChatSession over real DSL Gemma 4 runtime dispatches scripted tool call to calculator`() {
        val weights = buildWeights()
        val dslModel = GemmaNetworkLoader.fromWeights(ctx, weights)
        val dslRuntime = OptimizedLLMRuntime(
            model = dslModel,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class
        )

        val tokenizer = ByteTokenizer()
        val tool = FixedCalculatorTool(expectedExpression = "3+4", result = "7")

        val template = Gemma4ChatTemplate()
        val round1Messages = listOf(
            ChatMessage(ChatRole.SYSTEM, "You are a helpful assistant with access to tools."),
            ChatMessage(ChatRole.USER, "what is 3 + 4?")
        )
        val round1Prompt = template.apply(
            round1Messages,
            listOf(tool.definition),
            addGenerationPrompt = true
        )
        val promptLen = tokenizer.encode(round1Prompt).size

        val toolCallText =
            "I'll compute that.\n<|tool_call>{\"name\":\"calculator\",\"args\":{\"expression\":\"3+4\"}}<tool_call|>"
        val scriptedOutput = tokenizer.encode(toolCallText)

        val scripted = ScriptedDslRuntime(dslRuntime, promptLen, scriptedOutput)

        val metadata = ModelMetadata(architecture = "gemma4", family = "gemma4")
        val session = ChatSession(scripted, tokenizer, metadata)

        assertEquals("gemma4", session.providerFamily)
        assertTrue(session.chatTemplate is Gemma4ChatTemplate)

        val toolCallsSeen = mutableListOf<ToolCall>()
        val toolResultsSeen = mutableListOf<Pair<String, String>>()
        val listener = object : AgentListener {
            override fun onToolCalls(calls: List<ToolCall>) { toolCallsSeen += calls }
            override fun onToolResult(call: ToolCall, result: String) {
                toolResultsSeen += call.name to result
            }
        }

        session.runSingleTurn(
            prompt = "what is 3 + 4?",
            tools = listOf(tool),
            maxTokens = scriptedOutput.size + 4,
            temperature = 0.0f,
            listener = listener
        )

        assertEquals(1, toolCallsSeen.size, "expected exactly 1 parsed tool call, got ${toolCallsSeen.size}")
        assertEquals("calculator", toolCallsSeen[0].name)
        assertEquals("3+4", toolCallsSeen[0].arguments["expression"]?.jsonPrimitive?.content)
        assertTrue(tool.invoked, "calculator tool was never executed")
        assertEquals("3+4", tool.receivedExpression)
        assertEquals("calculator" to "7", toolResultsSeen[0])
    }
}
