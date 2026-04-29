package sk.ainet.apps.kgemma

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.chat.Gemma4ChatTemplate
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.model.QuantPolicy
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.llm.api.ChatOptions
import sk.ainet.llm.providers.SkaiNetChatModel

/**
 * One-call factory that wires a Gemma 4 SafeTensors checkpoint into the
 * Spring-AI-style [SkaiNetChatModel] surface for text-only chat.
 *
 * Composes the four already-existing pieces:
 *  1. [Gemma4Ingestion.loadDslRuntimeFromSafeTensors] — produces an
 *     `InferenceRuntime<FP32>` from the DSL `gemmaNetwork()` definition.
 *  2. [GGUFTokenizer.fromTokenizerJson] — loads HF `tokenizer.json` next to
 *     the index file (same loader the kgemma CLI uses).
 *  3. [Gemma4ChatTemplate] — emits the official `<bos><|turn>…<turn|>` format.
 *  4. [SkaiNetChatModel] — the neutral `ChatModel` / `StreamingChatModel` adapter.
 *
 * Caller experience:
 *
 * ```
 * val model = Gemma4ChatModel.fromSafeTensors("/path/to/model.safetensors.index.json")
 * println(model.call(ChatRequest("Hello!")).result.text)
 * ```
 *
 * **Scope (v1):**
 *  - FP32 SafeTensors only. Q4_K_M GGUF parity vs llama.cpp is a known open
 *    issue and is intentionally not exercised by this factory.
 *  - Text-only — no vision / audio.
 *  - Tool calling parses correctly on the template side, but real-checkpoint
 *    end-to-end tool calling is gated on the same parity bug.
 *
 * **Memory:** A full FP32 E2B checkpoint is ~20 GB resident — run with a heap
 * sized for it (`-Xmx24g` or larger). The supplied [ExecutionContext] backs
 * tensor data with a [MemorySegmentTensorDataFactory] by default; callers that
 * already own a context can pass it in.
 *
 * **Threading:** [SkaiNetChatModel] is documented as not thread-safe — KV-cache
 * state is mutated across forward passes. Use one instance per concurrent
 * request.
 */
public object Gemma4ChatModel {

    /**
     * Build a [SkaiNetChatModel] for a Gemma 4 SafeTensors checkpoint.
     *
     * @param indexPath Either the `model.safetensors.index.json` (sharded
     *   layout, recommended) OR the directory containing it / a single
     *   `model.safetensors`. The factory resolves the model directory and
     *   the actual index file from this argument.
     * @param ctx Execution context for tensor allocation. A
     *   [DirectCpuExecutionContext] backed by [MemorySegmentTensorDataFactory]
     *   is the default — same construction the kgemma CLI uses.
     * @param options Default [ChatOptions] applied when a [sk.ainet.llm.api.ChatRequest]
     *   does not override them.
     * @param modelId Optional identifier surfaced on every [sk.ainet.llm.api.ChatResponse].
     * @param enableThinking Forwarded to [Gemma4ChatTemplate] — opt-in `<|think|>`
     *   priming on the system turn. Off by default.
     */
    public fun fromSafeTensors(
        indexPath: String,
        ctx: ExecutionContext = DirectCpuExecutionContext(
            tensorDataFactory = MemorySegmentTensorDataFactory()
        ),
        options: ChatOptions = ChatOptions.DEFAULTS,
        modelId: String? = "gemma4",
        enableThinking: Boolean = false,
    ): SkaiNetChatModel<FP32> {
        val (resolvedIndex, modelDir) = resolveIndexAndDir(indexPath)
        require(resolvedIndex.exists()) {
            "SafeTensors index not found: $resolvedIndex"
        }

        val ingestion = Gemma4Ingestion<FP32>(
            ctx = ctx,
            dtype = FP32::class,
            config = Gemma4LoadConfig(
                quantPolicy = QuantPolicy.DEQUANTIZE_TO_FP32,
                allowQuantized = false,
            ),
        )
        val runtime = runBlocking {
            ingestion.loadDslRuntimeFromSafeTensors(resolvedIndex.toString())
        }

        val tokenizerFile = modelDir.resolve("tokenizer.json")
        require(tokenizerFile.exists()) {
            "tokenizer.json not found alongside checkpoint at $modelDir"
        }
        val tokenizer = GGUFTokenizer.fromTokenizerJson(Files.readString(tokenizerFile))

        val chatTemplate = Gemma4ChatTemplate(enableThinking = enableThinking)

        // Gemma4ChatTemplate emits `<bos>` itself as the very first character of
        // the rendered prompt, and the tokenizer maps that string to the BOS
        // token id. SkaiNetChatModel.runGenerationLoop only prepends BOS if it
        // is not already first, so the natural construction does not double up.
        return SkaiNetChatModel(
            runtime = runtime,
            tokenizer = tokenizer,
            chatTemplate = chatTemplate,
            defaultOptions = options,
            modelId = modelId,
        )
    }

    private fun resolveIndexAndDir(input: String): Pair<Path, Path> {
        val p = Paths.get(input)
        return when {
            p.isDirectory() -> {
                val idx = p.resolve("model.safetensors.index.json")
                val single = p.resolve("model.safetensors")
                val chosen = if (idx.exists()) idx else single
                chosen to p
            }
            p.fileName?.toString() == "model.safetensors.index.json" -> {
                p to (p.parent ?: Paths.get("."))
            }
            p.fileName?.toString() == "model.safetensors" -> {
                p to (p.parent ?: Paths.get("."))
            }
            else -> {
                // Caller passed a custom path; treat its parent as the model dir.
                p to (p.parent ?: Paths.get("."))
            }
        }
    }
}
