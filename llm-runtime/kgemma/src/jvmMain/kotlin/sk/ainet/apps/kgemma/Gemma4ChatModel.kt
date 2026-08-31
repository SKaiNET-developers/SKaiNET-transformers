package sk.ainet.apps.kgemma

import java.lang.foreign.Arena
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import kotlin.io.path.exists
import kotlin.io.path.isDirectory
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.chat.Gemma4ChatTemplate
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.llm.api.ChatOptions
import sk.ainet.llm.api.ChatRequest
import sk.ainet.llm.api.ChatResponse
import sk.ainet.llm.api.ChatResponseChunk
import sk.ainet.llm.api.StreamingChatModel
import sk.ainet.llm.providers.SkaiNetChatModel
import sk.ainet.models.gemma.Gemma4SafeTensorsMappedPle
import sk.ainet.models.gemma.Gemma4SafeTensorsWeightLoader
import sk.ainet.models.gemma.Gemma4Weights

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
     * Build a [SkaiNetChatModel] for a Gemma 4 GGUF checkpoint.
     *
     * Composes the same four pieces as [fromSafeTensors], sourced from GGUF instead:
     *  1. [Gemma4Ingestion.loadDslRuntimeStreaming] — streams the GGUF via a random-access
     *     source through the engine loader; quantized (K-quant) tensors stay packed by default
     *     (see [Gemma4LoadConfig.weightForm]), same mechanism [sk.ainet.apps.kllama.java.KLlamaJava]
     *     uses for Llama.
     *  2. [GGUFTokenizer.fromRandomAccessSource] — reads the tokenizer baked into the GGUF itself
     *     (no separate `tokenizer.json` needed). Called directly rather than through
     *     `sk.ainet.apps.llm.tokenizer.TokenizerFactory.fromGgufSource`, which delegates to the
     *     engine's `tokenizer.ggml.model`-gated factory — that allowlist doesn't have a "gemma4"
     *     case yet, even though `GGUFTokenizer` itself already handles Gemma 4's GGUF shape.
     *  3. [Gemma4ChatTemplate] — same official turn format as the SafeTensors path.
     *  4. [SkaiNetChatModel] — the neutral `ChatModel` / `StreamingChatModel` adapter.
     *
     * **Known limitation vs [fromSafeTensors]**: a GGUF only carries a single
     * `tokenizer.ggml.eos_token_id` (unlike `generation_config.json`'s `eos_token_id` array, which
     * on real Gemma 4 checkpoints lists multiple stop ids, e.g. `[1, 106]`). Passing only the
     * tokenizer's single id risks the same "keeps emitting `<turn|>` past the natural boundary"
     * failure mode [fromSafeTensors]'s doc comment warns about, if the GGUF's own EOS id isn't the
     * one that matters for a given prompt shape. Verify empirically per checkpoint; a future
     * revision could scan the GGUF's tokenizer vocab for `<turn|>`/`<end_of_turn>`-shaped entries.
     *
     * @param path Path to a `.gguf` file.
     */
    public fun fromGguf(
        path: String,
        ctx: ExecutionContext = DirectCpuExecutionContext(
            tensorDataFactory = MemorySegmentTensorDataFactory()
        ),
        options: ChatOptions = ChatOptions.DEFAULTS,
        modelId: String? = "gemma4",
        enableThinking: Boolean = false,
    ): StreamingChatModel {
        require(Paths.get(path).exists()) { "GGUF not found: $path" }

        val ingestion = Gemma4Ingestion<FP32>(
            ctx = ctx,
            dtype = FP32::class,
            config = Gemma4LoadConfig(),
        )
        val runtime = runBlocking {
            ingestion.loadDslRuntimeStreaming { JvmRandomAccessSource.open(path) }
        }

        // sk.ainet.apps.llm.tokenizer.TokenizerFactory.fromGgufSource delegates to the engine's
        // sk.ainet.io.tokenizer.TokenizerFactory, which gates on tokenizer.ggml.model and doesn't
        // recognize "gemma4" yet (throws UnsupportedTokenizerException) — even though GGUFTokenizer
        // itself already has explicit Gemma-4-aware handling (see its addSpacePrefix doc). Call it
        // directly to bypass the stale model-string allowlist.
        val tokenizer = JvmRandomAccessSource.open(path).use { source ->
            GGUFTokenizer.fromRandomAccessSource(source)
        }

        val chatTemplate = Gemma4ChatTemplate(enableThinking = enableThinking)
        // GGUF carries only a single eos id; resolve the full Gemma 4 stop set
        // ({<eos>, <turn|>, chat-end} per generation_config.json) from the vocab.
        val eosTokenIds = Gemma4StopTokens.resolve(tokenizer)

        val delegate = SkaiNetChatModel(
            runtime = runtime,
            tokenizer = tokenizer,
            chatTemplate = chatTemplate,
            defaultOptions = options,
            eosTokenIds = eosTokenIds,
            modelId = modelId,
        )
        return delegate
    }

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
    ): StreamingChatModel {
        val (resolvedIndex, modelDir) = resolveIndexAndDir(indexPath)
        require(resolvedIndex.exists()) {
            "SafeTensors index not found: $resolvedIndex"
        }

        val ingestion = Gemma4Ingestion<FP32>(
            ctx = ctx,
            dtype = FP32::class,
            config = Gemma4LoadConfig(),
        )

        // Two-step load so we can inject the mmap'd PLE token-embedding
        // table between weight loading and DSL build. The eager loader
        // skips that tensor when it exceeds the 2 GB ByteArray limit
        // (4.7 GB BF16 on Gemma 4 E2B), and `injectIfMissing` re-attaches
        // it via FileChannel.map. The shared arena owns the mmap'd region's
        // lifetime — it intentionally outlives the model on a process basis;
        // a follow-up will plumb explicit close via a wrapping ChatModel.
        @Suppress("UNCHECKED_CAST")
        val rawWeights = runBlocking {
            Gemma4SafeTensorsWeightLoader(resolvedIndex.toString())
                .loadToMap(ctx, FP32::class)
        } as Gemma4Weights<FP32, Float>
        val pleArena = Arena.ofShared()
        val weights = Gemma4SafeTensorsMappedPle.injectIfMissing(
            weights = rawWeights,
            indexPath = resolvedIndex.toString(),
            ctx = ctx,
            arena = pleArena,
        )
        val runtime = ingestion.buildDslRuntime(weights)

        val tokenizerFile = modelDir.resolve("tokenizer.json")
        require(tokenizerFile.exists()) {
            "tokenizer.json not found alongside checkpoint at $modelDir"
        }
        val tokenizer = GGUFTokenizer.fromTokenizerJson(Files.readString(tokenizerFile))

        val chatTemplate = Gemma4ChatTemplate(enableThinking = enableThinking)

        // Gemma 4 ships multiple stop ids (`<eos>=1`, `<turn|>=106`, and the
        // chat-end variant `50` per the released `generation_config.json`).
        // Without all of them, greedy decoding will keep emitting `<turn|>`
        // tokens past the natural turn boundary until `maxTokens` runs out.
        val eosTokenIds = readEosTokenIds(modelDir)
            ?: setOf(tokenizer.eosTokenId)

        // Gemma4ChatTemplate emits `<bos>` itself as the very first character of
        // the rendered prompt, and the tokenizer maps that string to the BOS
        // token id. SkaiNetChatModel.runGenerationLoop only prepends BOS if it
        // is not already first, so the natural construction does not double up.
        val delegate = SkaiNetChatModel(
            runtime = runtime,
            tokenizer = tokenizer,
            chatTemplate = chatTemplate,
            defaultOptions = options,
            eosTokenIds = eosTokenIds,
            modelId = modelId,
        )
        return ArenaScopedStreamingChatModel(delegate, pleArena)
    }

    /**
     * Wraps a [SkaiNetChatModel] together with a JVM [Arena] that owns the
     * memory-mapped PLE token-embedding region. Closing this propagates to
     * both — `delegate.close()` resets the runtime, then `arena.close()`
     * unmaps the file region, releasing the underlying file handle.
     *
     * Without this wrapper, the only way the mmap'd region got cleaned up
     * was at JVM exit (the arena was unreachable but bound to a long-lived
     * native segment). Fine for tests; not fine for any process that
     * builds and discards multiple chat models.
     */
    private class ArenaScopedStreamingChatModel(
        private val delegate: StreamingChatModel,
        private val arena: Arena,
    ) : StreamingChatModel {
        override val defaultOptions: ChatOptions get() = delegate.defaultOptions
        override fun call(request: ChatRequest): ChatResponse = delegate.call(request)
        override fun stream(request: ChatRequest): Flow<ChatResponseChunk> = delegate.stream(request)
        override fun close() {
            try {
                delegate.close()
            } finally {
                runCatching { arena.close() }
            }
        }
    }

    /**
     * Read `eos_token_id` from `generation_config.json` next to the
     * checkpoint. Accepts either a scalar (`"eos_token_id": 1`) or a list
     * (`"eos_token_id": [1, 106, 50]`); returns `null` if the file is
     * missing or unparseable so the caller can fall back to the
     * tokenizer's default.
     */
    private fun readEosTokenIds(modelDir: Path): Set<Int>? {
        val genConfig = modelDir.resolve("generation_config.json")
        if (!genConfig.exists()) return null
        return runCatching {
            val text = Files.readString(genConfig)
            val arrayMatch = Regex("""\"eos_token_id\"\s*:\s*\[\s*([^\]]*)\]""").find(text)
            if (arrayMatch != null) {
                arrayMatch.groupValues[1]
                    .split(',')
                    .mapNotNull { it.trim().toIntOrNull() }
                    .toSet()
                    .takeIf { it.isNotEmpty() }
            } else {
                Regex("""\"eos_token_id\"\s*:\s*(\d+)""").find(text)
                    ?.groupValues?.get(1)?.toIntOrNull()
                    ?.let { setOf(it) }
            }
        }.getOrNull()
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
