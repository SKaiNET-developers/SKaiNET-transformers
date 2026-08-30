@file:JvmName("KLlamaJava")

package sk.ainet.apps.kllama.java

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.kllama.*
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.backend.api.kernel.KernelPacks
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.FfmRowMajorKernelPack
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.ExperimentalMemoryApi
import sk.ainet.lang.tensor.data.MemorySegmentTensorDataFactory
import sk.ainet.lang.types.FP32
import sk.ainet.models.llama.DecoderGgufWeightLoader
import sk.ainet.models.llama.DecoderSafeTensorsLoader
import sk.ainet.models.llama.LlamaConfigParser
import sk.ainet.models.llama.LlamaNetworkLoader
import java.nio.file.Path
import java.util.concurrent.atomic.AtomicBoolean
import kotlin.io.path.exists
import kotlin.io.path.readText

/**
 * Java-friendly facade for loading and running LLaMA models.
 *
 * Handles all the internal orchestration (Arena, MemorySegment, context,
 * loader, runtime, tokenizer) behind a simple API.
 *
 * Both `loadGGUF` and `loadSafeTensors` build a `llamaNetwork()` DSL
 * module + `OptimizedLLMRuntime` DIRECT mode — same path the kllama CLI
 * uses (PR #122). Numerical equivalence with the legacy `LlamaRuntime`
 * is pinned by `QwenDslLegacyParityTest` (#120).
 *
 * Example usage from Java:
 * ```java
 * try (KLlamaSession session = KLlamaJava.loadGGUF(Path.of("model.gguf"))) {
 *     String response = session.generate("What is the meaning of life?",
 *         GenerationConfig.builder().maxTokens(128).temperature(0.7f).build());
 *     System.out.println(response);
 * }
 * ```
 */
public object KLlamaJava {

    private val LLAMA_FAMILY: Set<String> = setOf("llama", "mistral")

    private val kernelsInstalled = AtomicBoolean(false)

    /**
     * Installs the 0.51 view-keyed kernel tiers exactly once per process (#338 arc): the
     * reference + best-provider FP32/prepacked kernels into `KernelDispatch`, and the FFM
     * row-major pack that serves canonical packed weights — mapped or un-prepacked heap —
     * zero-copy. [DecoderGgufWeightLoader]'s default `WeightForm` is MAPPED, and without this
     * install the mapped/keep-packed weights this facade produces fall to the decoding
     * reference kernel: correct, but dramatically slower per matmul (mirrors skainet-cli's
     * `Main.kt` install, which this facade previously lacked).
     */
    @OptIn(ExperimentalMemoryApi::class)
    private fun ensureKernelPacksInstalled() {
        if (!kernelsInstalled.compareAndSet(false, true)) return
        KernelPacks.install()
        FfmRowMajorKernelPack.install()
    }

    /**
     * Load a GGUF model and return a ready-to-use session.
     *
     * Accepts Llama / Mistral architectures. Qwen-family GGUFs are not
     * accepted here — use the kllama CLI (`kllama-cli`) which has Qwen
     * dispatch, or extend this facade.
     *
     * @param modelPath Path to the .gguf model file.
     * @param systemPrompt Optional system prompt to prepend to all user inputs.
     * @return A KLlamaSession that implements AutoCloseable.
     */
    @JvmStatic
    @JvmOverloads
    public fun loadGGUF(modelPath: Path, systemPrompt: String? = null): KLlamaSession {
        ensureKernelPacksInstalled()
        val memSegFactory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

        val loader = DecoderGgufWeightLoader(
            randomAccessProvider = { JvmRandomAccessSource.open(modelPath.toString()) },
            acceptedArchitectures = LLAMA_FAMILY,
        )
        // The engine loader keeps quantized tensors packed for SIMD dispatch.
        val weights = runBlocking { loader.loadToMapStreaming<FP32, Float>(ctx) }

        val model = LlamaNetworkLoader.fromWeights(weights)
        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
            bos = weights.metadata.bosTokenId,
        )

        // Embedded GGUF tokenizer (auto-dispatches Qwen / GPT-2 BPE → upstream)
        val tokenizer = JvmRandomAccessSource.open(modelPath.toString()).use { source ->
            TokenizerFactory.fromGgufSource(source)
        }

        return KLlamaSession(
            runtime = runtime,
            tokenizer = tokenizer,
            eosTokenId = tokenizer.eosTokenId,
            systemPrompt = systemPrompt,
            closeAction = Runnable {
                memSegFactory.close()
            }
        )
    }

    /**
     * Load a SafeTensors model from a HuggingFace directory.
     *
     * The directory must contain `model.safetensors`, `config.json`,
     * and `tokenizer.json`.
     *
     * @param modelDir Path to the HuggingFace model directory.
     * @param systemPrompt Optional system prompt to prepend to all user inputs.
     * @return A KLlamaSession that implements AutoCloseable.
     */
    @JvmStatic
    @JvmOverloads
    public fun loadSafeTensors(modelDir: Path, systemPrompt: String? = null): KLlamaSession {
        ensureKernelPacksInstalled()
        val memSegFactory = MemorySegmentTensorDataFactory()
        val ctx = DirectCpuExecutionContext(tensorDataFactory = memSegFactory)

        val configFile = modelDir.resolve("config.json")
        require(configFile.exists()) { "config.json not found in $modelDir" }
        val configJson = configFile.readText()
        val metadata = LlamaConfigParser.parse(configJson)
        val tiedEmbeddings = LlamaConfigParser.isTiedEmbeddings(configJson)

        val safetensorsPath = modelDir.resolve("model.safetensors")
        require(safetensorsPath.exists()) { "model.safetensors not found in $modelDir" }

        val safeLoader = DecoderSafeTensorsLoader<FP32>(ctx, FP32::class, metadata, tiedEmbeddings)
        val weights = safeLoader.loadToMap {
            JvmRandomAccessSource.open(safetensorsPath.toString())
        }

        val model = LlamaNetworkLoader.fromWeights(weights)
        val runtime = OptimizedLLMRuntime(
            model = model,
            ctx = ctx,
            mode = OptimizedLLMMode.DIRECT,
            dtype = FP32::class,
            bos = weights.metadata.bosTokenId,
        )

        // Tokenizer from tokenizer.json
        val tokenizerPath = modelDir.resolve("tokenizer.json")
        require(tokenizerPath.exists()) { "tokenizer.json not found in $modelDir" }
        val tokenizer = GGUFTokenizer.fromTokenizerJson(tokenizerPath.readText())

        return KLlamaSession(
            runtime = runtime,
            tokenizer = tokenizer,
            eosTokenId = tokenizer.eosTokenId,
            systemPrompt = systemPrompt,
            closeAction = Runnable {
                memSegFactory.close()
            }
        )
    }
}
