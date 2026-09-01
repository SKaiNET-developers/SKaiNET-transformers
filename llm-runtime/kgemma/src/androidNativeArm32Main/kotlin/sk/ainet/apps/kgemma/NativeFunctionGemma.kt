package sk.ainet.apps.kgemma

import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import sk.ainet.apps.llm.InferenceRuntime
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.generate
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.gguf.createRandomAccessSource
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP32
import sk.ainet.models.gemma.GemmaWeightLoader
import sk.ainet.models.gemma.GemmaNetworkLoader
import sk.ainet.apps.kllama.chat.ChatMessage
import sk.ainet.apps.kllama.chat.ChatRole
import sk.ainet.transformers.gemma.iree.CompactCodec
import sk.ainet.transformers.gemma.iree.FunctionGemmaChatTemplate
import sk.ainet.transformers.gemma.iree.ToolCall
import kotlin.random.Random

/**
 * androidNativeArm32's [FunctionGemma]: identical contract (`fromGguf(path).call(text)`),
 * same `gemmaNetwork()` DSL + `DirectCpuExecutionContext` as the JVM facade — only the
 * file-access layer differs, because `JvmRandomAccessSource`/`java.io.File` don't exist
 * here. Tokenizer loading mirrors [sk.ainet.transformers.gemma.iree.GemmaKvDecoder]'s own
 * native GGUF read (`SystemFileSystem`/`kotlinx-io`); weight loading uses
 * `sk.ainet.io.gguf.createRandomAccessSource` — POSIX `pread` via
 * `Posix32PreadRandomAccessSource` on this 32-bit target (see that class's own doc for why
 * it isn't the same implementation the 64-bit natives share).
 *
 * No compile-leg export here (unlike the JVM facade's `exportCompiled`) — that's host
 * tooling (`:llm-inference:functiongemma`), out of scope for an on-device eager CLI.
 */
public class NativeFunctionGemma private constructor(
    private val runtime: InferenceRuntime<FP32>,
    private val tokenizer: GGUFTokenizer,
    private val bos: Int,
    private val eot: Int,
    private val eos: Int,
) {
    public data class Turn(val text: String, val calls: List<ToolCall>)

    private val chatTemplate = FunctionGemmaChatTemplate()

    public fun call(userText: String, maxTokens: Int = 24): Turn {
        val prompt = chatTemplate.apply(listOf(ChatMessage(ChatRole.USER, userText)))
        val ptoks = tokenizer.encode(prompt)
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

    public companion object {
        public fun fromGguf(gguf: String, partialRotary: Float = 1.0f): NativeFunctionGemma = runBlocking {
            val tok = GGUFTokenizer.fromSource(SystemFileSystem.source(Path(gguf)).buffered())
            val ctx = DirectCpuExecutionContext.create()
            val weights = GemmaWeightLoader(
                randomAccessProvider = {
                    createRandomAccessSource(gguf)
                        ?: error("could not open $gguf for random access (pread failed?)")
                },
                // Mirrors the JVM facade: the engine replaced QuantPolicy with WeightForm, and
                // narrow-float handling moved to dtypePolicy. Keeping BF16 packed halves the bytes
                // a decode step reads on a checkpoint stored that way.
                dtypePolicy = DTypePolicy.Prefer(BF16),
            ).loadToMapStreaming<FP32, Float>(ctx, FP32::class)
            val patched = weights.copy(
                metadata = weights.metadata.copy(
                    ropeParametersFull = weights.metadata.ropeParametersFull.copy(partialRotaryFactor = partialRotary),
                ),
            )
            val model = GemmaNetworkLoader.fromWeights(ctx, patched, FP32::class)
            val runtime = OptimizedLLMRuntime(model, ctx, OptimizedLLMMode.DIRECT, FP32::class, random = Random.Default)
            NativeFunctionGemma(
                runtime = runtime,
                tokenizer = tok,
                bos = tok.bosTokenId,
                eot = tok.encode("<end_of_turn>").single(),
                eos = tok.eosTokenId,
            )
        }
    }
}
