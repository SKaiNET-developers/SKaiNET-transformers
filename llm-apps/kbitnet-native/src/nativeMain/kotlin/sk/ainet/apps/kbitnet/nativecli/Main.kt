package sk.ainet.apps.kbitnet.nativecli

import kotlinx.coroutines.runBlocking
import sk.ainet.apps.llm.OptimizedLLMMode
import sk.ainet.apps.llm.OptimizedLLMRuntime
import sk.ainet.apps.llm.UnifiedModelLoader
import sk.ainet.apps.llm.tokenizer.TokenizerFactory
import sk.ainet.backend.api.kernel.KernelDispatch
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.exec.kernel.NativeKnTernaryF32Gemv
import sk.ainet.exec.kernel.NativeKnTernaryLmhead
import sk.ainet.exec.kernel.installNativeKernels
import sk.ainet.io.gguf.I2sGgufLayout
import sk.ainet.io.gguf.createRandomAccessSource
import sk.ainet.lang.types.FP32
import sk.ainet.models.bitnet.BitNetRuntimeWeights
import sk.ainet.models.bitnet.BitNetStage1Kernel
import sk.ainet.models.bitnet.BitNetWeightLoader
import sk.ainet.models.bitnet.bitnetPlanesHead
import sk.ainet.models.bitnet.generateTwoStage
import kotlin.time.TimeSource

/** usage: kbitnet <model.gguf> <steps> <prompt> — greedy, two-stage lm_head. */
fun main(args: Array<String>) = runBlocking {
    require(args.size >= 3) { "usage: kbitnet <model.gguf> <steps> <prompt>" }
    val path = args[0]; val steps = args[1].toInt(); val prompt = args.drop(2).joinToString(" ")
    fun source() = createRandomAccessSource(path) ?: error("cannot open $path")

    val t0 = TimeSource.Monotonic.markNow()
    installNativeKernels()
    println("ternary gemv: ${NativeKnTernaryF32Gemv.install { System_err(it) }}, lmhead: ${NativeKnTernaryLmhead.install { System_err(it) }}")
    KernelDispatch.ensureInstalled()

    val info = UnifiedModelLoader.peek { source() }
    println("Architecture: ${info.architecture}, ${info.embeddingLength}d, ${info.blockCount} layers, vocab=${info.vocabSize}")
    val ctx = DirectCpuExecutionContext()
    val w = BitNetWeightLoader.loadRuntimeWeights(ctx, sourceProvider = { source() }, i2sLayout = I2sGgufLayout.GROUP_128)
    val model = BitNetRuntimeWeights(w.metadata, w.tensors).toModule()
    val head = bitnetPlanesHead(model) ?: error("lm_head did not load as BITNET_PLANES")
    val runtime = OptimizedLLMRuntime(model = model, ctx = ctx, mode = OptimizedLLMMode.DIRECT, dtype = FP32::class, bos = w.metadata.bosTokenId)
    val tokenizer = TokenizerFactory.fromGgufFields(info.fields)
    println("load: ${t0.elapsedNow()}")

    val tokens = intArrayOf(tokenizer.bosTokenId) + tokenizer.encode(prompt)
    print(prompt)
    val t1 = TimeSource.Monotonic.markNow()
    var first: kotlin.time.Duration? = null
    runtime.generateTwoStage(
        prompt = tokens, steps = steps, temperature = 0f, head = head,
        native = BitNetStage1Kernel(NativeKnTernaryLmhead::lmheadStage1),
    ) { id -> if (first == null) first = t1.elapsedNow(); print(tokenizer.decode(id)) }
    val total = t1.elapsedNow()
    val f = first ?: total
    println("\n---")
    println("prefill+first token: $f, decode: ${(steps - 1) / (total - f).inWholeMilliseconds.toDouble() * 1000} tok/s, overall: ${steps / total.inWholeMilliseconds.toDouble() * 1000} tok/s")
}

private fun System_err(msg: String) = println("[warn] $msg")
