package sk.ainet.models.moonshine

import sk.ainet.apps.llm.tokenizer.GGUFTokenizer
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.types.FP32
import java.io.File
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

/**
 * Phase C — fully in-stack CPU proof of the DSL DECODER (no HF, no vendor decoder binary).
 *
 * Greedy re-decode: our [moonshineDecoder] (real weights, eager on [DirectCpuExecutionContext])
 * attends to the REFERENCE encoder memory for `test.wav` and must transcribe "One, two, three.".
 * "Re-decode" = no KV cache; each step feeds the full token prefix `[1, t, dim]` and takes the
 * last-position argmax (the simplest graph shape — KV-cache export is Phase D). The token
 * embedding is looked up host-side from `decoder.embed_tokens.weight` (the board decoder likewise
 * consumes `inputs_embeds`).
 *
 * The encoder memory is produced by the reference ONNX frontend+encoder (`enc_frontend`→
 * `enc_xformer`) — this test isolates the DECODER; Phase C's encoder half swaps in our eager
 * encoder. So the only Moonshine-specific compute under test here is OUR decoder + weights.
 *
 * Gated on env (skips cleanly if unset):
 *   MOON_REF        dir with `memory.bin` (reference encoder memory `[1,165,288]` f32)
 *   DEC_CHECKPOINT  dir with per-tensor `.bin` decoder weights (from convert_moonshine_weights.py)
 *   MOON_TOKENIZER  path to moonshine-tiny `tokenizer.json`
 */
class MoonshineDecoderE2ECpuTest {
    @Test
    fun greedyDecodeReferenceMemoryTranscribes() {
        val refDir = System.getenv("MOON_REF") ?: return skip("MOON_REF")
        val ckpt = System.getenv("DEC_CHECKPOINT") ?: return skip("DEC_CHECKPOINT")
        val tokJson = System.getenv("MOON_TOKENIZER") ?: return skip("MOON_TOKENIZER")

        val cfg = MoonshineConfig()
        val ctx = DirectCpuExecutionContext.create()
        val decoder = moonshineDecoder<FP32, Float>(cfg, FP32::class)
        val baked = bakeDecoderWeights(decoder, DecDirBinWeightSource(ckpt), FP32::class, ctx as ExecutionContext)
        println("baked $baked decoder params")

        // Reference encoder memory [1, frames, dim] (from the ONNX frontend+encoder).
        val memData = readF32(File(refDir, "memory.bin"))
        val frames = memData.size / cfg.dim
        val memory = ctx.fromFloatArray<FP32, Float>(Shape(1, frames, cfg.dim), FP32::class, memData)

        val text = decodeToText(decoder, memory, ckpt, tokJson, cfg, ctx)
        println("OUR decoder (reference memory) → \"$text\"")
        assertEquals("One, two, three.", text, "our DSL decoder must transcribe the reference audio")
    }

    /** Full in-stack: OUR encoder (eager) → OUR decoder (eager), no ONNX/HF/vendor in the model path. */
    @Test
    fun ourEncoderOurDecoderTranscribes() {
        val refDir = System.getenv("MOON_REF") ?: return skip("MOON_REF")
        val ckpt = System.getenv("DEC_CHECKPOINT") ?: return skip("DEC_CHECKPOINT")
        val tokJson = System.getenv("MOON_TOKENIZER") ?: return skip("MOON_TOKENIZER")

        val cfg = MoonshineConfig()
        val ctx = DirectCpuExecutionContext.create()

        // OUR encoder, real HF weights, eager on the reference conv-frontend features.
        val encoder = moonshineEncoder<FP32, Float>(cfg, FP32::class)
        bakeMoonshineWeights(encoder, DecDirBinWeightSource(ckpt), ::encoderHfNameFor, FP32::class, ctx as ExecutionContext)
        val featData = readF32(File(refDir, "features.bin"))
        val frames = featData.size / cfg.dim
        val features = ctx.fromFloatArray<FP32, Float>(Shape(1, frames, cfg.dim), FP32::class, featData)
        val ourMemory = encoder.forward(features, ctx)

        // Sanity: our encoder memory matches the reference encoder (cosine ≈ 1.0).
        val cos = cosine(ourMemory.data.copyToFloatArray(), readF32(File(refDir, "memory.bin")))
        println("our-encoder vs reference memory cosine = $cos")
        assertTrue(cos > 0.99, "our encoder memory must match the reference (cos=$cos)")

        // OUR decoder on OUR memory.
        val decoder = moonshineDecoder<FP32, Float>(cfg, FP32::class)
        bakeDecoderWeights(decoder, DecDirBinWeightSource(ckpt), FP32::class, ctx)
        val text = decodeToText(decoder, ourMemory, ckpt, tokJson, cfg, ctx)
        println("OUR encoder+decoder → \"$text\"")
        assertEquals("One, two, three.", text, "fully in-stack encoder+decoder must transcribe the audio")
    }

    /** Two-graph KV loop: prefill → decoder_with_past steps must reproduce the Phase C transcript. */
    @Test
    fun twoGraphKvLoopTranscribes() {
        val refDir = System.getenv("MOON_REF") ?: return skip("MOON_REF")
        val ckpt = System.getenv("DEC_CHECKPOINT") ?: return skip("DEC_CHECKPOINT")
        val tokJson = System.getenv("MOON_TOKENIZER") ?: return skip("MOON_TOKENIZER")

        val cfg = MoonshineConfig()
        val ctx = DirectCpuExecutionContext.create()
        val model = moonshineDecoder<FP32, Float>(cfg, FP32::class)
        bakeDecoderWeights(model, DecDirBinWeightSource(ckpt), FP32::class, ctx as ExecutionContext)

        val memData = readF32(File(refDir, "memory.bin"))
        val frames = memData.size / cfg.dim
        val memory = ctx.fromFloatArray<FP32, Float>(Shape(1, frames, cfg.dim), FP32::class, memData)
        val embed = readF32(File(ckpt, "decoder.embed_tokens.weight.bin"))

        val start = 1; val end = 2; val maxTokens = 16
        // Prefill on the START token → logits + self/cross caches.
        val prefill = model.forwardPrefill(embedRow(embed, start, cfg, ctx), memory, ctx)
        var selfK = prefill.selfK; var selfV = prefill.selfV
        val crossK = prefill.crossK; val crossV = prefill.crossV
        var next = argmaxRow(prefill.logits.data.copyToFloatArray(), row = 0, cols = cfg.vocabSize)

        val ids = mutableListOf(start)
        var pos = 1
        while (next != end && ids.size <= maxTokens) {
            ids.add(next)
            val (cos, sin) = model.buildRopeCosSin(pos, 1, ctx) // runtime-position RoPE tables, host-built
            val past = model.forwardWithPast(embedRow(embed, next, cfg, ctx), cos, sin, selfK, selfV, crossK, crossV, ctx)
            selfK = past.selfK; selfV = past.selfV
            next = argmaxRow(past.logits.data.copyToFloatArray(), row = 0, cols = cfg.vocabSize)
            pos++
        }
        val text = GGUFTokenizer.fromTokenizerJson(File(tokJson).readText()).decode(ids.drop(1).toIntArray()).trim()
        println("two-graph KV loop → ids=${ids.drop(1)}  text=\"$text\"")
        assertEquals("One, two, three.", text, "prefill + decoder_with_past loop must match the transcript")
    }

    private fun embedRow(embed: FloatArray, token: Int, cfg: MoonshineConfig, ctx: ExecutionContext): sk.ainet.lang.tensor.Tensor<FP32, Float> {
        val row = FloatArray(cfg.dim)
        System.arraycopy(embed, token * cfg.dim, row, 0, cfg.dim)
        return ctx.fromFloatArray<FP32, Float>(Shape(1, 1, cfg.dim), FP32::class, row)
    }

    /** Greedy re-decode: [memory] is the encoder output; returns the detokenized transcript. */
    private fun decodeToText(
        decoder: MoonshineDecoderModel<FP32, Float>,
        memory: sk.ainet.lang.tensor.Tensor<FP32, Float>,
        ckpt: String,
        tokJson: String,
        cfg: MoonshineConfig,
        ctx: ExecutionContext,
    ): String {
        val embed = readF32(File(ckpt, "decoder.embed_tokens.weight.bin")) // [vocab, dim]
        val start = 1; val end = 2; val maxTokens = 16
        val ids = mutableListOf(start)
        for (step in 0 until maxTokens) {
            val emb = FloatArray(ids.size * cfg.dim)
            for ((i, t) in ids.withIndex()) System.arraycopy(embed, t * cfg.dim, emb, i * cfg.dim, cfg.dim)
            val embeds = ctx.fromFloatArray<FP32, Float>(Shape(1, ids.size, cfg.dim), FP32::class, emb)
            val logits = decoder.forward(embeds, memory, ctx)
            val next = argmaxRow(logits.data.copyToFloatArray(), row = ids.size - 1, cols = cfg.vocabSize)
            if (next == end) break
            ids.add(next)
        }
        val tokenizer = GGUFTokenizer.fromTokenizerJson(File(tokJson).readText())
        return tokenizer.decode(ids.drop(1).toIntArray()).trim()
    }

    private fun cosine(a: FloatArray, b: FloatArray): Double {
        val n = minOf(a.size, b.size)
        var dot = 0.0; var na = 0.0; var nb = 0.0
        for (i in 0 until n) { dot += a[i] * b[i]; na += a[i] * a[i].toDouble(); nb += b[i] * b[i].toDouble() }
        return dot / (kotlin.math.sqrt(na) * kotlin.math.sqrt(nb))
    }

    private fun skip(missing: String) = println("SKIP moonshine E2E: set $missing")

    private fun argmaxRow(flat: FloatArray, row: Int, cols: Int): Int {
        val base = row * cols
        var best = 0
        var bestV = flat[base]
        for (c in 1 until cols) {
            val v = flat[base + c]
            if (v > bestV) { bestV = v; best = c }
        }
        return best
    }

    private fun readF32(f: File): FloatArray {
        val b = f.readBytes()
        return FloatArray(b.size / 4) { i ->
            var bits = 0
            for (k in 0 until 4) bits = bits or ((b[i * 4 + k].toInt() and 0xFF) shl (8 * k))
            Float.fromBits(bits)
        }
    }
}
