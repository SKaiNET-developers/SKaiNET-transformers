package sk.ainet.transformers.gemma.iree

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.time.TimeSource
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import kotlinx.io.readByteArray
import kotlinx.io.write
import platform.posix.getenv
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer

/**
 * FunctionGemma-270M **KV-cache** decoder — the 2-graph (prefill + with_past) board loop that replaces
 * the fixed seq re-decode's O(seq²) recompute with O(1 new position)/step (the perf-program Phase-2 win).
 *
 * Drives two vmfbs (from `compile-gemma.sh GEMMA_KV=1`, each with its own parameter archive —
 * `gemma-prefill.irpa` / `gemma-with-past.irpa` — because every trace numbers its externals itself):
 *   gemma_prefill    : `{seq}xi32` tokens -> argMax `{seq}xi32` + 18 self-K + 18 self-V `[1,1,seq,256]`
 *   gemma_with_past  : token[1]i32 + per-base cos/sin[1,256] + 18 past K/V `[1,1,pos,256]`
 *                      -> 18 extended K/V `[1,1,pos+1,256]` + token'[1]i32
 * RoPE position is a runtime input (cos/sin built host-side); the with_past cache dim is dynamic so ONE
 * vmfb serves every position.
 *
 * BOARD-VERIFIED on the SL2610 (2026-08-11, see docs/GEMMA-KV-BOARD-LOOP.md): the full prefill→with_past
 * loop reproduces the oracle `[262146,236769,3255,718,498,1373,262152,106]` token-for-token.
 *   1. [kFirstInOutput] = true — both graphs emit K THEN V per block (the earlier return-SSA "V,K" hint
 *      was wrong; confirmed in the emitted MLIR — layer-0 pair is concat(K_in,·), concat(V_in,·) — and by
 *      board oracle parity with the K-first mapping).
 *   2. `--output=@file.bin` writes RAW little-endian bytes (extension-driven; `.npy` would add a header).
 * Input arg order (token, per-base cos/sin introduced on first use, per-block K-then-V) is trace-confirmed
 * and board-confirmed; `--task_topology_group_count` is accepted by the board `iree-run-module` (g165).
 */
@OptIn(kotlin.native.runtime.NativeRuntimeApi::class, ExperimentalForeignApi::class)
public class GemmaKvDecoder(
    private val prefillVmfb: String,
    private val withPastVmfb: String,
    irpa: String,
    private val gguf: String,
    // Each traced graph numbers its "model" external parameters independently (t0, t10, …), so the
    // redecode archive does NOT serve the KV graphs — each needs the archive written for ITS trace
    // (compile-gemma.sh GEMMA_KV=1 emits gemma-prefill.irpa / gemma-with-past.irpa). Board-verified:
    // binding the wrong archive fails loudly with NOT_FOUND on the first missing `tN` key.
    private val prefillIrpa: String = irpa.replace("gemma-gen.irpa", "gemma-prefill.irpa"),
    private val withPastIrpa: String = irpa.replace("gemma-gen.irpa", "gemma-with-past.irpa"),
    private val seq: Int = 24,
    private val maxNewTokens: Int = 20,
    private val workDir: String = "/tmp/gemma-kv",
    ireeBin: String = "iree-run-module",
) {
    private companion object {
        const val N_LAYERS = 18
        const val HEAD_DIM = 256
        const val ROW = HEAD_DIM              // nKV=1 -> one KV row is headDim floats
        const val SLIDING_BASE = 10_000f
        const val GLOBAL_BASE = 1_000_000f
        // Per-block output K-vs-V order. BOARD-VERIFIED true (SL2610, 2026-08-11): both graphs return
        // K THEN V per block — layer-0 return pair in the emitted MLIR is (concat(K_in,·), concat(V_in,·))
        // where K is the RoPE'd+normed projection — and the K-first loop reproduces the oracle
        // token-for-token. (The draft's return-SSA "V,K" hint was wrong.)
        const val kFirstInOutput = true
    }

    private val taskGroups: Int? =
        (getenv("GEMMA_TASK_GROUPS")?.toKString()?.trim()?.toIntOrNull() ?: 2).takeIf { it > 0 }
    private val profile: Boolean =
        getenv("VOICECC_PROFILE")?.toKString()?.let { it == "1" || it.equals("true", true) } ?: false
    private val rt = IreeRuntime(ireeBin = ireeBin, taskTopologyGroupCount = taskGroups)

    public fun generate(userText: String): GemmaDecoder.Generation {
        val prompt = "<start_of_turn>user\n$userText<end_of_turn>\n<start_of_turn>model\n"
        val eot: Int
        val eos: Int
        val ptoks: List<Int>
        run {
            val tok = GGUFTokenizer.fromSource(SystemFileSystem.source(Path(gguf)).buffered())
            eot = tok.encode("<end_of_turn>").single()
            eos = tok.eosTokenId
            ptoks = listOf(tok.bosTokenId) + tok.encode(prompt).toList()
        }
        kotlin.native.runtime.GC.collect()
        val p = ptoks.size
        if (p > seq - 4) return GemmaDecoder.Generation("(skipped: prompt $p tokens > ${seq - 4})", emptyList())
        SystemFileSystem.createDirectories(Path(workDir))

        val genStart = if (profile) TimeSource.Monotonic.markNow() else null

        // --- PREFILL: pad to seq, run, read argMax + 36 K/V, slice caches to the real length p ---
        val padded = IntArray(seq) { if (it < p) ptoks[it] else 0 }
        val preOut = kvOutFiles("pre")
        val preTokFile = "$workDir/pre_tok.bin"
        val preRes = rt.invokeFiles(
            prefillVmfb, "gemma_prefill",
            inputSpecs = listOf("${seq}xi32=" + padded.joinToString(",")),
            outputFiles = preOut + preTokFile,
            parameters = mapOf("model" to prefillIrpa), parameterMode = "file",
        )
        if (!preRes.ok) { println("[gemma-kv] prefill failed: ${preRes.stdout.take(400)}"); return GemmaDecoder.Generation("(prefill failed)", emptyList()) }
        var next = Bin.readI32(preTokFile, seq)[p - 1]
        // slice each [1,1,seq,256] output to [1,1,p,256] (drop padded tail positions).
        val kCache = Array(N_LAYERS) { Bin.readF32(kFile(preOut, it), seq * ROW).copyOfRange(0, p * ROW) }
        val vCache = Array(N_LAYERS) { Bin.readF32(vFile(preOut, it), seq * ROW).copyOfRange(0, p * ROW) }

        // --- DECODE: one token/step over the growing cache ---
        val gen = mutableListOf<Int>()
        var pos = p
        var step = 0
        while (step < maxNewTokens) {
            if (next == eos) break
            gen.add(next)
            if (next == eot) break
            val t0 = if (profile) TimeSource.Monotonic.markNow() else null

            // cos/sin for both RoPE bases at this position (full-rotary split-half, headDim denom).
            val (cosS, sinS) = splitHalfCosSin(pos, SLIDING_BASE)
            val (cosG, sinG) = splitHalfCosSin(pos, GLOBAL_BASE)
            Bin.writeF32("$workDir/cosS.bin", cosS); Bin.writeF32("$workDir/sinS.bin", sinS)
            Bin.writeF32("$workDir/cosG.bin", cosG); Bin.writeF32("$workDir/sinG.bin", sinG)
            for (i in 0 until N_LAYERS) {
                Bin.writeF32("$workDir/k_in_$i.bin", kCache[i])
                Bin.writeF32("$workDir/v_in_$i.bin", vCache[i])
            }

            val wpOut = kvOutFiles("wp")
            val wpTokFile = "$workDir/wp_tok.bin"
            val res = rt.invokeFiles(
                withPastVmfb, "gemma_with_past",
                inputSpecs = withPastInputSpecs(next, pos),
                outputFiles = wpOut + wpTokFile,
                parameters = mapOf("model" to withPastIrpa), parameterMode = "file",
            )
            if (!res.ok) { println("[gemma-kv] with_past failed at pos=$pos: ${res.stdout.take(400)}"); break }
            next = Bin.readI32(wpTokFile, 1)[0]
            val newLen = (pos + 1) * ROW
            for (i in 0 until N_LAYERS) {
                kCache[i] = Bin.readF32(kFile(wpOut, i), newLen)
                vCache[i] = Bin.readF32(vFile(wpOut, i), newLen)
            }
            if (t0 != null) println("[perf] gemma-kv step $step: ${t0.elapsedNow().inWholeMilliseconds} ms")
            pos++
            step++
        }
        if (genStart != null) {
            val total = genStart.elapsedNow().inWholeMilliseconds
            val n = if (gen.isNotEmpty()) gen.size else 1
            println("[perf] gemma-kv total: $total ms, ${gen.size} tokens, ${total / n} ms/token")
        }

        val tok2 = GGUFTokenizer.fromSource(SystemFileSystem.source(Path(gguf)).buffered())
        val text = tok2.decode(gen.toIntArray())
        return GemmaDecoder.Generation(text, CompactCodec.parse(text))
    }

    /** Sign-baked split-half cos/sin `[headDim]` at [position], full rotary, freqDenom=headDim (matches
     *  RoPE.buildSplitHalfCosSin for FunctionGemma: factor=1 no-op, partialRotary=1.0). */
    private fun splitHalfCosSin(position: Int, base: Float): Pair<FloatArray, FloatArray> {
        val half = HEAD_DIM / 2
        val c = FloatArray(HEAD_DIM)
        val s = FloatArray(HEAD_DIM)
        for (i in 0 until half) {
            val freq = 1.0 / base.toDouble().pow(2.0 * i / HEAD_DIM)
            val a = position * freq
            val cv = cos(a).toFloat()
            val sv = sin(a).toFloat()
            c[i] = cv; c[half + i] = cv
            s[i] = -sv; s[half + i] = sv
        }
        return c to s
    }

    /** with_past inputs in the trace-confirmed arg order: token, then per-base cos/sin introduced on
     *  first use of each layer type, with each block's K THEN V. */
    private fun withPastInputSpecs(token: Int, pos: Int): List<String> {
        val kv = "1x1x${pos}x256xf32"
        val specs = mutableListOf("1xi32=$token")
        var introS = false
        var introG = false
        for (i in 0 until N_LAYERS) {
            val isGlobal = i % 6 == 5
            if (isGlobal && !introG) { specs += "1x256xf32=@$workDir/cosG.bin"; specs += "1x256xf32=@$workDir/sinG.bin"; introG = true }
            if (!isGlobal && !introS) { specs += "1x256xf32=@$workDir/cosS.bin"; specs += "1x256xf32=@$workDir/sinS.bin"; introS = true }
            specs += "$kv=@$workDir/k_in_$i.bin"
            specs += "$kv=@$workDir/v_in_$i.bin"
        }
        return specs
    }

    // Output layout: 36 K/V (block order, two per block) then the token file (appended by the caller).
    private fun kvOutFiles(tag: String): List<String> = (0 until 2 * N_LAYERS).map { "$workDir/${tag}_out_$it.bin" }
    private fun kFile(outs: List<String>, layer: Int): String = outs[2 * layer + if (kFirstInOutput) 0 else 1]
    private fun vFile(outs: List<String>, layer: Int): String = outs[2 * layer + if (kFirstInOutput) 1 else 0]

    /** Raw little-endian binary tensor I/O (matches iree-run-module `=@file` / `--output=@file`). */
    private object Bin {
        fun writeF32(path: String, data: FloatArray) {
            val b = ByteArray(data.size * 4)
            for (i in data.indices) {
                val v = data[i].toRawBits()
                b[i * 4] = (v and 0xFF).toByte()
                b[i * 4 + 1] = ((v ushr 8) and 0xFF).toByte()
                b[i * 4 + 2] = ((v ushr 16) and 0xFF).toByte()
                b[i * 4 + 3] = ((v ushr 24) and 0xFF).toByte()
            }
            SystemFileSystem.sink(Path(path)).buffered().use { it.write(b) }
        }

        fun readF32(path: String, count: Int): FloatArray {
            val b = SystemFileSystem.source(Path(path)).buffered().use { it.readByteArray(count * 4) }
            return FloatArray(count) { i ->
                val v = (b[i * 4].toInt() and 0xFF) or
                    ((b[i * 4 + 1].toInt() and 0xFF) shl 8) or
                    ((b[i * 4 + 2].toInt() and 0xFF) shl 16) or
                    ((b[i * 4 + 3].toInt() and 0xFF) shl 24)
                Float.fromBits(v)
            }
        }

        fun readI32(path: String, count: Int): IntArray {
            val b = SystemFileSystem.source(Path(path)).buffered().use { it.readByteArray(count * 4) }
            return IntArray(count) { i ->
                (b[i * 4].toInt() and 0xFF) or
                    ((b[i * 4 + 1].toInt() and 0xFF) shl 8) or
                    ((b[i * 4 + 2].toInt() and 0xFF) shl 16) or
                    ((b[i * 4 + 3].toInt() and 0xFF) shl 24)
            }
        }
    }
}
