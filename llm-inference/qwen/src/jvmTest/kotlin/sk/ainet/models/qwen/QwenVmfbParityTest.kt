package sk.ainet.models.qwen

import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import org.junit.jupiter.api.Assumptions.assumeTrue

/**
 * L3 — host-CPU vmfb greedy parity for the qwen-kv-v1 contract (the `WhisperVmfbParityTest`
 * pattern). Exports all three graphs of Qwen3-0.6B exactly as they ship (bf16 weights, host-gather
 * `emb` argument, padded argMax, head-shared chunk mask), converts each graph's safetensors to an
 * `.irpa`, compiles for `llvm-cpu` host with the dockerized IREE toolchain
 * (`skainet/iree-compiler:3.11.0`), and drives them through `iree-run-module` the way the native
 * session (`iree_kv_jni.c`) does:
 *
 *  1. `qwen_prefill_at` over the first [PREFIX] prompt tokens, zero-padded to [SEQ], K/V sliced to
 *     the real rows afterwards;
 *  2. ONE `qwen_prefill_with_past` chunk call over the remaining prompt tokens, zero-padded to
 *     [CHUNK], against that cache, with host-built split-half RoPE tables and the head-shared
 *     additive mask — its selected token must be the oracle's first token;
 *  3. `qwen_with_past` decode steps until [STEPS] tokens exist.
 *
 * Every token is compared with the mainline llama.cpp greedy oracle in
 * `qwen3-06b/golden-greedy-06b.txt` (smallest top-1/top-2 gap 2.35 nats, so a flip means a real
 * defect, not cross-implementation noise). Embedding rows are read from the exported archive
 * (the bf16-truncated table the device reads), RoPE tables and the mask are built with the same
 * formulas as the native session. Writes an L3 evidence record next to the artefacts.
 *
 * Gated: skips without `QWEN3_06B_GGUF` or without docker and the toolchain image. Artefacts land in
 * `QWEN_L3_OUT` (default `build/qwen-l3`), ~7 GB while it runs.
 *   QWEN3_06B_GGUF=/path/Qwen3-0.6B-Q8_0.gguf ./gradlew :llm-inference:qwen:jvmTest \
 *     --tests "*QwenVmfbParityTest*" -PqwenTestMaxHeap=24g
 */
class QwenVmfbParityTest {

    private data class Fixture(val prompt: List<Int>, val oracle: List<Int>, val steps: Int)

    private fun fixture(): Fixture {
        val raw = checkNotNull(javaClass.getResourceAsStream("/qwen3-06b/golden-greedy-06b.txt")) { "fixture missing" }
            .bufferedReader().readText()
        val kv = raw.lines().filter { it.isNotBlank() && !it.startsWith("#") && '=' in it }
            .associate { it.substringBefore('=') to it.substringAfter('=') }
        fun ints(k: String) = kv.getValue(k).split(',').map { it.trim().toInt() }
        return Fixture(ints("prompt_tokens"), ints("oracle_tokens"), kv.getValue("steps").toInt())
    }

    @Test
    fun vmfbGreedyMatchesLlamaCpp() {
        val gguf = System.getenv("QWEN3_06B_GGUF")
        assumeTrue(gguf != null && File(gguf).exists(), "QWEN3_06B_GGUF not set -- skipping")
        val out = File(System.getenv("QWEN_L3_OUT") ?: "build/qwen-l3").absoluteFile.apply { mkdirs() }
        val tc = Toolchain(out)
        assumeTrue(tc.available(), "docker / ${Toolchain.IMAGE} unavailable -- skipping")
        val fx = fixture()

        // ---- export, parameters, compile ----
        val r = QwenExportHarness.exportAll(gguf!!, out.path, SEQ, CHUNK, bf16 = true, hostGather = true)
        val a = r.arch
        val graphs = listOf("qwen-prefill-at", "qwen-prefill-with-past", "qwen-with-past")
        for (g in graphs) {
            if (!File(out, "$g.irpa").exists()) {
                val (c, log) = tc.run("convert-parameters", "--parameters=model=$g.safetensors", "--output=$g.irpa")
                assertTrue(c == 0, "convert-parameters $g failed:\n${log.takeLast(3000)}")
            }
            val (c, log) = tc.run(
                "compile", "--", "--iree-input-type=stablehlo", "--iree-hal-target-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host", "$g.mlir", "-o", "$g-host.vmfb",
            )
            assertTrue(c == 0, "iree-compile $g failed:\n${log.takeLast(3000)}")
        }
        val emb = EmbeddingTable(File(out, "qwen-prefill-at.mlir"), File(out, "qwen-prefill-at.safetensors"), a.hiddenSize)

        val nL = a.nLayers; val nKv = a.nKvHeads; val hd = a.headDim; val h = a.hiddenSize
        val kvNames = (0 until nL).flatMap { listOf("l$it.k", "l$it.v") }
        val tokens = mutableListOf<Int>()

        // ---- 1. prefill_at over the first PREFIX prompt tokens ----
        val pre = fx.prompt.take(PREFIX)
        val preIds = IntArray(SEQ) { if (it < pre.size) pre[it] else 0 }
        Npy.writeInt(File(out, "pa_tokens.npy"), preIds, intArrayOf(SEQ))
        Npy.writeFloat(File(out, "pa_emb.npy"), emb.rows(preIds), intArrayOf(SEQ, h))
        Npy.writeFloat(File(out, "pa_select.npy"), FloatArray(SEQ) { if (it == pre.size - 1) 1f else 0f }, intArrayOf(1, SEQ))
        val paIn = mapOf("tokens" to "pa_tokens.npy", "emb" to "pa_emb.npy", "select" to "pa_select.npy")
        run(tc, "qwen-prefill-at-host.vmfb", "qwen-prefill-at.irpa", QwenKvContract.FN_PREFILL_AT,
            QwenKvContract.prefillAtArgs().map { paIn.getValue(it) }, kvNames.map { "pa_$it.npy" } + "pa_token.npy")
        var cache = kvNames.associateWith { sliceRows(Npy.readFloat(File(out, "pa_$it.npy")), nKv, SEQ, pre.size, hd) }
        var past = pre.size

        // ---- 2. one chunk call over the rest of the prompt ----
        val rest = fx.prompt.drop(PREFIX); val n = rest.size
        val chIds = IntArray(CHUNK) { if (it < n) rest[it] else 0 }
        Npy.writeInt(File(out, "ch_tokens.npy"), chIds, intArrayOf(CHUNK))
        Npy.writeFloat(File(out, "ch_emb.npy"), emb.rows(chIds), intArrayOf(CHUNK, h))
        val (cs, sn) = ropeTable(past, CHUNK, a.ropeBase, hd)
        Npy.writeFloat(File(out, "ch_cos.npy"), cs, intArrayOf(CHUNK, hd))
        Npy.writeFloat(File(out, "ch_sin.npy"), sn, intArrayOf(CHUNK, hd))
        Npy.writeFloat(File(out, "ch_mask.npy"), chunkMask(past, n), intArrayOf(1, QwenKvContract.MASK_HEADS, CHUNK, past + CHUNK))
        Npy.writeFloat(File(out, "ch_select.npy"), FloatArray(CHUNK) { if (it == n - 1) 1f else 0f }, intArrayOf(1, CHUNK))
        for (k in kvNames) Npy.writeFloat(File(out, "ch_in_$k.npy"), cache.getValue(k), intArrayOf(1, nKv, past, hd))
        val chIn = mapOf("tokens" to "ch_tokens.npy", "emb" to "ch_emb.npy", "cos" to "ch_cos.npy", "sin" to "ch_sin.npy",
            "mask" to "ch_mask.npy", "select" to "ch_select.npy") + kvNames.associateWith { "ch_in_$it.npy" }
        run(tc, "qwen-prefill-with-past-host.vmfb", "qwen-prefill-with-past.irpa", QwenKvContract.FN_PREFILL_WITH_PAST,
            QwenKvContract.prefillWithPastArgs(a).map { chIn.getValue(it) }, kvNames.map { "ch_$it.npy" } + "ch_token.npy")
        cache = kvNames.associateWith { sliceRows(Npy.readFloat(File(out, "ch_$it.npy")), nKv, past + CHUNK, past + n, hd) }
        past += n
        tokens += Npy.readInt(File(out, "ch_token.npy"))[0]

        // ---- 3. with_past decode steps ----
        while (tokens.size < fx.steps) {
            val t = tokens.last()
            Npy.writeInt(File(out, "wp_token.npy"), intArrayOf(t), intArrayOf(1))
            Npy.writeFloat(File(out, "wp_emb.npy"), emb.rows(intArrayOf(t)), intArrayOf(1, h))
            val (c1, s1) = ropeTable(past, 1, a.ropeBase, hd)
            Npy.writeFloat(File(out, "wp_cos.npy"), c1, intArrayOf(1, hd))
            Npy.writeFloat(File(out, "wp_sin.npy"), s1, intArrayOf(1, hd))
            for (k in kvNames) Npy.writeFloat(File(out, "wp_in_$k.npy"), cache.getValue(k), intArrayOf(1, nKv, past, hd))
            val wpIn = mapOf("token" to "wp_token.npy", "emb" to "wp_emb.npy", "cos" to "wp_cos.npy", "sin" to "wp_sin.npy") +
                kvNames.associateWith { "wp_in_$it.npy" }
            run(tc, "qwen-with-past-host.vmfb", "qwen-with-past.irpa", QwenKvContract.FN_WITH_PAST,
                QwenKvContract.withPastArgs(a).map { wpIn.getValue(it) }, kvNames.map { "wp_$it.npy" } + "wp_token.npy")
            cache = kvNames.associateWith { Npy.readFloat(File(out, "wp_$it.npy")) }
            past += 1
            tokens += Npy.readInt(File(out, "wp_token.npy"))[0]
        }

        val firstDivergence = tokens.indices.firstOrNull { tokens[it] != fx.oracle[it] }
        writeEvidence(out, tokens, fx, firstDivergence)
        println("L3 qwen3-0.6b host vmfb tokens: $tokens\n                      oracle: ${fx.oracle}")
        assertEquals(fx.oracle[0], tokens[0], "qwen_prefill_with_past chunk call: first token must equal the oracle's")
        assertEquals(fx.oracle, tokens, "vmfb greedy diverged from llama.cpp at step $firstDivergence")
    }

    // ------------------------------------------------------------------ helpers

    /** Keeps the first [real] of [rows] rows of a `[1, nKv, rows, hd]` cache. */
    private fun sliceRows(x: FloatArray, nKv: Int, rows: Int, real: Int, hd: Int): FloatArray {
        val o = FloatArray(nKv * real * hd)
        for (hh in 0 until nKv) System.arraycopy(x, hh * rows * hd, o, hh * real * hd, real * hd)
        return o
    }

    /** Sign-baked split-half cos/sin for positions `pos .. pos+n-1` (same formula as `iree_kv_jni.c`). */
    private fun ropeTable(pos: Int, n: Int, base: Float, hd: Int): Pair<FloatArray, FloatArray> {
        val half = hd / 2; val c = FloatArray(n * hd); val s = FloatArray(n * hd)
        for (r in 0 until n) for (i in 0 until half) {
            val ang = (pos + r) / base.toDouble().pow(2.0 * i / hd)
            val cv = cos(ang).toFloat(); val sv = sin(ang).toFloat()
            c[r * hd + i] = cv; c[r * hd + half + i] = cv
            s[r * hd + i] = -sv; s[r * hd + half + i] = sv
        }
        return c to s
    }

    /** Head-shared additive chunk mask `[1, 1, C, past+C]`, the `chunk_mask` rule of `iree_kv_jni.c`. */
    private fun chunkMask(past: Int, nReal: Int): FloatArray {
        val k = past + CHUNK; val m = FloatArray(CHUNK * k)
        for (i in 0 until CHUNK) for (j in 0 until k) {
            val ok = if (i < nReal) (j < past || j - past <= i) else (j == past + i)
            m[i * k + j] = if (ok) 0f else MASK_NEG
        }
        return m
    }

    private fun run(tc: Toolchain, vmfb: String, irpa: String, fn: String, inputs: List<String>, outputs: List<String>) {
        val args = mutableListOf("run-module", "--device=local-task", "--module=$vmfb", "--parameters=model=$irpa", "--function=$fn")
        inputs.forEach { args += "--input=@$it" }
        outputs.forEach { args += "--output=@$it" }
        val (c, log) = tc.run(*args.toTypedArray())
        assertTrue(c == 0, "iree-run-module $fn failed:\n${log.takeLast(4000)}")
    }

    private fun writeEvidence(out: File, tokens: List<Int>, fx: Fixture, firstDivergence: Int?) {
        val result = if (firstDivergence == null) "pass" else "fail"
        File(out, "L3-qwen3-06b-host-vmfb.json").writeText(
            """
            |{
            |  "rung": "L3",
            |  "id": "qwen3-06b-host-vmfb-greedy32",
            |  "subject": { "model": "Qwen3-0.6B-Q8_0", "contract": "${QwenKvContract.CONTRACT_ID}", "weights": "bf16", "host_gather": true },
            |  "oracle": { "kind": "llama.cpp", "build": "0.3.0 build 10621 commit c1d0e7a00", "fixture": "qwen3-06b/golden-greedy-06b.txt", "min_top2_gap_nats": 2.35 },
            |  "runner": { "tool": "iree-run-module", "image": "${Toolchain.IMAGE}", "target": "llvm-cpu host", "flow": "prefill_at($PREFIX) + chunk($CHUNK) + with_past" },
            |  "metric": { "tokens_compared": ${tokens.size}, "first_divergence": ${firstDivergence ?: "null"}, "tokens": $tokens },
            |  "tolerance": 0,
            |  "result": "$result"
            |}
            |""".trimMargin(),
        )
    }

    /** Rows of the exported token-embedding table (bf16 in the archive, widened exactly like the device does). */
    private class EmbeddingTable(mlir: File, st: File, private val hidden: Int) {
        private val raf = RandomAccessFile(st, "r")
        private val dataStart: Long
        private val offset: Long
        private val bf16: Boolean
        init {
            val key = Regex("""util\.global private @(\w+) = #flow\.parameter\.named<"model"::"(\w+)"> : tensor<\d+x${hidden}x(bf16|f32)>""")
                .findAll(mlir.readText()).firstOrNull { Regex("""tensor<(\d+)x""").find(it.value)!!.groupValues[1].toInt() > 100_000 }
                ?: error("no vocab x hidden embedding global in ${mlir.name}")
            val name = key.groupValues[2]; bf16 = key.groupValues[3] == "bf16"
            val lenBuf = ByteArray(8); raf.readFully(lenBuf)
            val hLen = ByteBuffer.wrap(lenBuf).order(ByteOrder.LITTLE_ENDIAN).long
            val hdr = ByteArray(hLen.toInt()); raf.readFully(hdr)
            dataStart = 8 + hLen
            val m = Regex(""""$name":\{[^}]*"data_offsets":\[(\d+),""").find(hdr.decodeToString()) ?: error("no $name in ${st.name}")
            offset = m.groupValues[1].toLong()
        }
        fun rows(ids: IntArray): FloatArray {
            val bpe = if (bf16) 2 else 4
            val out = FloatArray(ids.size * hidden); val buf = ByteArray(hidden * bpe)
            for ((r, id) in ids.withIndex()) {
                raf.seek(dataStart + offset + id.toLong() * hidden * bpe); raf.readFully(buf)
                val bb = ByteBuffer.wrap(buf).order(ByteOrder.LITTLE_ENDIAN)
                for (i in 0 until hidden) out[r * hidden + i] =
                    if (bf16) Float.fromBits((bb.getShort(i * 2).toInt() and 0xFFFF) shl 16) else bb.getFloat(i * 4)
            }
            return out
        }
    }

    private object Npy {
        private fun write(f: File, descr: String, shape: IntArray, body: ByteArray) {
            val shapeStr = if (shape.size == 1) "(${shape[0]},)" else shape.joinToString(", ", "(", ")")
            var header = "{'descr': '$descr', 'fortran_order': False, 'shape': $shapeStr, }"
            val pad = 16 - (10 + header.length + 1) % 16
            header += " ".repeat(if (pad == 16) 0 else pad) + "\n"
            val bb = ByteBuffer.allocate(10 + header.length).order(ByteOrder.LITTLE_ENDIAN)
            bb.put(0x93.toByte()); bb.put("NUMPY".toByteArray()); bb.put(1); bb.put(0); bb.putShort(header.length.toShort()); bb.put(header.toByteArray())
            f.outputStream().buffered().use { it.write(bb.array()); it.write(body) }
        }
        fun writeFloat(f: File, d: FloatArray, shape: IntArray) =
            write(f, "<f4", shape, ByteBuffer.allocate(d.size * 4).order(ByteOrder.LITTLE_ENDIAN).also { b -> d.forEach { b.putFloat(it) } }.array())
        fun writeInt(f: File, d: IntArray, shape: IntArray) =
            write(f, "<i4", shape, ByteBuffer.allocate(d.size * 4).order(ByteOrder.LITTLE_ENDIAN).also { b -> d.forEach { b.putInt(it) } }.array())
        private fun body(f: File): ByteBuffer {
            val b = f.readBytes(); require(b[0] == 0x93.toByte()) { "not a .npy: ${f.name}" }
            val hl = (b[8].toInt() and 0xff) or ((b[9].toInt() and 0xff) shl 8)
            return ByteBuffer.wrap(b, 10 + hl, b.size - 10 - hl).slice().order(ByteOrder.LITTLE_ENDIAN)
        }
        fun readFloat(f: File): FloatArray = body(f).let { bb -> FloatArray(bb.remaining() / 4) { bb.getFloat(it * 4) } }
        fun readInt(f: File): IntArray = body(f).let { bb -> IntArray(bb.remaining() / 4) { bb.getInt(it * 4) } }
    }

    private class Toolchain(private val workDir: File) {
        fun available(): Boolean = try {
            ProcessBuilder("docker", "image", "inspect", IMAGE).redirectErrorStream(true).start()
                .also { it.inputStream.readBytes() }.waitFor() == 0
        } catch (_: Exception) { false }
        fun run(vararg args: String): Pair<Int, String> {
            val p = ProcessBuilder(listOf("docker", "run", "--rm", "-v", "${workDir.absolutePath}:/work", "-w", "/work", IMAGE) + args)
                .redirectErrorStream(true).start()
            val o = p.inputStream.bufferedReader().readText()
            return p.waitFor() to o
        }
        companion object { const val IMAGE = "skainet/iree-compiler:3.11.0" }
    }

    private companion object {
        const val SEQ = 16
        const val CHUNK = QwenKvContract.DEFAULT_CHUNK
        const val PREFIX = 3
        const val MASK_NEG = -1.0e30f
    }
}
