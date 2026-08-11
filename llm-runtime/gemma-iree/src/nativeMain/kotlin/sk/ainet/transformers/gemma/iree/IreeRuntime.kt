package sk.ainet.transformers.gemma.iree

import kotlinx.cinterop.ByteVar
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.allocArray
import kotlinx.cinterop.memScoped
import kotlinx.cinterop.toKString
import platform.posix.fgets
import platform.posix.pclose
import platform.posix.popen

/**
 * Board-side IREE runtime binding. The SL2610 ships a statically-linked
 * `iree-run-module` (Torq fork g165e12a) and no shared libiree C API, so we
 * drive it as a subprocess via `popen` — the runtime/ binding the plan calls
 * the interim. It is enough to load a vmfb, bind weights from an `.irpa`
 * parameter archive, set inputs, invoke, and read back result tensors.
 *
 * vmfbs MUST be compiled with the Torq-fork iree-compile (g165e12a) — the
 * stock IREE 3.x compiler emits a bytecode feature ("Ch") the board runtime
 * rejects with INVALID_ARGUMENT.
 */
@OptIn(ExperimentalForeignApi::class)
public class IreeRuntime(
    private val ireeBin: String = "iree-run-module",
    private val device: String = "local-task",
    // When set, emit `--task_topology_group_count=N` so the local-task HAL
    // spreads work over N worker groups (= N cores). Default null keeps IREE's
    // auto-topology; the board driver (GemmaDecoder) passes the A55 core count.
    // Gated so a board that rejects the flag can be reverted without recompiling.
    private val taskTopologyGroupCount: Int? = null,
) {
    public data class Result(val exitCode: Int, val stdout: String) {
        val ok: Boolean get() = exitCode == 0
    }

    /**
     * Invoke [function] in [module]. [parameters] maps an .irpa scope name
     * (e.g. "model") to its file path; [inputs] are IREE `--input=` specs
     * such as `"1x4xi32=2,887,506,2214"`.
     */
    public fun invoke(
        module: String,
        function: String,
        inputs: List<String>,
        parameters: Map<String, String> = emptyMap(),
        // "file" mmaps the .irpa on demand (required on the 1.9GB-RAM board for
        // the 1.74GB FP32 gemma archive); "preload" wires it all into RAM.
        parameterMode: String? = null,
    ): Result {
        val args = buildList {
            add(ireeBin)
            add("--device=$device")
            taskTopologyGroupCount?.let { add("--task_topology_group_count=$it") }
            if (parameterMode != null) add("--parameter_mode=$parameterMode")
            add("--module=$module")
            add("--function=$function")
            for ((scope, path) in parameters) add("--parameters=$scope=$path")
            for (i in inputs) add("--input=$i")
        }
        return exec(args)
    }

    /**
     * File-I/O invoke for the KV-cache 2-graph decode (many large tensor I/O per step). Each entry of
     * [inputSpecs] is a full `--input=` value — either a literal (`"1xi32=262146"`, `"24xi32=2,887,…"`)
     * or a raw-bin file (`"1x1x7x256xf32=@/tmp/k0.bin"`); each [outputFiles] path becomes `--output=@path`
     * (raw little-endian bytes). Weights bind via [parameters] + [parameterMode] like [invoke].
     *
     * BOARD-VERIFIED (SL2610 g165 `iree-run-module`, 2026-08-11): the `--output=@file` format is
     * extension-driven — `@file.bin` writes RAW little-endian bytes (what this path uses), `@file.npy`
     * would add a NumPy header.
     */
    public fun invokeFiles(
        module: String,
        function: String,
        inputSpecs: List<String>,
        outputFiles: List<String>,
        parameters: Map<String, String> = emptyMap(),
        parameterMode: String? = "file",
    ): Result {
        val args = buildList {
            add(ireeBin)
            add("--device=$device")
            taskTopologyGroupCount?.let { add("--task_topology_group_count=$it") }
            if (parameterMode != null) add("--parameter_mode=$parameterMode")
            add("--module=$module")
            add("--function=$function")
            for ((scope, path) in parameters) add("--parameters=$scope=$path")
            for (spec in inputSpecs) add("--input=$spec")
            for (f in outputFiles) add("--output=@$f")
        }
        return exec(args)
    }

    /**
     * Run [module]/[function] and compute the per-position argmax of a single
     * `[1, seqLen, vocab]` f32 result by STREAMING stdout — never buffering the
     * full ~vocab*seqLen float dump. Essential on the board: holding the 1M-float
     * gemma result in memory on top of iree-run-module's ~1.44GB working set
     * OOMs the 1.9GB device. Returns one argmax per sequence position, or null
     * if the run failed / no f32 result line was seen.
     */
    public fun argmaxLogits(
        module: String,
        function: String,
        inputs: List<String>,
        vocab: Int,
        seqLen: Int,
        parameters: Map<String, String> = emptyMap(),
        parameterMode: String? = "file",
    ): IntArray? {
        val args = buildList {
            add(ireeBin); add("--device=$device")
            taskTopologyGroupCount?.let { add("--task_topology_group_count=$it") }
            if (parameterMode != null) add("--parameter_mode=$parameterMode")
            add("--module=$module"); add("--function=$function")
            for ((s, p) in parameters) add("--parameters=$s=$p")
            for (i in inputs) add("--input=$i")
        }
        val cmd = args.joinToString(" ") { shellQuote(it) } + " 2>&1"
        val fp = popen(cmd, "r") ?: return null

        val bestVal = FloatArray(seqLen) { Float.NEGATIVE_INFINITY }
        val bestIdx = IntArray(seqLen) { -1 }
        var afterEq = false          // past the "xf32=" marker of the result line
        var floatIdx = 0
        val tok = StringBuilder()    // current partial number across chunk boundaries
        val tail = StringBuilder()   // small rolling buffer to detect "xf32=" before data
        var sawF32 = false

        fun consumeNumber() {
            if (tok.isEmpty()) return
            val v = tok.toString().toFloatOrNull()
            tok.setLength(0)
            if (v == null) return
            val pos = floatIdx / vocab
            if (pos < seqLen && v > bestVal[pos]) { bestVal[pos] = v; bestIdx[pos] = floatIdx % vocab }
            floatIdx++
        }

        memScoped {
            val n = 65536
            val buf = allocArray<ByteVar>(n)
            while (fgets(buf, n, fp) != null) {
                val chunk = buf.toKString()
                for (c in chunk) {
                    if (!afterEq) {
                        tail.append(c)
                        if (tail.length > 8) tail.deleteAt(0)
                        if (tail.endsWith("xf32=")) { afterEq = true; sawF32 = true }
                    } else if (c == '\n') {
                        consumeNumber(); afterEq = false  // result line ended
                    } else if (c == ' ' || c == '\t' || c == '\r') {
                        consumeNumber()
                    } else {
                        tok.append(c)
                    }
                }
            }
            consumeNumber()
        }
        val status = pclose(fp)
        val code = if (status == -1) -1 else (status shr 8) and 0xff
        return if (code == 0 && sawF32 && bestIdx.none { it < 0 }) bestIdx else null
    }

    private fun exec(args: List<String>): Result {
        // 2>&1 so runtime errors (verifier rejects, OOM) land in stdout too.
        val cmd = args.joinToString(" ") { shellQuote(it) } + " 2>&1"
        val fp = popen(cmd, "r") ?: return Result(-1, "popen failed: $cmd")
        val sb = StringBuilder()
        memScoped {
            val n = 8192
            val buf = allocArray<ByteVar>(n)
            while (fgets(buf, n, fp) != null) sb.append(buf.toKString())
        }
        val status = pclose(fp)
        // pclose returns the wait status; shift to the exit code (POSIX).
        val code = if (status == -1) -1 else (status shr 8) and 0xff
        return Result(code, sb.toString())
    }

    private fun shellQuote(s: String): String =
        if (s.all { it.isLetterOrDigit() || it in "-_=/.,:+@" }) s
        else "'" + s.replace("'", "'\\''") + "'"

    public companion object {
        /**
         * Parse iree-run-module's textual result lines, e.g.
         *   result[0]: hal.buffer_view
         *   4xf32=4 5 6 7
         * into FloatArrays (one per `=`-bearing line). Robust enough for the
         * smoke path; the real decode loop reads `--output=@file.npy` instead.
         */
        /** Parse a single `NxiNN=` integer result line, e.g. `24xi32=3617 107 ...`. */
        public fun parseIntResult(stdout: String): IntArray? =
            stdout.lineSequence()
                .firstOrNull { it.substringBefore('=').contains("xi32") }
                ?.substringAfter('=')?.trim()?.split(Regex("\\s+"))
                ?.let { p -> runCatching { IntArray(p.size) { p[it].toInt() } }.getOrNull() }

        public fun parseFloatResults(stdout: String): List<FloatArray> =
            stdout.lineSequence()
                .mapNotNull { line ->
                    val eq = line.indexOf('=')
                    if (eq < 0 || !line.substringBefore('=').contains("xf32")) return@mapNotNull null
                    val nums = line.substring(eq + 1).trim().split(Regex("\\s+"))
                    runCatching { FloatArray(nums.size) { nums[it].toFloat() } }.getOrNull()
                }
                .toList()
    }
}
