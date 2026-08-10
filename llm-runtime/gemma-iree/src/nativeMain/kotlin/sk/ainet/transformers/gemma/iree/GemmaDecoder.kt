package sk.ainet.transformers.gemma.iree

import kotlin.time.TimeSource
import kotlinx.cinterop.ExperimentalForeignApi
import kotlinx.cinterop.toKString
import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
import platform.posix.getenv
import sk.ainet.apps.llm.tokenizer.GGUFTokenizer

/**
 * FunctionGemma-270M decoder: OUR DSL->StableHLO->IREE f16 vmfb (CPU), driven on
 * the SL2610 board through [IreeRuntime]. Self-contained and reusable — give it
 * a vmfb + its `.irpa` weights + the matching GGUF (for the tokenizer), then
 * call [generate] with user text; it applies the Octopus-v2 chat template, runs
 * the greedy decode loop, and returns the tool-call text + parsed [ToolCall]s.
 *
 * The vmfb is a fixed seq=[seq] prefill graph with an in-graph per-position
 * argmax tail, so each step returns small token ids (not the 262153-wide logits
 * that OOM the board's result formatter); causal masking makes padding to [seq]
 * safe as the sequence grows.
 */
@OptIn(kotlin.native.runtime.NativeRuntimeApi::class, ExperimentalForeignApi::class)
public class GemmaDecoder(
    private val vmfb: String,
    private val irpa: String,
    private val gguf: String,
    private val seq: Int = 24,
    ireeBin: String = "iree-run-module",
) {
    // Number of local-task worker groups (= cores). The SL2610 has 2 A55 cores,
    // so default to 2; override/disable via GEMMA_TASK_GROUPS (0 or empty = let
    // IREE auto-pick, i.e. drop the flag — the escape hatch if the board rejects it).
    private val taskGroups: Int? =
        (getenv("GEMMA_TASK_GROUPS")?.toKString()?.trim()?.toIntOrNull() ?: 2).takeIf { it > 0 }

    // Per-step latency profiling; set VOICECC_PROFILE=1 to print a `[perf]`
    // timing breakdown (Phase-0 perf harness). Safe on this driver's stdout —
    // token parsing reads the iree-run-module SUBPROCESS pipe, not our stdout.
    private val profile: Boolean =
        getenv("VOICECC_PROFILE")?.toKString()?.let { it == "1" || it.equals("true", true) } ?: false

    private val rt = IreeRuntime(ireeBin = ireeBin, taskTopologyGroupCount = taskGroups)

    /** The model's output for one prompt. */
    public data class Generation(
        /** Decoded special-token text, e.g. `<tool_0>(state="on")<end>`. */
        val toolCallText: String,
        /** [toolCallText] parsed by [CompactCodec]. */
        val calls: List<ToolCall>,
    )

    public fun generate(userText: String): Generation {
        val prompt = "<start_of_turn>user\n$userText<end_of_turn>\n<start_of_turn>model\n"

        // Hold the tokenizer ONLY to encode, then free it — the vocab (262153)
        // resident alongside the ~905MB gen subprocess OOMs the 1.9GB board.
        // Scope it in run {} so the reference is dropped before GC.collect();
        // reload it after gen, just to detokenize.
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
        if (ptoks.size > seq - 4) {
            // Too long for the fixed seq=$seq prefill graph -> not a short command; skip
            // gracefully rather than throwing (a listen daemon must survive this).
            return Generation("(skipped: prompt ${ptoks.size} tokens > ${seq - 4})", emptyList())
        }

        // NOTE: no `drop_caches` here. Dropping the page cache forced a COLD mmap
        // of the 831MB irpa on every re-decode step; keeping it warm across steps
        // is exactly what we want (the re-spawn already re-mmaps, but from cache).
        val buf = IntArray(seq) { if (it < ptoks.size) ptoks[it] else 0 }
        val gen = mutableListOf<Int>()
        var step = 0
        val genStart = if (profile) TimeSource.Monotonic.markNow() else null
        while (ptoks.size + step < seq) {
            val t0 = if (profile) TimeSource.Monotonic.markNow() else null
            val r = rt.invoke(
                vmfb, "gemma",
                listOf("1x${seq}xi32=" + buf.joinToString(",")),
                mapOf("model" to irpa), "file",
            )
            val arr = IreeRuntime.parseIntResult(r.stdout) ?: break
            val next = arr[ptoks.size - 1 + step]
            if (t0 != null) println("[perf] gemma step $step: ${t0.elapsedNow().inWholeMilliseconds} ms")
            if (next == eos) break
            gen.add(next)
            if (next == eot) break
            buf[ptoks.size + step] = next
            step++
        }
        if (genStart != null) {
            val total = genStart.elapsedNow().inWholeMilliseconds
            val n = if (gen.isNotEmpty()) gen.size else 1
            println("[perf] gemma total: $total ms, ${gen.size} tokens, ${total / n} ms/token")
        }

        // reload tokenizer (gen subprocess has exited, RAM freed) to detokenize
        val tok2 = GGUFTokenizer.fromSource(SystemFileSystem.source(Path(gguf)).buffered())
        val toolCall = tok2.decode(gen.toIntArray())
        return Generation(toolCall, CompactCodec.parse(toolCall))
    }
}
