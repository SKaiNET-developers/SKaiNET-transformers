package sk.ainet.transformers.gemma.iree

import kotlinx.io.buffered
import kotlinx.io.files.Path
import kotlinx.io.files.SystemFileSystem
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
@OptIn(kotlin.native.runtime.NativeRuntimeApi::class)
public class GemmaDecoder(
    private val vmfb: String,
    private val irpa: String,
    private val gguf: String,
    private val seq: Int = 24,
    ireeBin: String = "iree-run-module",
) {
    private val rt = IreeRuntime(ireeBin = ireeBin)

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

        platform.posix.system("sync; echo 3 > /proc/sys/vm/drop_caches 2>/dev/null")
        val buf = IntArray(seq) { if (it < ptoks.size) ptoks[it] else 0 }
        val gen = mutableListOf<Int>()
        var step = 0
        while (ptoks.size + step < seq) {
            val r = rt.invoke(
                vmfb, "gemma",
                listOf("1x${seq}xi32=" + buf.joinToString(",")),
                mapOf("model" to irpa), "file",
            )
            val arr = IreeRuntime.parseIntResult(r.stdout) ?: break
            val next = arr[ptoks.size - 1 + step]
            if (next == eos) break
            gen.add(next)
            if (next == eot) break
            buf[ptoks.size + step] = next
            step++
        }

        // reload tokenizer (gen subprocess has exited, RAM freed) to detokenize
        val tok2 = GGUFTokenizer.fromSource(SystemFileSystem.source(Path(gguf)).buffered())
        val toolCall = tok2.decode(gen.toIntArray())
        return Generation(toolCall, CompactCodec.parse(toolCall))
    }
}
