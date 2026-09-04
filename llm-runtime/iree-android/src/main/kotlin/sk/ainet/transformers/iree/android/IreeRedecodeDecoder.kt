package sk.ainet.transformers.iree.android

import android.content.Context
import java.io.File

/**
 * Token-ids-in/token-ids-out facade over [IreeRedecodeSession]: the `gemma-iree`
 * `GemmaDecoder`-style greedy re-decode loop (fixed-`seq` vmfb, invoked once per new
 * token over a growing, causally-masked-safe padded buffer). No tokenizer here — pair
 * this with whatever BPE tokenizer your app already has.
 *
 * **The redecode graph contract** any (vmfb, irpa, functionName) triple must satisfy to
 * work with this class: the compiled function is `tensor<1xSEQxi32> -> tensor<SEQxi32>` —
 * one fixed-length token-id row in, one fixed-length predicted-next-token-id row out, with
 * the DSL's in-graph `argMax` already applied (no host-side argmax over a `[SEQ, vocab]`
 * logits tensor). `:llm-inference:smollm2`'s `SmolLm2ExportHarness` is the first producer
 * of a compatible triple (docs: `docs/smollm2-vmfb.md`).
 *
 * `fromAssets` copies the bundled vmfb + irpa to `filesDir` on first run (IREE maps them
 * from plain file paths) and creates the native session — blocking, call off the main thread.
 */
public class IreeRedecodeDecoder(
    private val native: IreeRedecodeSession,
    public val seq: Int,
) : AutoCloseable {

    /**
     * Greedy-decodes from [promptIds] until [eosTokenId] or [maxNewTokens], whichever
     * comes first. [promptIds] must be shorter than [seq] (the vmfb's fixed sequence
     * length) — there must be room for at least one generated token.
     */
    public fun generate(
        promptIds: IntArray,
        eosTokenId: Int,
        maxNewTokens: Int = seq - promptIds.size,
    ): IntArray {
        require(promptIds.size < seq) { "prompt (${promptIds.size} tokens) must be < seq ($seq)" }
        val buf = IntArray(seq)
        promptIds.copyInto(buf)
        val generated = mutableListOf<Int>()
        var step = 0
        while (promptIds.size + step < seq && step < maxNewTokens) {
            val out = native.step(buf) ?: break
            val next = out[promptIds.size - 1 + step]
            if (next == eosTokenId) break
            generated.add(next)
            buf[promptIds.size + step] = next
            step++
        }
        return generated.toIntArray()
    }

    override fun close(): Unit = native.close()

    public companion object {
        /**
         * Build from app-bundled assets.
         *
         * @param vmfbAsset asset path of the compiled redecode vmfb (e.g. `"smollm2/smollm2-gen-arm64.vmfb"`)
         * @param irpaAsset asset path of the external-weights `.irpa`
         * @param functionName the exported function's fully-qualified name (e.g. `"module.smollm2"`)
         * @param seq the vmfb's fixed sequence length
         * @param cacheDirName subdirectory of `filesDir` to copy the assets into
         * @param device IREE HAL driver — [IreeRedecodeSession.DEFAULT_DEVICE] (CPU) or
         *   [IreeRedecodeSession.VULKAN_DEVICE] (GPU, needs a Vulkan-built `.so` + vmfb)
         * @param taskTopologyGroupCount local-task worker groups (run-time core knob, SKEEP-005);
         *   defaults to the `SKAINET_TASK_GROUPS` environment value, `null` = IREE auto topology
         */
        public fun fromAssets(
            context: Context,
            vmfbAsset: String,
            irpaAsset: String,
            functionName: String,
            seq: Int,
            cacheDirName: String,
            device: String = IreeRedecodeSession.DEFAULT_DEVICE,
            taskTopologyGroupCount: Int? = IreeTaskTopology.fromEnv(),
        ): IreeRedecodeDecoder {
            val app = context.applicationContext
            val dir = File(app.filesDir, cacheDirName).apply { mkdirs() }
            val vmfb = File(dir, File(vmfbAsset).name)
            val irpa = File(dir, File(irpaAsset).name)
            copyAssetIfMissing(app, vmfbAsset, vmfb)
            copyAssetIfMissing(app, irpaAsset, irpa)
            val session = IreeRedecodeSession(vmfb.absolutePath, irpa.absolutePath, functionName, device, taskTopologyGroupCount)
            return IreeRedecodeDecoder(session, seq)
        }

        private fun copyAssetIfMissing(context: Context, asset: String, target: File) {
            if (target.exists() && target.length() > 0L) return
            context.assets.open(asset).use { input -> target.outputStream().use { input.copyTo(it) } }
        }
    }
}
