package sk.ainet.models.qwen

/**
 * The qwen-kv-v1 compiled-pipeline CONTRACT (SKaiNET-transformers#411) — `manifest.json`
 * emission, the FunctionGemma `manifest.json` pattern (see `FunctionGemmaContract`) adapted for
 * a model with no sliding/global layer split (every layer "global", one RoPE base) and grouped-
 * query attention (`nKvHeads` up to 8), and for the MERGED-MODULE archive shape this contract
 * uses instead of FunctionGemma's three independent ones (SKaiNET-transformers#411: three
 * ~1-1.2 GB bf16 archives would not fit a 32-bit process, so `qwen_prefill_at` /
 * `qwen_prefill_with_past` / `qwen_with_past` are three functions of ONE `qwen-kv.vmfb` sharing
 * ONE `qwen.irpa`).
 *
 * Unlike [FunctionGemmaContract]'s arg lists (which predate the host-gather addendum and omit
 * the embedding tensor those graphs actually take as an input — see `iree_kv_jni.c`'s header
 * comment), this contract lists `emb` explicitly: every arg/output list here matches exactly what
 * `iree_kv_jni.c`'s `ins[]`/`outs[]` assembly builds (`nativePrefill`/`nativeChunk`/`nativeStep`),
 * so `QwenExportHarness`'s trace + the native runtime's calls are checked against the SAME source
 * of truth.
 *
 * Pure Kotlin (no I/O) so the emission is testable without a checkpoint.
 */
public object QwenKvContract {

    /** Bump when the emitted graph I/O contract changes shape or order. */
    public const val CONTRACT_VERSION: Int = 1
    public const val CONTRACT_ID: String = "qwen-kv-v1"

    /** The parameter scope every graph's externals are named under — matches FunctionGemma's. */
    public const val PARAMETER_SCOPE: String = "model"

    public const val FN_PREFILL_AT: String = "qwen_prefill_at"
    public const val FN_PREFILL_WITH_PAST: String = "qwen_prefill_with_past"
    public const val FN_WITH_PAST: String = "qwen_with_past"

    /** Fixed chunk size an utterance is padded to for [FN_PREFILL_WITH_PAST] — one call per
     *  utterance instead of C [FN_WITH_PAST] steps. Matches `IreeKvSpec`'s `chunk` field and
     *  `FunctionGemmaContract.DEFAULT_CHUNK`. */
    public const val DEFAULT_CHUNK: Int = 32

    /**
     * The argMax tail's reduction extent must be a multiple of this: IREE 3.11's SPIR-V backend
     * (Vulkan, valhall4) cannot lower the fused argMax reduction otherwise — bisected with real
     * exports, see `QwenExportHarness.rewriteArgMaxPadded`. Qwen's 151936-entry vocab is not
     * (74.19 × 2048), FunctionGemma's 262144 is (128 × 2048), which is why only Qwen needs the pad.
     */
    public const val ARGMAX_REDUCTION_MULTIPLE: Int = 2048

    /**
     * Head count of `qwen_prefill_with_past`'s additive mask, `[1, MASK_HEADS, C, past+C]`. One
     * head-shared mask (its rows never depend on the head) that the graph broadcasts onto the
     * grouped-query scores: IREE 3.11 cannot lower a per-head mask there once the key length is
     * dynamic (see `QwenExportHarness.rewriteGqaMaskBroadcast`). Written to the manifest as
     * `maskHeads`, which `IreeKvSpec.fromManifest` passes to the native session.
     */
    public const val MASK_HEADS: Int = 1

    /** `vocab` rounded up to the next multiple of [ARGMAX_REDUCTION_MULTIPLE] (153600 for Qwen). */
    public fun paddedArgMaxExtent(vocab: Int): Int {
        require(vocab > 0) { "vocab must be positive, got $vocab" }
        val m = ARGMAX_REDUCTION_MULTIPLE
        return (vocab + m - 1) / m * m
    }

    /** The IREE runtime addresses functions by their module-qualified name (`module.qwen_with_past`). */
    public fun qualified(fn: String): String = if ('.' in fn) fn else "module.$fn"

    /** Every graph emits K THEN V per layer (matches FunctionGemma; `iree_kv_jni.c`'s `adopt_outputs`
     *  assumes this order — `outs[2*l]` is K, `outs[2*l+1]` is V). */
    public const val K_FIRST_IN_OUTPUT: Boolean = true

    /** `qwen_prefill_at` argument order: tokens, host-gathered embedding rows, then the one-hot
     *  position row (matches `iree_kv_jni.c`'s `nativePrefill`: `ins[0]=tokens, ins[1]=emb, ins[2]=select`). */
    public fun prefillAtArgs(): List<String> = listOf("tokens", "emb", "select")

    /** `qwen_prefill_at` result order: per-layer K,V caches then the single selected token LAST. */
    public fun prefillAtOutputs(arch: QwenKvArch): List<String> = perLayerKv(arch) + "token"

    /**
     * `qwen_prefill_with_past` argument order (the trace's first-use order): tokens, embedding
     * rows, the one RoPE base's cos/sin (introduced once, before the first layer's K/V — every
     * layer is "global"), then per layer that layer's K then V, the additive mask (introduced
     * once, right after the first layer's K/V — matches `iree_kv_jni.c`'s `nativeChunk` arg
     * assembly with `hasSliding = false`), and finally the one-hot select.
     */
    public fun prefillWithPastArgs(arch: QwenKvArch): List<String> {
        val args = mutableListOf("tokens", "emb", "cos", "sin")
        for (l in 0 until arch.nLayers) {
            args += "l$l.k"; args += "l$l.v"
            if (l == 0) args += "mask"
        }
        args += "select"
        return args
    }

    /** `qwen_prefill_with_past` result order: per-layer K,V (extended by the chunk) then the token LAST. */
    public fun prefillWithPastOutputs(arch: QwenKvArch): List<String> = perLayerKv(arch) + "token"

    /**
     * `qwen_with_past` argument order: token, embedding row, the one RoPE base's cos/sin
     * (introduced once, before the first layer), then per layer that layer's K then V. Matches
     * `iree_kv_jni.c`'s `nativeStep` arg assembly with `hasSliding = false`.
     */
    public fun withPastArgs(arch: QwenKvArch): List<String> {
        val args = mutableListOf("token", "emb", "cos", "sin")
        for (l in 0 until arch.nLayers) { args += "l$l.k"; args += "l$l.v" }
        return args
    }

    /** `qwen_with_past` result order: per-layer extended K,V caches then the next token LAST. */
    public fun withPastOutputs(arch: QwenKvArch): List<String> = perLayerKv(arch) + "token"

    /**
     * The complete `manifest.json` text for [arch]. Every key `IreeKvSpec.fromManifest` reads
     * (`nLayers, headDim, nKvHeads, nHeads, hiddenSize, vocabSize, slidingWindow,
     * globalLayerPeriod, chunk, slidingRopeBase, globalRopeBase`) is present; `slidingWindow`
     * is always 0 and `slidingRopeBase == globalRopeBase` (qwen-kv-v1: no sliding/global split,
     * see the class doc), which is what makes `IreeKvSession` build no sliding-side tensors at all.
     */
    public fun manifestJson(arch: QwenKvArch, chunk: Int = DEFAULT_CHUNK): String {
        fun arr(names: List<String>) = names.joinToString(",") { "\"$it\"" }
        return """
        |{
        |  "contract": "$CONTRACT_ID",
        |  "contractVersion": $CONTRACT_VERSION,
        |  "model": "${arch.architecture}",
        |  "architecture": "${arch.architecture}",
        |  "functions": { "prefillAt": "$FN_PREFILL_AT", "prefillWithPast": "$FN_PREFILL_WITH_PAST", "withPast": "$FN_WITH_PAST" },
        |  "module": "qwen-kv.vmfb",
        |  "nLayers": ${arch.nLayers},
        |  "headDim": ${arch.headDim},
        |  "nKvHeads": ${arch.nKvHeads},
        |  "nHeads": ${arch.nHeads},
        |  "hiddenSize": ${arch.hiddenSize},
        |  "vocabSize": ${arch.vocabSize},
        |  "slidingWindow": 0,
        |  "globalLayerPeriod": 1,
        |  "chunk": $chunk,
        |  "maskHeads": $MASK_HEADS,
        |  "slidingRopeBase": ${arch.ropeBase},
        |  "globalRopeBase": ${arch.ropeBase},
        |  "ropeBase": ${arch.ropeBase},
        |  "rmsNormEps": ${arch.rmsNormEps},
        |  "qkNorm": ${arch.qkNorm},
        |  "attnBias": ${arch.attnBias},
        |  "eos": ${arch.eos},
        |  "eot": ${arch.eot},
        |  "embeddingKey": "token_embd.weight",
        |  "kFirstInOutput": $K_FIRST_IN_OUTPUT,
        |  "parameterScope": "$PARAMETER_SCOPE",
        |  "parameters": { "shared": "qwen.irpa" },
        |  "prefillAtArgs": [${arr(prefillAtArgs())}],
        |  "prefillAtOutputs": [${arr(prefillAtOutputs(arch))}],
        |  "prefillWithPastArgs": [${arr(prefillWithPastArgs(arch))}],
        |  "prefillWithPastOutputs": [${arr(prefillWithPastOutputs(arch))}],
        |  "withPastArgs": [${arr(withPastArgs(arch))}],
        |  "withPastOutputs": [${arr(withPastOutputs(arch))}]
        |}
        |""".trimMargin()
    }

    private fun perLayerKv(arch: QwenKvArch): List<String> =
        (0 until arch.nLayers).flatMap { l -> listOf("l$l.k", "l$l.v") }
}
