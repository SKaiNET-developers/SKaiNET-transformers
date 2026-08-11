package sk.ainet.models.functiongemma

/**
 * The FunctionGemma compiled-pipeline CONTRACT — `manifest.json` emission (the
 * whisper contract-v2 pattern): function names, argument/result orders, dims and
 * the tool map, generated from the [FunctionGemmaSpec] so the board runtime
 * (`GemmaKvDecoder` in :llm-runtime:gemma-iree) consumes the orders mechanically
 * instead of hardcoding them.
 *
 * Arg/result order facts (BOARD-VERIFIED on the SL2610, 2026-08-11 — see
 * llm-runtime/gemma-iree/docs/GEMMA-KV-BOARD-LOOP.md):
 *  - `gemma`            args `[tokens 1x{seq}xi32]` -> results `[tokens {seq}xi32]`
 *  - `gemma_prefill`    args `[tokens {seq}xi32]` -> results = per layer l:
 *    `l.k`, `l.v` (`1x{nKV}x{seq}x{headDim}`), THEN `tokens {seq}xi32` last
 *  - `gemma_with_past`  args = `token 1xi32`, then per layer in order with each
 *    RoPE base's `cos/sin [1,{headDim}]` introduced on FIRST use of that layer
 *    type (sliding at layer 0, global at layer `period-1`), and each block's
 *    K THEN V (`1x{nKV}x?x{headDim}` dynamic); results = per layer `l.k`, `l.v`
 *    extended caches, THEN `token 1xi32` last.
 *
 * Pure Kotlin (no I/O) so the emission is testable without a checkpoint.
 */
public object FunctionGemmaContract {

    /** Bump when the emitted graph I/O contract changes shape or order. */
    public const val CONTRACT_VERSION: Int = 1

    /** The parameter scope every graph's externals are named under. */
    public const val PARAMETER_SCOPE: String = "model"

    public const val FN_REDECODE: String = "gemma"
    public const val FN_PREFILL: String = "gemma_prefill"
    public const val FN_WITH_PAST: String = "gemma_with_past"

    /** Both graphs emit K THEN V per block (board-verified; see GemmaKvDecoder.kFirstInOutput). */
    public const val K_FIRST_IN_OUTPUT: Boolean = true

    /** `gemma_prefill` / `gemma` argument order. */
    public fun prefillArgs(): List<String> = listOf("tokens")

    /** `gemma_prefill` result order: per-layer K,V caches then the argMax tokens LAST. */
    public fun prefillOutputs(spec: FunctionGemmaSpec): List<String> =
        perLayerKv(spec) + "tokens"

    /**
     * `gemma_with_past` argument order: token first, then per layer (K then V)
     * with each RoPE base's cos/sin introduced on FIRST use of that layer type —
     * exactly the first-unresolved-use order the tracer synthesizes.
     */
    public fun withPastArgs(spec: FunctionGemmaSpec): List<String> {
        val args = mutableListOf("token")
        var introSliding = false
        var introGlobal = false
        for (l in 0 until spec.nLayers) {
            val isGlobal = l % spec.globalLayerPeriod == spec.globalLayerPeriod - 1
            if (isGlobal && !introGlobal) {
                args += "cosGlobal"; args += "sinGlobal"; introGlobal = true
            }
            if (!isGlobal && !introSliding) {
                args += "cosSliding"; args += "sinSliding"; introSliding = true
            }
            args += "l$l.k"; args += "l$l.v"
        }
        return args
    }

    /** `gemma_with_past` result order: per-layer extended K,V caches then the next token LAST. */
    public fun withPastOutputs(spec: FunctionGemmaSpec): List<String> =
        perLayerKv(spec) + "token"

    /** The complete `manifest.json` text for [spec]. */
    public fun manifestJson(spec: FunctionGemmaSpec): String {
        fun arr(names: List<String>) = names.joinToString(",") { "\"$it\"" }
        val tools = spec.toolMap.entries.joinToString(",") { (k, v) ->
            "\"$k\": " + if (v == null) "null" else "\"$v\""
        }
        return """
        |{
        |  "contractVersion": $CONTRACT_VERSION,
        |  "model": "functiongemma-270m",
        |  "functions": { "redecode": "$FN_REDECODE", "prefill": "$FN_PREFILL", "withPast": "$FN_WITH_PAST" },
        |  "nLayers": ${spec.nLayers},
        |  "headDim": ${spec.headDim},
        |  "nKvHeads": ${spec.nKvHeads},
        |  "seq": ${spec.seq},
        |  "maxPositions": ${spec.maxPositions},
        |  "eot": ${spec.eot},
        |  "slidingRopeBase": ${spec.slidingRopeBase},
        |  "globalRopeBase": ${spec.globalRopeBase},
        |  "globalLayerPeriod": ${spec.globalLayerPeriod},
        |  "kFirstInOutput": $K_FIRST_IN_OUTPUT,
        |  "parameterScope": "$PARAMETER_SCOPE",
        |  "parameters": { "redecode": "gemma-gen.irpa", "prefill": "gemma-prefill.irpa", "withPast": "gemma-with-past.irpa" },
        |  "redecodeArgs": [${arr(prefillArgs())}],
        |  "redecodeOutputs": ["tokens"],
        |  "prefillArgs": [${arr(prefillArgs())}],
        |  "prefillOutputs": [${arr(prefillOutputs(spec))}],
        |  "withPastArgs": [${arr(withPastArgs(spec))}],
        |  "withPastOutputs": [${arr(withPastOutputs(spec))}],
        |  "toolMap": { $tools }
        |}
        |""".trimMargin()
    }

    private fun perLayerKv(spec: FunctionGemmaSpec): List<String> =
        (0 until spec.nLayers).flatMap { l -> listOf("l$l.k", "l$l.v") }
}
