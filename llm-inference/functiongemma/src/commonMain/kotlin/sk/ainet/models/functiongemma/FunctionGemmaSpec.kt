package sk.ainet.models.functiongemma

/** Weight materialization policy for the compiled export. */
public enum class FunctionGemmaQuant {
    /** Raw f32 externals — numeric bring-up only (doubles the archive). */
    FP32,

    /** bf16 externals + `stablehlo.convert bf16->f32` on load (compute stays f32).
     *  Bit-exact drop-in for the f16 vmfb (board A/B verified) at half the archive. */
    BF16,

    /** Per-row symmetric int8 for the 2-D matmul weights (+ per-row f32 scale,
     *  in-graph dequant); norms stay bf16. Redecode graph only (Phase 5, opt-in). */
    INT8,
}

/**
 * Everything the FunctionGemma compiled export needs, in ONE place — nothing
 * env-var-driven inside this module (the env CLI in [main] and the legacy
 * `:llm-runtime:kgemma:exportFunctionGemma` wrapper stay thin shells over this).
 *
 * The architecture constants ([nLayers], [headDim], [nKvHeads], the two RoPE
 * bases and the global-layer period) describe FunctionGemma-270M (gemma3) and
 * are emitted into `manifest.json` so the board runtime consumes the contract
 * mechanically instead of hardcoding it. The export contract tests cross-check
 * them against the shapes in the emitted MLIR.
 */
public data class FunctionGemmaSpec(
    /** Path to the FunctionGemma Q5_K_M `.gguf` checkpoint. */
    val gguf: String,
    /** Fixed prompt capacity of the redecode/prefill graphs (zero-padded, causal-masked). */
    val seq: Int = 24,
    /** Advisory decode-position cap for the dynamic with_past cache (the graph itself is unbounded). */
    val maxPositions: Int = 1024,
    /** gemma3 uses FULL rotary; the gguf omits the factor, so it is forced here. */
    val partialRotary: Float = 1.0f,
    /** Weight materialization for the emitted externals. */
    val quant: FunctionGemmaQuant = FunctionGemmaQuant.BF16,
    /** `<tool_N>` token -> tool-name map (a `null` name is an explicit no-op). */
    val toolMap: Map<String, String?> = DEFAULT_TOOL_MAP,
    /** `<end_of_turn>` token id — the greedy loop's stop token. */
    val eot: Int = DEFAULT_EOT,
    // --- FunctionGemma-270M architecture constants (mirrored into the manifest) ---
    val nLayers: Int = 18,
    val headDim: Int = 256,
    val nKvHeads: Int = 1,
    val slidingRopeBase: Float = 10_000f,
    val globalRopeBase: Float = 1_000_000f,
    /** Every [globalLayerPeriod]-th layer (i % period == period-1) is a global-RoPE layer. */
    val globalLayerPeriod: Int = 6,
) {
    init {
        require(seq > 0) { "seq must be positive, got $seq" }
        require(nLayers > 0 && headDim > 0 && nKvHeads > 0) { "bad architecture constants" }
        require(globalLayerPeriod > 0) { "globalLayerPeriod must be positive" }
    }

    public companion object {
        /** The stock FunctionGemma v10 tool vocabulary (six tools + explicit no-op). */
        public val DEFAULT_TOOL_MAP: Map<String, String?> = mapOf(
            "0" to "set_lights", "1" to "play_buzzer", "2" to "set_alarm",
            "3" to "cancel_alarm", "4" to "get_system_status", "5" to "respond",
            "none" to null,
        )

        /** gemma3 `<end_of_turn>` token id. */
        public const val DEFAULT_EOT: Int = 106
    }
}
