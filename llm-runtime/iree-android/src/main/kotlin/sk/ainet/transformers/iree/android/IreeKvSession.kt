package sk.ainet.transformers.iree.android

/**
 * Architecture constants the native KV session needs (mirrors the `manifest.json` written by
 * `FunctionGemmaContract.manifestJson` / `QwenKvContract.manifestJson`). Field names and types are
 * read by JNI via reflection (`GetFieldID`) — keep them.
 *
 * [nKvHeads] may now be less than [nHeads] (grouped-query attention: Qwen2.5 nKvHeads=2,
 * Qwen3 nKvHeads=8 vs FunctionGemma's plain multi-head nKvHeads=1) as long as `nHeads % nKvHeads
 * == 0` — the native session gathers per-KV-head cache row ranges with device-to-device copies
 * when `nKvHeads > 1` (SKaiNET-transformers#411). The chunk graph's additive mask is rank-4
 * `[1, maskHeads, C, past+C]`; its rows never depend on the head, so [maskHeads] is either
 * [nHeads] (`0`, the default: FunctionGemma's per-head mask) or `1` (qwen-kv-v1: one head-shared
 * mask the graph broadcasts, because IREE 3.11 cannot lower a per-head mask onto grouped-query
 * scores with a dynamic key length). [globalLayerPeriod] `== 1` means
 * every layer is "global" — the qwen-kv-v1 contract, no sliding-window/global split at all, in
 * which case [slidingWindow] and [slidingRopeBase] are unused ([globalRopeBase] doubles as the
 * model's one RoPE base) and the traced graph carries no sliding-side inputs at all.
 */
public class IreeKvSpec @JvmOverloads constructor(
    @JvmField public val nLayers: Int,
    @JvmField public val headDim: Int,
    @JvmField public val nKvHeads: Int,
    @JvmField public val nHeads: Int,
    @JvmField public val hiddenSize: Int,
    @JvmField public val vocabSize: Int,
    @JvmField public val slidingWindow: Int,
    @JvmField public val globalLayerPeriod: Int,
    @JvmField public val chunk: Int,
    @JvmField public val slidingRopeBase: Float,
    @JvmField public val globalRopeBase: Float,
    /** Head count of the chunk mask the native session builds: `0` = [nHeads], or `1`. */
    @JvmField public val maskHeads: Int = 0,
) {
    public companion object {
        /** FunctionGemma-270M with the contract's default chunk. */
        public fun functionGemma270m(chunk: Int = 32): IreeKvSpec = IreeKvSpec(
            nLayers = 18, headDim = 256, nKvHeads = 1, nHeads = 4, hiddenSize = 640, vocabSize = 262144,
            slidingWindow = 512, globalLayerPeriod = 6, chunk = chunk,
            slidingRopeBase = 10_000f, globalRopeBase = 1_000_000f,
        )

        /** Qwen2.5-0.5B-Instruct (arch `qwen2`): 24 layers, GQA 14/2 heads, headDim 64, RoPE base 1e6. */
        public fun qwen25_05bInstruct(chunk: Int = 32): IreeKvSpec = IreeKvSpec(
            nLayers = 24, headDim = 64, nKvHeads = 2, nHeads = 14, hiddenSize = 896, vocabSize = 151936,
            slidingWindow = 0, globalLayerPeriod = 1, chunk = chunk,
            slidingRopeBase = 1_000_000f, globalRopeBase = 1_000_000f, maskHeads = 1,
        )

        /** Qwen3-0.6B (arch `qwen3`): 28 layers, GQA 16/8 heads, headDim 128, RoPE base 1e6, QK-norm. */
        public fun qwen3_06b(chunk: Int = 32): IreeKvSpec = IreeKvSpec(
            nLayers = 28, headDim = 128, nKvHeads = 8, nHeads = 16, hiddenSize = 1024, vocabSize = 151936,
            slidingWindow = 0, globalLayerPeriod = 1, chunk = chunk,
            slidingRopeBase = 1_000_000f, globalRopeBase = 1_000_000f, maskHeads = 1,
        )

        /** Minimal parser for the fields above from a `manifest.json` string (no JSON dependency). */
        public fun fromManifest(json: String, chunkOverride: Int? = null): IreeKvSpec {
            fun int(key: String, def: Int): Int = Regex("\"$key\"\\s*:\\s*(-?\\d+)").find(json)?.groupValues?.get(1)?.toInt() ?: def
            fun flt(key: String, def: Float): Float = Regex("\"$key\"\\s*:\\s*(-?[0-9.]+(?:[eE][-+]?\\d+)?)").find(json)?.groupValues?.get(1)?.toFloat() ?: def
            val d = functionGemma270m()
            return IreeKvSpec(
                nLayers = int("nLayers", d.nLayers), headDim = int("headDim", d.headDim), nKvHeads = int("nKvHeads", d.nKvHeads),
                nHeads = int("nHeads", d.nHeads), hiddenSize = int("hiddenSize", d.hiddenSize), vocabSize = int("vocabSize", d.vocabSize),
                slidingWindow = int("slidingWindow", d.slidingWindow), globalLayerPeriod = int("globalLayerPeriod", d.globalLayerPeriod),
                chunk = chunkOverride ?: int("chunk", d.chunk),
                slidingRopeBase = flt("slidingRopeBase", d.slidingRopeBase), globalRopeBase = flt("globalRopeBase", d.globalRopeBase),
                maskHeads = int("maskHeads", d.maskHeads),
            )
        }
    }
}

/**
 * Stateful KV-cache session over three compiled graphs of a host-gather KV-cache contract (the
 * token embedding rows are read from the parameter archive natively, so callers pass token ids
 * only): `prefill(ids, n)` runs the catalog prefix once (`*_prefill_at`), `chunk(ids, n)` runs an
 * utterance in one call (`*_prefill_with_past`), `step(token)` generates one token (`*_with_past`).
 * The cache stays on the device; sliding layers (if any — [IreeKvSpec.globalLayerPeriod] `> 1`,
 * FunctionGemma's contract) only ever see their last `slidingWindow` positions (zero-copy tail
 * views). `snapshot()`/`restore()` retain the current cache so the catalog prefix is prefilled
 * once per process and restored per turn.
 *
 * Two contracts share this class:
 *  - **functiongemma-kv-v1**: three distinct archives (`vmfbWithPast`/`vmfbChunk`/`vmfbPrefill`
 *    each with their own `.irpa`), [IreeKvSpec.functionGemma270m], `nKvHeads == 1`, sliding +
 *    global layer split.
 *  - **qwen-kv-v1** (SKaiNET-transformers#411): the SAME vmfb+irpa path passed for all three
 *    graphs (three exported functions of one merged module sharing one parameter archive — three
 *    independent sessions would triple the resident bf16 parameter bytes, ~1-1.2 GB, in a 32-bit
 *    process), [IreeKvSpec.qwen25_05bInstruct] / [IreeKvSpec.qwen3_06b], GQA (`nKvHeads` up to 8),
 *    every layer "global" (`globalLayerPeriod == 1`, no sliding-side graph inputs at all). Pass
 *    `fnWithPast = "module.qwen_with_past"`, `fnChunk = "module.qwen_prefill_with_past"`,
 *    `fnPrefill = "module.qwen_prefill_at"` for this contract's function names.
 *
 * Every native failure throws an [IllegalStateException] with the formatted IREE status.
 * The package/class name is the JNI symbol contract with `libskainet_iree_kv.so` — do not move/rename.
 */
public class IreeKvSession(
    spec: IreeKvSpec,
    device: String,
    vmfbWithPast: String, irpaWithPast: String,
    vmfbChunk: String, irpaChunk: String,
    vmfbPrefill: String?, irpaPrefill: String?,
    fnWithPast: String = "module.gemma_with_past",
    fnChunk: String = "module.gemma_prefill_with_past",
    fnPrefill: String = "module.gemma_prefill_at",
) : AutoCloseable {
    public val spec: IreeKvSpec = spec
    private var handle: Long = 0

    init {
        handle = nativeCreate(device, spec, vmfbWithPast, irpaWithPast, fnWithPast, vmfbChunk, irpaChunk, fnChunk, vmfbPrefill, irpaPrefill, fnPrefill)
        check(handle != 0L) { "IreeKvSession: native create failed (device='$device')" }
    }

    /** Absolute position of the next token (= rows in the global layers' cache). */
    public val position: Int get() = nativePosition(handle)

    /** Prefill [n] real tokens of [tokens] (zero-padded to the prefill graph's SEQ); returns the first generated token. */
    public fun prefill(tokens: IntArray, n: Int = tokens.size): Int = nativePrefill(handle, tokens, n)

    /** Run [n] ≤ `spec.chunk` tokens against the cache in one call; returns the token after the last real one. */
    public fun chunk(tokens: IntArray, n: Int = tokens.size): Int = nativeChunk(handle, tokens, n)

    /** Append [token] and return the next one. */
    public fun step(token: Int): Int = nativeStep(handle, token)

    /** Drop the prefill graph's session (its archive mapping) once the prefix snapshot exists. */
    public fun releasePrefill(): Unit = nativeReleasePrefill(handle)

    public fun snapshot(): Snapshot = Snapshot(nativeSnapshot(handle))
    public fun restore(s: Snapshot): Unit = nativeRestore(handle, s.handle)

    override fun close() { if (handle != 0L) { nativeDestroy(handle); handle = 0 } }

    /** A retained cache state; zero-copy (IREE outputs are fresh buffers, inputs are never mutated). */
    public inner class Snapshot internal constructor(internal val handle: Long) : AutoCloseable {
        override fun close() { nativeReleaseSnapshot(handle) }
    }

    private external fun nativeCreate(
        device: String, spec: IreeKvSpec,
        vmfbWithPast: String, irpaWithPast: String, fnWithPast: String,
        vmfbChunk: String, irpaChunk: String, fnChunk: String,
        vmfbPrefill: String?, irpaPrefill: String?, fnPrefill: String?,
    ): Long
    private external fun nativePrefill(handle: Long, tokens: IntArray, n: Int): Int
    private external fun nativeChunk(handle: Long, tokens: IntArray, n: Int): Int
    private external fun nativeStep(handle: Long, token: Int): Int
    private external fun nativeSnapshot(handle: Long): Long
    private external fun nativeRestore(handle: Long, snapshot: Long)
    private external fun nativeReleaseSnapshot(snapshot: Long)
    private external fun nativeReleasePrefill(handle: Long)
    private external fun nativePosition(handle: Long): Int
    private external fun nativeDestroy(handle: Long)

    public companion object {
        init { System.loadLibrary("skainet_iree_kv") }
        public const val DEFAULT_DEVICE: String = "local-task"
        public const val VULKAN_DEVICE: String = "vulkan"
    }
}
