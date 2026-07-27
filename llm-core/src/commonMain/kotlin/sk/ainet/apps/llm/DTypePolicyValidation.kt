package sk.ainet.apps.llm

import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32

/**
 * Eager-validation helper for `DTypePolicy` carried by SKaiNET-transformers loaders.
 *
 * SKaiNET 0.25.0 introduced `DTypePolicy` (`Any | Require | Prefer | OneOf`) as the
 * generalised execution-side dtype constraint surface. Its own loaders
 * (`StreamingGgufParametersLoader.withPolicy`, `SafeTensorsParametersLoader.withPolicy`)
 * validate the policy at construction so callers fail fast on impossible
 * requirements; this validator is the same boundary for the transformer-repo
 * loader chains (`DecoderGgufWeightLoader`, `DecoderSafeTensorsLoader`, …).
 *
 * ## What a chain can promise
 *
 * Every chain produces FP32, so `Require(FP32)` always passes. A `Require` naming a
 * **narrow float** (BF16 / FP16) is a promise that the weights reach the kernel in their
 * on-disk 2-bytes-per-element layout — the KEEP_NATIVE path. Only a chain that actually
 * implements KEEP_NATIVE for that format can honour it, so each caller declares its
 * capability via [keepNative] rather than the validator guessing from the format alone.
 *
 * The two narrow formats are tracked **separately and never interchangeably**. They are
 * different bit layouts at the same width, so mis-tagging F16 bytes as BF16 decodes to
 * plausible-looking garbage instead of throwing. A chain that keeps BF16 native but
 * widens F16 declares exactly that, and `Require(FP16)` against it fails loudly.
 *
 * Note that a `Require(BF16)` chain still **widens F16 sources to FP32** — the policy
 * names the format to preserve, not a conversion target. Neither narrow format can be
 * re-encoded into the other without a lossy round-trip, and the loaders do not try.
 *
 * `Any`, `Prefer`, and `OneOf` always pass: they are soft constraints that a chain is
 * free to satisfy or ignore per tensor.
 *
 * Throws [IllegalArgumentException] on `Require(target)` for targets the caller cannot
 * produce.
 */
public object DTypePolicyValidation {

    /**
     * Validates a [DTypePolicy] for one transformer-repo loader chain.
     *
     * @param policy the policy supplied by the caller
     * @param loaderName loader name for error messages (e.g. `"LlamaNetworkLoader.fromGguf"`)
     * @param keepNative the narrow-float dtypes this chain hands through in their on-disk
     *   packed layout. Empty (the default) means the chain widens every narrow float to
     *   FP32, so any `Require` naming one is rejected.
     */
    public fun validate(
        policy: DTypePolicy,
        loaderName: String,
        keepNative: Set<DType> = emptySet(),
    ) {
        when (policy) {
            DTypePolicy.Any -> Unit
            is DTypePolicy.Prefer -> Unit
            is DTypePolicy.OneOf -> Unit
            is DTypePolicy.Require -> validateRequire(policy.target, loaderName, keepNative)
        }
    }

    /**
     * BF16-only capability flag.
     *
     * @param allowBf16Require whether `Require(BF16)` is acceptable.
     */
    @Deprecated(
        "Narrow-float capability is per-format since engine 0.38.0 — a chain can keep FP16 " +
            "native too. Pass the set of formats it keeps packed.",
        ReplaceWith(
            "validate(policy, loaderName, if (allowBf16Require) setOf(BF16) else emptySet())",
            "sk.ainet.lang.types.BF16",
        ),
    )
    public fun validate(
        policy: DTypePolicy,
        loaderName: String,
        allowBf16Require: Boolean,
    ): Unit = validate(policy, loaderName, if (allowBf16Require) setOf(BF16) else emptySet())

    /**
     * Whether [policy] asks for [native] source tensors to stay in their on-disk 16-bit layout.
     *
     * The single decision point every transformer-repo loader chain shares, mirroring the
     * engine's `SafeTensorsParametersLoader.mapPolicyToNarrow` /
     * `StreamingGgufParametersLoader.keepsNative`. Only the format the policy actually names
     * is kept: `Require(BF16)` still widens F16 sources, because turning one narrow format
     * into the other needs a lossy re-encode.
     *
     * [native] is expected to be [BF16] or [FP16]; any other dtype answers `false`.
     */
    public fun keepsNative(policy: DTypePolicy, native: DType): Boolean = when (policy) {
        DTypePolicy.Any -> false
        is DTypePolicy.Require -> policy.target == native
        is DTypePolicy.Prefer -> policy.target == native
        is DTypePolicy.OneOf -> native in policy.allowed
    }

    private fun validateRequire(target: DType, loaderName: String, keepNative: Set<DType>) {
        when (target) {
            FP32 -> Unit
            BF16, FP16 -> if (target !in keepNative) {
                throw IllegalArgumentException(
                    "$loaderName: Require(${target.name}) is not supported by this loader chain — " +
                        "${target.name} sources are widened to FP32 at load " +
                        describeKeepNative(keepNative) + ". " +
                        "Use Any or Prefer(${target.name}) to accept the widening fallback."
                )
            }
            else -> throw IllegalArgumentException(
                "$loaderName: Require(${target.name}) is not satisfiable — the transformer-repo " +
                    "loader chain produces FP32 " + describeKeepNative(keepNative) + ". " +
                    "It cannot fabricate ${target.name} from arbitrary sources."
            )
        }
    }

    private fun describeKeepNative(keepNative: Set<DType>): String = when {
        keepNative.isEmpty() -> "(this chain keeps no narrow float packed)"
        else -> "(this chain keeps only ${keepNative.joinToString(" / ") { it.name }} packed)"
    }
}
