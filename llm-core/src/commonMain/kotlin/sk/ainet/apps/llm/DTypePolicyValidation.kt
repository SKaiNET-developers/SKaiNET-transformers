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
 * requirements.
 *
 * The transformer-repo loaders (`LlamaNetworkLoader`, `QwenNetworkLoader`, …) ship
 * their own weight-loading chain on top of `DecoderGgufWeightLoader` /
 * `DecoderSafeTensorsLoader`. Those chains do not yet plumb `DTypePolicy` through
 * to the underlying tensor producers — that's a separate follow-up. In the
 * meantime, accepting the policy on the public surface lets consumers express
 * intent today, and this validator ensures we reject impossible requirements at
 * the same boundary SKaiNET's own loaders do.
 *
 * Today the transformer-repo loaders only produce FP32 (after Q4/Q8/BF16/F16
 * dequant on the SafeTensors path; native quantization preservation on the GGUF
 * path). That matches the SKaiNET 0.25.0 `StreamingGgufParametersLoader`
 * validator. The BF16 KEEP_NATIVE SafeTensors path (`Require(BF16)`) is allowed
 * here even though the transformer-repo `DecoderSafeTensorsLoader` does not yet
 * honor it — when wired through, no API change is needed.
 *
 * Throws [IllegalArgumentException] on `Require(target)` for targets we cannot
 * produce. `Any`, `Prefer`, and `OneOf` always pass.
 */
public object DTypePolicyValidation {

    /**
     * Validates a [DTypePolicy] for the transformer-repo loader chain.
     *
     * @param policy the policy supplied by the caller
     * @param loaderName loader name for error messages (e.g. `"LlamaNetworkLoader.fromGguf"`)
     * @param allowBf16Require whether `Require(BF16)` is acceptable. SafeTensors-backed
     *   loaders set this to `true` (matches SKaiNET's `SafeTensorsParametersLoader`); GGUF-only
     *   loaders set it to `false` (matches SKaiNET's `StreamingGgufParametersLoader`).
     */
    public fun validate(
        policy: DTypePolicy,
        loaderName: String,
        allowBf16Require: Boolean,
    ) {
        when (policy) {
            DTypePolicy.Any -> Unit
            is DTypePolicy.Prefer -> Unit
            is DTypePolicy.OneOf -> Unit
            is DTypePolicy.Require -> validateRequire(policy.target, loaderName, allowBf16Require)
        }
    }

    private fun validateRequire(target: DType, loaderName: String, allowBf16Require: Boolean) {
        when (target) {
            FP32 -> Unit
            BF16 -> if (!allowBf16Require) {
                throw IllegalArgumentException(
                    "$loaderName: Require(BF16) is not supported by the GGUF loader chain — " +
                        "GGUF BF16 sources are dequanted to FP32 today (no KEEP_NATIVE GGUF path " +
                        "yet). Use Any or Prefer(BF16) to accept the dequant fallback."
                )
            }
            FP16 -> throw IllegalArgumentException(
                "$loaderName: Require(FP16) is not supported — the loader chain dequants F16 to " +
                    "FP32 (no Fp16DenseTensorData backing yet). Use Any or Prefer(FP16)."
            )
            else -> throw IllegalArgumentException(
                "$loaderName: Require(${target.name}) is not satisfiable — the transformer-repo " +
                    "loader chain produces FP32 (optionally BF16 on the SafeTensors KEEP_NATIVE " +
                    "path). It cannot fabricate ${target.name} from arbitrary sources."
            )
        }
    }
}
