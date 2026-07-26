package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Map/bake the Moonshine **v2** DECODER weights into [moonshineV2Decoder] (= [MoonshineDecoderModel]).
 * Source is the v2 float ONNX baked by `scripts/bake_moonshine_v2_decoder.py`; the `.bin` files are already
 * named by the DSL param (e.g. `dec.0.self_attn.q_proj.weight`), so this mapper is mostly identity — its job
 * is the **transpose flag**, which is the one subtlety (numerically validated to cos-sim 1.0 vs ONNX,
 * `validate_moonshine_v2_decoder.py`):
 *
 *  - `decoder_kv` linears are ONNX **`MatMul` `[in,out]`** (`x@W`) → the DSL `linearProject` wants `[out,in]`
 *    (`x@Wᵀ`) ⇒ **transpose=true** (self q,k,v,o; cross q,o; mlp fc1/fc2).
 *  - `cross_kv` linears are ONNX **`Gemm` `[out,in]`** (transB) → already `[out,in]` ⇒ **transpose=false**
 *    (cross **k,v** only).
 *  - LayerNorms are scale-only → `*_norm.bias` maps to null → zeros; `lm_head.weight` is the tied
 *    `[vocab,dim]=[out,in]` embedding ⇒ transpose=false; `lm_head.bias` absent (bias=false).
 *
 * Reuses the shared bake infra ([DecWeightSource] / [DecMap] / [bakeMoonshineWeights]).
 */
private val V2_DEC_LAYER = Regex("""^dec\.(\d+)\.(.+)$""")

internal fun v2DecoderSrcNameFor(dslName: String): DecMap? {
    when (dslName) {
        "dec_out_norm.weight" -> return DecMap("dec_out_norm.weight", false)
        "dec_out_norm.bias" -> return DecMap(null, false)             // scale-only norm
        "lm_head.weight" -> return DecMap("lm_head.weight", false)    // tied [vocab,dim]=[out,in]
        "lm_head.bias" -> return DecMap(null, false)                  // proj_out has no bias
    }
    val m = V2_DEC_LAYER.matchEntire(dslName) ?: return null
    val l = m.groupValues[1]
    return when (val p = m.groupValues[2]) {
        // scale-only norms (weight only; bias → zeros)
        "self_attn_norm.weight", "cross_attn_norm.weight", "mlp_norm.weight" -> DecMap("dec.$l.$p", false)
        "self_attn_norm.bias", "cross_attn_norm.bias", "mlp_norm.bias" -> DecMap(null, false)
        // decoder_kv MatMul projections + MLP weights — [in,out] → transpose
        "self_attn.q_proj.weight", "self_attn.k_proj.weight",
        "self_attn.v_proj.weight", "self_attn.o_proj.weight",
        "cross_attn.q_proj.weight", "cross_attn.o_proj.weight",
        "mlp_fc1.weight", "mlp_fc2.weight" -> DecMap("dec.$l.$p", true)
        "mlp_fc1.bias", "mlp_fc2.bias" -> DecMap("dec.$l.$p", false)
        // cross_kv Gemm K/V — [out,in] → NO transpose
        "cross_attn.k_proj.weight", "cross_attn.v_proj.weight" -> DecMap("dec.$l.$p", false)
        else -> null
    }
}

/** Overwrite every v2 decoder parameter with real weights from [src]. Fail-fast on any gap. */
internal fun <T : DType, V> bakeV2DecoderWeights(
    model: Module<T, V>,
    src: DecWeightSource,
    dtypeClass: KClass<T>,
    ctx: ExecutionContext,
): Int = bakeMoonshineWeights(model, src, ::v2DecoderSrcNameFor, dtypeClass, ctx)
