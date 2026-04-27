package sk.ainet.apps.llm

import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * RoPE rotation conventions used by GGUF-based models.
 *
 * The two conventions are mathematically equivalent under different weight
 * permutations and produce identical results IF the weights are stored in the
 * matching layout. Mismatching them silently corrupts attention.
 *
 * | Convention   | llama.cpp name      | Pair indexing                      | Used by                  |
 * |--------------|---------------------|------------------------------------|--------------------------|
 * | [INTERLEAVED]| `LLAMA_ROPE_TYPE_NORM` (mode 0) | `(buf[2i], buf[2i+1])` | LLaMA, Mistral, Gemma    |
 * | [HALF_SPLIT] | `LLAMA_ROPE_TYPE_NEOX` (mode 2) | `(buf[i], buf[i+ropeDim/2])` | Qwen 2/3, Phi, Falcon |
 *
 * llama.cpp picks the right convention per architecture via
 * `llm_arch_rope_type(arch)`. We mirror that mapping in [CpuAttentionBackend]
 * (and any other backend) so that GGUF tensors load as-is — no weight
 * permutation at conversion time.
 */
public enum class RopeType {
    /** Interleaved adjacent-pair rotation (llama.cpp NORM, mode 0). */
    INTERLEAVED,
    /** Half-split rotation (llama.cpp NEOX, mode 2) — first half rotates with second half. */
    HALF_SPLIT;

    public companion object {
        /**
         * Map a GGUF `general.architecture` string to the RoPE convention the
         * model was trained / converted with. Mirrors `llm_arch_rope_type()` in
         * `llama.cpp/src/llama-arch.cpp`.
         *
         * Defaults to [INTERLEAVED] for unknown architectures (the LLaMA-family
         * default), since most new families that need [HALF_SPLIT] derive from
         * Qwen / Phi / Falcon and should be added here explicitly.
         */
        public fun forArchitecture(arch: String): RopeType = when (arch.lowercase()) {
            "qwen2", "qwen3", "qwen35", "phi2", "phi3", "phi4", "falcon", "mpt", "stablelm", "starcoder2" -> HALF_SPLIT
            else -> INTERLEAVED
        }
    }
}

/**
 * Compute RoPE (Rotary Position Embedding) frequency for a given pair index and position.
 *
 * Formula: freq = pos / base^(2 * pair / dim)
 *
 * @param pair Pair index (0 to dim/2 - 1)
 * @param pos Sequence position
 * @param dim RoPE dimension (typically headSize)
 * @param base Base frequency (default 10000, Gemma global uses 1M)
 */
public fun ropeFrequency(pair: Int, pos: Int, dim: Int, base: Float = 10000f): Float {
    val exponent = (2f * pair) / dim
    return pos / base.pow(exponent)
}

/**
 * Compute RoPE cosine value for a given pair and position.
 */
public fun ropeCos(pair: Int, pos: Int, dim: Int, base: Float = 10000f): Float =
    cos(ropeFrequency(pair, pos, dim, base))

/**
 * Compute RoPE sine value for a given pair and position.
 */
public fun ropeSin(pair: Int, pos: Int, dim: Int, base: Float = 10000f): Float =
    sin(ropeFrequency(pair, pos, dim, base))

/**
 * Apply RoPE rotation in-place to a buffer of head vectors.
 *
 * Rotates pairs of elements (even, odd) using the RoPE formula:
 *   out[2i]   = in[2i] * cos(freq) - in[2i+1] * sin(freq)
 *   out[2i+1] = in[2i] * sin(freq) + in[2i+1] * cos(freq)
 *
 * Supports optional precomputed cos/sin tables for performance.
 * When [precomputedMatchBase] is provided, precomputed tables are only used
 * if [base] matches that value (useful for Gemma's dual-frequency RoPE).
 *
 * @param buf Buffer containing concatenated head vectors (mutated in-place)
 * @param nHeads Number of heads in the buffer
 * @param headSize Size of each head
 * @param ropeDim Number of dimensions to rotate (typically headSize)
 * @param pos Current sequence position
 * @param base RoPE base frequency
 * @param precomputedCos Optional precomputed cosine table [pos * ropeStride + pair]
 * @param precomputedSin Optional precomputed sine table [pos * ropeStride + pair]
 * @param ropeStride Stride in precomputed tables (typically headSize / 2)
 * @param precomputedMatchBase If set, only use precomputed tables when base equals this value
 */
public fun applyRopeRotation(
    buf: FloatArray,
    nHeads: Int,
    headSize: Int,
    ropeDim: Int,
    pos: Int,
    base: Float = 10000f,
    precomputedCos: FloatArray? = null,
    precomputedSin: FloatArray? = null,
    ropeStride: Int = ropeDim / 2,
    precomputedMatchBase: Float? = null,
    ropeType: RopeType = RopeType.INTERLEAVED
) {
    val usePrecomputed = precomputedCos != null && precomputedSin != null &&
            (precomputedMatchBase == null || base == precomputedMatchBase)
    val halfDim = ropeDim / 2

    for (h in 0 until nHeads) {
        val headOffset = h * headSize
        for (pair in 0 until halfDim) {
            val fcr: Float
            val fci: Float
            if (usePrecomputed) {
                fcr = precomputedCos!![pos * ropeStride + pair]
                fci = precomputedSin!![pos * ropeStride + pair]
            } else {
                fcr = ropeCos(pair, pos, ropeDim, base)
                fci = ropeSin(pair, pos, ropeDim, base)
            }
            // INTERLEAVED rotates (2i, 2i+1) — adjacent pairs (Llama / Gemma / Mistral).
            // HALF_SPLIT rotates (i, i + ropeDim/2) — first-half / second-half (Qwen / Phi / Falcon).
            val (idxA, idxB) = when (ropeType) {
                RopeType.INTERLEAVED -> headOffset + pair * 2 to headOffset + pair * 2 + 1
                RopeType.HALF_SPLIT -> headOffset + pair to headOffset + pair + halfDim
            }
            val v0 = buf[idxA]
            val v1 = buf[idxB]
            buf[idxA] = v0 * fcr - v1 * fci
            buf[idxB] = v0 * fci + v1 * fcr
        }
    }
}
