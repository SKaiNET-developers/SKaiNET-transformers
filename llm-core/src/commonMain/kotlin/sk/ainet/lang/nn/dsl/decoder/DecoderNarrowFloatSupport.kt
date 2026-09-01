package sk.ainet.lang.nn.dsl.decoder

import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DType
import sk.ainet.lang.types.FP16

/**
 * The narrow-float formats the shared decoder chain hands through in their on-disk
 * 2-bytes-per-element layout, rather than widening to FP32 at load.
 *
 * Both [DecoderGgufWeightLoader] and [DecoderSafeTensorsLoader] implement KEEP_NATIVE for BF16
 * and FP16 as of engine 0.38.0, so every `*NetworkLoader` built on them (LLaMA, Qwen, Voxtral)
 * declares the same capability to `DTypePolicyValidation`. Loaders with their own weight chains
 * — Gemma, Apertus — declare an empty set until those chains grow the same path.
 *
 * The set is a statement about *source* formats: a policy naming BF16 keeps BF16 tensors packed
 * and still widens F16 ones, because neither format can be re-encoded as the other without a
 * lossy round-trip.
 *
 * One caveat this coarse capability set cannot express: on the GGUF side, KEEP_NATIVE applies
 * only with an `FP32` element type — the packed tensor presents FP32 to consumers while
 * keeping the on-disk 16-bit bytes.
 */
public val DECODER_NARROW_KEEP_NATIVE: Set<DType> = setOf(BF16, FP16)
