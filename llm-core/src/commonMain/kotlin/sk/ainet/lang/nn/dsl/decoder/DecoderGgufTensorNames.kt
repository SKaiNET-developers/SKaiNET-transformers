package sk.ainet.lang.nn.dsl.decoder

import sk.ainet.io.gguf.TensorNameMapper

/**
 * Standard GGUF tensor naming for LLaMA-family models.
 *
 * Uses the canonical `blk.N.*` naming scheme from llama.cpp GGUF files.
 * This is also the default scheme for Mistral, Qwen, and other LLaMA derivatives.
 */
public object DecoderGgufTensorNames : TensorNameMapper {
    override fun tokenEmbedding(): String = DecoderTensorNames.TOKEN_EMBEDDINGS
    override fun outputNorm(): String = DecoderTensorNames.OUTPUT_NORM
    override fun outputWeight(): String = DecoderTensorNames.OUTPUT_WEIGHT

    override fun layerAttnNorm(layer: Int): String = DecoderTensorNames.attnNorm(layer)
    override fun layerAttnQ(layer: Int): String = DecoderTensorNames.attnQ(layer)
    override fun layerAttnK(layer: Int): String = DecoderTensorNames.attnK(layer)
    override fun layerAttnV(layer: Int): String = DecoderTensorNames.attnV(layer)
    override fun layerAttnO(layer: Int): String = DecoderTensorNames.attnOut(layer)

    override fun layerFfnNorm(layer: Int): String = DecoderTensorNames.ffnNorm(layer)
    override fun layerFfnGate(layer: Int): String = DecoderTensorNames.ffnGate(layer)
    override fun layerFfnUp(layer: Int): String = DecoderTensorNames.ffnUp(layer)
    override fun layerFfnDown(layer: Int): String = DecoderTensorNames.ffnDown(layer)
}
