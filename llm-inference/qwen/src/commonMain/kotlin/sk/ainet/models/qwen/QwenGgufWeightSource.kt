package sk.ainet.models.qwen

import sk.ainet.io.gguf.TensorNameMapper
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufTensorNames

/**
 * GGUF tensor name mapper for Qwen2 models.
 *
 * Qwen2 uses the standard llama.cpp GGUF naming convention (`blk.N.*`),
 * identical to LLaMA. This object delegates entirely to [DecoderGgufTensorNames].
 */
public object QwenGgufTensorNames : TensorNameMapper by DecoderGgufTensorNames
