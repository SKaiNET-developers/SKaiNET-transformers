package sk.ainet.models.qwen

import sk.ainet.io.gguf.TensorNameMapper
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufTensorNames

/**
 * GGUF tensor name mapper for the Qwen family (#346's `<F>TensorNames` row).
 *
 * Qwen uses the standard llama.cpp GGUF naming convention (`blk.N.*`),
 * identical to LLaMA. This object delegates entirely to [DecoderGgufTensorNames].
 */
public object QwenTensorNames : TensorNameMapper by DecoderGgufTensorNames

/** Renamed to [QwenTensorNames] (#346 naming convention). Kept one release. */
@Deprecated(
    message = "Renamed to QwenTensorNames (#346 naming convention).",
    replaceWith = ReplaceWith("QwenTensorNames"),
)
public typealias QwenGgufTensorNames = QwenTensorNames
