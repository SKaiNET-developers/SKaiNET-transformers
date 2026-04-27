@file:Suppress("DEPRECATION")

package sk.ainet.llm.providers

import sk.ainet.apps.llm.Tokenizer
import sk.ainet.lang.types.DType
import sk.ainet.llm.api.Embedding
import sk.ainet.llm.api.EmbeddingModel
import sk.ainet.llm.api.EmbeddingRequest
import sk.ainet.llm.api.EmbeddingResponse
import sk.ainet.llm.api.Usage
import sk.ainet.models.bert.BertRuntime

/**
 * Adapts a BERT-style encoder runtime (`BertRuntime`) + `Tokenizer` to the neutral
 * [EmbeddingModel] SPI.
 *
 * The runtime already does mean pooling + L2 normalization internally, so the
 * adapter is little more than a tensor-to-FloatArray copy.
 *
 * **Threading:** Like [SkaiNetChatModel], a single instance is **not** thread-safe.
 */
public class SkaiNetEmbeddingModel<T : DType>(
    private val runtime: BertRuntime<T>,
    private val tokenizer: Tokenizer,
    public override val dimensions: Int,
    private val modelId: String? = null,
) : EmbeddingModel {

    override fun call(request: EmbeddingRequest): EmbeddingResponse {
        var totalPromptTokens = 0
        val embeddings = request.inputs.mapIndexed { idx, text ->
            val tokens = tokenizer.encode(text)
            totalPromptTokens += tokens.size
            val tensor = runtime.encode(tokens)
            val vector = tensor.data.copyToFloatArray()
            Embedding(index = idx, vector = vector)
        }
        return EmbeddingResponse(
            embeddings = embeddings,
            usage = Usage(promptTokens = totalPromptTokens, completionTokens = 0),
            modelId = modelId,
        )
    }
}
