@file:JvmName("KBertJava")

package sk.ainet.models.bert.java

import kotlinx.coroutines.runBlocking
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.lang.types.FP32
import sk.ainet.models.bert.BertConfigParser
import sk.ainet.models.bert.BertEncoderRuntime
import sk.ainet.models.bert.BertNetworkLoader
import sk.ainet.models.bert.HuggingFaceTokenizer
import sk.ainet.models.bert.MDBR_LEAF_IR_CONFIG
import sk.ainet.models.bert.PooledExecutionContext
import sk.ainet.models.bert.createBertEncoderRuntime
import java.nio.file.Path
import kotlin.io.path.exists
import kotlin.io.path.readText

/**
 * Java-friendly facade for loading and running BERT embedding models on the
 * DSL path ([sk.ainet.models.bert.bertNetwork] + [BertEncoderRuntime]).
 *
 * Example usage from Java:
 * ```java
 * try (KBertSession session = KBertJava.loadSafeTensors(Path.of("model-dir"))) {
 *     float[] embedding = session.encode("Hello world");
 *     float similarity = session.similarity("query", "document");
 * }
 * ```
 */
public object KBertJava {

    /**
     * Load a BERT model from a HuggingFace directory containing
     * model.safetensors, vocab.txt, and optionally config.json plus the
     * sentence-transformers `2_Dense/` projection head.
     *
     * @param modelDir Path to the model directory.
     * @return A KBertSession that implements AutoCloseable.
     */
    @JvmStatic
    public fun loadSafeTensors(modelDir: Path): KBertSession {
        require(modelDir.exists()) { "Model directory not found: $modelDir" }

        val safetensorsPath = modelDir.resolve("model.safetensors")
        require(safetensorsPath.exists()) { "model.safetensors not found in $modelDir" }

        val vocabPath = modelDir.resolve("vocab.txt")
        require(vocabPath.exists()) { "vocab.txt not found in $modelDir" }

        val denseWeights = modelDir.resolve("2_Dense").resolve("model.safetensors")
        val denseConfig = modelDir.resolve("2_Dense").resolve("config.json")

        // Detect config from config.json (+ projection head) or use defaults.
        // The legacy facade ignored 2_Dense entirely and silently returned
        // unprojected embeddings for models that ship a projection.
        val configFile = modelDir.resolve("config.json")
        val config = if (configFile.exists()) {
            BertConfigParser.parse(
                configFile.readText(),
                if (denseConfig.exists()) denseConfig.readText() else null,
            )
        } else {
            MDBR_LEAF_IR_CONFIG
        }

        val tokenizer = HuggingFaceTokenizer.fromVocabTxt(vocabPath.readText())
        // Pool scratch buffers across encode() calls — embedding workloads
        // typically encode many strings in a row, so the SizeClassedScratchPool
        // returns real wins. With a single one-shot call the pool is no
        // worse than NoopScratchPool.
        val ctx = PooledExecutionContext(DirectCpuExecutionContext())

        val loaders = buildList {
            add(
                SafeTensorsParametersLoader(
                    sourceProvider = { JvmRandomAccessSource.open(safetensorsPath.toString()) },
                    onProgress = { _, _, _ -> }
                )
            )
            if (denseWeights.exists()) {
                add(
                    SafeTensorsParametersLoader(
                        sourceProvider = { JvmRandomAccessSource.open(denseWeights.toString()) },
                        onProgress = { _, _, _ -> }
                    )
                )
            }
        }

        val tensors = runBlocking { BertNetworkLoader.loadWeightTensors(loaders, ctx, FP32::class) }
        val runtime = createBertEncoderRuntime(config, tensors, ctx)

        return KBertSession(runtime, tokenizer)
    }
}

/**
 * Java-friendly session for BERT encoding and similarity.
 *
 * Implements AutoCloseable for try-with-resources usage.
 */
public class KBertSession(
    private val runtime: BertEncoderRuntime<FP32>,
    private val tokenizer: HuggingFaceTokenizer
) : AutoCloseable {

    /**
     * Encode text into an embedding vector.
     *
     * @param text The input text.
     * @return A float array representing the embedding.
     */
    public fun encode(text: String): FloatArray {
        val tokOutput = tokenizer.encodeWithMetadata(text)
        return runtime.encode(tokOutput.inputIds, tokOutput.attentionMask)
    }

    /**
     * Compute cosine similarity between two texts.
     *
     * @param textA First text.
     * @param textB Second text.
     * @return Cosine similarity in [-1, 1].
     */
    public fun similarity(textA: String, textB: String): Float {
        val embA = encode(textA)
        val embB = encode(textB)
        return cosineSimilarity(embA, embB)
    }

    private fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
        require(a.size == b.size) { "Embedding dimensions must match" }
        var dot = 0f
        var normA = 0f
        var normB = 0f
        for (i in a.indices) {
            dot += a[i] * b[i]
            normA += a[i] * a[i]
            normB += b[i] * b[i]
        }
        val denom = Math.sqrt(normA.toDouble()) * Math.sqrt(normB.toDouble())
        return if (denom < 1e-12) 0f else (dot / denom).toFloat()
    }

    override fun close() {
        // No-op for now — CPU context doesn't need explicit cleanup
    }
}
