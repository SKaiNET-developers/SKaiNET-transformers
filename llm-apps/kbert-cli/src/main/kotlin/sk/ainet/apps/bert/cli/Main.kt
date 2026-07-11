package sk.ainet.apps.bert.cli

import sk.ainet.llm.api.EmbeddingModel
import sk.ainet.llm.providers.BertEmbeddingModel
import java.nio.file.Path
import kotlin.io.path.isDirectory
import kotlin.system.exitProcess
import kotlin.time.measureTime

private fun usage(): Nothing {
    println("Usage: kbert <model-dir | hf-repo-id> \"query text\" [\"doc text\"]")
    println()
    println("  <model-dir>   Directory containing model.safetensors, vocab.txt, config.json")
    println("  <hf-repo-id>  Hugging Face repo (e.g. MongoDB/mdbr-leaf-mt) — downloaded and")
    println("                cached under ~/.cache/skainet/models on first use")
    println("  \"query text\"  Text to encode")
    println("  \"doc text\"    Optional second text — if given, prints cosine similarity")
    exitProcess(1)
}

fun main(args: Array<String>) {
    if (args.isEmpty()) usage()

    val modelRef = args[0]
    val textA = args.getOrNull(1) ?: usage()
    val textB = args.getOrNull(2)

    print("Loading model $modelRef... ")
    val model: EmbeddingModel
    val loadElapsed = measureTime { model = loadModel(modelRef) }
    println("done ($loadElapsed) — dimensions=${model.dimensions}")

    println("\nEncoding: \"$textA\"")
    val vecA: FloatArray
    val encodeElapsed = measureTime { vecA = model.embed(textA) }
    println("Encoded in $encodeElapsed")
    println("Embedding (first 8): ${vecA.take(8).joinToString(", ") { "%.6f".format(it) }}")
    println("Embedding dim: ${vecA.size}")

    if (textB != null) {
        println("\nEncoding: \"$textB\"")
        val vecB = model.embed(textB)
        println("Embedding (first 8): ${vecB.take(8).joinToString(", ") { "%.6f".format(it) }}")
        println("\nCosine similarity: %.6f".format(cosineSimilarity(vecA, vecB)))
    }
}

/** Local directory → fromSafeTensors; anything owner/name-shaped → Hugging Face. */
internal fun loadModel(modelRef: String): EmbeddingModel {
    val asPath = Path.of(modelRef)
    if (asPath.isDirectory()) return BertEmbeddingModel.fromSafeTensors(asPath)
    require(Regex("[\\w.-]+/[\\w.-]+").matches(modelRef)) {
        "Not a directory and not an owner/name Hugging Face repo id: $modelRef"
    }
    return BertEmbeddingModel.fromHuggingFace(modelRef)
}

internal fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
    require(a.size == b.size) { "Vectors must have same dimension" }
    var dot = 0f
    var normA = 0f
    var normB = 0f
    for (i in a.indices) {
        dot += a[i] * b[i]
        normA += a[i] * a[i]
        normB += b[i] * b[i]
    }
    val denom = kotlin.math.sqrt(normA) * kotlin.math.sqrt(normB)
    return if (denom > 0f) dot / denom else 0f
}
