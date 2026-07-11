package sk.ainet.apps.bert.cli

import sk.ainet.llm.api.EmbeddingModel
import java.util.Locale
import kotlin.system.exitProcess
import kotlin.time.measureTime

private fun usage(): Nothing {
    println("Usage: kbert-demo <model-dir | hf-repo-id> [--iterations N]")
    println()
    println("  <model-dir>    Directory containing model.safetensors, vocab.txt, config.json")
    println("  <hf-repo-id>   Hugging Face repo — downloaded and cached on first use")
    println("  --iterations N Number of embeddings for throughput test (default: 100)")
    exitProcess(1)
}

fun main(args: Array<String>) {
    if (args.isEmpty()) usage()

    val modelRef = args[0]
    var iterations = 100
    var i = 1
    while (i < args.size) {
        when (args[i]) {
            "--iterations" -> {
                iterations = args.getOrNull(i + 1)?.toIntOrNull()
                    ?: error("--iterations requires an integer argument")
                i += 2
            }
            else -> error("Unknown flag: ${args[i]}")
        }
    }

    println("=== SKaiNET LEAF IR Demo ===")
    print("Loading model $modelRef... ")
    val model: EmbeddingModel
    val loadElapsed = measureTime { model = loadModel(modelRef) }
    println("done ($loadElapsed)")
    println("Embedding dimensions: ${model.dimensions}")

    fun encode(text: String): FloatArray = model.embed(text)

    // ── Section 1: Single Embedding Sanity Check ──
    println()
    println("=== 1. Single Embedding Sanity Check ===")
    val query1 = "What is artificial intelligence?"
    val emb1 = encode(query1)
    println("Query: \"$query1\"")
    println("Embedding dim: ${emb1.size} (expected: ${model.dimensions})")
    if (emb1.size != model.dimensions) {
        println("FAIL: dimension mismatch!")
        exitProcess(1)
    }
    val nonZero = emb1.count { it != 0f }
    val allFinite = emb1.all { it.isFinite() }
    val min = emb1.min()
    val max = emb1.max()
    val mean = emb1.map { it.toDouble() }.average()
    println(String.format(Locale.ROOT, "Non-zero: %d/%d, All finite: %s", nonZero, emb1.size, allFinite))
    println(String.format(Locale.ROOT, "Range: [%.6f, %.6f], Mean: %.6f", min, max, mean))
    println("PASS")

    // ── Section 2: Semantic Similarity Check ──
    println()
    println("=== 2. Semantic Similarity Check ===")
    val query2 = "Define artificial intelligence"
    val query3 = "What is the weather today?"
    val emb2 = encode(query2)
    val emb3 = encode(query3)
    val sim12 = cosineSimilarity(emb1, emb2)
    val sim13 = cosineSimilarity(emb1, emb3)
    println(String.format(Locale.ROOT, "sim(\"%s\", \"%s\") = %.4f", query1, query2, sim12))
    println(String.format(Locale.ROOT, "sim(\"%s\", \"%s\") = %.4f", query1, query3, sim13))
    if (sim12 > sim13) {
        println("PASS: related pair scores higher than unrelated pair")
    } else {
        println("FAIL: expected related pair to score higher")
    }

    // ── Section 3: Small IR Retrieval Test ──
    println()
    println("=== 3. Small IR Retrieval Test ===")
    val baseQuery = "MongoDB is a NoSQL database"
    val candidates = listOf(
        "MongoDB stores data in documents",
        "PostgreSQL is a relational database",
        "The cat sat on the mat",
        "NoSQL databases are non-relational",
        "MongoDB uses BSON format"
    )
    val baseEmb = encode(baseQuery)
    println("Base query: \"$baseQuery\"")
    println()

    val scores = candidates.mapIndexed { idx, doc ->
        val docEmb = encode(doc)
        val sim = cosineSimilarity(baseEmb, docEmb)
        println(String.format(Locale.ROOT, "  [%d] sim=%.4f  %s", idx, sim, doc))
        idx to sim
    }
    val ranked = scores.sortedByDescending { it.second }
    val bestIdx = ranked.first().first
    println()
    println(String.format(Locale.ROOT, "Best match: [%d] score=%.4f", bestIdx, ranked.first().second))
    val mongoRelated = setOf(0, 3, 4)
    if (bestIdx in mongoRelated) {
        println("PASS: best match is MongoDB-related")
    } else {
        println("FAIL: expected a MongoDB-related doc to rank first")
    }

    // ── Section 4: Performance / Throughput Test ──
    println()
    println("=== 4. Performance / Throughput Test ($iterations iterations) ===")
    val elapsed = measureTime {
        for (j in 0 until iterations) {
            encode("$query1 $j")
        }
    }
    val avgMs = elapsed.inWholeNanoseconds / 1_000_000.0 / iterations
    println(String.format(Locale.ROOT, "Total: %s for %d embeddings", elapsed, iterations))
    println(String.format(Locale.ROOT, "Avg: %.3f ms/embedding", avgMs))

    println()
    println("=== Demo Complete ===")
}
