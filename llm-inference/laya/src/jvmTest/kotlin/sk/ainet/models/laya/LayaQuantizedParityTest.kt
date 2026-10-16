package sk.ainet.models.laya

import kotlinx.serialization.json.jsonObject
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.tensor.storage.PackedBlockStorage
import java.nio.file.Files
import kotlin.math.abs
import kotlin.test.Test
import kotlin.test.assertTrue
import kotlin.test.fail

/**
 * The decision-quality gate of the packed Q8_0 lane (SKEEP-007 Tier 2 consumer).
 *
 * FP32 parity (`LayaPredictFixtureTest`, 1.5e-4) is not the right contract for a quantized model;
 * what has to hold is that it **decides the same way**: every choice the FP32 agent makes on the
 * `predict.json` cases, the Q8_0 agent makes too, and no probability, score, noul value or
 * confidence moves by more than [maxProbabilityShift]. Measured on the multilingual checkpoint on
 * 2026-10-16: 5/5 choices agree, 45 values compared, max shift 0.035 — the gate sits at 0.05 so a
 * regression of the lane (a wrong block order, a mis-split projection) fails loudly while the
 * quantization noise of a 22-layer encoder passes. Needs `LAYA_MULTILINGUAL_PATH` with
 * `model.safetensors`; skipped otherwise, like the other model-gated tests.
 */
class LayaQuantizedParityTest {
    private val maxProbabilityShift = System.getenv("LAYA_Q8_TOL")?.toDoubleOrNull() ?: 0.05

    @Test
    fun `q8_0 lane decides like the fp32 lane on the predict fixtures`() {
        val dir = LayaFixtures.checkpointDir() ?: return
        if (!Files.exists(dir.resolve("model.safetensors"))) { println("[skip] $dir has no model.safetensors"); return }

        val t0 = System.nanoTime()
        val fp32 = LayaModelLoader.load(dir, DirectCpuExecutionContext())
        val t1 = System.nanoTime()
        val q8 = LayaModelLoader.load(dir, DirectCpuExecutionContext(), weightFormFor = LayaWeightForms.q8_0For)
        val t2 = System.nanoTime()
        println("[laya-q8] fp32 loaded in ${(t1 - t0) / 1_000_000} ms, q8_0 in ${(t2 - t1) / 1_000_000} ms")

        val packedBytes = q8.model.packedWeightBytes()
        assertTrue(packedBytes > 0, "the q8_0 lane must hold packed weights")
        println("[laya-q8] packed weight bytes: ${packedBytes / (1024 * 1024)} MB")

        val fixture = LayaFixtures.resource("predict.json")
        val failures = mutableListOf<String>()
        var choices = 0
        var agreed = 0
        var compared = 0
        var maxShift = 0.0
        var fp32Ms = 0L
        var q8Ms = 0L
        for ((name, element) in fixture["single"]!!.jsonObject) {
            val case = element.jsonObject
            val state = LayaState.of(case["state"]!!)
            val questions = case["questions"]!!.jsonObject.mapValues { (_, q) -> LayaFixtures.questionOf(q.jsonObject) }
            val a = System.nanoTime()
            val reference = fp32.decideAll(state, questions)
            val b = System.nanoTime()
            val quantized = q8.decideAll(state, questions)
            val c = System.nanoTime()
            fp32Ms += (b - a) / 1_000_000
            q8Ms += (c - b) / 1_000_000
            for ((qid, ref) in reference) {
                val got = quantized.getValue(qid).answer
                val exp = ref.answer
                fun shift(what: String, g: Double, e: Double) {
                    compared++
                    val d = abs(g - e)
                    if (d > maxShift) maxShift = d
                    if (d > maxProbabilityShift) failures += "$name/$qid $what: q8 ${"%.4f".format(g)} vs fp32 ${"%.4f".format(e)} (shift ${"%.3f".format(d)})"
                }
                when (exp) {
                    is LayaAnswer.Choice -> {
                        choices++
                        val g = got as? LayaAnswer.Choice ?: run { failures += "$name/$qid: answer kind differs"; null }
                        if (g != null) {
                            if (g.choice == exp.choice) agreed++ else failures += "$name/$qid choice: q8 ${g.choice} vs fp32 ${exp.choice}"
                            for ((key, p) in exp.byKey) shift("p[$key]", g.byKey.getValue(key), p)
                        }
                    }
                    is LayaAnswer.Score -> {
                        val g = got as LayaAnswer.Score
                        shift("score", g.score, exp.score)
                        for (i in exp.probabilities.indices) shift("p[$i]", g.probabilities[i], exp.probabilities[i])
                    }
                    is LayaAnswer.Noul -> shift("noul", (got as LayaAnswer.Noul).noul, exp.noul)
                }
                shift("confidence", got.confidence, exp.confidence)
            }
        }
        println("[laya-q8] $compared values compared, max shift ${"%.4f".format(maxShift)}, choices $agreed/$choices agree; fp32 $fp32Ms ms, q8_0 $q8Ms ms")
        if (choices > 0 && agreed < choices) failures += "choice agreement $agreed/$choices below the gate (every choice must agree)"
        if (failures.isNotEmpty()) fail(failures.joinToString("\n"))
    }
}

/** Sum of the packed bytes held by a decision model's parameters, to prove the lane is in effect. */
private fun LayaDecisionModel<*>.packedWeightBytes(): Long =
    (encoder.trainableParameters() + head.trainableParameters())
        .sumOf { p -> (p.value.data as? PackedBlockStorage)?.packedData?.size?.toLong() ?: 0L }
