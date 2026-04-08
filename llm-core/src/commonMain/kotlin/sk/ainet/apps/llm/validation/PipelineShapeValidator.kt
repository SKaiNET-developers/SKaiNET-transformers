package sk.ainet.apps.llm.validation

import sk.ainet.lang.tensor.Shape

/**
 * Validates shape compatibility across a multi-component inference pipeline.
 *
 * This is model-agnostic: it validates the data flow between pipeline stages
 * (backbone → projections → acoustic → codec) using only shape information.
 *
 * Usage:
 * ```
 * val validator = PipelineShapeValidator()
 * validator.stage("backbone.output", hiddenStates.shape)
 * validator.projection("acoustic.inputProj", inputProjShape, transpose = true)
 * validator.stage("acoustic.output", acousticOutputShape)
 * validator.matchCount("semantic_tokens", semanticCount, "acoustic_frames", acousticFrames)
 * val result = validator.validate()
 * ```
 */
public class PipelineShapeValidator {

    private data class StageInfo(
        val name: String,
        val shape: List<Int>
    )

    private data class ProjectionInfo(
        val name: String,
        val weightShape: List<Int>,
        val transpose: Boolean
    )

    private data class CountMatch(
        val nameA: String,
        val countA: Int,
        val nameB: String,
        val countB: Int
    )

    private val stages = mutableListOf<StageInfo>()
    private val projections = mutableListOf<Pair<String, ProjectionInfo>>() // afterStage → projection
    private val countMatches = mutableListOf<CountMatch>()
    private var lastStageName: String? = null

    /**
     * Record a pipeline stage with its output shape.
     */
    public fun stage(name: String, shape: Shape): PipelineShapeValidator {
        stages.add(StageInfo(name, shape.dimensions.toList()))
        lastStageName = name
        return this
    }

    /**
     * Record a pipeline stage with its output shape as a list.
     */
    public fun stage(name: String, shape: List<Int>): PipelineShapeValidator {
        stages.add(StageInfo(name, shape))
        lastStageName = name
        return this
    }

    /**
     * Record a projection (matmul) between the last stage and the next.
     * @param weightShape Shape of the projection weight matrix
     * @param transpose Whether the weight is transposed before matmul
     */
    public fun projection(name: String, weightShape: List<Int>, transpose: Boolean = false): PipelineShapeValidator {
        val after = lastStageName ?: "input"
        projections.add(after to ProjectionInfo(name, weightShape, transpose))
        return this
    }

    /**
     * Validate that two counts match (e.g., semantic tokens == acoustic frames).
     */
    public fun matchCount(nameA: String, countA: Int, nameB: String, countB: Int): PipelineShapeValidator {
        countMatches.add(CountMatch(nameA, countA, nameB, countB))
        return this
    }

    /**
     * Run all validations and return results.
     */
    public fun validate(): ValidationResult {
        val errors = mutableListOf<String>()
        val trace = mutableListOf<String>()

        // Validate projections
        for ((afterStage, proj) in projections) {
            val stage = stages.find { it.name == afterStage }
            if (stage != null) {
                val stageLastDim = stage.shape.lastOrNull()
                val projInputDim = if (proj.transpose) proj.weightShape.lastOrNull() else proj.weightShape.firstOrNull()

                trace.add("${stage.name} ${stage.shape} → ${proj.name} ${proj.weightShape}${if (proj.transpose) "^T" else ""}")

                if (stageLastDim != null && projInputDim != null && stageLastDim != projInputDim) {
                    errors.add(
                        "${proj.name}: matmul dim mismatch — " +
                            "${stage.name} outputs dim=$stageLastDim but " +
                            "${proj.name} expects dim=$projInputDim " +
                            "(weight shape ${proj.weightShape}${if (proj.transpose) " transposed" else ""})"
                    )
                }
            }
        }

        // Validate count matches
        for (match in countMatches) {
            trace.add("${match.nameA}=${match.countA} ↔ ${match.nameB}=${match.countB}")
            if (match.countA != match.countB) {
                errors.add(
                    "Frame count mismatch: ${match.nameA}=${match.countA} " +
                        "but ${match.nameB}=${match.countB}"
                )
            }
        }

        return ValidationResult(errors, trace)
    }

    public data class ValidationResult(
        val errors: List<String>,
        val trace: List<String>
    ) {
        val isValid: Boolean get() = errors.isEmpty()

        fun printSummary(prefix: String = "  ") {
            if (isValid) {
                println("${prefix}Pipeline shape validation passed")
            } else {
                println("${prefix}Pipeline shape validation FAILED:")
                errors.forEach { println("${prefix}  ERROR: $it") }
            }
            trace.forEach { println("${prefix}  $it") }
        }
    }
}
