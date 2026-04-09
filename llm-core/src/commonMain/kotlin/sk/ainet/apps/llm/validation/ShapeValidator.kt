package sk.ainet.apps.llm.validation

import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.ResidualAdd
import sk.ainet.lang.nn.transformer.SwiGLUFFN
import sk.ainet.lang.types.DType

/**
 * Dry-run shape validation for DSL module pipelines.
 *
 * Propagates symbolic shapes through the module tree without executing
 * any computation. Catches dimension mismatches at construction/load time
 * rather than deep into a forward pass.
 *
 * Also identifies optimization entry points (fusible ops, quantization
 * candidates, layout transitions).
 *
 * Usage:
 * ```
 * val result = ShapeValidator.validate(model, inputShape = listOf(seqLen, dim))
 * if (!result.isValid) {
 *     result.errors.forEach { println("SHAPE ERROR: ${it.message}") }
 * }
 * result.optimizationHints.forEach { println("HINT: ${it.description}") }
 * ```
 */
public object ShapeValidator {

    /**
     * Result of a shape validation pass.
     */
    public data class ValidationResult(
        val errors: List<ShapeError>,
        val trace: List<ShapeStep>,
        val optimizationHints: List<OptimizationHint>
    ) {
        val isValid: Boolean get() = errors.isEmpty()

        fun printSummary() {
            if (isValid) {
                println("Shape validation passed (${trace.size} modules)")
            } else {
                println("Shape validation FAILED (${errors.size} errors):")
                errors.forEach { println("  ERROR: ${it.message}") }
            }
            if (optimizationHints.isNotEmpty()) {
                println("Optimization hints:")
                optimizationHints.forEach { println("  ${it.type}: ${it.description}") }
            }
        }
    }

    public data class ShapeError(
        val moduleName: String,
        val expected: String,
        val actual: String,
        val message: String
    )

    public data class ShapeStep(
        val moduleName: String,
        val moduleType: String,
        val inputShape: List<Int>,
        val outputShape: List<Int>,
        val paramShapes: Map<String, List<Int>>
    )

    public enum class HintType {
        FUSION, QUANTIZATION, LAYOUT, MEMORY
    }

    public data class OptimizationHint(
        val moduleName: String,
        val type: HintType,
        val description: String
    )

    /**
     * Validate a module tree by propagating shapes through it.
     *
     * @param model The root module
     * @param inputShape The input tensor shape (e.g., [seqLen, dim])
     * @return Validation result with errors, trace, and optimization hints
     */
    public fun validate(
        model: Module<*, *>,
        inputShape: List<Int>
    ): ValidationResult {
        val errors = mutableListOf<ShapeError>()
        val trace = mutableListOf<ShapeStep>()
        val hints = mutableListOf<OptimizationHint>()

        propagate(model, inputShape, errors, trace, hints)

        return ValidationResult(errors, trace, hints)
    }

    /**
     * Validate that loaded weight shapes match module parameter expectations.
     *
     * @param model Module tree with weights loaded
     * @return List of shape mismatches between parameters and expected dimensions
     */
    public fun validateWeights(model: Module<*, *>): List<ShapeError> {
        val errors = mutableListOf<ShapeError>()
        walkModules(model) { module ->
            val rule = ShapeInferenceRegistry.getRule(module::class)
            rule?.validateParams(module, errors)
        }
        return errors
    }

    private fun propagate(
        module: Module<*, *>,
        inputShape: List<Int>,
        errors: MutableList<ShapeError>,
        trace: MutableList<ShapeStep>,
        hints: MutableList<OptimizationHint>
    ): List<Int> {
        val rule = ShapeInferenceRegistry.getRule(module::class)

        // Check input constraints
        rule?.checkInput(module, inputShape)?.let { error ->
            errors.add(error)
        }

        // Collect param shapes for diagnostics
        val paramShapes = collectParamShapes(module)

        // Collect optimization hints
        rule?.collectHints(module, inputShape, hints)

        // If this module has children, propagate through them sequentially
        val children = module.modules
        if (children.isNotEmpty()) {
            var currentShape = inputShape
            for (child in children) {
                currentShape = propagate(child, currentShape, errors, trace, hints)
            }
            val step = ShapeStep(
                moduleName = module.name,
                moduleType = module::class.simpleName ?: "?",
                inputShape = inputShape,
                outputShape = currentShape,
                paramShapes = paramShapes
            )
            trace.add(step)
            return currentShape
        }

        // Leaf module: infer output shape
        val outputShape = rule?.inferOutput(module, inputShape) ?: inputShape

        val step = ShapeStep(
            moduleName = module.name,
            moduleType = module::class.simpleName ?: "?",
            inputShape = inputShape,
            outputShape = outputShape,
            paramShapes = paramShapes
        )
        trace.add(step)

        return outputShape
    }

    private fun collectParamShapes(@Suppress("UNUSED_PARAMETER") module: Module<*, *>): Map<String, List<Int>> {
        // Parameter shape collection requires modules to expose shapes explicitly
        // (e.g. via a ModuleParameters interface). No reflection-based fallback.
        return emptyMap()
    }

    private fun walkModules(module: Module<*, *>, visitor: (Module<*, *>) -> Unit) {
        visitor(module)
        for (child in module.modules) {
            walkModules(child, visitor)
        }
    }
}
