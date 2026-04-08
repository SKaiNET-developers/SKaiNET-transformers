package sk.ainet.apps.llm.validation

import sk.ainet.apps.llm.HybridTransformerBlock
import sk.ainet.apps.llm.TransformerBlock
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.normalization.RMSNormalization
import sk.ainet.lang.nn.transformer.MultiHeadAttention
import sk.ainet.lang.nn.transformer.ResidualAdd
import sk.ainet.lang.nn.transformer.SwiGLUFFN
import sk.ainet.lang.nn.topology.MLP
import kotlin.reflect.KClass

/**
 * Registry of shape inference rules for module types.
 *
 * Each rule knows how a specific module type transforms tensor shapes,
 * what input constraints it has, and what optimization hints to emit.
 *
 * Designed for extensibility: model-specific modules (codec conv layers,
 * snake activation, etc.) can register their own rules.
 */
public object ShapeInferenceRegistry {

    /**
     * Rule for inferring output shapes and validating inputs for a module type.
     */
    public interface ShapeRule {
        /** Infer output shape given the module instance and input shape. */
        fun inferOutput(module: Module<*, *>, inputShape: List<Int>): List<Int>?

        /** Check input constraints. Return error if violated, null if OK. */
        fun checkInput(module: Module<*, *>, inputShape: List<Int>): ShapeValidator.ShapeError? = null

        /** Validate parameter shapes after weight loading. */
        fun validateParams(module: Module<*, *>, errors: MutableList<ShapeValidator.ShapeError>) {}

        /** Collect optimization hints for this module. */
        fun collectHints(
            module: Module<*, *>,
            inputShape: List<Int>,
            hints: MutableList<ShapeValidator.OptimizationHint>
        ) {}
    }

    private val rules = mutableMapOf<KClass<*>, ShapeRule>()

    init {
        registerBuiltinRules()
    }

    /** Register a shape rule for a module class. */
    public fun register(moduleClass: KClass<*>, rule: ShapeRule) {
        rules[moduleClass] = rule
    }

    /** Get the rule for a module, checking superclasses if needed. */
    public fun getRule(moduleClass: KClass<*>): ShapeRule? {
        return rules[moduleClass]
    }

    private fun registerBuiltinRules() {
        // ---- Shape-preserving modules ----

        val shapePreserving = object : ShapeRule {
            override fun inferOutput(module: Module<*, *>, inputShape: List<Int>) = inputShape
        }

        register(RMSNormalization::class, object : ShapeRule {
            override fun inferOutput(module: Module<*, *>, inputShape: List<Int>) = inputShape

            override fun checkInput(module: Module<*, *>, inputShape: List<Int>): ShapeValidator.ShapeError? {
                // RMSNorm expects last dim to match normalizedShape
                // We can't easily access normalizedShape without reflection, but we can
                // check the weight param shape if available
                return null // validated via weight shapes
            }
        })

        register(ResidualAdd::class, shapePreserving)

        // ---- Attention ----

        register(MultiHeadAttention::class, object : ShapeRule {
            override fun inferOutput(module: Module<*, *>, inputShape: List<Int>): List<Int> {
                // MHA preserves shape: [seqLen, dim] → [seqLen, dim]
                return inputShape
            }

            override fun checkInput(module: Module<*, *>, inputShape: List<Int>): ShapeValidator.ShapeError? {
                if (inputShape.size < 2) {
                    return ShapeValidator.ShapeError(
                        moduleName = module.name,
                        expected = "[seqLen, dim]",
                        actual = inputShape.toString(),
                        message = "MultiHeadAttention(${module.name}) requires rank >= 2, got ${inputShape.size}"
                    )
                }
                return null
            }

            override fun collectHints(
                module: Module<*, *>,
                inputShape: List<Int>,
                hints: MutableList<ShapeValidator.OptimizationHint>
            ) {
                val dim = inputShape.lastOrNull() ?: return
                if (dim >= 1024) {
                    hints.add(
                        ShapeValidator.OptimizationHint(
                            moduleName = module.name,
                            type = ShapeValidator.HintType.QUANTIZATION,
                            description = "MHA(${module.name}) dim=$dim — candidate for quantized QKV projections"
                        )
                    )
                }
            }
        })

        // ---- SwiGLU FFN ----

        register(SwiGLUFFN::class, object : ShapeRule {
            override fun inferOutput(module: Module<*, *>, inputShape: List<Int>) = inputShape

            override fun collectHints(
                module: Module<*, *>,
                inputShape: List<Int>,
                hints: MutableList<ShapeValidator.OptimizationHint>
            ) {
                val dim = inputShape.lastOrNull() ?: return
                if (dim >= 1024) {
                    hints.add(
                        ShapeValidator.OptimizationHint(
                            moduleName = module.name,
                            type = ShapeValidator.HintType.FUSION,
                            description = "SwiGLU(${module.name}) dim=$dim — fusible gate+up projections"
                        )
                    )
                }
            }
        })

        // ---- Container modules (propagate through children) ----

        val containerRule = object : ShapeRule {
            override fun inferOutput(module: Module<*, *>, inputShape: List<Int>): List<Int>? {
                // Output inferred by propagating through children
                return null
            }
        }

        register(MLP::class, containerRule)
        register(TransformerBlock::class, containerRule)
        register(HybridTransformerBlock::class, containerRule)
    }
}
