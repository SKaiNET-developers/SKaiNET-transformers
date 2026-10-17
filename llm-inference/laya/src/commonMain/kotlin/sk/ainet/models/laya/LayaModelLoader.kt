package sk.ainet.models.laya

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.memory.plan.EncodingRequest
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.memory.plan.WeightShapeOrientation
import sk.ainet.lang.tensor.storage.TensorEncoding
import sk.ainet.lang.types.FP32

/**
 * Loads a Laya checkpoint — the four side files as text ([LayaCheckpointFiles]) and
 * `model.safetensors` through a [RandomAccessSource] — into a [LayaAgent].
 *
 * Common code since engine 0.61 (SKEEP-007): the JVM, Android and browser builds all run this
 * function; only how the bytes are obtained differs (a file, an app's files directory, a download
 * held in memory). The FP32 reference path (REQ-EXE-008) is the default: F16 weights are up-cast
 * on load, which is what the Python reference does on CPU.
 *
 * [weightForm] / [weightFormFor] are passed through to the engine's SafeTensors loader. With
 * [LayaWeightForms.q8_0For] the encoder and head projections and the token table are quantized to
 * Q8_0 at load — about a quarter of the FP32 bytes, served by the packed kernels on every tier —
 * and the decision-quality gate (`LayaQuantizedParityTest`) is the contract that lane keeps.
 */
public object LayaModelLoader {
    public suspend fun load(
        files: LayaCheckpointFiles,
        weights: () -> RandomAccessSource,
        ctx: ExecutionContext,
        exactGelu: Boolean = true,
        weightForm: WeightForm? = null,
        weightFormFor: ((String) -> WeightForm?)? = null,
    ): LayaAgent<FP32> {
        val preprocessor = LayaPreprocessor.fromFiles(files)
        val raw = ArrayList<WeightTensor<FP32, Float>>()
        SafeTensorsParametersLoader(sourceProvider = weights, weightForm = weightForm, weightFormFor = weightFormFor)
            .load<FP32, Float>(ctx, FP32::class) { name, tensor ->
                raw += WeightTensor(name, tensor.shape.dimensions.toList(), tensor)
            }
        val prepared = LayaWeightPreparation.prepare(raw, preprocessor.encoderConfig, preprocessor.agentConfig, ctx, FP32::class)
        val model = LayaDecisionModel.build(preprocessor.encoderConfig, preprocessor.agentConfig, prepared, ctx, FP32::class, exactGelu)
        return LayaAgent(preprocessor, model)
    }
}

/**
 * Weight forms for a Laya checkpoint (SKEEP-007 Tier 2 consumer).
 *
 * The matrices that carry the compute — every encoder projection (`Wqkv`, `Wo`, `Wi`, `mlp.Wo`),
 * the head's attention and feed-forward weights — and the 256k × 768 token table are the ones
 * worth packing; norms, biases, the three-row type embedding and the two scorer layers stay dense,
 * where they cost nothing and where a packed form would only complicate the small ops that read
 * them (`narrow` on the type embedding, the final Linear 768→1).
 */
public object LayaWeightForms {
    private val q8_0: WeightForm = WeightForm(
        encoding = EncodingRequest.RequantizeTo(TensorEncoding.Q8_0),
        shape = WeightShapeOrientation.OUT_IN,
    )

    private val packedSuffixes = listOf(
        ".attn.Wqkv.weight", ".attn.Wo.weight", ".mlp.Wi.weight", ".mlp.Wo.weight",
        ".self_attn.in_proj_weight", ".self_attn.out_proj.weight", ".linear1.weight", ".linear2.weight",
    )

    /** `true` for the tensors [q8_0For] packs. */
    public fun isPacked(name: String): Boolean =
        name == "encoder.embeddings.tok_embeddings.weight" || packedSuffixes.any { name.endsWith(it) }

    /** Per-tensor form: Q8_0 for the projection matrices and the token table, dense for everything else. */
    public val q8_0For: (String) -> WeightForm? = { name -> if (isPacked(name)) q8_0 else null }
}
