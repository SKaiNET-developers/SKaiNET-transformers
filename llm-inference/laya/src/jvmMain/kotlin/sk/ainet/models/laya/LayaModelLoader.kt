package sk.ainet.models.laya

import kotlinx.coroutines.runBlocking
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.io.weights.WeightTensor
import sk.ainet.lang.types.FP32
import java.nio.file.Files
import java.nio.file.Path

/**
 * Loads a checkpoint directory (`rl_agent_config.json`, `encoder/config.json`, `tokenizer/`,
 * `model.safetensors`) into a [LayaAgent] on the FP32 CPU reference path (REQ-EXE-008). F16
 * weights are up-cast on load, which is what the Python reference does on CPU.
 */
public object LayaModelLoader {

    public fun load(directory: Path, ctx: ExecutionContext, exactGelu: Boolean = true): LayaAgent<FP32> {
        val preprocessor = LayaPreprocessor.fromFiles(LayaCheckpointFiles.read(directory))
        val weights = directory.resolve("model.safetensors")
        require(Files.exists(weights)) { "no model.safetensors in $directory" }
        val raw = ArrayList<WeightTensor<FP32, Float>>()
        runBlocking {
            SafeTensorsParametersLoader(sourceProvider = { JvmRandomAccessSource.open(weights.toString()) })
                .load<FP32, Float>(ctx, FP32::class) { name, tensor ->
                    raw += WeightTensor(name, tensor.shape.dimensions.toList(), tensor)
                }
        }
        val prepared = LayaWeightPreparation.prepare(raw, preprocessor.encoderConfig, preprocessor.agentConfig, ctx, FP32::class)
        val model = LayaDecisionModel.build(preprocessor.encoderConfig, preprocessor.agentConfig, prepared, ctx, FP32::class, exactGelu)
        return LayaAgent(preprocessor, model)
    }
}
