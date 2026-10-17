package sk.ainet.models.laya

import kotlinx.coroutines.runBlocking
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.types.FP32
import java.nio.file.Files
import java.nio.file.Path

/**
 * Loads a checkpoint directory (`rl_agent_config.json`, `encoder/config.json`, `tokenizer/`,
 * `model.safetensors`) into a [LayaAgent] — the JVM convenience over the common
 * [LayaModelLoader.load]. Blocking; call off the UI thread.
 *
 * [weightFormFor] selects the lane: `null` is the FP32 reference path (REQ-EXE-008, F16 up-cast on
 * load), [LayaWeightForms.q8_0For] the packed Q8_0 lane.
 */
public fun LayaModelLoader.load(
    directory: Path,
    ctx: ExecutionContext,
    exactGelu: Boolean = true,
    weightFormFor: ((String) -> WeightForm?)? = null,
): LayaAgent<FP32> {
    val weights = directory.resolve("model.safetensors")
    require(Files.exists(weights)) { "no model.safetensors in $directory" }
    val files = LayaCheckpointFiles.read(directory)
    return runBlocking {
        load(
            files = files,
            weights = { JvmRandomAccessSource.open(weights.toString()) },
            ctx = ctx,
            exactGelu = exactGelu,
            weightFormFor = weightFormFor,
        )
    }
}
