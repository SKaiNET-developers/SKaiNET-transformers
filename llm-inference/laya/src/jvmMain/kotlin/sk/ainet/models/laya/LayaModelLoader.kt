package sk.ainet.models.laya

import kotlinx.coroutines.runBlocking
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.types.FP32
import java.nio.file.Files
import java.nio.file.Path

public actual object LayaModelLoader {
    public actual suspend fun load(
        files: LayaCheckpointFiles,
        weights: () -> RandomAccessSource,
        ctx: ExecutionContext,
        exactGelu: Boolean,
        weightForm: WeightForm?,
        weightFormFor: ((String) -> WeightForm?)?,
    ): LayaAgent<FP32> = loadLayaAgent(files, weights, ctx, exactGelu, weightForm, weightFormFor)

    /**
     * Loads a checkpoint directory (`rl_agent_config.json`, `encoder/config.json`, `tokenizer/`,
     * `model.safetensors`) into a [LayaAgent]: the JVM convenience over the common [load].
     * Blocking; call off the UI thread.
     *
     * [weightFormFor] selects the lane: `null` is the FP32 reference path (REQ-EXE-008, F16 up-cast on
     * load), [LayaWeightForms.q8_0For] the packed Q8_0 lane. `@JvmOverloads` keeps the 0.61
     * `load(Path, ExecutionContext, boolean)` signature for Java callers.
     */
    @JvmOverloads
    public fun load(
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
}
