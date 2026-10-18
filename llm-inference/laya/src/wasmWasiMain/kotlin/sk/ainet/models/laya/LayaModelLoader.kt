package sk.ainet.models.laya

import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.memory.plan.WeightForm
import sk.ainet.lang.types.FP32

public actual object LayaModelLoader {
    public actual suspend fun load(
        files: LayaCheckpointFiles,
        weights: () -> RandomAccessSource,
        ctx: ExecutionContext,
        exactGelu: Boolean,
        weightForm: WeightForm?,
        weightFormFor: ((String) -> WeightForm?)?,
    ): LayaAgent<FP32> = loadLayaAgent(files, weights, ctx, exactGelu, weightForm, weightFormFor)
}
