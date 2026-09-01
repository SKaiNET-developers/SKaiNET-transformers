package sk.ainet.apps.kbitnet

import kotlinx.io.Source
import sk.ainet.context.ExecutionContext
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.I2sGgufLayout
import sk.ainet.lang.types.FP32
import sk.ainet.models.bitnet.BITNET_ARCHITECTURES
import sk.ainet.models.bitnet.BitNetRuntimeWeights
import sk.ainet.models.bitnet.BitNetWeightLoader
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader

/**
 * Thin facade around the BitNet loaders (#346's `<F>Ingestion` row, transformers#359) — the
 * family's runtime entry point, mirroring [sk.ainet.apps.kapertus.ApertusIngestion].
 *
 * The primary path is [loadStreaming]: the packed I2_S load through the engine loader —
 * ternary projections stay `BITNET_B1_58` (0.25 B/weight) and the lm_head arrives as
 * `BITNET_PLANES` (both the `output.weight` and tied-2B4T lanes), which is what gates the
 * two-stage decode. [load] is the sequential, fully-dequantizing fallback for small files.
 */
public class BitNetIngestion(
    private val ctx: ExecutionContext,
) {

    /**
     * Load BitNet runtime weights from a GGUF via streaming random access — any size, packed.
     *
     * @param i2sLayout the converter flavor of the file's I2_S payloads (BitNet.cpp x86 =
     *   `GROUP_128`, ARM = `GROUP_64`, NeoGPU = `SEQUENTIAL`) — see the engine's `I2sGgufLayout`
     * @param planesLmHead serve the lm_head as `BITNET_PLANES` (default; pass `false` for the
     *   exact dense head)
     */
    public suspend fun loadStreaming(
        randomAccessProvider: () -> RandomAccessSource,
        i2sLayout: I2sGgufLayout = I2sGgufLayout.GROUP_128,
        planesLmHead: Boolean = true,
    ): BitNetRuntimeWeights = BitNetWeightLoader.loadRuntimeWeights(
        ctx = ctx,
        sourceProvider = randomAccessProvider,
        i2sLayout = i2sLayout,
        planesLmHead = planesLmHead,
    )

    /**
     * Load BitNet runtime weights from a sequential GGUF source — models under 2GB, always
     * dequantizes to dense FP32 (the exact baseline; no packed tensors, no planes head).
     */
    public suspend fun load(sourceProvider: () -> Source): BitNetRuntimeWeights {
        val weights = DecoderGgufWeightLoader(
            sourceProvider = sourceProvider,
            acceptedArchitectures = BITNET_ARCHITECTURES,
        ).loadToMap<FP32, Float>(ctx)
        return BitNetRuntimeWeights(weights.metadata, weights.tensors)
    }
}
