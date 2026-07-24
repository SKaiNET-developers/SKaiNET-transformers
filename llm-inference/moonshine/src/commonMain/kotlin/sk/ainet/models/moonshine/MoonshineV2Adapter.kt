package sk.ainet.models.moonshine

import sk.ainet.context.ExecutionContext
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.hooks.withForwardHooks
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.nn.layers.EmbeddingAdapter
import sk.ainet.lang.nn.normalization.LayerNormalization
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.Tensor
import sk.ainet.lang.tensor.VoidOpsTensor
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.tensor.operators.bind
import sk.ainet.lang.types.DType
import kotlin.reflect.KClass

/**
 * Moonshine **v2 adapter** — bridges the **position-free** v2 encoder to the **position-aware** decoder by
 * injecting a learned absolute positional embedding, then normalizing (Moonshine v2 paper: "an adapter layer
 * bridges the position-free encoder to a position-aware decoder by injecting learned positional embeddings").
 *
 * ```
 *   adapted[b, t, :] = LayerNorm( encoderMemory[b, t, :] + posEmbed[t, :] )
 * ```
 *
 * The decoder's cross-attention then consumes `adapted` exactly as it consumes v1's RoPE encoder memory
 * (`MoonshineDecoderLayer.forward(input, encoderMemory, ctx)`), so no decoder change is needed.
 *
 * NOTE: the exact adapter form is not yet confirmed against a released v2 checkpoint — the paper names a
 * learned positional embedding; a projection or gating may also be present. Verify + bake once weights exist.
 * dtype-portable like the encoder/decoder (the element type flows through).
 */
public class MoonshineV2Adapter<T : DType, V>(
    cfg: MoonshineV2Config,
    /** Encoder memory length (frames after the conv frontend). Sizes the positional table. */
    maxFrames: Int,
    private val dtype: KClass<T>,
) : Module<T, V>() {

    override val name: String = "v2_adapter"

    private val dim = cfg.dim

    // Learned absolute positional embedding [maxFrames, dim], wrapped as a Module<T, V>. Void-initialised so
    // the module traces to StableHLO before real weights load (the DSL `embedding` builder's placeholder pattern).
    private val posEmbed = EmbeddingAdapter(
        Embedding<T, V>(
            numEmbeddings = maxFrames,
            embeddingDim = dim,
            initWeight = voidWeight(maxFrames, dim),
            name = "v2_adapter.pos_embed",
        ),
    )

    private val norm = LayerNormalization<T, V>(
        normalizedShape = intArrayOf(dim),
        eps = cfg.layerNormEps.toDouble(),
        name = "v2_adapter.norm",
        dtype = dtype,
    )

    override val modules: List<Module<T, V>> = listOf(posEmbed, norm)

    /**
     * [encoderMemory] = position-free encoder output `[·, frames, dim]`;
     * [positions] = frame indices `[·, frames]` as (float-encoded) ids 0..frames-1.
     * Returns position-aware memory `[·, frames, dim]` for the decoder's cross-attention.
     */
    public fun forward(
        encoderMemory: Tensor<T, V>,
        positions: Tensor<T, V>,
        ctx: ExecutionContext,
    ): Tensor<T, V> {
        val mem = encoderMemory.bind(ctx)
        return withForwardHooks(ctx, this, mem) {
            val pos = posEmbed.forward(positions, ctx)             // [·, frames, dim]
            norm.forward(ctx.ops.add(mem, pos), ctx)
        }
    }

    @Suppress("UNCHECKED_CAST")
    private fun voidWeight(rows: Int, cols: Int): Tensor<T, V> = VoidOpsTensor(
        object : TensorData<T, V> {
            override val shape = Shape(rows, cols)
            override fun get(vararg indices: Int): V = 0.0f as V
            override fun set(vararg indices: Int, value: V) {}
        },
        dtype,
    )
}
