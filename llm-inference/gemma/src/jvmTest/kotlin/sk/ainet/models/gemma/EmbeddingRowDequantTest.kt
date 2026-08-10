package sk.ainet.models.gemma

import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.lang.nn.RowDequantSource
import sk.ainet.lang.nn.layers.Embedding
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int32
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Regression lock for the packed token-embedding gather path: the core [Embedding] layer must route a
 * [RowDequantSource]-backed weight through per-row dequant — dequantising ONLY the rows actually looked up
 * this step — instead of materialising the whole table to FP32. That is what keeps FunctionGemma's Q8_0
 * `token_embd` at its ~178 MB packed footprint on the 1.9 GB board instead of inflating to ~0.67 GB and OOMing
 * (see `GemmaPackedWeights.dequantNoTranspose`, which wraps row-sliceable Q-quant token_embd as a
 * [GemmaPerLayerTokenEmbedTensorData] — a `RowDequantSource`).
 *
 * The spy weight FAILS LOUDLY on any full-table access (`get`), so if [Embedding] ever regressed to the dense
 * path this test would throw rather than silently pass. Lives in the gemma module for the `DirectCpuExecutionContext`
 * test infra; the behaviour under test is the shared `transformer-core` [Embedding].
 */
class EmbeddingRowDequantTest {

    /** A [RowDequantSource] weight that records which rows were dequantised and throws if fully materialised. */
    private class SpyRowEmbed(rows: Int, private val cols: Int) : TensorData<FP32, Float>, RowDequantSource {
        override val shape: Shape = Shape(rows, cols)
        val touched = mutableListOf<Int>()
        override fun dequantRow(rowIdx: Int): FloatArray {
            touched.add(rowIdx)
            return FloatArray(cols) { (rowIdx * 100 + it).toFloat() }   // deterministic, row-distinct
        }
        override fun get(vararg indices: Int): Float =
            error("full-table materialisation (get) — Embedding must use dequantRow for a RowDequantSource")
        override fun set(vararg indices: Int, value: Float): Unit = error("read-only")
    }

    @Test
    fun embeddingGathersRowDequantOnlyForLookedUpRows() {
        val ctx = DirectCpuExecutionContext()
        val rows = 4
        val cols = 8
        val spy = SpyRowEmbed(rows, cols)
        val weight = ctx.fromData(spy, FP32::class)
        val emb = Embedding<FP32, Float>(
            numEmbeddings = rows,
            embeddingDim = cols,
            initWeight = weight,
            name = "tok_embd",
        )

        val lookup = intArrayOf(1, 3, 3)   // includes a repeat; row 0/2 never touched
        val out = emb.forward(ctx.fromIntArray(Shape(lookup.size), Int32::class, lookup), ctx)
        val got = out.data.copyToFloatArray()   // [lookup.size, cols]

        // 1) each output row equals that token's dequantRow — the gather is correct.
        for ((r, token) in lookup.withIndex()) {
            for (c in 0 until cols) {
                assertEquals((token * 100 + c).toFloat(), got[r * cols + c], "row $r (token $token) col $c")
            }
        }
        // 2) ONLY the looked-up rows were dequantised — the whole table was never materialised
        //    (and `get` would have thrown if it had been).
        assertEquals(listOf(1, 3, 3), spy.touched, "one dequantRow per gathered index, no full-table pass")
    }
}
