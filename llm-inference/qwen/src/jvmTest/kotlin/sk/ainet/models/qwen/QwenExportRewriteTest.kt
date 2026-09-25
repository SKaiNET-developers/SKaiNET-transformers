package sk.ainet.models.qwen

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFalse
import kotlin.test.assertTrue

/**
 * Pure (no checkpoint) tests for [QwenExportHarness]'s post-emit Vulkan-portability rewrites on
 * the exact converter output shape the harness emits (copied from a real Qwen3-0.6B
 * `qwen_with_past` export, SSA names shortened). Both rewrites were bisected against real
 * valhall4 compiles — this pins the text transformation those compiles were run on.
 */
class QwenExportRewriteTest {

    private val argMaxTail = """
        |  func.func @qwen_with_past(%arg0: tensor<1xi32>, %arg1: tensor<1x128xf32>) -> (tensor<1x8x?x128xf32>, tensor<1xi32>) {
        |    %v1 = stablehlo.dot_general %v0, %v9, contracting_dims = [1] x [0] : (tensor<1x1024xf32>, tensor<1024x151936xf32>) -> tensor<1x151936xf32>
        |    %v2 = stablehlo.constant dense<0xFF800000> : tensor<f32>
        |    %v3 = stablehlo.reduce(%v1 init: %v2) applies stablehlo.maximum across dimensions = [1] : (tensor<1x151936xf32>, tensor<f32>) -> tensor<1xf32>
        |    %v4 = stablehlo.broadcast_in_dim %v3, dims = [0] : (tensor<1xf32>) -> tensor<1x151936xf32>
        |    %v5 = stablehlo.compare EQ, %v1, %v4 : (tensor<1x151936xf32>, tensor<1x151936xf32>) -> tensor<1x151936xi1>
        |    %v6 = stablehlo.iota dim = 1 : tensor<1x151936xi32>
        |    %v7 = stablehlo.constant dense<151936> : tensor<1x151936xi32>
        |    %v8 = stablehlo.select %v5, %v6, %v7 : tensor<1x151936xi1>, tensor<1x151936xi32>
        |    %v10 = stablehlo.constant dense<151936> : tensor<i32>
        |    %v11 = stablehlo.reduce(%v8 init: %v10) applies stablehlo.minimum across dimensions = [1] : (tensor<1x151936xi32>, tensor<i32>) -> tensor<1xi32>
        |    return %k, %v11 : tensor<1x8x?x128xf32>, tensor<1xi32>
        |  }
        |""".trimMargin()

    @Test
    fun paddedExtent_isNextMultipleOf2048() {
        assertEquals(153600, QwenKvContract.paddedArgMaxExtent(151936))
        assertEquals(262144, QwenKvContract.paddedArgMaxExtent(262144))
        assertEquals(2048, QwenKvContract.paddedArgMaxExtent(1))
        assertEquals(0, 153600 % QwenKvContract.ARGMAX_REDUCTION_MULTIPLE)
    }

    @Test
    fun argMaxPad_padsLogitsAndRetargetsTheWholeTail() {
        val out = QwenExportHarness.rewriteArgMaxPadded(argMaxTail, 151936)
        assertTrue(out.contains("%v1_pad_fill = stablehlo.constant dense<-3.000000e+38> : tensor<f32>"))
        assertTrue(
            out.contains(
                "%v1_pad = stablehlo.pad %v1, %v1_pad_fill, low = [0, 0], high = [0, 1664], interior = [0, 0] : " +
                    "(tensor<1x151936xf32>, tensor<f32>) -> tensor<1x153600xf32>",
            ),
        )
        assertTrue(out.contains("stablehlo.reduce(%v1_pad init: %v2) applies stablehlo.maximum across dimensions = [1] : (tensor<1x153600xf32>"))
        assertTrue(out.contains("stablehlo.compare EQ, %v1_pad, %v4 : (tensor<1x153600xf32>, tensor<1x153600xf32>) -> tensor<1x153600xi1>"))
        assertTrue(out.contains("stablehlo.iota dim = 1 : tensor<1x153600xi32>"))
        assertTrue(out.contains("dense<153600> : tensor<1x153600xi32>"))
        assertTrue(out.contains("(tensor<1x153600xi32>, tensor<i32>) -> tensor<1xi32>"))
        // the LM head itself, the function signature and the return are untouched
        assertTrue(out.contains("-> tensor<1x151936xf32>\n    %v2 = stablehlo.constant dense<0xFF800000>"))
        assertTrue(out.contains("return %k, %v11 : tensor<1x8x?x128xf32>, tensor<1xi32>"))
        assertTrue(out.contains("-> (tensor<1x8x?x128xf32>, tensor<1xi32>) {"))
        // nothing of the old extent survives inside the tail
        val tail = out.substring(out.indexOf("%v1_pad_fill"), out.indexOf("    return"))
        assertFalse(tail.contains("1x151936xi"), "tail must not keep an unpadded i32/i1 extent:\n$tail")
        assertEquals(1, Regex("1x151936xf32>, tensor<f32>\\) -> tensor<1x153600xf32>").findAll(tail).count())
    }

    @Test
    fun argMaxPad_isANoOpForAnAlignedVocab() {
        val aligned = argMaxTail.replace("151936", "262144")
        assertEquals(aligned, QwenExportHarness.rewriteArgMaxPadded(aligned, 262144))
    }

    @Test
    fun argMaxPad_handlesEveryFunctionOfAMergedModule() {
        val merged = argMaxTail + argMaxTail.replace("@qwen_with_past", "@qwen_prefill_at")
        val out = QwenExportHarness.rewriteArgMaxPadded(merged, 151936)
        assertEquals(2, Regex("stablehlo\\.pad %v1, %v1_pad_fill").findAll(out).count())
        assertEquals(0, Regex("1x151936xi32").findAll(out).count())
    }

    private val gatherModule = """
        |module {
        |  util.global private @t0 = #flow.parameter.named<"model"::"t0"> : tensor<151936x1024xf32>
        |  func.func @qwen_with_past(%arg0: tensor<1xi32>, %arg1: tensor<1x128xf32>) -> (tensor<1xi32>) {
        |    %v0 = util.global.load @t0 : tensor<151936x1024xf32>
        |    %v311 = "stablehlo.gather"(%v0, %arg0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1024>, indices_are_sorted = false}> : (tensor<151936x1024xf32>, tensor<1xi32>) -> tensor<1x1024xf32>
        |    return %arg0 : tensor<1xi32>
        |  }
        |  func.func @qwen_prefill_at(%arg0: tensor<64xi32>, %arg1: tensor<1x64xf32>) -> (tensor<1xi32>) {
        |    %v0 = util.global.load @t0 : tensor<151936x1024xf32>
        |    %v423 = "stablehlo.gather"(%v0, %arg0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1024>, indices_are_sorted = false}> : (tensor<151936x1024xf32>, tensor<64xi32>) -> tensor<64x1024xf32>
        |    return %arg0 : tensor<1xi32>
        |  }
        |}
        |""".trimMargin()

    @Test
    fun hostGather_replacesEveryFunctionsGatherWithAnEmbArg() {
        val out = QwenExportHarness.rewriteHostGather(gatherModule)
        assertFalse(out.contains("stablehlo.gather"))
        assertTrue(out.contains("func.func @qwen_with_past(%arg0: tensor<1xi32>, %emb: tensor<1x1024xf32>, %arg1: tensor<1x128xf32>)"))
        assertTrue(out.contains("    %ez = stablehlo.constant dense<0.0> : tensor<1x1024xf32>\n    %v311 = stablehlo.add %emb, %ez : tensor<1x1024xf32>\n"))
        assertTrue(out.contains("func.func @qwen_prefill_at(%arg0: tensor<64xi32>, %emb: tensor<64x1024xf32>, %arg1: tensor<1x64xf32>)"))
        assertTrue(out.contains("    %ez = stablehlo.constant dense<0.0> : tensor<64x1024xf32>\n    %v423 = stablehlo.add %emb, %ez : tensor<64x1024xf32>\n"))
        // the embedding table global stays (the tied LM head still reads it)
        assertTrue(out.contains("util.global private @t0"))
    }

    @Test
    fun hostGather_failsLoudlyWhenThereIsNothingToRewrite() {
        val e = runCatching { QwenExportHarness.rewriteHostGather("module {}") }.exceptionOrNull()
        assertTrue(e is IllegalArgumentException && e.message!!.contains("no `\"stablehlo.gather\""), "$e")
    }
}
