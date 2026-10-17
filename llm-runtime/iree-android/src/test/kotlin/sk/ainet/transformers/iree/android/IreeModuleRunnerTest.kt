package sk.ainet.transformers.iree.android

import org.junit.jupiter.api.Test
import kotlin.test.assertContentEquals
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

class IreeModuleRunnerTest {
    @Test
    fun bareFunctionNamesAreModuleQualified() {
        assertEquals("module.yolov8_detect", IreeModuleRunner.qualify("yolov8_detect"))
        assertEquals("other.main", IreeModuleRunner.qualify("other.main"))
        assertFailsWith<IllegalArgumentException> { IreeModuleRunner.qualify(" ") }
    }

    @Test
    fun aTensorMustHaveAsManyValuesAsItsShape() {
        FloatTensor(FloatArray(6), intArrayOf(1, 2, 3))
        assertFailsWith<IllegalArgumentException> { FloatTensor(FloatArray(5), intArrayOf(1, 2, 3)) }
        assertFailsWith<IllegalArgumentException> { FloatTensor(FloatArray(0), intArrayOf()) }
        assertFailsWith<IllegalArgumentException> { FloatTensor(FloatArray(0), intArrayOf(0, 3)) }
        assertFailsWith<IllegalArgumentException> { FloatTensor(FloatArray(1), IntArray(9) { 1 }) }
    }

    @Test
    fun theNativeShapeBufferIsRankThenDims() {
        val t = FloatTensor.fromNative(FloatArray(84 * 8400), intArrayOf(3, 1, 84, 8400, 0, 0, 0, 0, 0))
        assertContentEquals(intArrayOf(1, 84, 8400), t.shape)
        assertFailsWith<IllegalArgumentException> { FloatTensor.fromNative(FloatArray(1), intArrayOf(0, 0, 0, 0, 0, 0, 0, 0, 0)) }
    }
}
