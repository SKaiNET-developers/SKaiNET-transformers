package sk.ainet.transformers.iree.android

import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Test

class IreeTaskTopologyTest {
    @Test fun unsetBlankZeroAndGarbageMeanAutoTopology() {
        assertNull(IreeTaskTopology.parse(null))
        assertNull(IreeTaskTopology.parse(""))
        assertNull(IreeTaskTopology.parse(" 0 "))
        assertNull(IreeTaskTopology.parse("-3"))
        assertNull(IreeTaskTopology.parse("four"))
    }
    @Test fun positiveCountsPassThrough() {
        assertEquals(4, IreeTaskTopology.parse(" 4 "))
        assertEquals(12, IreeTaskTopology.fromEnv { key -> if (key == IreeTaskTopology.ENV) "12" else null })
        assertNull(IreeTaskTopology.fromEnv { null })
    }
    @Test fun sequentialScheduleIsOneGroup() {
        assertEquals(1, IreeTaskTopology.groupCountFor(1))   // Schedule.Sequential.parallelism
        assertEquals(1, IreeTaskTopology.groupCountFor(0))
        assertEquals(8, IreeTaskTopology.groupCountFor(8))
    }
}
