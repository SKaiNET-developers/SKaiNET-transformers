package sk.ainet.models.bert

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith

/** [BertConfigParser.parsePooling] against real sentence-transformers `1_Pooling/config.json` shapes. */
class BertPoolingConfigTest {

    @Test
    fun clsPooling_bgeShape() {
        val json = """
            {
              "word_embedding_dimension": 384,
              "pooling_mode_cls_token": true,
              "pooling_mode_mean_tokens": false,
              "pooling_mode_max_tokens": false,
              "pooling_mode_mean_sqrt_len_tokens": false
            }
        """.trimIndent()
        assertEquals(BertPooling.CLS, BertConfigParser.parsePooling(json))
    }

    @Test
    fun meanPooling_e5Shape() {
        val json = """
            {
              "word_embedding_dimension": 384,
              "pooling_mode_cls_token": false,
              "pooling_mode_mean_tokens": true,
              "pooling_mode_max_tokens": false,
              "pooling_mode_mean_sqrt_len_tokens": false
            }
        """.trimIndent()
        assertEquals(BertPooling.MEAN, BertConfigParser.parsePooling(json))
    }

    @Test
    fun missingFile_defaultsToMean() {
        assertEquals(BertPooling.MEAN, BertConfigParser.parsePooling(null))
    }

    @Test
    fun unsupportedModes_areRejectedNotSilentlyMispooled() {
        assertFailsWith<IllegalArgumentException> {
            BertConfigParser.parsePooling("""{"pooling_mode_max_tokens": true}""")
        }
        assertFailsWith<IllegalArgumentException> {
            BertConfigParser.parsePooling("""{"pooling_mode_mean_sqrt_len_tokens": true}""")
        }
    }
}
