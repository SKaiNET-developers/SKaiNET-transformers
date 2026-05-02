package sk.ainet.apps.llm

import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.export.GGUFWriter
import sk.ainet.io.gguf.export.GgufWriteRequest
import java.nio.file.Files
import kotlin.test.Test
import kotlin.test.assertEquals

/**
 * Regression tests for [UnifiedModelLoader.peek] handling of GGUF metadata
 * fields stored as unsigned integer types.
 *
 * Before the fix, `(fields[...] as? Number)?.toInt()` silently returned null
 * for `UInt`/`ULong` values (they are not subtypes of `Number` in Kotlin),
 * causing every modern GGUF — which uses uint32 dimensions — to fall back
 * to the defaults: contextLength=4096, blockCount=0, embeddingLength=0.
 * A blockCount of 0 yields a model with zero transformer layers.
 */
class UnifiedModelLoaderUIntMetadataTest {

    @Test
    fun peek_reads_uint32_metadata_fields() {
        val bytes = buildGgufBytes(
            arch = "apertus",
            metadata = mapOf(
                "apertus.context_length" to 8192u,
                "apertus.block_count" to 32u,
                "apertus.embedding_length" to 4096u,
                "apertus.vocab_size" to 128256u
            )
        )

        val info = peekFromBytes(bytes)

        assertEquals("apertus", info.architecture)
        assertEquals(8192, info.contextLength)
        assertEquals(32, info.blockCount)
        assertEquals(4096, info.embeddingLength)
        assertEquals(128256, info.vocabSize)
    }

    @Test
    fun peek_reads_uint64_metadata_fields() {
        val bytes = buildGgufBytes(
            arch = "apertus",
            metadata = mapOf(
                "apertus.context_length" to 8192uL,
                "apertus.block_count" to 32uL,
                "apertus.embedding_length" to 4096uL,
                "apertus.vocab_size" to 128256uL
            )
        )

        val info = peekFromBytes(bytes)

        assertEquals(8192, info.contextLength)
        assertEquals(32, info.blockCount)
        assertEquals(4096, info.embeddingLength)
        assertEquals(128256, info.vocabSize)
    }

    @Test
    fun peek_reads_int32_metadata_fields() {
        val bytes = buildGgufBytes(
            arch = "apertus",
            metadata = mapOf(
                "apertus.context_length" to 8192,
                "apertus.block_count" to 32,
                "apertus.embedding_length" to 4096,
                "apertus.vocab_size" to 128256
            )
        )

        val info = peekFromBytes(bytes)

        assertEquals(8192, info.contextLength)
        assertEquals(32, info.blockCount)
        assertEquals(4096, info.embeddingLength)
        assertEquals(128256, info.vocabSize)
    }

    @Test
    fun peek_falls_back_to_defaults_when_fields_missing() {
        val bytes = buildGgufBytes(
            arch = "apertus",
            metadata = emptyMap()
        )

        val info = peekFromBytes(bytes)

        assertEquals("apertus", info.architecture)
        assertEquals(4096, info.contextLength) // default
        assertEquals(0, info.blockCount)
        assertEquals(0, info.embeddingLength)
        assertEquals(0, info.vocabSize)
    }

    private fun buildGgufBytes(arch: String, metadata: Map<String, Any>): ByteArray {
        val merged = LinkedHashMap<String, Any>()
        merged["general.architecture"] = arch
        merged.putAll(metadata)
        val request = GgufWriteRequest(
            metadata = merged,
            tensors = emptyList(),
            tensorMap = emptyMap()
        )
        return GGUFWriter.writeToByteArray(request).second
    }

    private fun peekFromBytes(bytes: ByteArray): GGUFModelInfo {
        val tempFile = Files.createTempFile("uint-meta", ".gguf").toFile()
        tempFile.deleteOnExit()
        tempFile.writeBytes(bytes)
        return UnifiedModelLoader.peek { JvmRandomAccessSource.open(tempFile) }
    }
}
