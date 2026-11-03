package sk.ainet.apps.klfm2

import kotlinx.serialization.json.*
import java.io.File
import java.nio.file.Files
import kotlin.test.*

class Lfm2IreeChatCliTest {
    private val request = """{"schema_version":1,"requests":[{"id":"example","messages":[{"role":"user","content":"世界🙂"}],"maxNewTokens":8,"decoderPolicy":"HUGGING_FACE_5_COMPAT"}]}"""

    @Test fun requestsPreserveMessageBytesAndRequireExplicitSupportedPolicy() {
        assertFails { parseIreeChatRequests(request.replace("\"id\":\"example\"", "\"id\":\"first\",\"id\":\"example\"")) }
        assertFails { parseIreeObject(byteArrayOf(123, 34, -1, 34, 58, 49, 125)) }
        val parsed = parseIreeChatRequests(request).single()
        assertEquals("世界🙂", parsed.messages.single().content)
        assertEquals(8, parsed.maxNewTokens)
        assertEquals(Lfm2DecodingPolicy.HUGGING_FACE_5_COMPAT, parsed.decoderPolicy)
        assertFails { parseIreeChatRequests(request.replace("HUGGING_FACE_5_COMPAT", "unspecified")) }
        assertFails { parseIreeChatRequests(request.replace("\"maxNewTokens\":8,", "")) }
        assertFails { parseIreeChatRequests(request.replace("\"maxNewTokens\":8", "\"maxNewTokens\":0")) }
        assertFails { parseIreeChatRequests(request.replace("\"id\":\"example\"", "\"temperature\":0,\"id\":\"example\"")) }
        assertFails { parseIreeChatRequests(request.replace("\"role\":\"user\"", "\"role\":\"tool\"")) }
        assertFails { parseIreeChatRequests(request.replace("\"content\":\"世界🙂\"", "\"content\":3")) }
    }

    @Test fun inventoryIsRelativeDeterministicAndDetectsChangedBytesAndSymlinks() {
        val directory = kotlin.io.path.createTempDirectory("iree-inventory-test").toFile()
        try {
            File(directory, "nested").mkdir()
            val source = File(directory, "nested/value.bin").apply { writeBytes(byteArrayOf(1, 2, 3)) }
            File(directory, "first.bin").writeBytes(byteArrayOf(4))
            val first = inventoryIreeDirectory(directory)
            assertEquals(listOf("first.bin", "nested/value.bin"), first.map { it.jsonObject.getValue("file").jsonPrimitive.content })
            assertEquals(first, inventoryIreeDirectory(directory))
            source.writeBytes(byteArrayOf(1, 2, 4))
            assertNotEquals(first, inventoryIreeDirectory(directory))
            Files.createSymbolicLink(File(directory, "link").toPath(), source.toPath())
            assertFailsWith<IllegalArgumentException> { inventoryIreeDirectory(directory) }
        } finally { directory.deleteRecursively() }
    }
}
