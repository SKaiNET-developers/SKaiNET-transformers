package sk.ainet.apps.voxtral.cli

import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader

fun main() {
    val path = "/Users/A9973957/projects/mistral/SKaiNET-transformers/Voxtral-4B-TTS-2603-Q8_0.gguf"
    println("Opening GGUF: $path")
    StreamingGGUFReader.open(JvmRandomAccessSource.open(path)).use { reader ->
        println("GGUF version: ${reader.version}, tensors: ${reader.tensorCount}, fields: ${reader.fields.size}")
        println()

        val keywords = listOf("token", "vocab", "model")
        val matching = reader.fields.keys.sorted().filter { key ->
            keywords.any { kw -> key.contains(kw, ignoreCase = true) }
        }

        println("=== Fields matching 'token', 'vocab', or 'model' (${matching.size}) ===")
        matching.forEach { key ->
            val value = reader.fields[key]
            val preview = when (value) {
                is String -> "\"${value.take(120)}\""
                is List<*> -> "List(${value.size})"
                is ByteArray -> "ByteArray(${value.size})"
                is Array<*> -> "Array(${value.size})"
                else -> value?.toString()?.take(120) ?: "null"
            }
            println("  $key = $preview")
        }

        println()
        println("=== All field names (${reader.fields.size}) ===")
        reader.fields.keys.sorted().forEach { key ->
            println("  $key")
        }
    }
}
