package sk.ainet.models.gemma

/**
 * Shared checkpoint location for the real-GGUF FunctionGemma-270M tests in this
 * module — env-first, mirroring `:llm-runtime:kgemma`'s fixture of the same name.
 *
 * Set `GEMMA_GGUF` to point at the checkpoint; the fallback below is the
 * documented dev-box location of the v10 Q5_K_M fine-tune:
 * `SKaiNET-embedded/sl2610-function-calling/models/functiongemma-physical-ai-v10-Q5_K_M.gguf`.
 * Tests skip (each via its own guard) when the file is absent.
 */
internal object FunctionGemmaFixture {

    /** FunctionGemma-270M Q5_K_M checkpoint; override with `GEMMA_GGUF`. */
    val gguf: String = System.getenv("GEMMA_GGUF")
        ?: "/home/miso/projects/coral/SKaiNET-embedded/sl2610-function-calling/models/functiongemma-physical-ai-v10-Q5_K_M.gguf"
}
