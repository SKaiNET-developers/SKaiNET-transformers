package sk.ainet.lang.nn.dsl.decoder

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue
import sk.ainet.lang.types.FP32

class DecoderTransformerNetworkTest {

    private fun metadata(
        layers: Int = 1,
        ropeFreqBase: Float = 10_000f,
        rmsNormEps: Float = 1e-5f,
    ) = TestDecoderMetadata(
        embeddingLength = 8,
        contextLength = 32,
        blockCount = layers,
        headCount = 2,
        kvHeadCount = 2,
        feedForwardLength = 16,
        ropeDimensionCount = 4,
        vocabSize = 16,
        ropeFreqBase = ropeFreqBase,
        rmsNormEps = rmsNormEps,
        bosTokenId = 1,
        eosTokenId = 2,
    )

    @Test
    fun `builds expected top-level module tree`() {
        val model = decoderTransformerNetwork<FP32, Float>(metadata(layers = 2))

        val topLevelNames = model.modules.map { it.name }
        assertTrue("token_embd" in topLevelNames, "missing token_embd; got $topLevelNames")
        assertTrue("blk.0" in topLevelNames, "missing blk.0")
        assertTrue("blk.1" in topLevelNames, "missing blk.1; layers param not honored")
        assertTrue("output_norm" in topLevelNames, "missing output_norm")
        assertTrue("output" in topLevelNames, "missing output")
        assertEquals(5, topLevelNames.size, "unexpected extras in module tree: $topLevelNames")
    }

    @Test
    fun `qkNorm flag flips MHA q_norm and k_norm submodules on`() {
        val withoutQkNorm = decoderTransformerNetwork<FP32, Float>(metadata(), qkNorm = false)
        val withQkNorm = decoderTransformerNetwork<FP32, Float>(metadata(), qkNorm = true)

        // The qkNorm submodules are nested inside the transformer block's MHA module.
        // Searching the recursive child tree by name suffix is enough to confirm
        // the parameter actually wires through.
        assertTrue(
            findAnyName(withoutQkNorm.modules) { it.endsWith(".q_norm") || it.endsWith(".k_norm") } == null,
            "qkNorm=false should NOT add q_norm / k_norm modules"
        )
        assertTrue(
            findAnyName(withQkNorm.modules) { it.endsWith(".q_norm") } != null,
            "qkNorm=true should add a q_norm module"
        )
        assertTrue(
            findAnyName(withQkNorm.modules) { it.endsWith(".k_norm") } != null,
            "qkNorm=true should add a k_norm module"
        )
    }

    @Test
    fun `defaults pull ropeBase and eps from metadata`() {
        // No assertion on the actual numeric ropeBase used inside RoPE
        // (it's stored as a private field). What we can assert is that
        // a metadata with non-default ropeFreqBase and rmsNormEps doesn't
        // throw and produces the same structure.
        val qwenLikeMeta = metadata(ropeFreqBase = 1_000_000f, rmsNormEps = 1e-6f)
        val model = decoderTransformerNetwork<FP32, Float>(qwenLikeMeta)
        assertTrue(model.modules.size == 4, "structure should be invariant to ropeBase / eps")
    }

    /** Recursively scan module tree for a name matching [predicate]. */
    private fun findAnyName(
        roots: List<sk.ainet.lang.nn.Module<*, *>>,
        predicate: (String) -> Boolean,
    ): String? {
        for (m in roots) {
            if (predicate(m.name)) return m.name
            val found = findAnyName(m.modules, predicate)
            if (found != null) return found
        }
        return null
    }
}

private data class TestDecoderMetadata(
    override val embeddingLength: Int,
    override val contextLength: Int,
    override val blockCount: Int,
    override val headCount: Int,
    override val kvHeadCount: Int,
    override val feedForwardLength: Int,
    override val ropeDimensionCount: Int?,
    override val vocabSize: Int,
    override val ropeFreqBase: Float,
    override val rmsNormEps: Float,
    override val bosTokenId: Int,
    override val eosTokenId: Int,
) : DecoderModelMetadata
