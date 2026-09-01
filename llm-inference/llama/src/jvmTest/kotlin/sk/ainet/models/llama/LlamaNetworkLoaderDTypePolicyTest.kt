package sk.ainet.models.llama

import sk.ainet.lang.nn.dsl.decoder.DECODER_NARROW_KEEP_NATIVE
import sk.ainet.lang.nn.dsl.decoder.DecoderGgufWeightLoader
import sk.ainet.lang.nn.dsl.decoder.DecoderSafeTensorsLoader
import sk.ainet.lang.nn.dsl.decoder.GgufDecoderMetadata

import kotlinx.io.Source
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertFailsWith
import kotlin.test.assertSame
import sk.ainet.io.RandomAccessSource
import sk.ainet.lang.types.BF16
import sk.ainet.lang.types.DTypePolicy
import sk.ainet.lang.types.FP16
import sk.ainet.lang.types.FP32
import sk.ainet.lang.types.Int8

/**
 * Pins the `DTypePolicy` validation contract on `LlamaNetworkLoader`:
 *
 *   - the default value is [DTypePolicy.Any];
 *   - `withDtypePolicy(Require(FP32))` always succeeds (it's the loader's
 *     native output dtype);
 *   - `withDtypePolicy(Require(BF16))` and `Require(FP16)` succeed on
 *     **both** paths as of engine 0.38.0: `DecoderSafeTensorsLoader` and
 *     `DecoderGgufWeightLoader` each keep narrow-float sources in their
 *     on-disk 2-bytes-per-element layout (see
 *     [DECODER_NARROW_KEEP_NATIVE]). Before that, GGUF rejected BF16 and
 *     both paths rejected FP16 for want of an `Fp16DenseTensorData`;
 *   - `withDtypePolicy(Require(Int8))` etc. still reject — the loader
 *     doesn't fabricate dtypes the source files don't carry;
 *   - `Prefer` / `OneOf` arms never raise (they're soft constraints).
 *
 * No model files are read — these tests only construct loader instances
 * to exercise the validation boundary added in PR #144 (`*NetworkLoader.
 * withDtypePolicy`).
 */
class LlamaNetworkLoaderDTypePolicyTest {

    private val noopSourceProvider: () -> Source = { error("source not used in validation tests") }
    private val noopRandomAccessProvider: () -> RandomAccessSource = { error("source not used in validation tests") }

    private val anyMetadata = GgufDecoderMetadata(
        architecture = "llama",
        embeddingLength = 4,
        contextLength = 8,
        blockCount = 1,
        headCount = 1,
        kvHeadCount = 1,
        feedForwardLength = 4,
        ropeDimensionCount = 4,
        vocabSize = 4,
        ropeFreqBase = 10_000f,
        rmsNormEps = 1e-5f,
    )

    @Test
    fun `default policy is Any`() {
        val loader = LlamaNetworkLoader.fromGguf(sourceProvider = noopSourceProvider)
        assertSame(DTypePolicy.Any, loader.dtypePolicy)
    }

    @Test
    fun `Require(FP32) is accepted on both GGUF and SafeTensors paths`() {
        val gguf = LlamaNetworkLoader.fromGguf(sourceProvider = noopSourceProvider)
            .withDtypePolicy(DTypePolicy.Require(FP32))
        assertEquals(DTypePolicy.Require(FP32), gguf.dtypePolicy)

        val safetensors = LlamaNetworkLoader.fromSafeTensors(
            metadata = anyMetadata, randomAccessProvider = noopRandomAccessProvider,
        ).withDtypePolicy(DTypePolicy.Require(FP32))
        assertEquals(DTypePolicy.Require(FP32), safetensors.dtypePolicy)
    }

    @Test
    fun `Require(BF16) is accepted on both GGUF and SafeTensors paths`() {
        val safetensors = LlamaNetworkLoader.fromSafeTensors(
            metadata = anyMetadata, randomAccessProvider = noopRandomAccessProvider,
        ).withDtypePolicy(DTypePolicy.Require(BF16))
        assertEquals(DTypePolicy.Require(BF16), safetensors.dtypePolicy)

        val gguf = LlamaNetworkLoader.fromGguf(sourceProvider = noopSourceProvider)
            .withDtypePolicy(DTypePolicy.Require(BF16))
        assertEquals(DTypePolicy.Require(BF16), gguf.dtypePolicy)
    }

    @Test
    fun `Require(FP16) is accepted on both GGUF and SafeTensors paths`() {
        val gguf = LlamaNetworkLoader.fromGguf(sourceProvider = noopSourceProvider)
            .withDtypePolicy(DTypePolicy.Require(FP16))
        assertEquals(DTypePolicy.Require(FP16), gguf.dtypePolicy)

        val safetensors = LlamaNetworkLoader.fromSafeTensors(
            metadata = anyMetadata, randomAccessProvider = noopRandomAccessProvider,
        ).withDtypePolicy(DTypePolicy.Require(FP16))
        assertEquals(DTypePolicy.Require(FP16), safetensors.dtypePolicy)
    }

    @Test
    fun `Require(Int8) is rejected on both paths`() {
        assertFailsWith<IllegalArgumentException> {
            LlamaNetworkLoader.fromGguf(sourceProvider = noopSourceProvider)
                .withDtypePolicy(DTypePolicy.Require(Int8))
        }
        assertFailsWith<IllegalArgumentException> {
            LlamaNetworkLoader.fromSafeTensors(
                metadata = anyMetadata, randomAccessProvider = noopRandomAccessProvider,
            ).withDtypePolicy(DTypePolicy.Require(Int8))
        }
    }

    @Test
    fun `Prefer and OneOf are always accepted regardless of target`() {
        // Prefer is a soft constraint — never raises.
        LlamaNetworkLoader.fromGguf(sourceProvider = noopSourceProvider)
            .withDtypePolicy(DTypePolicy.Prefer(BF16))
        LlamaNetworkLoader.fromGguf(sourceProvider = noopSourceProvider)
            .withDtypePolicy(DTypePolicy.Prefer(FP16))

        // OneOf is a restricted set; non-empty is the only invariant.
        LlamaNetworkLoader.fromSafeTensors(
            metadata = anyMetadata, randomAccessProvider = noopRandomAccessProvider,
        ).withDtypePolicy(DTypePolicy.OneOf(setOf(FP32, BF16)))
    }
}
