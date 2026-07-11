package sk.ainet.llm.providers

import kotlinx.coroutines.runBlocking
import kotlinx.io.buffered
import kotlinx.io.files.SystemFileSystem
import sk.ainet.context.DirectCpuExecutionContext
import sk.ainet.context.ExecutionContext
import sk.ainet.data.source.CachePolicy
import sk.ainet.data.source.DataSourceException
import sk.ainet.data.source.DataSourceRequest
import sk.ainet.data.source.JvmDataSourceResolver
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.SafeTensorsParametersLoader
import sk.ainet.llm.api.EmbeddingModel
import sk.ainet.models.bert.BertConfigParser
import sk.ainet.models.bert.BertExecutionMode
import sk.ainet.models.bert.BertNetworkLoader
import sk.ainet.models.bert.HuggingFaceTokenizer
import sk.ainet.models.bert.PooledExecutionContext
import sk.ainet.models.bert.createBertEncoderRuntime
import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.StandardCopyOption
import kotlin.io.path.createDirectories
import kotlin.io.path.exists
import kotlin.io.path.name
import kotlin.io.path.readText
import kotlinx.io.files.Path as KotlinxPath

/**
 * One-call factories for BERT sentence-embedding models behind the neutral
 * [EmbeddingModel] SPI.
 *
 * ```kotlin
 * // From the Hugging Face Hub — downloads and caches on first use:
 * val model = BertEmbeddingModel.fromHuggingFace("MongoDB/mdbr-leaf-mt")
 *
 * // From a local snapshot directory:
 * val model = BertEmbeddingModel.fromSafeTensors(Path.of("/models/mdbr-leaf-mt"))
 *
 * val vector = model.embed("How do I reset a password?")
 * ```
 *
 * Auto-detects inside the snapshot: the base weights file
 * (`model.safetensors` / `pytorch_model.safetensors` / any `*.safetensors`),
 * the optional sentence-transformers projection head
 * (`2_Dense/model.safetensors` + `2_Dense/config.json`), the model config
 * (`config.json`), and the tokenizer (`vocab.txt`, falling back to
 * `tokenizer.json`).
 */
public object BertEmbeddingModel {

    /**
     * Load a HuggingFace sentence-transformers BERT snapshot directory.
     *
     * @param modelDir directory containing the snapshot
     * @param ctx execution context; defaults to a pooled CPU context (scratch
     *   buffers reused across `embed` calls)
     * @param mode eager [BertExecutionMode.DIRECT] (default) or traced/fused
     *   [BertExecutionMode.OPTIMIZED]
     * @param modelId reported via [EmbeddingModel] response metadata
     */
    @JvmStatic
    @JvmOverloads
    public fun fromSafeTensors(
        modelDir: Path,
        ctx: ExecutionContext = PooledExecutionContext(DirectCpuExecutionContext()),
        mode: BertExecutionMode = BertExecutionMode.DIRECT,
        modelId: String? = modelDir.fileName?.toString(),
    ): EmbeddingModel {
        val weightsFile = resolveWeightsFile(modelDir)
        val denseWeights = modelDir.resolve("2_Dense").resolve("model.safetensors")
        val denseConfig = modelDir.resolve("2_Dense").resolve("config.json")

        val configFile = modelDir.resolve("config.json")
        require(configFile.exists()) { "config.json not found in $modelDir" }
        val config = BertConfigParser.parse(
            configFile.readText(),
            if (denseConfig.exists()) denseConfig.readText() else null,
        )

        val tokenizer = loadTokenizer(modelDir)

        val loaders = buildList {
            add(SafeTensorsParametersLoader(sourceProvider = { JvmRandomAccessSource.open(weightsFile.toString()) }))
            if (denseWeights.exists()) {
                add(SafeTensorsParametersLoader(sourceProvider = { JvmRandomAccessSource.open(denseWeights.toString()) }))
            }
        }

        val tensors = runBlocking { BertNetworkLoader.loadWeightTensors(loaders, ctx, sk.ainet.lang.types.FP32::class) }
        val runtime = createBertEncoderRuntime(config, tensors, ctx, mode = mode)

        return SkaiNetEmbeddingModel(
            runtime = runtime,
            tokenizer = tokenizer,
            modelId = modelId,
        )
    }

    /**
     * Load a BERT sentence-transformers model straight from the Hugging Face
     * Hub, downloading missing snapshot files into
     * `~/.cache/skainet/models/<owner>_<repo>` on first use (offline-safe
     * afterwards). Gated repos pick up `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN`.
     *
     * @param repoId Hub repo, e.g. `"MongoDB/mdbr-leaf-mt"`
     * @param revision branch/tag/commit; default `main`
     */
    @JvmStatic
    @JvmOverloads
    public fun fromHuggingFace(
        repoId: String,
        revision: String = "main",
        ctx: ExecutionContext = PooledExecutionContext(DirectCpuExecutionContext()),
        mode: BertExecutionMode = BertExecutionMode.DIRECT,
    ): EmbeddingModel {
        val modelDir = downloadSnapshot(repoId, revision)
        return fromSafeTensors(modelDir, ctx, mode, modelId = repoId)
    }

    // ------------------------------------------------------------------

    /** Required snapshot files; download fails loudly when one is missing. */
    private val REQUIRED_FILES = listOf("config.json", "model.safetensors")

    /** Optional snapshot files; absence is fine (bias-free heads, tokenizer variants). */
    private val OPTIONAL_FILES = listOf(
        "vocab.txt",
        "tokenizer.json",
        "2_Dense/config.json",
        "2_Dense/model.safetensors",
    )

    private fun downloadSnapshot(repoId: String, revision: String): Path {
        val home = System.getProperty("user.home")
            ?: error("user.home not set — cannot locate the SKaiNET model cache")
        val targetDir = Path.of(home, ".cache", "skainet", "models", repoId.replace('/', '_'))
        val missingRequired = REQUIRED_FILES.filterNot { targetDir.resolve(it).exists() }
        val missingOptional = OPTIONAL_FILES.filterNot { targetDir.resolve(it).exists() }
        if (missingRequired.isEmpty() && missingOptional.isEmpty()) return targetDir
        // A previously completed download is detected by the required files +
        // tokenizer being present; only then do we skip the network entirely.
        if (missingRequired.isEmpty() && targetDir.resolve(DOWNLOAD_MARKER).exists()) return targetDir

        val resolver = JvmDataSourceResolver(useEnvironmentHuggingFaceToken = true)
        runBlocking {
            missingRequired.forEach { file ->
                downloadFile(resolver, repoId, revision, file, targetDir, required = true)
            }
            missingOptional.forEach { file ->
                downloadFile(resolver, repoId, revision, file, targetDir, required = false)
            }
        }
        require(targetDir.resolve("vocab.txt").exists() || targetDir.resolve("tokenizer.json").exists()) {
            "Neither vocab.txt nor tokenizer.json available in $repoId — cannot build a tokenizer"
        }
        Files.write(targetDir.resolve(DOWNLOAD_MARKER), ByteArray(0))
        return targetDir
    }

    private suspend fun downloadFile(
        resolver: JvmDataSourceResolver,
        repoId: String,
        revision: String,
        file: String,
        targetDir: Path,
        required: Boolean,
    ) {
        val target = targetDir.resolve(file)
        target.parent.createDirectories()
        // Stream to a .part sibling and move atomically so an interrupted
        // download retries instead of leaving a truncated file behind.
        val tmp = target.resolveSibling(target.name + ".part")
        try {
            val artifact = resolver.resolve(
                DataSourceRequest(
                    uri = "hf://$repoId@$revision/$file",
                    // targetDir itself is the cache; the existence check above
                    // is the cache hit, so skip the resolver's flat store.
                    cachePolicy = CachePolicy.Bypass,
                )
            )
            SystemFileSystem.sink(KotlinxPath(tmp.toString())).buffered().use { sink ->
                artifact.copyTo(sink)
            }
            Files.move(tmp, target, StandardCopyOption.REPLACE_EXISTING)
        } catch (e: Exception) {
            Files.deleteIfExists(tmp)
            if (required) {
                throw DataSourceException("Failed to download required file $file from $repoId", e)
            }
            // Optional file (bias-free 2_Dense heads lack linear.bias etc.) — skip.
        }
    }

    private const val DOWNLOAD_MARKER = ".skainet-snapshot-complete"

    private fun resolveWeightsFile(modelDir: Path): Path {
        listOf("model.safetensors", "pytorch_model.safetensors").forEach { name ->
            val p = modelDir.resolve(name)
            if (p.exists()) return p
        }
        modelDir.toFile().listFiles()
            ?.firstOrNull { it.extension == "safetensors" }
            ?.let { return it.toPath() }
        error("No .safetensors file found in $modelDir")
    }

    private fun loadTokenizer(modelDir: Path): HuggingFaceTokenizer {
        val vocab = modelDir.resolve("vocab.txt")
        if (vocab.exists()) return HuggingFaceTokenizer.fromVocabTxt(vocab.readText())
        val tokenizerJson = modelDir.resolve("tokenizer.json")
        if (tokenizerJson.exists()) return HuggingFaceTokenizer.fromTokenizerJson(tokenizerJson.readText())
        error("Neither vocab.txt nor tokenizer.json found in $modelDir")
    }
}
