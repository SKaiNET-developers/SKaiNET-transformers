package sk.ainet.models.moonshine

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.float
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import sk.ainet.context.ExecutionContext
import sk.ainet.io.JvmRandomAccessSource
import sk.ainet.io.safetensors.StreamingSafeTensorsReader
import sk.ainet.lang.nn.Module
import sk.ainet.lang.nn.topology.ModuleParameter
import sk.ainet.lang.tensor.Shape
import sk.ainet.lang.tensor.data.DenseFloatArrayTensorData
import sk.ainet.lang.tensor.data.TensorData
import sk.ainet.lang.types.DType
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.reflect.KClass

/**
 * A Hugging Face `moonshine_streaming` snapshot on disk: `config.json` + `model.safetensors` (and, for
 * [MoonshineV2ExportHarness.writeVocab], `tokenizer.json`). Everything the DSL model needs is read from it —
 * the geometry from the config, the weights through [MoonshineV2HfWeightMap] — so one code path serves every
 * checkpoint of the family (English and German tiny; the split-width small ones).
 *
 * Tracks which tensors were read, so an export can prove that no weight of the checkpoint was silently dropped
 * ([unconsumed]).
 */
public class MoonshineV2Checkpoint(public val dir: File) : AutoCloseable {

    private val source = JvmRandomAccessSource.open(File(dir, "model.safetensors").also {
        require(it.isFile) { "no model.safetensors in $dir — expected a Hugging Face moonshine_streaming snapshot" }
    })
    private val reader = StreamingSafeTensorsReader.open(source)
    private val infos = reader.tensors.associateBy { it.name }
    private val consumed = HashSet<String>()

    public val config: MoonshineV2Config = configFromHf(File(dir, "config.json").also {
        require(it.isFile) { "no config.json in $dir" }
    }.readText())

    public val tensorNames: Set<String> get() = infos.keys

    /** The output-head tensor: an explicit one when the checkpoint ships it, else the tied embedding table. */
    public val outputHead: String =
        MoonshineV2HfWeightMap.OUTPUT_HEADS.firstOrNull { it in infos } ?: MoonshineV2HfWeightMap.EMBED_TOKENS

    /** Rows of the adapter's learned positional table = the longest encoder memory the checkpoint supports. */
    public val maxPositions: Int get() = shapeOf("model.decoder.pos_emb.weight")[0]

    public fun shapeOf(name: String): List<Int> =
        (infos[name] ?: error("checkpoint has no tensor '$name'")).shape.map { it.toInt() }

    /** Row-major f32 values of [name]. F32, F16 and BF16 checkpoints are widened to f32. */
    public fun floats(name: String): FloatArray {
        val info = infos[name] ?: error("checkpoint has no tensor '$name'")
        consumed += name
        val buf = ByteBuffer.wrap(reader.loadTensorData(info)).order(ByteOrder.LITTLE_ENDIAN)
        return when (info.dtype.uppercase()) {
            "F32" -> FloatArray(buf.remaining() / 4) { buf.getFloat(it * 4) }
            "F16" -> FloatArray(buf.remaining() / 2) { java.lang.Float.float16ToFloat(buf.getShort(it * 2)) }
            "BF16" -> FloatArray(buf.remaining() / 2) { bf16ToFloat(buf.getShort(it * 2).toInt() and 0xFFFF) }
            else -> error("unsupported dtype '${info.dtype}' for '$name'")
        }
    }

    /** Checkpoint tensors nothing has read yet. Empty after a full export — anything else is a gap in the map. */
    public fun unconsumed(): List<String> = (infos.keys - consumed).sorted()

    /**
     * Overwrite every parameter of [model] with the checkpoint tensor [map] names for it. Fails on a parameter
     * without a mapping, a missing tensor, or an element-count mismatch.
     */
    public fun <T : DType, V> bake(
        model: Module<T, V>,
        map: (String) -> MoonshineV2WeightRef?,
        dtype: KClass<T>,
        ctx: ExecutionContext,
    ): Int {
        var baked = 0
        for (p in parametersOf(model)) {
            val ref = map(p.name) ?: error("no checkpoint mapping for DSL parameter '${p.name}'")
            val shape = p.value.shape.dimensions
            val n = shape.fold(1) { a, b -> a * b }
            var data = ref.hfName?.let(::floats) ?: FloatArray(n)
            require(data.size == n) { "'${ref.hfName}' has ${data.size} elements, DSL parameter '${p.name}' ${shape.toList()} needs $n" }
            when (ref.transform) {
                MoonshineV2WeightRef.Transform.NONE -> Unit
                MoonshineV2WeightRef.Transform.PLUS_ONE -> for (i in data.indices) data[i] += 1.0f
                MoonshineV2WeightRef.Transform.TRANSPOSE -> {
                    require(shape.size == 2) { "TRANSPOSE needs a 2-D parameter, '${p.name}' is ${shape.toList()}" }
                    data = transpose(data, rows = shape[1], cols = shape[0])
                }
            }
            @Suppress("UNCHECKED_CAST")
            (p as ModuleParameter<T, V>).value =
                ctx.fromData(DenseFloatArrayTensorData<T>(Shape(*shape), data) as TensorData<T, V>, dtype)
            baked++
        }
        return baked
    }

    override fun close() { source.close() }

    private fun parametersOf(m: Module<*, *>, out: MutableList<ModuleParameter<*, *>> = ArrayList()): List<ModuleParameter<*, *>> {
        out.addAll(m.params)
        for (child in m.modules) parametersOf(child, out)
        return out
    }

    private fun transpose(src: FloatArray, rows: Int, cols: Int): FloatArray {
        val dst = FloatArray(src.size)
        for (r in 0 until rows) for (c in 0 until cols) dst[c * rows + r] = src[r * cols + c]
        return dst
    }

    private fun bf16ToFloat(bits: Int): Float {
        val sign = (bits ushr 15) shl 31
        val exponent = ((bits ushr 7) and 0xFF) shl 23
        val mantissa = (bits and 0x7F) shl 16
        return Float.fromBits(sign or exponent or mantissa)
    }

    public companion object {
        private val json = Json { ignoreUnknownKeys = true }

        /**
         * `config.json` of a `moonshine_streaming` checkpoint → [MoonshineV2Config]. The top level describes the
         * decoder, `encoder_config` the encoder; `encoder_config.sliding_windows` is taken verbatim, so the
         * per-layer attention bands are always the checkpoint's own.
         */
        public fun configFromHf(configJson: String): MoonshineV2Config {
            val top = json.parseToJsonElement(configJson).jsonObject
            val enc = top["encoder_config"]?.jsonObject ?: error("config.json has no encoder_config — not a moonshine_streaming checkpoint")
            fun JsonObject.int(key: String): Int = (this[key] ?: error("config.json: missing '$key'")).jsonPrimitive.int
            val base = MoonshineV2Config()
            val rope = top["rope_parameters"]?.jsonObject
            val dim = enc.int("hidden_size")
            val heads = enc.int("num_attention_heads")
            val cfg = base.copy(
                dim = dim,
                encoderLayers = enc.int("num_hidden_layers"),
                nHeads = heads,
                headDim = enc["head_dim"]?.jsonPrimitive?.int ?: (dim / heads),
                ffnDim = enc.int("intermediate_size"),
                vocabSize = top.int("vocab_size"),
                decoderDim = top.int("hidden_size"),
                decoderFfnDim = top.int("intermediate_size"),
                decoderLayers = top.int("num_hidden_layers"),
                slidingWindows = enc["sliding_windows"]?.jsonArray?.map {
                    val pair = it.jsonArray
                    pair[0].jsonPrimitive.int to pair[1].jsonPrimitive.int
                },
                ropeBase = rope?.get("rope_theta")?.jsonPrimitive?.float ?: base.ropeBase,
                partialRotaryFactor = rope?.get("partial_rotary_factor")?.jsonPrimitive?.float ?: base.partialRotaryFactor,
            )
            cfg.slidingWindows?.let {
                require(it.size == cfg.encoderLayers) { "sliding_windows has ${it.size} entries, the encoder ${cfg.encoderLayers} layers" }
            }
            return cfg
        }
    }
}
