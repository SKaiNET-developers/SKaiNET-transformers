package sk.ainet.apps.llm.families

import sk.ainet.context.ExecutionContext
import sk.ainet.frontend.FamilyFrontEnds
import sk.ainet.frontend.FrontEndSet
import sk.ainet.frontend.text.SpecialTokens
import sk.ainet.frontend.text.Tokenizer
import sk.ainet.model.CheckpointView
import sk.ainet.model.ModelFamilyDescriptor
import sk.ainet.model.ModelFamilyRegistry
import sk.ainet.model.ModelFormat
import sk.ainet.model.ModelIdentity
import sk.ainet.model.NameMap

/**
 * A model family this repository runs, as an engine [ModelFamilyDescriptor] plus the front-end
 * policy of [FamilyFrontEnds] (SKEEP-006 §SKaiNET-transformers after migration).
 *
 * Policy, never algorithm: the family contributes the stop markers its chat format uses
 * ([stopTokenStrings], resolved through the checkpoint's own vocabulary) and the tool-calling
 * facts the agent layer reads. Weight name maps are the engine's where it ships them.
 */
public open class TransformersFamily(
    final override val family: String,
    final override val architectures: Set<String>,
    public val displayName: String,
    public val supportsToolCalling: Boolean,
    public val chatTemplateFamily: String?,
    /** Literal end-of-turn markers; those present in the vocab become `SpecialTokens.extraStop`. */
    public val stopTokenStrings: List<String> = emptyList(),
) : ModelFamilyDescriptor, FamilyFrontEnds {

    override fun nameMap(format: ModelFormat): NameMap? =
        ModelFamilyRegistry.builtIn.descriptorForFamily(family)?.nameMap(format)

    override fun decorate(view: CheckpointView, identity: ModelIdentity, base: FrontEndSet, ctx: ExecutionContext?): FrontEndSet {
        val text = base.text ?: return base
        val vocabulary = text.tokenizer.vocabulary
        val extra = stopTokenStrings.mapNotNull { vocabulary.tokenToId(it) }.toSet() - setOfNotNull(text.tokenizer.specialTokens.eos)
        if (extra.isEmpty()) return base
        return base.copy(text = text.copy(tokenizer = WithExtraStop(text.tokenizer, extra)))
    }

    override fun toString(): String = "TransformersFamily($family)"
}

/** The base tokenizer with the family's end-of-turn ids added to its stop set. */
internal class WithExtraStop(private val base: Tokenizer, extra: Set<Int>) : Tokenizer by base {
    override val specialTokens: SpecialTokens = base.specialTokens.copy(extraStop = base.specialTokens.extraStop + extra)
    override val vocabSize: Int get() = base.vocabSize
    override val bosTokenId: Int? get() = specialTokens.bos
    override val eosTokenId: Int? get() = specialTokens.eos
}

public object LlamaFamily : TransformersFamily(
    family = "llama",
    architectures = setOf("llama", "llama4", "mistral", "smollm", "smollm2", "tinyllama"),
    displayName = "LLaMA / Mistral",
    supportsToolCalling = true,
    chatTemplateFamily = "llama3",
    stopTokenStrings = listOf("<|eot_id|>", "<|eom_id|>", "<|im_end|>"),
)

public object QwenFamily : TransformersFamily(
    family = "qwen",
    architectures = setOf("qwen2", "qwen2.5", "qwen3", "qwen35"),
    displayName = "Qwen",
    supportsToolCalling = true,
    chatTemplateFamily = "qwen",
    stopTokenStrings = listOf("<|im_end|>", "<|endoftext|>"),
)

public object GemmaFamily : TransformersFamily(
    family = "gemma",
    architectures = setOf("gemma", "gemma2", "gemma3", "gemma3n", "gemma3_text", "gemma4"),
    displayName = "Gemma",
    supportsToolCalling = true,
    chatTemplateFamily = "gemma",
    stopTokenStrings = listOf("<end_of_turn>", "<turn|>", "<|chat_end|>"),
)

public object ApertusFamily : TransformersFamily(
    family = "apertus",
    architectures = setOf("apertus"),
    displayName = "Apertus",
    supportsToolCalling = true,
    chatTemplateFamily = "apertus",
    stopTokenStrings = listOf("<|assistant_end|>"),
)

public object BitNetFamily : TransformersFamily(
    family = "bitnet",
    // "bitnet" is BitNet.cpp's llama.cpp arch id; "bitnet-25" its 2B4T-era variant;
    // "bitnet-b1.58" appears in community conversions.
    architectures = setOf("bitnet", "bitnet-25", "bitnet-b1.58"),
    displayName = "BitNet b1.58",
    supportsToolCalling = false,
    chatTemplateFamily = "bitnet",
    stopTokenStrings = listOf("<|eot_id|>"),
)

public object BertFamily : TransformersFamily(
    family = "bert",
    architectures = setOf("bert", "modernbert", "distilbert", "roberta"),
    displayName = "BERT",
    supportsToolCalling = false,
    chatTemplateFamily = null,
)

public object VoxtralFamily : TransformersFamily(
    family = "voxtral",
    architectures = setOf("voxtral"),
    displayName = "Voxtral",
    supportsToolCalling = false,
    chatTemplateFamily = null,
)

public object WhisperFamily : TransformersFamily("whisper", setOf("whisper"), "Whisper", false, null)
public object MoonshineFamily : TransformersFamily("moonshine", setOf("moonshine"), "Moonshine", false, null)
public object T5Family : TransformersFamily("t5", setOf("t5", "mt5"), "T5", false, null)

/**
 * The family registry of this repository: the engine's built-in families replaced by the
 * descriptors above and extended with the families only this repository runs. Pass it to
 * `FrontEndResolver` / `TokenizerFactory.resolve`; compose further with `+`.
 */
public object TransformersFamilies {
    public val all: List<TransformersFamily> = listOf(
        LlamaFamily, QwenFamily, GemmaFamily, ApertusFamily, BitNetFamily, BertFamily, VoxtralFamily,
        WhisperFamily, MoonshineFamily, T5Family,
    )

    public val registry: ModelFamilyRegistry = ModelFamilyRegistry.builtIn + all

    /** The family descriptor claiming [architecture], or `null`. */
    public fun forArchitecture(architecture: String): TransformersFamily? =
        registry.descriptorForArchitecture(architecture) as? TransformersFamily

    /** The family descriptor named [family], or `null`. */
    public fun forFamily(family: String): TransformersFamily? = registry.descriptorForFamily(family) as? TransformersFamily
}
