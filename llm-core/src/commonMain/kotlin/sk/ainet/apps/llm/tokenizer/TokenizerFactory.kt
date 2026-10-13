package sk.ainet.apps.llm.tokenizer

import sk.ainet.apps.llm.Tokenizer
import sk.ainet.apps.llm.families.TransformersFamilies
import sk.ainet.frontend.FrontEndResolver
import sk.ainet.frontend.text.TextFrontEnd
import sk.ainet.frontend.tokenizer.TextFrontEnds
import sk.ainet.io.RandomAccessSource
import sk.ainet.io.gguf.StreamingGGUFReader
import sk.ainet.model.CheckpointView
import sk.ainet.model.ModelFamilyRegistry

/**
 * Builds a [Tokenizer] for the runtimes in this repository from one tokenizer source.
 *
 * Since 0.58 (SKEEP-006) every path is the engine's: byte-level BPE, Metaspace BPE,
 * SentencePiece, Unigram, WordPiece and Tekken come from `skainet-frontend-text`, selected by
 * `TextFrontEnds` (the dispatch table FE-R-02) and decorated by [TransformersFamilies] — the
 * special-token splitting and the `add_space_prefix` detection this factory used to patch in
 * downstream now live upstream, where the goldens are. The object stays as the bridge to this
 * repository's [Tokenizer] sub-interface; it is deprecated in 0.60 and removed in 0.61, together
 * with the sub-interface.
 *
 * ```kotlin
 * val info = UnifiedModelLoader.peek { sourceProvider() }
 * val tokenizer = TokenizerFactory.fromGgufFields(info.fields)         // GGUF metadata map
 * val tokenizer = TokenizerFactory.fromGgufSource(source)              // GGUF RandomAccessSource
 * val tokenizer = TokenizerFactory.fromTokenizerJsonString(json, cfg)  // tokenizer.json (+ tokenizer_config.json)
 * val text = TokenizerFactory.resolve(view)                            // any CheckpointView, family-decorated
 * ```
 */
public object TokenizerFactory {

    /** Create a tokenizer from a pre-parsed GGUF metadata field map (`GGUFModelInfo.fields`). */
    public fun fromGgufFields(fields: Map<String, Any?>): Tokenizer = adapt(TextFrontEnds.fromGguf(fields))

    /** Open a [StreamingGGUFReader] over [source], parse the metadata and build via [fromGgufFields]. */
    public fun fromGgufSource(source: RandomAccessSource): Tokenizer =
        StreamingGGUFReader.open(source).use { reader -> fromGgufFields(reader.fields) }

    /** Create a tokenizer from a Hugging Face `tokenizer.json` string and its optional `tokenizer_config.json`. */
    public fun fromTokenizerJsonString(json: String, configJson: String? = null): Tokenizer =
        adapt(TextFrontEnds.fromTokenizerJson(json, configJson))

    /** Create the Mistral Tekken tokenizer from a `tekken.json` string. */
    public fun fromTekkenJson(json: String): Tokenizer = adapt(TextFrontEnds.fromTekkenJson(json))

    /**
     * Resolve the text front end of any checkpoint view (GGUF, Hugging Face directory, in-memory)
     * through [registry] — this repository's families by default, so stop ids and templates come
     * decorated.
     */
    public fun resolve(view: CheckpointView, registry: ModelFamilyRegistry = TransformersFamilies.registry): TextFrontEnd =
        resolver(registry).resolveText(view)

    /** As [resolve], returning the tokenizer as this repository's [Tokenizer]. */
    public fun fromCheckpoint(view: CheckpointView, registry: ModelFamilyRegistry = TransformersFamilies.registry): Tokenizer =
        adapt(resolve(view, registry).tokenizer)

    /** A text-only [FrontEndResolver] over [registry]. */
    public fun resolver(registry: ModelFamilyRegistry = TransformersFamilies.registry): FrontEndResolver = TextFrontEnds.resolver(registry)

    /** Present an engine tokenizer as this repository's [Tokenizer] (identity when it already is one). */
    public fun adapt(engine: sk.ainet.frontend.text.Tokenizer): Tokenizer = engine as? Tokenizer ?: UpstreamTokenizerAdapter(engine)
}
