package sk.ainet.apps.kllama.chat

/**
 * Tool-calling output format for Llama 3 / 3.1 / 3.2 family models.
 *
 * Llama 3.x ships two formats Meta has documented for custom tool calling.
 * The model expects a *specific* one based on how the system prompt is
 * phrased — there is no auto-detection on the model side, so the prompt
 * and the parser must agree.
 *
 * | Format         | Model response shape                                  | Recommended for                      |
 * |----------------|-------------------------------------------------------|--------------------------------------|
 * | [JSON]         | `{"name": "fn", "parameters": {"k": "v"}}`            | Llama **3.2** 1B/3B (default)        |
 * | [FUNCTION_TAG] | `<function=fn>{"k": "v"}</function>`                  | Llama **3.1** legacy / fallback      |
 *
 * Pass the chosen format to [Llama3ChatTemplate] (and to
 * [Llama3ToolCallingSupport] when constructing one explicitly). The
 * default — what `kllama --demo --template=llama3` resolves to — is
 * [JSON], because that is the format Llama 3.2 1B/3B was fine-tuned on
 * for custom tools.
 *
 * See `docs/llama3-tool-calling.md` for the full prompt-template shape
 * Meta publishes for each format.
 */
public enum class Llama3ToolFormat {
    /**
     * Bare JSON response: `{"name": "fn_name", "parameters": {...}}`.
     * Default for Llama 3.2 custom tools (matches Meta's `llama3_2/text_prompt_format.md`).
     * Also accepted on 3.1 — use this unless you specifically need 3.1 legacy compat.
     */
    JSON,

    /**
     * Tag-wrapped JSON response: `<function=fn_name>{"arg": "value"}</function>`.
     * Llama 3.1 legacy format, kept selectable for compatibility with prompts
     * trained against earlier 3.1 docs. Llama 3.2 will follow this format
     * if asked, but Meta's 3.2 docs recommend [JSON].
     */
    FUNCTION_TAG
}
