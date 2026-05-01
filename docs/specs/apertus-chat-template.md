# Apertus chat-template format

Spec for the `ApertusChatTemplate` implementation that PR 3 of the Apertus rollout (see `APERTUS_ROLLOUT.md`) will land. Source: HuggingFace `swiss-ai/Apertus-8B-Instruct-2509` `chat_template.jinja` (14601 bytes), `tokenizer_config.json`, `special_tokens_map.json` — fetched 2026-05-01.

> The Apertus chat template is **NOT** chatml-compatible. `ModelRegistry.kt`'s previous setting of `chatTemplateFamily = "chatml"` was a fallback guess; PR 2 of the rollout (this spec lands alongside it) corrects it to `"apertus"`. PR 3 implements the dedicated `ApertusChatTemplate` against this spec.

## Special tokens

| Purpose                | Token                  | Notes                                      |
| ---------------------- | ---------------------- | ------------------------------------------ |
| BOS                    | `<s>`                  | SentencePiece-style; emitted once at start |
| EOS                    | `<|assistant_end|>`    | Closes assistant turn                      |
| PAD                    | `<pad>`                |                                            |
| UNK                    | `<unk>`                |                                            |
| System turn open       | `<|system_start|>`     |                                            |
| System turn close      | `<|system_end|>`       |                                            |
| Developer block open   | `<|developer_start|>`  | Auto-emitted; carries tool capabilities    |
| Developer block close  | `<|developer_end|>`    |                                            |
| User turn open         | `<|user_start|>`       |                                            |
| User turn close        | `<|user_end|>`         |                                            |
| Assistant turn open    | `<|assistant_start|>`  | One opens at first assistant message       |
| Assistant turn close   | `<|assistant_end|>`    | Same as EOS; closes a complete turn        |
| Inner prefix           | `<|inner_prefix|>`     | Wraps `thoughts` / deliberation block      |
| Inner suffix           | `<|inner_suffix|>`     | Closes inner block; emits before response  |
| Tool calls prefix      | `<|tools_prefix|>`     | Prefixes the JSON tool-call array          |
| Tool calls suffix      | `<|tools_suffix|>`     | Closes the tool-call array                 |
| Image                  | `<|image|>`            | Reserved for multimodal (not used in 2509) |

The `add_bos_token` field in `tokenizer_config.json` is `true`; the chat template emits `{{ bos_token }}` at the very start.

## Turn structure

```
<s>
<|system_start|> <system_text> <|system_end|>
<|developer_start|>
Deliberation: enabled|disabled
Tool Capabilities:
<typescript-tool-defs>     # or "Tool Capabilities: disabled"
<|developer_end|>
<|user_start|> <user_text> <|user_end|>
<|assistant_start|>
   ... assistant content (see below) ...
<|assistant_end|>
<|user_start|> ...                   # next turn
```

### System message

If the caller doesn't supply a system message, the template emits a default:

```
You are Apertus, a helpful assistant created by the SwissAI initiative.
Knowledge cutoff: 2024-04
Current date: <YYYY-MM-DD>          # filled via strftime_now('%Y-%m-%d')
```

### Developer block (auto-injected after system)

Always emitted, even if the caller didn't supply tools or `enable_thinking`:

```
Deliberation: enabled       # if `enable_thinking=true`, else "disabled\n"
Tool Capabilities:
<typescript-tool-defs>      # if tools present
```

or

```
Deliberation: disabled
Tool Capabilities: disabled
```

### Tool-capability rendering (TypeScript-style)

Tools are NOT rendered as JSON Schema. The Jinja macro `render_tools` emits TypeScript-like type declarations:

```
// <tool description>
type <tool_name> = (_: {
    // <param description>
    <param_name>: <typescript_type>,
    <optional_param>?: <typescript_type>,    // default: <value>
}) => any;
```

Type mapping (from `render_typescript_type` macro):

| JSON Schema type       | TypeScript                              |
| ---------------------- | --------------------------------------- |
| `"string"`             | `string` (or `"a" \| "b"` if `enum`)    |
| `"number"`/`"integer"` | `number`                                |
| `"boolean"`            | `boolean`                               |
| `"array"` of primitive | `string[]` / `number[]` / `boolean[]`   |
| `"object"`             | `{ prop: type, ... }` if `properties` else `object` |
| `"oneOf"` (objects)    | `any` (multi-variant unions collapse)   |
| `nullable: true`       | appends ` | null` to the type           |

Tool calls without parameters render as `() => any;`. Multiple tools are joined with newlines.

## Assistant content

Assistant messages can come in two shapes; the template chooses based on `message.content` type:

### Shape 1 — string content

```
<|assistant_start|> <plain assistant text> <|assistant_end|>
```

Used when `message.content` is a string. Simplest path; what most chat frameworks emit by default.

### Shape 2 — block content (`message.content.blocks` array)

```
<|assistant_start|>
[<|inner_prefix|> <thoughts text> <|inner_suffix|>]?       # optional thoughts block
[<|tools_prefix|>[{"<tool>": <args_json>}, ...]<|tools_suffix|> [<output1>, <output2>] ]*   # zero or more tool-call+output cycles
[<response text>]?                                          # optional final response block
<|assistant_end|>
```

Block types:
- `thoughts` — wraps text in `<|inner_prefix|>...<|inner_suffix|>`. Used for deliberation when `enable_thinking=true`.
- `tool_calls` — emits `<|tools_prefix|>[{"name": args_json}, ...]<|tools_suffix|>`. The args are stringified JSON, NOT a parsed object. Multiple calls comma-separated inside the array.
- `tool_outputs` — emits `[output1, output2, ...]` (a literal-bracket-comma list, NOT JSON). Pairs with the preceding `tool_calls`.
- `response` — emits the final visible answer text. If a prior `thoughts` block opened `<|inner_prefix|>`, the `response` first emits `<|inner_suffix|>` to close the thinking section.

### Tool role messages (alternative tool-output encoding)

When the caller passes `role: "tool"` messages between assistant turns (instead of bundling outputs in an assistant `tool_outputs` block), the template encodes the outputs inline:

```
<|tools_prefix|>[{"calc": ...}]<|tools_suffix|>
[<tool_output_1>, <tool_output_2>, ...]
<assistant continues, e.g., with response text>
<|assistant_end|>
```

The `[...]` after `<|tools_suffix|>` is the tool result; multiple tool messages stack into one comma-separated bracket.

## Tool-call output format from the model

The model emits tool calls in the same shape the template renders historical calls:

```
<|tools_prefix|>[{"<tool_name>": {"<arg>": <value>, ...}}]<|tools_suffix|>
```

- `<|tools_prefix|>` and `<|tools_suffix|>` are special tokens (single token each).
- The bracket contains a JSON array.
- Each element is a JSON object with **one** key (the tool name) whose value is the args object.
- Multiple parallel tool calls = multiple objects in the array, comma-separated.
- Args are rendered as proper JSON (not stringified TypeScript).

Parser strategy for `ApertusToolCallParserStrategy` (PR 3):

1. Scan model output for `<|tools_prefix|>`.
2. Read until `<|tools_suffix|>`.
3. Parse the inner string as JSON array.
4. For each element, the single key is the tool name; the value is the args dict.
5. Emit one `ToolCall(name, args)` per element.
6. After `<|tools_suffix|>`, the next assistant text (until `<|assistant_end|>` or another marker) is the response.

## Generation prompt

When the caller sets `add_generation_prompt = true`, the template appends a final `<|assistant_start|>` to prompt the model to begin a new assistant turn. This is the standard "open the next turn for the model to fill" pattern.

## Comparison vs other chat templates we support

| Aspect                   | chatml                | llama3                  | gemma2                | **Apertus**                  |
| ------------------------ | --------------------- | ----------------------- | --------------------- | ---------------------------- |
| Role open token          | `<\|im_start\|>role`  | `<\|start_header_id\|>role<\|end_header_id\|>` | `<start_of_turn>role` | `<\|<role>_start\|>`         |
| Role close token         | `<\|im_end\|>`        | `<\|eot_id\|>`          | `<end_of_turn>`       | `<\|<role>_end\|>`           |
| Auto developer/tool block| no                    | no                      | no                    | **yes** (`<\|developer_…\|>`)|
| Tool-def serialization   | JSON Schema           | JSON Schema             | JSON Schema           | **TypeScript types**         |
| Tool-call format         | `<\|tool_call\|>{…}`  | `<\|python_tag\|>{…}`   | `<\|tool_call\|>…`    | `<\|tools_prefix\|>[{…}]<\|tools_suffix\|>` |
| Inner thoughts marker    | n/a                   | n/a                     | n/a                   | **`<\|inner_prefix\|>…<\|inner_suffix\|>`** |
| Default system on absent | none                  | none                    | none                  | **emits a default**          |

Apertus shares no markup with the existing chat templates; it deserves its own `ApertusChatTemplate.kt` and `ApertusToolCallingSupport.kt` (PR 3 of the rollout) rather than reusing any existing class.

## Implementation notes for PR 3

- The default system message is **always** emitted when the caller omits a system message — `ApertusChatTemplate` should mirror this. Don't silently drop the default; the model is trained on it.
- The developer block is emitted **always**, even with no tools and `enable_thinking=false` (renders `Deliberation: disabled\nTool Capabilities: disabled`). If `enable_thinking` isn't surfaced at the SKaiNET layer yet, default it to `false` and emit `disabled` in the developer block.
- The TypeScript-style tool renderer is non-trivial — recursive type lowering, `oneOf` collapse to `any`, nullable handling. Either port the Jinja macro logic verbatim into Kotlin, OR (simpler) keep an embedded Jinja template + Pebble/jinjava and just substitute the Tools list. The kgemma chat-template work (`Gemma4ChatTemplate.kt`) ports Jinja to hand-coded Kotlin; Apertus's renderer is significantly more involved, so a Jinja-runtime approach may be the lower-risk path.
- `<|inner_prefix|>` / `<|inner_suffix|>` is the deliberation/CoT marker. If the agent loop doesn't emit `thoughts` blocks today, the parser for assistant output should at least know to skip everything between these tokens before looking for tool calls or the final response.
- `<|assistant_end|>` is BOTH the EOS token AND the assistant-turn close. The agent loop's stop condition is unchanged (still EOS-driven).
- Tool-call args are JSON. Multiple calls in one assistant turn = multiple `{"name": {...}}` objects in the JSON array between `<|tools_prefix|>` and `<|tools_suffix|>`.

## Verification artifacts

- `chat_template.jinja` (14601 bytes) — fetched from `swiss-ai/Apertus-8B-Instruct-2509@main`. Pin this exact byte content as the parity reference in `ApertusChatTemplateHfParityTest` (PR 3, mirroring `Gemma4ChatTemplateHfParityTest` shape).
- The four canonical parity cases to assert byte-for-byte against the Jinja:
  1. user-only (no system, no tools) — exercises default system + disabled developer block
  2. system + user — exercises caller-supplied system
  3. system + user + assistant string content — exercises Shape 1 assistant
  4. system + user + assistant block content with `tool_calls` + `tool_outputs` + `response` — exercises Shape 2 assistant + tool call rendering
