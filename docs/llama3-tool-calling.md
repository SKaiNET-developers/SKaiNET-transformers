# Llama 3 / 3.1 / 3.2 tool calling

This page describes how `kllama` formats tool-calling prompts for the
Llama 3 family and how it parses the model's responses. It also explains
the two response formats Meta has documented and which one to pick for
which model.

> **TL;DR** — for Llama 3.2 1B / 3B (and any Llama 3.x in 2025) leave the
> defaults alone. The default format is `Llama3ToolFormat.JSON`, which is
> what Llama 3.2 was fine-tuned on for custom tools. Switch to
> `Llama3ToolFormat.FUNCTION_TAG` only if you are running an older
> Llama 3.1 prompt that expects the tag-wrapped form.

## Quick start

```sh
# Build
./gradlew :llm-apps:kllama-cli:shadowJar

# Run the demo against a Llama 3.x GGUF (auto-detects the family)
java --enable-preview --add-modules jdk.incubator.vector \
     -jar llm-apps/kllama-cli/build/libs/kllama-all.jar \
     -m models/Llama-3.2-1B-Instruct-Q8_0.gguf \
     --demo --template=llama3 \
     -s 256 -k 0.0 \
     "What files are in /tmp?"
```

The demo registers two tools (`list_files`, `calculator`) and runs the
agent loop until the model produces a final assistant message.

## The two formats

Llama 3.x supports two response shapes for custom tool calls. They are
**not auto-negotiated** between the model and the harness — the system
prompt declares which one the model should emit, and the parser must be
told to look for the same one. `Llama3ChatTemplate` and
`Llama3ToolCallingSupport` take a single [`Llama3ToolFormat`][fmt] that
both sides share.

### `Llama3ToolFormat.JSON` (default)

What Llama 3.2 1B / 3B was fine-tuned on for **custom** tool calling.
Meta documents this in `llama-models/models/llama3_2/text_prompt_format.md`.

Model emits a single JSON object on one line (no surrounding prose):

```
{"name": "list_files", "parameters": {"path": "/tmp"}}
```

System prompt the template builds:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a helpful assistant with tool calling capabilities.
When you receive a tool call response, use the output to format an answer to the original user question.

You have access to the following functions:

{"name":"list_files","description":"...","parameters":{...}}

If you choose to call a function, your reply MUST be a single JSON object on one line in the following format and nothing else:
{"name": <function-name>, "parameters": <arguments-object>}
Do not write the function definition. Do not include any prose. Do not use variables.<|eot_id|>
<|start_header_id|>user<|end_header_id|>

What files are in /tmp?<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>

```

Parser ([`Llama31ToolCallParserStrategy`][p1]) accepts:
- The Meta-documented `"parameters"` key, or `"arguments"` (Hermes-style alias).
- A leading `<|python_tag|>` marker (used by Llama 3.2's built-in tools; tolerated here too).
- Trailing prose after the JSON object (small models often append "I hope that helps!").

### `Llama3ToolFormat.FUNCTION_TAG` (Llama 3.1 legacy)

Tag-wrapped JSON. Documented in early Llama 3.1 prompt-format material.
Llama 3.2 will follow this format if asked, but Meta no longer recommends
it for custom tools on 3.2.

Model emits:

```
<function=list_files>{"path": "/tmp"}</function>
```

System prompt the template builds:

```
...
If you choose to call a function, ONLY reply in the format below and nothing else:
<function=function_name>{"arg_name": "arg_value"}</function>
Function calls MUST be on a single line. Required parameters MUST be specified.
...
```

Parser: [`Llama3FunctionTagParserStrategy`][p2]. Multiple `<function=...>`
blocks in a single response are extracted in order.

## Picking a format programmatically

```kotlin
val support = Llama3ToolCallingSupport(format = Llama3ToolFormat.JSON)
val template = support.createChatTemplate()           // Llama3ChatTemplate(JSON)
val calls    = support.parseToolCalls(modelOutput)     // tries Hermes → function-tag → JSON
```

`ToolCallParser.parse` tries every registered strategy and returns the
first non-empty hit. The three default strategies are disjoint by surface
form, so you never get a double-parse:

| Surface form                                    | Strategy                                 |
|-------------------------------------------------|------------------------------------------|
| `<tool_call>{...}</tool_call>`                  | `HermesToolCallParserStrategy`           |
| `<function=name>{...}</function>`               | `Llama3FunctionTagParserStrategy`        |
| Bare `{"name": ..., "arguments"\|"parameters": ...}` | `Llama31ToolCallParserStrategy`     |

That means you can safely select either Llama 3 format on the prompt side
without touching the parser registration — the parser will pick up
whichever the model actually emits.

## Why two formats exist

- **Llama 3.1** shipped with the `<function=...>...</function>` tag form
  in the early prompt-format docs. Meta later updated the docs to also
  show the bare-JSON format alongside it.
- **Llama 3.2** released in late 2024 with built-in tools (`brave_search`,
  `wolfram_alpha`, `code_interpreter`) that use the `<|python_tag|>`-prefixed
  bare-JSON format; for custom tools the docs canonicalise plain bare JSON
  with `"parameters"`. The 1B and 3B Instruct variants are fine-tuned for
  this format.

So: if you're running Llama 3.2, default JSON is the trained-on format
and gives the best chance of a clean call. If you're running an older
Llama 3.1 prompt or you have prompt material specifically calling for
the tag form, switch to `FUNCTION_TAG`.

## Model-size caveat

Llama 3.2 1B is a **small** model. Even with the correct format and
prompt it can:
- Echo back the tool schema instead of producing a call (treat with a
  few-shot example added to the system prompt).
- Hallucinate a tool *result* directly without calling the tool.
- Append commentary after the JSON (the parser handles this).

3B is meaningfully better; 8B (Llama 3.1) is the sweet spot for tool
calling on commodity hardware. Drop the temperature to `0.0` for
deterministic tool-call generation.

## Related files

- `llm-agent/.../chat/Llama3ChatTemplate.kt` — prompt builder.
- `llm-agent/.../chat/Llama3ToolFormat.kt` — format enum.
- `llm-agent/.../chat/ToolCallParser.kt` — both Llama 3 parser strategies + Hermes.
- `llm-agent/.../chat/ToolCallingSupport.kt` — `Llama3ToolCallingSupport`
  pulls everything together.
- `llm-runtime/kllama/.../cli/ToolCallingDemo.kt` — the `--demo` runner.

[fmt]: ../llm-agent/src/commonMain/kotlin/sk/ainet/apps/kllama/chat/Llama3ToolFormat.kt
[p1]: ../llm-agent/src/commonMain/kotlin/sk/ainet/apps/kllama/chat/ToolCallParser.kt
[p2]: ../llm-agent/src/commonMain/kotlin/sk/ainet/apps/kllama/chat/ToolCallParser.kt
