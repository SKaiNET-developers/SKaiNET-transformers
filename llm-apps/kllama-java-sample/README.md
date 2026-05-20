# SKaiNET Transformers — Java Starter Sample

This is the **fastest first-run path** for SKaiNET Transformers. It is a pure-Java
sample (`Main.java`) that loads a GGUF model and runs a tool-calling conversation
through the kllama Java surface — no Kotlin, no suspend functions.

## Prerequisites

- **JDK 21+** (Java 25 preferred — the runtime uses the Vector API as an incubator module).
- A local **GGUF model file**. Use a small quantized model for the first run, e.g.
  `tinyllama-1.1b-chat-v1.0.Q8_0.gguf`. This sample does not download a model for you.
- Enough RAM for the model (a Q4/Q8 1B model is comfortable on 8 GB).

## Run

```bash
./gradlew :llm-apps:kllama-java-sample:run \
  --args="/absolute/path/to/model.gguf 'What is 17 * 23?'"
```

- The first argument is the **absolute path** to the `.gguf` file (required).
- The second argument is the prompt (optional; defaults to `What is 17 * 23?`).

Running through `./gradlew` applies the required Vector API JVM flags
(`--enable-preview --add-modules jdk.incubator.vector`) automatically.

## Success signal

The sample loads the model, streams generated tokens to stdout, and then prints:

```
---
Final assistant response:
<the model's answer, e.g. 391>
```

If you see a streamed response followed by the `Final assistant response:` block,
SKaiNET Transformers works on your machine.

## Common first-run problems

| Problem | What to check |
|---|---|
| `Usage: kllama-java-sample <model.gguf> [prompt]` | No model path was passed — supply an absolute path as the first argument. |
| Model file not found | Use an absolute path to the `.gguf` file. |
| `ClassCastException` / scalar fallback | Run via `./gradlew ...:run` so the Vector API flags are applied. |
| Out of memory | Use a smaller quantized model and close memory-heavy apps. |

## Next steps

- Try a different prompt or your own tool.
- Move from the sample to the unified `skainet-cli`.
- Read the [Java getting-started tutorial](../../docs/modules/ROOT/pages/tutorials/getting-started-java.adoc).
