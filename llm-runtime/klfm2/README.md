# LFM2 JVM text runtime

`sk.ainet.transformers:skainet-transformers-runtime-klfm2` connects the [LFM2 model](../../llm-inference/lfm2/README.md) to the published `InferenceRuntime<FP32>` interface and provides a checkpoint-specific chat frontend and greedy text loop. The initial target is JVM CPU.

```kotlin
import java.io.File
import sk.ainet.apps.klfm2.*
import sk.ainet.models.lfm2.*

val directory = File("model")
val config = Lfm2Config.parse(File(directory, "config.json").readText())
val loaded = loadLfm2Model(config, File(directory, "model.safetensors"),
    contextLimit = 256, expectedCheckpointSha256 = checkpointSha256)
val frontend = loadLfm2TextFrontend(directory)
val generator = Lfm2TextGenerator(loaded.model, frontend)
val result = generator.generate(
    listOf(Lfm2ChatMessage("user", "What is two plus two?")),
    Lfm2GenerationOptions(maxNewTokens = 8),
    isCancelled = { cancelled.get() },
    onToken = { event -> displayCompletion(event.text) },
)
println(result.text)
```

Here `checkpointSha256`, `cancelled`, and `displayCompletion` are supplied by the application. Obtain model files separately under their own license. No model files or downloads are bundled with the runtime.

## Text policy and lifecycle

The frontend supports the LFM2 text-conversation subset. By default `loadLfm2TextFrontend` verifies the exact bytes of `config.json`, `generation_config.json`, `tokenizer_config.json`, `tokenizer.json`, `chat_template.jinja`, and `LICENSE` against `Lfm2MetadataPins.LFM25_350M` (`LiquidAI/LFM2.5-350M` at revision `9e6c6ccf47cd318696e137d381a7ded8fe4df09f`) before constructing the tokenizer. Passing `pins = null` accepts other checkpoint revisions through the same structural tokenizer/generation validation, and `Lfm2TextFrontend.metadataPinned` records which mode loaded it. In both modes the `chat_template.jinja` bytes must hash to a template revision this module implements: the renderer is a checked Kotlin implementation of specific verified templates, not a general Jinja interpreter, and that gate is never relaxed. Tools, multimodal content, and unverified template revisions are unsupported.

The engine 0.62.0 JVM tokenizer receives an idempotent in-memory Unicode Split character-class adaptation. Original metadata stays unchanged and both original and effective hashes are exposed. This compatibility layer remains until a release containing the engine-side fix is adopted and revalidated.

`CHECKPOINT` decoding is the default. `HUGGING_FACE_5_COMPAT` explicitly reproduces the pinned reference's observed added-token specialness policy. These policies differ for ordinary added pieces as well as thinking/tool markers, so consumers must choose deliberately. Neither treats the ordinary piece `null` as special.

Generation is greedy only. Each call opens a fresh session, checks the prompt plus requested decode budget, retains EOS in returned token IDs/events, and stops before forwarding the last emitted EOS/length-bound token. `stateInputTokens` counts tokens actually forwarded. A callback receives **cumulative text**, which can revise an incomplete UTF-8 suffix; replacing the displayed completion is safe, while blindly appending each callback's text is not. The loop decodes the full generated token sequence on each step.

Cancellation is cooperative at forward boundaries. It cannot interrupt an in-progress tensor kernel. Callback, model, and tokenizer failures propagate after session cleanup. Calls are synchronous and thread-confined; callers must serialize shared model/context access. Results use `eos`, `length`, or `cancelled` stop reasons.

## Token-level integration

```kotlin
Lfm2InferenceRuntime(loaded.model).use { owner ->
    val runtime: sk.ainet.apps.llm.InferenceRuntime<sk.ainet.lang.types.FP32> = owner
    val logits = runtime.forwardBatched(promptTokenIds) // [vocabulary]
    runtime.forward(selectedTokenId)
    runtime.reset()
}
```

`forwardBatched` consumes the whole input and returns only the final position's logits. Reset clears this adapter's state. Close is idempotent and terminal without disposing shared weights. The general `InferenceRuntime` interface itself has no close contract, so keep the owning adapter reference.

The dedicated text loop preserves cumulative UTF-8, EOS-inclusive IDs and exact state counts. The generic generation helper has different semantics and is not substituted automatically. Family metadata identifies `model_type=lfm2`, but declares no generic agent chat-template provider or tool support.

## Tests

`./gradlew :llm-runtime:klfm2:jvmTest` runs without downloaded weights. Tests cover pinned metadata rejection, template rules, decoder policies, Unicode adaptation, cumulative UTF-8 behavior, greedy/EOS/length behavior, cancellation and failure cleanup, and the published token-runtime contract against a synthetic hybrid model. Model quality and target performance require separate checkpoint-specific evaluation.

The small pinned reference fixture also checks all 38 rendered chat cases by UTF-8 bytes in the default suite. To opt into full tokenizer and three short greedy comparisons, supply the exact local checkpoint directory and run:

```bash
LFM2_MODEL_DIR=/model ./gradlew :llm-runtime:klfm2:jvmTest -PincludeIntegration --tests sk.ainet.apps.klfm2.Lfm2ReferenceFixtureTest
```

This optional test validates the fixture's checkpoint hash and all six metadata hashes, permits an 8 GiB test heap, and downloads nothing. It compares reference behavior rather than answer quality; the pinned conversation corpus includes an incorrect arithmetic answer.

## Optional IREE Vulkan runtime

`Lfm2IreeGenerationRuntime` is a separate backend for a compiled fixed-capacity LFM2 graph. It requires an explicit local compiled bundle, Python executable, IREE runtime 3.11.0 installation, and Vulkan device URI. It neither installs packages nor selects a CPU fallback. Kotlin starts the bundled Python worker directly with `-I` and argument-list invocation, without a shell.

```kotlin
val gpuOptions = Lfm2IreeOptions(
    bundleDirectory = File("compiled-generation"),
    pythonExecutable = File("/usr/bin/python3"),
    packageDirectory = File("iree-python-packages"),
    deviceUri = "vulkan://DEVICE_UUID",
)
Lfm2IreeGenerationRuntime(gpuOptions).use { gpu ->
    val generator = gpu.textGenerator(frontend, Lfm2DecodingPolicy.HUGGING_FACE_5_COMPAT)
    val completion = generator.generate(messages, Lfm2GenerationOptions(maxNewTokens = 8))
    println(completion.text)
}
```

Replace the example paths and device identifier with explicit local choices. The compiled bundle's `runtime-manifest.json` binds its graph ABI, dimensions, module and parameter hashes. The worker verifies these identities before announcing readiness. Kotlin independently checks readiness against the requested device, vocabulary, capacity and artifact hashes; every forward verifies response identity, accepted token count, cumulative device steps, logit byte count and finite FP32 values.

The persistent worker keeps model parameters and convolution/KV state on the selected device. The host constructs token/rotary/mask controls and receives final logits for greedy selection and text decoding. A batch currently invokes the compiled single-token graph for each input token. This interface supplies no kernel-level prefill/decode timing or throughput measurement.

Every text generation obtains an exclusive reset session. Session cleanup clears its state and releases the lease; the worker stays loaded for the next request. Closing the runtime owns worker termination. A concurrent close interrupts an outstanding worker exchange; ordinary generation cancellation remains cooperative at forward boundaries. Worker I/O defaults to a configurable ten-minute timeout, with a separate five-second cleanup budget. Protocol/backend failures make that runtime terminal rather than reusing possibly partially advanced state.

`workerInfo` records the accepted worker identity and `implementationSourcePins` includes the exact bundled Python and Kotlin source hashes. Default JVM tests use synthetic protocol responses and owned Java child processes to check malformed responses, input bounds, session ownership, timeout/reaping, and concurrent close without requiring Python, IREE, or a GPU. Device execution and fidelity require separate validation of the compiled bundle.

### Standalone chat runner

Build a portable distribution with `./gradlew :llm-runtime:klfm2:installDist`. Copy `build/install/klfm2-iree-chat` to the target, using independent file copies so later builds cannot alter measured JARs. The target needs Java 21 and the explicitly supplied Python/IREE installation; it does not need Gradle or source checkpoint weights.

Create a UTF-8 request file with explicit policies, for example:

```json
{
  "schema_version": 1,
  "requests": [{
    "id": "example",
    "messages": [{"role": "user", "content": "Hello!"}],
    "maxNewTokens": 8,
    "decoderPolicy": "CHECKPOINT"
  }]
}
```

Run the installed `bin/lfm2-iree-chat --help` for argument names. The required inputs are `--bundle`, `--python`, `--device`, `--model-metadata`, `--requests`, and an exclusive new `--output` JSON file; `--packages` and `--timeout-ms` are optional. The metadata directory supplies only the six pinned frontend files. Use `HUGGING_FACE_5_COMPAT` explicitly when comparing that reference policy.

The report contains computed prompt/generated IDs, cumulative text events, stop/state counts, accepted worker identity and source hashes. It inventories the compiled bundle, metadata, requests, runtime JARs, Python executable/private packages, and Java executable/release/modules/libjvm before and after execution, failing if those inputs change. These are identity and behavioral records, not a benchmark or answer-quality evaluation. Generation/backend failures retain completed/failed samples in the output report and return a nonzero exit status. Invalid command-line arguments or missing inputs can fail before a report is created.

`Lfm2IreeGenerationRuntime.preflight(options)` performs a separate bounded Python/IREE/Vulkan device-initialization check without loading the model/archive or dispatching kernels. Its `device_initialization_only` scope must not be presented as proof of successful model execution.

### Pinned IREE host-copy ownership

The worker copies logits through `copy_device_f32_to_host(array, rt, np, device=...)`. For IREE 3.11.0 it reads the private `DeviceArray._buffer_view` handle, then uses public HAL APIs for direct mapping or a synchronized staging-buffer copy. A scoped Python `memoryview` feeds an owning NumPy copy; the helper never invokes `DeviceArray.to_host()` or `MappedMemory.asarray()`. This avoids the mapped-memory/keep-alive leak observed with the latter path, including when its returned array was immediately copied. Device state feedback remains unchanged and is never mapped during generation.

The helper is tied to the runtime version checked at startup. Offline tests exercise both capability branches and ownership. `tests/python/check_iree_host_copy.py` additionally performs 200 direct maps and 200 public-HAL staging copies on `local-task`; because that CPU allocator promotes all buffers to mappable memory, only its capability predicate is forced for the staging branch. Capture process stderr as well as stdout when running this optional check: leaked native-instance diagnostics at shutdown are a failure. This CPU-only ownership test does not qualify GPU execution or establish a general memory bound for a model.

Generated Python `__pycache__` directories and `.pyc` files are excluded from published JVM resources.
