# LFM2 model and StableHLO export

This module implements the dense, bias-free LFM2 hybrid architecture using SKaiNET tensor operations. It publishes a JVM target as `sk.ainet.transformers:skainet-transformers-inference-lfm2`; the tensor blocks live in `commonMain`, while checkpoint loading and export tooling run on the JVM. It uses the repository's engine BOM, currently 0.63.0.

The supported configuration has short convolution and grouped-query attention layers, split-half RoPE, SwiGLU feed-forward blocks, and a tied embedding/output projection. Unsupported architectural choices fail validation. This does not establish support for every checkpoint marketed as LFM, LFM2, or LFM2.5; MoE, multimodal paths, projection biases, scaled RoPE, quantized execution, and untied heads are outside this implementation.

## Load and execute

Provide local `config.json` and a single SafeTensors checkpoint. Nothing downloads automatically, and model weights are not included in this repository. Obtain weights under their own license; the repository's software license does not relicense them.

```kotlin
import java.io.File
import sk.ainet.models.lfm2.*

val config = Lfm2Config.parse(File("model/config.json").readText())
val loaded = loadLfm2Model(
    config = config,
    checkpoint = File("model/model.safetensors"),
    contextLimit = 256,
    expectedCheckpointSha256 = checkpointSha256,
)
Lfm2Session(loaded.model).use { session ->
    val logits = session.forwardBatched(promptTokenIds) // final position, [vocabulary]
    val next = session.forward(selectedTokenId)
    session.reset()
}
```

The loader checks the configured tensor inventory, shapes, supported BF16/F16/F32 source types, contiguous nonoverlapping extents, total payload size, and finite converted weights. It records checkpoint SHA-256 plus each FP32 tensor payload hash. All arithmetic and state are FP32 after loading; this increases memory relative to BF16 storage. For reproducible runs supply the expected checkpoint hash.

`HybridLfm2Model.forward` returns logits for every input position and updates a caller-owned `HybridModelState`. `Lfm2Session` returns the final position and owns an independent state. Session close is terminal and idempotent; it releases session state while retaining shared weights. Calls are synchronous and thread-confined. A caller sharing an execution context or model must serialize its use.

`BALANCED_FP32` RMS reduction — the policy the complete model was qualified with — is the uniform default for the model, the individual blocks, and `rmsNormalize`; `SERIAL_FP32` remains available explicitly for operator-level comparisons against serial references. `CURRENT_BATCHED` is the model's attention default; `DIRECT_KV_PER_HEAD` is an explicit alternative. Numerical policy changes require new fidelity measurements.

For checkpoint-specific chat and greedy text generation, use [`klfm2`](../../llm-runtime/klfm2/README.md).

## Export a verified checkpoint

The host exporter traces unreadable runtime tensors, lowers with the strict StableHLO converter, and keeps input/result ordering explicit. Runtime inputs include activations, convolution history, KV state, rotary tensors, and masks as appropriate. An export does not perform IREE compilation or imply accelerator execution.

```bash
./gradlew :llm-inference:lfm2:exportLfm2 --args='--help'

./gradlew :llm-inference:lfm2:exportLfm2 --args='--kind prefill --config /model/config.json --checkpoint /model/model.safetensors --checkpoint-sha256 SHA256 --length 16 --context-limit 256 --output /exports/prefill-16'

./gradlew :llm-inference:lfm2:exportLfm2 --args='--kind attention --config /model/config.json --checkpoint /model/model.safetensors --checkpoint-sha256 SHA256 --layer 2 --past 16 --length 1 --attention-layout DIRECT_KV_PER_HEAD --context-limit 256 --output /exports/attention-16-1'
```

Replace `SHA256` with the checkpoint's 64-digit digest and use a new output directory each time. Additional modes are `short-conv` and `conv-block`, each requiring `--layer`. The example paths are placeholders. The export task allows an 8 GiB JVM heap; actual memory needs depend on checkpoint and graph size.

Each bundle contains:

- One MLIR module with a fixed-shape ABI.
- A streamed external FP32 parameter archive, verified against the loaded tensor hashes before publication.
- A manifest with MLIR/archive hashes, parameter-key-to-payload hashes, ordered ABI names/shapes, checkpoint/config hashes, numerical policies, and portable implementation source hashes generated into the published JAR.

The archive uses flat SafeTensors shapes; tensor layouts remain in MLIR. Parameter keys are generated per graph. Share an archive between graphs only after verifying their key-to-payload mappings. Publication uses same-directory hard links so existing archives are never replaced; the filesystem must support hard links. The manifest is written last, and a failed export may leave an incomplete directory without a manifest.

`Lfm2ExportHarness`, `traceLfm2Function`, `writeLfm2ParameterArchive`, and `writeLfm2ExportBundle` are reusable JVM APIs. `Lfm2PrefillNetwork` exposes the pure complete prefill tensor network; embedding lookup and position scheduling remain caller responsibilities for this export mode.

## Verification

```bash
./gradlew :llm-inference:lfm2:jvmTest :llm-runtime:klfm2:jvmTest
./gradlew :llm-core:jvmTest --tests sk.ainet.apps.llm.families.TransformersFamiliesTest
```

Default tests require no checkpoint or network. They cover an independent synthetic short-convolution fixture, scalar attention/head-order/RoPE and closed-form block oracles, FP32 reduction loss, streaming/reset/independent state, SafeTensors validation, strict graph lowering, explicit state ports, complete prefill/eager agreement, streamed parameter integrity, and exclusive bundle publication. The small fixture under `src/jvmTest/resources` contains generated parameters and its reference provenance; it contains no downloaded checkpoint weights.

Synthetic tests verify implementation mechanics. They do not establish model quality, long-context accuracy, throughput, NPU support, or GPU generation acceptance. Real checkpoint and target validation must bind the exact model, source, compiler, runtime, and artifact identities.

## Fixed-capacity generation graph

`Lfm2CachedTensorNetwork` and `traceLfm2CachedStep` expose one token step with fixed-capacity convolution/KV state. The GPU-oriented path uses balanced FP32 RMS reduction and direct per-head KV attention. A one-hot token control selects rows through presliced embedding parameters, and the same shards form the tied output projection. Slicing happens outside tracing; the exporter verifies their concatenated bytes against the original embedding hash.

```bash
./gradlew :llm-inference:lfm2:exportLfm2Generation --args='--help'
```

The generation CLI takes local config/checkpoint paths with mandatory hashes, `--capacity`, and a new output directory. It emits the shared export bundle plus `generation-contract.json`. The host must compile and bind the resulting module/archive into a verified runtime manifest before using the [optional IREE runtime](../../llm-runtime/klfm2/README.md#optional-iree-vulkan-runtime). Export and synthetic state-parity tests establish graph structure and tensor semantics; accelerator execution still requires target validation.
