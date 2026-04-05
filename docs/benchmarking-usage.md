# Running Benchmarks

This repository exposes benchmarking through the `llm-performance` module.

## Prerequisites

- JDK 21 or newer
- Java 25 preferred for this repository
- enough RAM for the selected model
- a local GGUF model file

## List Available Scenarios

```bash
./gradlew :llm-performance:jvmRun --args='list-scenarios'
```

## Resolve A Model Path

```bash
./gradlew :llm-performance:jvmRun \
  --args='resolve-model --model-path /absolute/path/to/model.gguf'
```

## Run The Llama Throughput Benchmark

```bash
./gradlew :llm-performance:jvmRun \
  --args='run --scenario llama-runtime-throughput --model-path /absolute/path/to/tinyllama-1.1b-chat-v1.0.Q8_0.gguf'
```

## Model Configuration Precedence

The benchmark resolves models in this order:

1. CLI `--model-path` or `--model`
2. system property `-Dskainet.model.path=...`
3. environment variable `SKAINET_MODEL_PATH`

Examples:

```bash
SKAINET_MODEL_PATH=/absolute/path/to/model.gguf \
./gradlew :llm-performance:jvmRun --args='run --scenario llama-runtime-throughput'
```

```bash
./gradlew :llm-performance:jvmRun \
  -Dskainet.model.path=/absolute/path/to/model.gguf \
  --args='run --scenario llama-runtime-throughput'
```

## Useful Runtime Options

```bash
./gradlew :llm-performance:jvmRun \
  --args='run --scenario llama-runtime-throughput --model-path /absolute/path/to/model.gguf --warmup-runs 1 --measured-runs 3 --steps 16,64'
```

Parameters:

- `--scenario`: benchmark scenario id
- `--model` / `--model-path`: model reference or explicit local path
- `--warmup-runs`: warmup iterations per case
- `--measured-runs`: measured iterations per case
- `--steps`: comma-separated generation step counts
- `--format`: `console` or `json`

## Current Scenario

`llama-runtime-throughput` compares:

- `LlamaRuntime`
- `DIRECT`
- `OPTIMIZED`

## Building & Running the Fat JAR

Instead of `./gradlew :llm-performance:jvmRun` (which recompiles on every invocation), you can
build a self-contained JAR once and run it directly.

### Build

```bash
./gradlew :llm-performance:shadowJar
```

The output JAR is at:

```
llm-performance/build/libs/llm-performance-all.jar
```

### Run

```bash
java \
  --enable-preview --add-modules jdk.incubator.vector \
  -Xms2g -Xmx12g \
  -jar llm-performance/build/libs/llm-performance-all.jar \
  run --scenario llama-runtime-throughput \
      --model-path /absolute/path/to/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

All CLI options (`--warmup-runs`, `--measured-runs`, `--steps`, `--format`) work
the same way as with `jvmRun`.

### Quick examples

```bash
# List scenarios
java --enable-preview --add-modules jdk.incubator.vector \
  -jar llm-performance/build/libs/llm-performance-all.jar list-scenarios

# Minimal run (1 warmup, 1 measured, 16 steps only)
java --enable-preview --add-modules jdk.incubator.vector \
  -Xms2g -Xmx12g \
  -jar llm-performance/build/libs/llm-performance-all.jar \
  run --scenario llama-runtime-throughput \
      --model-path /path/to/model.gguf \
      --warmup-runs 1 --measured-runs 1 --steps 16
```

## Progress Logging

Progress is logged to **stderr** during benchmark execution. You will see output like:

```
[BENCH] Resolved model: /path/to/model.gguf (source: cli)
[BENCH] === Runtime 1/3: LlamaRuntime ===
[BENCH]   LlamaRuntime | prompt=short steps=16 | warming up (3 runs)...
[BENCH]     warmup 1/3 done
[BENCH]     ...
[BENCH]   LlamaRuntime | prompt=short steps=16 | measuring (3 runs)...
[BENCH]     measured 1/3: 1234ms
[BENCH]     ...
[BENCH]   LlamaRuntime | prompt=short steps=16 | median=1200ms throughput=13.33 tok/s
```

To capture only the final results (no progress), redirect stderr:

```bash
java ... -jar llm-performance-all.jar run ... 2>/dev/null
```

## Native macOS Benchmarks (CPU / Metal / MLX)

The `llm-performance` module also builds a native macOS ARM64 binary that benchmarks
CPU, Metal, and MLX backends head-to-head. This requires the `skainet-backend-metal`
and `skainet-backend-mlx` artifacts published to `mavenLocal()`.

### Prerequisites

- macOS on Apple Silicon (ARM64)
- `skainet-backend-metal` and `skainet-backend-mlx` published to local maven:
  ```bash
  cd /path/to/SKaiNET && ./gradlew publishAllPublicationsToMavenLocalRepository
  ```

### Build the native binary

```bash
./gradlew :llm-performance:linkReleaseExecutableMacosArm64
```

The binary is at:

```
llm-performance/build/bin/macosArm64/releaseExecutable/llm-performance.kexe
```

### Run

```bash
./llm-performance/build/bin/macosArm64/releaseExecutable/llm-performance.kexe \
  run --scenario native-backend-throughput \
      --model-path /absolute/path/to/tinyllama-1.1b-chat-v1.0.Q8_0.gguf
```

### Available options

```bash
# List scenarios
./llm-performance.kexe list-scenarios

# Minimal quick run
./llm-performance.kexe \
  run --scenario native-backend-throughput \
      --model-path /path/to/model.gguf \
      --warmup-runs 1 --measured-runs 1 --steps 16

# Full run with JSON output
./llm-performance.kexe \
  run --scenario native-backend-throughput \
      --model-path /path/to/model.gguf \
      --warmup-runs 3 --measured-runs 5 --steps 16,64,128 \
      --format json
```

### Scenario

`native-backend-throughput` compares three backends using the same `LlamaRuntime`:

| Backend | ExecutionContext        | Attention          |
|---------|------------------------|--------------------|
| CPU     | DirectCpuExecutionContext | CpuAttentionBackend  |
| Metal   | MetalExecutionContext   | GpuAttentionBackend (Metal bridge) |
| MLX     | MlxExecutionContext     | GpuAttentionBackend (MLX bridge)   |

If a backend is not available or fails to initialize, it is automatically skipped.

### Debug build (faster compilation)

```bash
./gradlew :llm-performance:linkDebugExecutableMacosArm64
./llm-performance/build/bin/macosArm64/debugExecutable/llm-performance.kexe \
  run --scenario native-backend-throughput --model-path /path/to/model.gguf
```

## Notes

- The JVM benchmark compares runtime strategies (LlamaRuntime, DIRECT, OPTIMIZED) using CPU backend.
- The native macOS benchmark compares backends (CPU, Metal, MLX) using the same LlamaRuntime.
- Large FP32 model loads are memory-heavy.
- Docker or otherwise memory-constrained environments may fail even when the benchmark is correct.
- A normal host machine with more RAM is the preferred environment for real throughput measurements.
