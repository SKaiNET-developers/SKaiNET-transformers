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

## Notes

- The current implementation is JVM-first even though the module is Kotlin Multiplatform.
- Large FP32 model loads are memory-heavy.
- Docker or otherwise memory-constrained environments may fail even when the benchmark is correct.
- A normal host machine with more RAM is the preferred environment for real throughput measurements.
