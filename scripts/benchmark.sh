#!/usr/bin/env bash
# Benchmark old vs new LLM runtime on TinyLlama 1.1B
#
# Usage:
#   ./scripts/benchmark.sh                              # uses SKAINET_MODEL_PATH or -Dskainet.model.path
#   ./scripts/benchmark.sh /path/to/model.gguf          # explicit model path
#   ./scripts/benchmark.sh --steps 16,64 --warmup 3     # custom steps / warmup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="$PROJECT_DIR/benchmarks"
DATE=$(date +%Y-%m-%d)
OUTPUT_FILE="$OUTPUT_DIR/$DATE.txt"

# ---------- parse args ----------
MODEL_PATH=""
STEPS="16,64"
WARMUP=3
MEASURED=3

while [[ $# -gt 0 ]]; do
  case "$1" in
    --steps)     STEPS="$2"; shift 2 ;;
    --warmup)    WARMUP="$2"; shift 2 ;;
    --measured)  MEASURED="$2"; shift 2 ;;
    -*)          echo "Unknown option: $1"; exit 1 ;;
    *)           MODEL_PATH="$1"; shift ;;
  esac
done

# ---------- resolve model ----------
MODEL_ARGS=()
if [[ -n "$MODEL_PATH" ]]; then
  MODEL_ARGS+=(--args="run --scenario llama-runtime-throughput --model-path $MODEL_PATH --steps $STEPS --warmup-runs $WARMUP --measured-runs $MEASURED")
else
  MODEL_ARGS+=(--args="run --scenario llama-runtime-throughput --steps $STEPS --warmup-runs $WARMUP --measured-runs $MEASURED")
fi

cd "$PROJECT_DIR"

eval "$(jenv init -)" 2>/dev/null || true

mkdir -p "$OUTPUT_DIR"

echo "=== SKaiNET Runtime Benchmark ==="
echo "Date: $DATE"
echo "Steps: $STEPS  Warmup: $WARMUP  Measured: $MEASURED"
echo ""

echo "Building and running benchmark via llm-performance..."
echo ""

./gradlew :llm-performance:jvmRun \
  "${MODEL_ARGS[@]}" 2>&1 | tee "$OUTPUT_FILE"

echo ""
echo "Results saved to: $OUTPUT_FILE"
