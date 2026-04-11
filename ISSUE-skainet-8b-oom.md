# Issue: Qwen3-8B OOM on 48GB Mac

## Problem

Running Qwen3-8B-Q4_K_M.gguf (4.7GB on disk) on a 48GB Mac fails with OOM during weight loading, both via kllama and the unified skainet CLI.

## Root Cause

The current loading path uses `DEQUANTIZE_TO_FP32`, which expands Q4 weights 8x:

| Component                | Size      |
|--------------------------|-----------|
| Quantized weights (disk) | 4.7 GB    |
| Dequantized FP32 weights | ~37-40 GB |
| KV cache (2048 context)  | 512 MB    |
| Embeddings, norms        | ~1 GB     |
| JVM + tokenizer          | ~2 GB     |
| **Total**                | **~41 GB** |

48GB barely fits, and the JVM needs headroom for temporary buffers during dequantization, so it OOMs.

## What Already Exists in the Codebase

### 1. NATIVE_OPTIMIZED quant policy (best option)

`QuantPolicy.NATIVE_OPTIMIZED` keeps weights in quantized form and uses SIMD-accelerated matmul kernels. `MemSegWeightConverter` converts raw Q4/Q8 bytes to 64-byte-aligned MemorySegment-backed tensors for Vector API dispatch.

- Memory: ~5GB for the 8B model (vs 40GB with FP32)
- Speed: 1-3 tok/s (proven on Qwen2/3 via kqwen runner)
- Already works for Qwen2/3 in kllama Main.kt (the `isQwen` path)

**Why it doesn't work today for the 8B:** The kllama `isQwen` path loads with `NATIVE_OPTIMIZED` but then creates `LlamaRuntime` which still transposes weight matrices to FP32 during init (`LlamaRuntime.kt:74`). This transpose step allocates FP32 copies.

### 2. Lazy per-layer dequantization (Apertus pattern)

`ApertusQuantizedRuntime` keeps weights quantized and dequantizes one projection at a time during `runLayer()`:

```
Resident: ~3.5GB (quantized) + ~100MB (norms/embeddings)
Per-layer temp: ~50MB (one projection, discarded after matmul)
```

This is the llama.cpp approach. Not yet available for LLaMA/Qwen runtimes.

### 3. Memory-mapped loading (F32 only)

`MmapLlamaLoader` maps the GGUF file via `MappedByteBuffer` for zero-copy tensor access. Only works for F32 models — Q4 models need dequantization which defeats the zero-copy benefit.

## Proposed Solutions (ordered by effort)

### Solution A: Fix NATIVE_OPTIMIZED path for 8B models (small effort)

The kllama Main.kt Qwen path already loads with `NATIVE_OPTIMIZED`. The problem is `LlamaRuntime` constructor transposes weights to FP32. Fix:

1. Skip transpose for quantized tensors in `LlamaRuntime` init
2. Or use `OptimizedLLMRuntime` which doesn't transpose (the DSL path)
3. Ensure SIMD matmul kernels handle Q4_K_M format (Q4_K dispatch exists in `MemSegWeightConverter`)

**Expected result:** 8B Q4 loads in ~5GB, runs at 1-3 tok/s.

**Files to change:**
- `llm-inference/llama/.../LlamaRuntime.kt` -- skip transpose for quantized MemSeg tensors
- Or migrate Qwen path in `kllama/cli/Main.kt` to `OptimizedLLMRuntime` + `llamaNetwork()`

### Solution B: Port lazy dequant from Apertus to LLaMA (medium effort)

Port the `ApertusQuantizedRuntime` pattern to a `LlamaQuantizedRuntime`:

1. Store projections as `QuantizedTensor` (quantized bytes + metadata)
2. In `runLayer()`, dequantize one weight matrix at a time, matmul, discard
3. Keep embeddings and norms as FP32 (small, need element access)

**Expected result:** 8B Q4 loads in ~5GB, runs at ~1 tok/s (dequant overhead per layer).

**Files to create:**
- `llm-inference/llama/.../LlamaQuantizedRuntime.kt` (new, based on Apertus pattern)
- `llm-runtime/kllama/.../LlamaQuantizedWeights.kt` (new, mixed storage)

### Solution C: SIMD-native matmul without dequantization (larger effort, best perf)

The SIMD backend (`skainet-backend-cpu`) already has Q4/Q8 matmul kernels via Vector API. The issue is the runtime layer doesn't use them directly. Changes needed in skainet core:

1. `skainet-backend-cpu`: Ensure `matmul(FP32, Q4_K)` kernel exists and dispatches correctly
2. `LlamaRuntime` or `OptimizedLLMRuntime`: Accept mixed-precision weight tensors (Q4 weights, FP32 activations)
3. Skip the `MemSegWeightConverter` step entirely — use raw quantized MemorySegments

**Expected result:** 8B Q4 loads in ~5GB, runs at 2-5 tok/s (no dequant overhead).

**Files to change (in skainet core):**
- `skainet-backend-cpu`: Q4_K matmul kernel (may already exist)
- `skainet-lang-core`: Mixed-precision tensor support in matmul dispatch

### Solution D: Memory-mapped quantized tensors (largest effort)

Extend `MmapLlamaLoader` to support quantized formats:

1. Map the GGUF file to virtual memory
2. Create quantized tensor views that reference mmap regions
3. Dequantize on-the-fly during matmul (like lazy dequant but zero-copy from disk)

**Expected result:** Load time near-zero, ~5GB virtual (OS manages paging).

**Files to change:**
- `llm-inference/llama/.../MmapLlamaLoader.kt` -- extend to Q4/Q8 formats
- Requires `skainet-io-core` changes for mmap quantized tensor views

## Recommended Path

**Start with Solution A** — it's the smallest change and uses code that already works for Qwen2/3. The `NATIVE_OPTIMIZED` + `MemSegWeightConverter` path is proven; the only blocker is `LlamaRuntime`'s constructor transposing weights to FP32.

If that's not enough, **add Solution B** (lazy dequant) which gives the most control over memory at a known performance cost.

Solution C is the long-term goal (best performance) but requires skainet core changes.
