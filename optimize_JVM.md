# JVM Inference Optimization — Product Requirements Document

**Module:** `llm-inference/apertus` (extends to all model runtimes)
**Baseline:** Lazy dequantization runtime (`ApertusQuantizedRuntime`, commit `8c63b60`)
**Target JVM:** 21+ (25 recommended), with `jdk.incubator.vector` and `java.lang.foreign`

---

## Context

The lazy dequant runtime reduces resident memory from ~28 GB to ~3.5 GB for a 7B Q4_0 model by storing weights as raw quantized bytes and dequantizing per-layer at execution time. This PRD defines four follow-up optimizations that build on that foundation, ordered by impact-to-effort ratio.

### Current bottlenecks (profiled on 7B Q4_0, single-threaded JVM)

| Bottleneck | % of token latency | Root cause |
|---|---|---|
| Dequantization | ~15% | `DequantOps.dequantFromBytes` is scalar Java, no SIMD |
| FP32 matmul | ~65% | `TensorOps.matmul` is scalar loops on `FloatArray` |
| GC pauses | ~8% | Per-layer FP32 temps (~50 MB) allocated on-heap, G1 humongous regions |
| Memory bandwidth | ~12% | Quantized bytes copied from heap array → dequant → heap array → matmul |

---

## Phase 1: Off-Heap Memory via Foreign Memory API

### Goal
Eliminate GC pressure from weight storage and per-layer temporaries by moving all large buffers off-heap using `java.lang.foreign.MemorySegment`.

### Background
- `java.lang.foreign` (JEP 454) finalized in Java 22, available as preview in Java 21
- `MemorySegment` supports mmap, >2 GB allocations, deterministic deallocation via `Arena`
- Current code uses `FloatArray` (on-heap) and `ByteArray` (on-heap) for all tensor data
- Build already sets `--enable-preview` and `MaxDirectMemorySize=12g`

### Requirements

#### 1.1 — `OffHeapQuantizedTensor` (jvmMain)
- New class replacing `QuantizedTensor` on JVM, backed by `MemorySegment`
- Constructor accepts `MemorySegment` slice (zero-copy from mmap) + quant metadata
- `dequantToFloat(arena: Arena): MemorySegment` — dequantizes into an arena-scoped off-heap float buffer
- `dequantToFloatArray(): FloatArray` — fallback that copies to heap (for compatibility with existing `TensorOps`)
- Implements `AutoCloseable`; no-op if backed by a shared arena (mmap case)

#### 1.2 — mmap-based GGUF loading
- New `MmapGGUFReader` that memory-maps the entire GGUF file via `FileChannel.map(READ_ONLY, 0, size, arena)`
- Returns `OffHeapQuantizedTensor` instances whose `MemorySegment` is a slice of the mapped file
- Zero allocation for weight storage — OS page cache handles residency
- Shared `Arena` owns the mapping; closed when the model is unloaded

#### 1.3 — Off-heap execution temporaries
- Per-layer dequantized FP32 buffers allocated in a confined `Arena`
- Arena opened at the start of `runLayer()`, closed at the end
- Deterministic deallocation — no GC involvement for the ~50 MB per-layer temp
- Requires a `MemorySegment`-aware matmul path (or copy to `FloatArray` as interim)

#### 1.4 — KMP expect/actual abstraction
- `expect` interface in `commonMain`:
  ```
  expect class PlatformQuantizedTensor : AutoCloseable {
      fun dequantToFloatArray(): FloatArray
      val quantType: GGMLQuantizationType
      val shape: Shape
      val nElements: Int
  }
  ```
- `actual` on JVM: `OffHeapQuantizedTensor` with `MemorySegment`
- `actual` on other targets: delegates to `QuantizedTensor` (heap `ByteArray`)
- `ApertusQuantizedRuntime` uses the expect type — platform-transparent

### Acceptance criteria
- All existing tests pass with off-heap backend
- JVM heap usage for weights is <10 MB for a 7B model (all weight data off-heap)
- No `OutOfMemoryError` for models up to 70B on a 64 GB machine
- GC pause time reduced by >80% vs on-heap baseline (measure with `-Xlog:gc`)

### Risks
- `MemorySegment` API is preview in Java 21; users on older JVMs need the heap fallback
- mmap on macOS has 4 KB page granularity; unaligned tensor slices need offset handling
- `Arena.ofConfined()` is single-threaded; batch inference needs `Arena.ofShared()` with synchronization

---

## Phase 2: Async Layer-Prefetch Pipeline

### Goal
Hide dequantization latency behind matmul compute by prefetching the next layer's weights while the current layer executes.

### Background
- Current `runLayer()` is strictly sequential: dequant wq → matmul wq → dequant wk → matmul wk → ...
- Dequantization is memory-bound (byte unpacking); matmul is compute-bound (FP32 FMA)
- These can overlap on separate threads with minimal contention

### Requirements

#### 2.1 — `DequantPrefetcher`
- Manages a single-threaded `CoroutineDispatcher` dedicated to dequantization
- API: `prefetch(qt: QuantizedTensor): Deferred<FloatArray>` — starts dequant on the prefetch thread
- API: `prefetchLayer(layer: ApertusQuantizedLayerWeights): DequantedLayer` — prefetches all 6 weight matrices for a layer
- Uses a bounded buffer (2 layers max) to avoid unbounded memory growth

#### 2.2 — Pipelined `runLayer()`
- At the start of `runLayer(i)`, issue `prefetcher.prefetchLayer(layers[i+1])` (if not last layer)
- Use `layer[i]`'s already-prefetched results (`.await()`) for the current layer's matmuls
- Output projection uses synchronous dequant (single tensor, not worth pipelining)

#### 2.3 — Memory budget control
- Constructor parameter `maxPrefetchLayers: Int = 1` controls how far ahead to prefetch
- At `maxPrefetchLayers = 1`: peak memory = 2 layers' FP32 temps (~100 MB for 7B)
- At `maxPrefetchLayers = 0`: disable prefetch, fall back to synchronous (current behavior)

### Acceptance criteria
- Token latency reduced by 10-20% for Q4_K models (dequant hidden behind compute)
- Peak memory increase is bounded to `maxPrefetchLayers * layerFP32Size`
- Thread-safety: no data races (verified with `-Xcheck:jni` and concurrent test)
- Graceful degradation: single-core machines fall back to synchronous path

### Dependencies
- Phase 1 (off-heap) is recommended but not required; works with heap `FloatArray` too

---

## Phase 3: SIMD-Accelerated Dequantization via Vector API

### Goal
Accelerate Q4_0/Q4_K/Q8_0 dequantization by 4-8x using the JDK Vector API (`jdk.incubator.vector`).

### Background
- `jdk.incubator.vector` is already on the module path (`--add-modules jdk.incubator.vector`)
- Current `DequantOps` is scalar Java — processes one element at a time
- Q4_0 dequant is embarrassingly parallel: each 32-element block is independent
- Vector API provides `FloatVector.SPECIES_256` (8 floats) or `SPECIES_512` (16 floats) on AVX2/AVX-512
- Apple Silicon (M-series) supports 128-bit NEON via `SPECIES_128` (4 floats)

### Requirements

#### 3.1 — `VectorDequantOps` (jvmMain)
- New class in `skainet-io-gguf` or `llm-inference/apertus` (jvmMain source set)
- Implements vectorized dequant for the most common quantization types:

  | Type | Block size | Priority | Expected speedup |
  |---|---|---|---|
  | Q8_0 | 32 | P0 | 6-8x (trivial: scale * int8) |
  | Q4_0 | 32 | P0 | 4-6x (nibble unpack + scale) |
  | Q4_K | 256 | P1 | 3-5x (super-block with min/scale) |
  | Q6_K | 256 | P1 | 3-4x |
  | F16/BF16 | 1 | P0 | 4-8x (half→float conversion) |

- Each method signature matches `DequantOps` but operates on `MemorySegment` or `ByteArray`
- Runtime feature detection: check `FloatVector.SPECIES_PREFERRED` and fall back to scalar

#### 3.2 — Vectorized F16/BF16 conversion
- `halfToFloat` currently processes one value at a time
- Vectorized: load 8 shorts → bit-shift to float bits → reinterpret as `FloatVector`
- BF16 is especially cheap: `(short << 16)` reinterpreted as float, 8-wide

#### 3.3 — Integration with `QuantizedTensor.dequantToFloat()`
- `QuantizedTensor` delegates to `VectorDequantOps` when available (JVM), scalar `DequantOps` otherwise
- expect/actual or service-loader pattern for platform dispatch
- Benchmark harness: JMH benchmarks for each quant type, scalar vs vector, varying array sizes

### Acceptance criteria
- Q8_0 dequant throughput: >8 GB/s on M-series, >12 GB/s on AVX2 (vs ~1.5 GB/s scalar)
- Q4_0 dequant throughput: >5 GB/s on M-series, >8 GB/s on AVX2
- F16 conversion: >10 GB/s
- Bit-exact output vs scalar `DequantOps` (no precision loss)
- Graceful fallback on JVMs without Vector API support

### Risks
- Vector API is still incubating (not finalized as of JDK 25); API may change
- Auto-vectorization by C2 JIT may already optimize some scalar loops — benchmark first
- Apple Silicon NEON lane width (128-bit) limits theoretical speedup to 4x for float ops

---

## Phase 4: Fused Quantized Matmul (Q4xF32 / Q8xF32)

### Goal
Eliminate the dequantization step entirely by fusing it into the matrix multiplication kernel. Dequantize each quantized block on the fly during the dot product, avoiding the FP32 intermediate buffer.

### Background
- This is what llama.cpp does with `ggml_vec_dot_q4_0_q8_0` and similar kernels
- Current flow: `dequant(Q4 bytes) → FP32 array → matmul(FP32, FP32)`
- Fused flow: `matmul_q4_f32(Q4 bytes, FP32 activations)` — no intermediate
- Saves memory bandwidth (read Q4 once, never write FP32 weights) and memory (no temp buffer)

### Requirements

#### 4.1 — `QuantizedMatmul` kernel interface
```
interface QuantizedMatmul {
    // y[m, n] = x[m, k] @ W_quant[n, k]^T
    // W is stored as quantized bytes, x and y are FP32
    fun matmulQ4_0(
        x: FloatArray, xOffset: Int, m: Int, k: Int,
        wBytes: ByteArray, wOffset: Int, n: Int,
        out: FloatArray, outOffset: Int
    )
    fun matmulQ8_0(...)
    fun matmulQ4_K(...)
}
```

#### 4.2 — Scalar reference implementation
- For each output element: iterate over quantized blocks, dequant-and-dot in one pass
- Q4_0: for each 32-element block, load scale + 16 packed bytes, unpack nibbles, multiply-accumulate
- Q8_0: for each 32-element block, load scale + 32 int8 values, scale and dot
- This alone may be faster than separate dequant+matmul due to better cache locality

#### 4.3 — SIMD-vectorized implementation
- Combine Vector API with fused dequant:
  - Load quantized block into `ByteVector`
  - Unpack/convert to `FloatVector` (or `ShortVector` for Q8_0 intermediate)
  - FMA with activation `FloatVector`
  - Horizontal reduce for dot product
- Target: match or exceed llama.cpp's throughput for single-threaded inference

#### 4.4 — Integration with `ApertusQuantizedRuntime`
- New `runLayer()` path that calls `QuantizedMatmul` directly on `QuantizedTensor.data`
- No `dequantToFloat()` call — weight bytes go directly into the fused kernel
- Falls back to dequant+matmul when fused kernel isn't available for a given quant type

#### 4.5 — Multi-threaded matmul (stretch goal)
- Partition the output dimension across `ForkJoinPool` or virtual threads
- Each thread handles a slice of rows, reading from the same quantized weight data
- Virtual threads (JDK 21+) for lightweight parallelism without thread pool tuning

### Acceptance criteria
- Single-threaded token/s for 7B Q4_0: >5 tok/s on M2 Pro (vs ~2 tok/s with dequant+matmul)
- Memory: zero FP32 weight temporaries during inference (only activation buffers)
- Bit-exact output vs dequant-then-matmul path (within FP32 rounding tolerance, ulp <= 2)
- Works with all quantization types in the `GGMLQuantizationType` enum (fallback for unsupported)

### Risks
- Fused kernels are complex and error-prone; need extensive numerical validation
- Performance is highly sensitive to memory access patterns and cache line alignment
- Different quant types have different block structures — each needs a dedicated kernel
- JIT warmup: fused kernels may be slow for the first few hundred calls until C2 compiles them

### Dependencies
- Phase 3 (Vector API dequant) provides the SIMD building blocks reused here
- Phase 1 (off-heap) enables `MemorySegment`-based weight access (avoids array bounds checks)

---

## Implementation Order and Dependencies

```
Phase 1: Off-Heap Memory ──────────┐
                                    ├──> Phase 4: Fused Quantized Matmul
Phase 3: SIMD Dequant ─────────────┘

Phase 2: Async Prefetch (independent, can parallelize with 1 or 3)
```

| Phase | Effort | Memory impact | Latency impact | Dependency |
|---|---|---|---|---|
| 1. Off-Heap | Medium | Eliminates GC pauses, enables >32 GB models | ~8% (GC reduction) | None |
| 2. Async Prefetch | Low | +50 MB per prefetch layer | 10-20% | None (benefits from Phase 1) |
| 3. SIMD Dequant | Medium | None | 10-15% (dequant portion) | None |
| 4. Fused Matmul | High | Eliminates ~50 MB per-layer temp | 40-60% (combined) | Phase 3 recommended |

**Recommended order:** Phase 1 → Phase 3 → Phase 2 → Phase 4

Phase 1 unblocks large model loading and removes the GC bottleneck. Phase 3 gives immediate dequant speedup with moderate effort. Phase 2 is low-hanging fruit that can be done in parallel. Phase 4 is the endgame — highest impact but requires the most engineering.

---

## Success Metrics

| Metric | Baseline (lazy dequant) | Target (all phases) |
|---|---|---|
| 7B Q4_0 resident memory | ~3.5 GB heap | <200 MB heap + OS page cache |
| 7B Q4_0 tok/s (M2 Pro, 1 thread) | ~2 tok/s | >5 tok/s |
| GC pause p99 | ~200 ms | <5 ms |
| Max model size (64 GB machine) | ~13B (FP32 matmul temps) | 70B+ (mmap + fused matmul) |
| Time to first token (7B) | ~3s | <1.5s |
