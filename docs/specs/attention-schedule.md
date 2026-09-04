# Schedule-driven attention: parallel heads and copy-free K/V (SKaiNET SKEEP-005)

**Repository:** `SKaiNET-developers/SKaiNET-transformers` — modules `transformer-core`, `llm-core`, `llm-inference/llama`, `llm-inference/qwen`
**Depends on:** SKaiNET engine `Schedule` API — [SKEEP-005](https://skainet-developers.github.io/SKaiNET/skainet/skeep/005-schedules-structured-concurrency.html) (`sk.ainet.context.schedule`, `DirectCpuExecutionContext(schedule = …)`, `CoroutineSchedule`)
**Issues:** transformers #412 (per-head fused attention copies the K/V prefix per layer per token), #413 (attention is single-threaded)
**Labels:** enhancement, performance
**Milestone:** 0.54.0 (lock-step with engine 0.54.0)

---

## Summary

Attention in the eager decode path was ≈40 % of every generated token on a 3B model at 600
tokens of context — scalar, single-threaded, and copying the entire K/V prefix out of the cache
per layer per token. Multi-head attention is embarrassingly parallel across heads, and the engine
now exposes *how* independent work is mapped onto cores as a first-class, dependency-free
`Schedule` on the `ExecutionContext` (Halide's algorithm/schedule split).

This design makes `MultiHeadAttention` the first transformer-level consumer of that schedule:

- the fused attention kernel runs **one task per head (or per GQA group)** under
  `ctx.schedule`, with results **bit-identical** to the sequential path;
- the fused path now covers **batched prefill and sliding-window layers**, not only decode, so
  `repeatKVHeads` / `permute` / `reshape` disappear from the hot path;
- `PositionalKVCache` (and its shared / padded / read-only wrappers) hand the kernel a
  **copy-free view** of their buffers (`KVBufferView`), and Llama / Qwen can opt into that cache
  with `withKVCacheKind(POSITIONAL)`.

The DSL is untouched: no schedule vocabulary appears inside `network {}`; the schedule is a
deployment property of the context.

## Motivation

The JFR profile behind [transformers#413] (Llama-3.2-3B Q4_K_M, i7-9750H, 622-token prompt):

| Bucket | Share of a decode token | Threads |
|---|---|---|
| native Q4_K gemv (`ffm-rowmajor-Q4_K`) | ≈50 % | 4 (compile-time pool) |
| `fusedDecodeAttention` + `scaledDotProductAttention` | ≈40 % | 1 |
| allocation / GC (K/V copies, reshapes) | ≈10 % | — |

`fusedDecodeAttention` copied `currentView()` of every layer's cache into fresh `FloatArray`s per
token — 111 MB per token at 622 context — before touching a single score ([transformers#412]).

## Scope

In:
- `AttentionSchedulePolicy` (Sequential / PerHead / PerKVGroup / Auto) and `HeadPlan`;
- `ScalarHeadAttentionKernel` (decode: legacy fused rounding order; prefill: engine SDPA order);
- `KVBufferView` + `KVCache.updateInPlace` overrides;
- fused batched prefill and sliding-window path in `MultiHeadAttention`;
- `DecoderKVCacheKind` and the `positionalKvCache` DSL clause; loader `withKVCacheKind`;
- golden-gate env switches, `AttentionScheduleSpeedProfile`, docs.

Out (follow-ups):
- Panama-vectorised inner dot products (changes summation order → tolerance-tested, separate PR);
- growable `PositionalKVCache` buffers (today pre-sized to `maxInferenceLen`);
- a schedule for the FFN / norm tail; consuming `skainet.schedule` metadata in the IREE lane.

## Design

```
OptimizedLLMRuntime ── ctx (schedule = CoroutineSchedule.hardware() on the JVM)
   └─ MultiHeadAttention.attentionImpl
        ├─ q/k/v projections, RoPE                        (unchanged, coordinator thread)
        ├─ cache.updateInPlace(k, v)  → KVBufferView       (PositionalKVCache & wrappers; Append best-effort)
        │     └─ null → cache.update + copiedView          (segment-backed data, recording, cross-attention)
        └─ fusedAttention(q, seqQ, kv, scale, ctx)
              plan = schedulePolicy.plan(nHeads, nKVHeads, seqKV, schedule.parallelism)
              schedule.forRange(plan.units, plan.grain) { start, end ->
                  scores = scoresScratch[start]           // coordinator-owned, one slot per task
                  for unit in start until end: for head in unit's heads:
                      decode  → ScalarHeadAttentionKernel.decodeHead   (Σ e·v, then ·1/sum — legacy order)
                      prefill → ScalarHeadAttentionKernel.prefillRows  (divide, then Σ — SDPA order)
              }
              out [seqQ, qDim] → ctx.fromData(...)         (coordinator)
        └─ o_proj (unchanged)
```

Rules for the worker lambda (the engine's `Schedule.forRange` contract): disjoint ranges; no
allocation through `ctx` / `ops`; no `PhaseProfile`, `mhaDumpStat`, or `ForwardScope` access; all
writes are visible when `forRange` returns. `PhaseProfile.time("attn.fused_compute")` wraps the
whole region on the coordinator.

**Bit-identity.** Per head, the loop order and rounding are exactly those of the previous
sequential kernels, so the parallel result is bit-identical to the sequential one. Two rounding
orders coexist on purpose: decode keeps the fused order the golden gates were validated against;
prefill uses the engine SDPA order, so it is bit-identical to `ops.scaledDotProductAttention`.
Masking is by loop bounds (`exp(-inf) = 0f` exactly); a sliding-window row whose whole band was
trimmed away reproduces the engine's uniform softmax.

**Policy.** `Auto(minSeqKV = 64)` chooses `PerKVGroup` when `nRep > 1 && nKVHeads ≥ parallelism`
(each task walks one KV head's rows for all its query heads — better locality), else `PerHead`.
Below `minSeqKV` keys or `parallelism ≤ 1` the plan is `null` and the coordinator runs the loop
inline — tiny decode steps never pay a fork/join.

**Copy-free views.** `KVBufferView(keys, values, length, headStride, rowStride, headDim)` describes
where head `g`'s row `t` lives. `PositionalKVCache` views its `keyBuf/valueBuf` (`headStride =
maxSeqLen × headDim`); `PaddedSharedPositionalKVCache` sets `rowStride` to the delegate's padded
head dim; `AppendKVCache` views its concatenated `FloatArrayTensorData` buffer when it has one and
returns `null` (→ copied path) on segment-backed data. Every override returns `null` while
`ctx.isRecording`, so tracing / compile stay on the tensor-op path.

## Acceptance criteria

1. `MultiHeadAttentionScheduleParityTest` — every `{(8,8),(8,2),(6,3)} × {none, append,
   positional}` combination: scheduled (shuffled pool) == sequential bit-for-bit; fused prefill ==
   general SDPA bit-for-bit; decode within rounding; in-place view == copied view;
   sliding-window layers == general path.
2. `KVCacheInPlaceViewTest` — every cache variant's view equals `currentView()`; `null` under
   recording.
3. Golden gates (`LlamaGoldenTokenParityTest`, `QwenGoldenTokenParityTest`) pass with
   `SKAINET_ATTN_SCHEDULE=sequential|parallel` × `SKAINET_KV_CACHE=append|positional`.
   Verified 2026-09-04: Llama-3.2-1B Q8_0, Qwen2.5-0.5B Q8_0 and Qwen3-1.7B Q8_0 under
   `sequential/append` and `parallel/positional` (`-PincludeIntegration` for the Qwen class).
4. `AttentionScheduleSpeedProfile` prints tok/s and the `attn.*` buckets for all four
   combinations and asserts identical greedy tokens.
5. `apiCheck` green: all changes additive (`<init>` signatures unchanged, trailing defaulted
   parameters only).

## Work items

| Id | Item | Status |
|---|---|---|
| AS-1 | `AttentionSchedulePolicy`, `HeadPlan`, `plan()` | done |
| AS-2 | `ScalarHeadAttentionKernel.decodeHead` (legacy order) | done |
| AS-3 | `KVBufferView`, `updateInPlace` overrides | done |
| AS-4 | `ScalarHeadAttentionKernel.prefillRows` + fused prefill / sliding-window path | done |
| AS-5 | `DecoderKVCacheKind`, `positionalKvCache` clause, loader `withKVCacheKind` | done |
| AS-6 | parity tests, golden-gate switches, speed profile | done |
| AS-7 | docs (this spec, explanation, tutorial), changelog, API dumps | done |
| AS-8 | vectorised inner dots, growable positional cache | follow-up |

Checkpoints: CP-1 engine `Schedule` API available (SKaiNET `feature/skeep-005-schedules`);
CP-2 transformer-core parity green; CP-3 golden gates green under every switch; CP-4 both
repositories released as 0.54.0 in lock-step (until then: build transformers with
`-PuseLocalSkainet=true`).

[transformers#412]: https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/412
[transformers#413]: https://github.com/SKaiNET-developers/SKaiNET-transformers/issues/413
