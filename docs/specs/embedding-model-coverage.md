# Embedding model coverage: BGE + multilingual E5

**Status:** Phase 1 in progress · **Owner:** @michalharakal · **Created:** 2026-07-13

Extend the BERT embedding stack beyond LEAF to the two most-used compact retrieval
models, establishing the model matrix for the upcoming embedding **performance
benchmarks** (`llm-performance` will compare models × execution modes on one harness):

| Target model | Dims | Size | Blocking gaps |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | 130 MB | CLS pooling; query instruction prefix |
| `intfloat/multilingual-e5-small` | 384 | 118 MB | Unigram (XLM-R) tokenizer; `query:`/`passage:` prefixes |

Verified facts (HF configs, 2026-07-13): both are plain 12-layer BERTs (hidden 384,
heads 12, FFN 1536, absolute positions, `type_vocab_size` 2, **no** `2_Dense` head,
modules `Transformer → Pooling → Normalize`). `bertNetwork()` runs both **unchanged** —
every gap is peripheral (tokenizer, pooling, text prefixes).

## Design

### D1. Pooling modes (BGE)

`BertEncoderRuntime` hard-codes masked mean pooling. Add:

```kotlin
public enum class BertPooling { MEAN, CLS }
```

- Constructor/`createBertEncoderRuntime` parameter, default `MEAN` (source- and
  behavior-compatible; BCV dumps regenerate for the new signatures).
- `encode()`: `CLS` takes hidden-state row 0; `MEAN` keeps the existing masked mean.
  Pooling stays **outside** the traced encoder graph (unchanged export/OPTIMIZED path).
- Detection: sentence-transformers `1_Pooling/config.json` (`pooling_mode_cls_token`
  / `pooling_mode_mean_tokens`) parsed next to `BertConfigParser`; both factories wire
  it through. Absent file → `MEAN` (status quo).

### D2. Query/document asymmetry (E5 requires, BGE recommends)

Retrieval models embed queries and documents differently (instruction prefixes).

- **SPI (additive):** `EmbeddingModel` gains `embedQuery(text)` and
  `embedDocument(text)` / `embedDocuments(texts)` with defaults delegating to
  `embed(...)` — existing implementations and consumers are untouched.
- **`PrefixedEmbeddingModel`** decorator (llm-providers): prepends per-role prefixes;
  `embed(...)` uses the document role (indexing is the common bulk path).
- **`EmbeddingModelProfiles`** registry keyed by repo-id prefix:
  `intfloat/multilingual-e5-*` → `("query: ", "passage: ")`;
  `BAAI/bge-*-en-*` → `("Represent this sentence for searching relevant passages: ", "")`.
  `fromHuggingFace` applies the profile automatically; `fromSafeTensors` accepts
  explicit prefix parameters (no repo id to key on).

### D3. Unigram tokenizer (E5) — Phase 2

`HuggingFaceTokenizer` is WordPiece-only with hard-coded `[CLS]`/`[SEP]`. E5 needs
SentencePiece **Unigram** (Viterbi over a 250k scored vocab) plus XLM-R special tokens
(`<s>`/`</s>`, ids 0/2).

- Source of truth: `tokenizer.json` (embeds the scored vocab + post-processor —
  no protobuf `.model` parsing).
- Normalizer: XLM-R uses a `Precompiled` charsmap; we approximate with NFKC and
  **quantify drift** with a multilingual parity corpus vs HF `tokenizers` (CP-3).
- Placement decision at CP-3 (see checkpoints): transformers-side
  (`llm-inference/bert`, no engine release needed) vs engine
  (`sk.ainet.io.tokenizer`, where SentencePiece-BPE lives, requires engine release).
  Default: transformers-side first, upstream to the engine later.

### Out of scope

Max/sqrt-len pooling modes, two-segment (cross-encoder) inputs, matryoshka dims,
serving-side batching.

## Traceable implementation plan

Gitflow: `feature/*` → `develop` → `release/x.y.z` → tag (publishes to Maven Central).

| ID | Work item | Repo / branch | Release | Status |
|---|---|---|---|---|
| E5BGE-0 | This design doc | SK-tr `feature/embedding-pooling-profiles` | 0.37.0 | in progress |
| E5BGE-1 | `BertPooling` (MEAN/CLS) + `1_Pooling` detection + tests | SK-tr, same branch | 0.37.0 | in progress |
| E5BGE-2 | SPI `embedQuery`/`embedDocument` + `PrefixedEmbeddingModel` + profiles | SK-tr, same branch | 0.37.0 | in progress |
| E5BGE-3 | BGE end-to-end verification (CLS, 384 dims, prefix) + smoke entry | SK-tr, same branch | 0.37.0 | pending |
| E5BGE-4 | Unigram tokenizer spike → placement decision | SK-tr (or engine per CP-3) | 0.38.0 | pending |
| E5BGE-5 | Unigram impl + multilingual parity + special-token generalization | per CP-3 | 0.38.0 | pending |
| E5BGE-6 | E5 end-to-end verification + docs + benchmark matrix entry | SK-tr | 0.38.0 | pending |
| E5BGE-7 | leaf-cli: `embedQuery`/`embedDocument` wiring + dim recorded in index | SK-leaf `feature/*` → `develop` | after 0.37.0 | pending |

## Checkpoints (upstream-change gates)

- **CP-1 — Phase 1 code complete** (E5BGE-1..2): full build + `apiCheck` green,
  pooling/prefix unit tests pass. *Engine changes required: none.*
- **CP-2 — BGE verified** (E5BGE-3): `fromHuggingFace("BAAI/bge-small-en-v1.5")`
  produces 384-dim CLS-pooled embeddings; parity vs sentence-transformers reference
  vectors; PR to `develop`; ship as **0.37.0**. *Gate: confirm no engine release is on
  the critical path — if anything surfaced, stop and file engine issues first.*
  **Gate outcome (2026-07-14):** one engine gap surfaced — `SafeTensorsParametersLoader`
  cannot skip BGE's persisted I64 `embeddings.position_ids` buffer
  ([SKaiNET#822](https://github.com/SKaiNET-developers/SKaiNET/issues/822)). Bridged by
  the interim `FloatSafeTensorsLoader` in llm-providers (permute-handler pattern from
  0.36.0: local, documented, dropped when the engine fix ships). **No engine release on
  the 0.37.0 critical path.**
- **CP-3 — Tokenizer placement decision** (E5BGE-4): spike Unigram-from-tokenizer.json;
  measure normalizer drift on a multilingual corpus. **Decision recorded here**:
  transformers-side vs engine-side. *This is the explicit "are upstream-engine changes
  required?" gate for Phase 2.*
- **CP-4 — E5 verified** (E5BGE-5..6): parity ≤ 1e-5 vs sentence-transformers on the
  parity corpus (drift from the normalizer approximation documented), smoke green;
  ship as **0.38.0**.

## Benchmark baseline hook

Once CP-2 lands, the embedding benchmark matrix becomes
`{mdbr-leaf-ir, bge-small-en-v1.5} × {DIRECT, OPTIMIZED}`, extended by
`multilingual-e5-small` at CP-4 — measured with the existing `llm-performance`
harness plus the `PhaseProfile` diagnostic (branch `perf/llama-packed-load-memory`)
for the non-matmul tail.
