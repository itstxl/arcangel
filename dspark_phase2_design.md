# Phase 2 Design Spec: Native DSpark (`dspark-Inkling-2-Mini`) in ArcAngel

Source-verified architecture specification, GGUF inventory, target hidden-state tap mapping, forward-pass equations, Lunar Lake bandwidth budget, and proposed C++/SYCL module design for integrating Inkling 2 Mini's native **DSpark** drafter into [`alpha2beta/arcangel`](https://github.com/alpha2beta/arcangel/tree/lunar-lake-optimizations).

> **Ground-truth sources verified:** `ggml-org/llama.cpp` ([`src/models/dflash.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/dflash.cpp), [`src/models/dspark.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/dspark.cpp), [`src/models/inkling2.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/inkling2.cpp), [`src/models/inkling2-loop.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/inkling2-loop.cpp), [`src/llama-model.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp), [`common/speculative.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/common/speculative.cpp)), [`thinkingmachines/Inkling-2-Mini/config.json`](https://huggingface.co/thinkingmachines/Inkling-2-Mini/raw/main/config.json), and the parsed GGUF header of [`unsloth/Inkling-2-Mini-GGUF/dspark-Inkling-2-Mini-Q8_0.gguf`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).

---

## 1. Verified Model Identity & Differences vs. Generic DFlash/DSpark

Generic DFlash/DSpark descriptions (e.g., for Qwen3 or DeepSeek-V4) differ in several critical ways from Inkling 2 Mini's actual `dspark-Inkling-2-Mini-Q8_0.gguf`:

1. **Mainline vs. Legacy GGUF:** Target the **mainline 58-tensor format** (`dspark-Inkling-2-Mini-Q8_0.gguf`, 503.9 MB, `general.size_label = 368M`; or `dspark/dspark-Inkling-2-Mini-BF16-mainline.gguf_file`, 0.75 GB), **not** the older 54-tensor Unsloth-pin format (`dspark-Inkling-2-Mini-Q8_0_Old.gguf`, 402.3 MB). Mainline uses unpadded `[256, 200058]` `markov_w1`/`markov_w2`, explicit `markov_w2.weight` + `markov_w2.bias`, and pre-softmaxed `ctx_mix` weights.
2. **No shared `fc.weight`:** Instead of projecting concatenated target states through a single `fc.weight`, each of the 3 draft layers (plus the Markov RNN head) computes its own weighted sum (`ctx_mix` / `markov_ctx_mix`) over the 4 tapped target hidden states, applies `RMSNorm`, and projects into `attn_k_ctx` / `attn_v_ctx`.
3. **No confidence head (`has_confidence_head = False`):** Draft length $K \in \{1..7\}$ is configured explicitly (e.g. `K = 3` or `K = 7`, or dynamically capped only by `n_ctx` / `max_new`).
4. **Bonus anchor (`sample_from_anchor = False`):** The block has $1 + K$ slots (`slot 0` = last committed token, `slots 1..K` = learned `mask_embd[0..K-1]`). `slot 0` is an anchor context slot only; drafted tokens are predicted at slots `1..K` (max $K = 7$ since `block_size = 8`).
5. **Interleaved SWA + Full Attention (`sliding_window = 1025`):** `sliding_window_pattern = [True, False, True]` — **layers 0 and 2** use a 1025-token sliding window, while **layer 1** uses full attention.
6. **NeoX RoPE (`LLAMA_ROPE_TYPE_NEOX`):** Unlike the target Inkling 2 Mini model (which has no RoPE and uses relative-position bias), the DSpark drafter uses **standard NeoX RoPE** (`rope.dimension_count = 128`, `rope.freq_base = 1000000.0`).

---

## 2. Complete GGUF Metadata & 58-Tensor Inventory

### 2.1 Observed GGUF Metadata (`dspark-Inkling-2-Mini-Q8_0.gguf`)

| Key | Value | Notes |
|---|---|---|
| `general.architecture` | `"dflash"` | Identified as DSpark via `markov_w1.weight` |
| `general.size_label` | `"368M"` | 503,889,952 B (~480.5 MiB) on disk (`Q8_0` + `BF16` + `F32`) |
| `dflash.block_count` | `3` | 3 decoder layers |
| `dflash.context_length` | `1048576` | Max supported RoPE position |
| `dflash.embedding_length` | `2304` | Matches target `D = 2304` |
| `dflash.feed_forward_length` | `9216` | Dense SwiGLU (`4 * D`) |
| `dflash.attention.head_count` | `24` | Query heads (`head_dim = 128`, GQA 3) |
| `dflash.attention.head_count_kv` | `8` | KV heads |
| `dflash.attention.key_length` / `value_length` | `128` | `HD = 128` |
| `dflash.rope.dimension_count` / `freq_base` | `128` / `1000000.0` | `LLAMA_ROPE_TYPE_NEOX` (per `llama-model.cpp`) |
| `dflash.attention.layer_norm_rms_epsilon` | `1e-6` | Matches target |
| `dflash.attention.sliding_window` | `1025` | `n_swa = 1025` |
| `dflash.attention.sliding_window_pattern` | `[True, False, True]` | Layer 0: SWA, Layer 1: Full, Layer 2: SWA |
| `dflash.final_logit_softcapping` | `30.0` | Applied **after** adding Markov bias |
| `dflash.block_size` | `8` | 1 anchor + up to 7 draft slots (`K <= 7`) |
| `dflash.target_layers` | `[5, 23, 41, 48]` | 4 tapped target feature vectors per token |
| `dflash.sample_from_anchor` | `False` | Predictions start at slot 1 |
| `dflash.has_confidence_head` | `False` | No `conf_proj` tensor |
| `dflash.has_markov_context` | `True` | Uses `f_rnn` from `markov_ctx_mix` |

### 2.2 Complete 58-Tensor Table (`dspark-Inkling-2-Mini-Q8_0.gguf`)

All dimensions are listed in GGML order `[ne[0], ne[1]]` (`ne[0]` is the contiguous inner / input-reduction dimension):

#### Global Tensors (10 tensors)

| Tensor Name | GGML Shape | File Type | Size (MiB) | Role |
|---|---|---|---:|---|
| `token_embd_norm.weight` | `[2304]` | `F32` | 0.01 | Applied to anchor token embedding only |
| `mask_embd.weight` | `[2304, 7]` | `F32` | 0.06 | Learned embeddings for draft slots `1..7` |
| `output_norm.weight` | `[2304]` | `F32` | 0.01 | Final RMSNorm before shared target LM head |
| `markov_ctx_mix.weight` | `[4]` | `F32` | < 0.01 | Pre-softmaxed weights over the 4 target taps |
| `markov_ctx_norm.weight` | `[2304]` | `F32` | 0.01 | RMSNorm weight for `f_rnn` |
| `joint_proj.weight` | `[5120, 768]` | `BF16` | 7.50 | RNN head gate/cand/out (`5120 = 256+256+2304+2304`) |
| `joint_proj.bias` | `[768]` | `F32` | < 0.01 | RNN head bias (`3 * 256`) |
| `markov_w1.weight` | `[256, 200058]` | `BF16` | 97.68 | Low-rank token embedding (`rank = 256`, unpadded vocab) |
| `markov_w2.weight` | `[256, 200058]` | `BF16` | 97.68 | Low-rank logit bias projection |
| `markov_w2.bias` | `[200058]` | `F32` | 0.76 | Per-token logit bias |

#### Per-Layer Tensors (`blk.{0,1,2}.*`, 16 tensors × 3 layers = 48 tensors)

| Tensor Suffix | GGML Shape | File Type | Size / Layer (MiB) | Role |
|---|---|---|---:|---|
| `ctx_mix.weight` | `[4]` | `F32` | < 0.01 | Weights over 4 target taps |
| `ctx_norm.weight` | `[2304]` | `F32` | 0.01 | RMSNorm after `ctx_mix` |
| `attn_k_ctx.weight` | `[2304, 1024]` | `Q8_0` | 2.39 | Injected context K projection |
| `attn_v_ctx.weight` | `[2304, 1024]` | `Q8_0` | 2.39 | Injected context V projection |
| `attn_k_ctx_norm.weight` | `[128]` | `F32` | < 0.01 | Per-head RMSNorm on `K_ctx` (before RoPE) |
| `attn_norm.weight` | `[2304]` | `F32` | 0.01 | Pre-attention RMSNorm |
| `attn_q.weight` | `[2304, 3072]` | `Q8_0` | 7.17 | Block query projection |
| `attn_k.weight` | `[2304, 1024]` | `Q8_0` | 2.39 | Block ephemeral key projection |
| `attn_v.weight` | `[2304, 1024]` | `Q8_0` | 2.39 | Block ephemeral value projection |
| `attn_q_norm.weight` | `[128]` | `F32` | < 0.01 | Per-head Q RMSNorm (before RoPE) |
| `attn_k_norm.weight` | `[128]` | `F32` | < 0.01 | Per-head K RMSNorm (before RoPE) |
| `attn_output.weight` | `[3072, 2304]` | `Q8_0` | 7.17 | Attention output projection |
| `ffn_norm.weight` | `[2304]` | `F32` | 0.01 | Pre-FFN RMSNorm |
| `ffn_gate.weight` | `[2304, 9216]` | `Q8_0` | 21.52 | SwiGLU gate |
| `ffn_up.weight` | `[2304, 9216]` | `Q8_0` | 21.52 | SwiGLU up |
| `ffn_down.weight` | `[9216, 2304]` | `Q8_0` | 21.52 | SwiGLU down |

- **Total draft weights:** ~203.7 MiB (global) + 3 × ~88.5 MiB (layers) ≈ **469.2 MiB** (+ ~11.4 MiB GGUF tokenizer header = **480.5 MiB / 503.9 MB**).
- **Shared target tensors (0 extra bytes):** Borrows `m_.tok_embd` and `m_.output` (already resident in UMA). Note that `m_.output` has `201024` padded rows; only the first `n_vocab = 200058` rows are computed/added to `markov_w2` output.

---

## 3. Target Hidden-State Tap Mapping in ArcAngel

In `llama.cpp` ([`inkling2-loop.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/inkling2-loop.cpp#L85-L122) and [`inkling2.cpp`](https://github.com/ggml-org/llama.cpp/blob/master/src/models/inkling2.cpp#L445)), `t_layer_inp[0..47]` records the residual stream **entering** each of the 48 layer visits, and `t_layer_inp[48]` records `output_norm(mix(blocks, x, output_res_query))` right before `m_.output` (the LM head).

For `dflash.target_layers = [5, 23, 41, 48]`, the exact correspondence in ArcAngel's `FastEngine` is:

| Tap Index | `target_layers[k]` | Schedule Visit / Stage | Exact ArcAngel Buffer at Capture Point |
|---:|---:|---|---|
| `0` | `5` | Entering visit `5` (physical layer 5, end of prelude) | `xa_` / `bxa_` / `vxa_` **before** `visit(vs[5])` |
| `1` | `23` | Entering visit `23` (pass 1, physical layer 11) | `xa_` / `bxa_` / `vxa_` **before** `visit(vs[23])` |
| `2` | `41` | Entering visit `41` (pass 2, physical layer 17) | `xa_` / `bxa_` / `vxa_` **before** `visit(vs[41])` |
| `3` | `48` | Final `output_norm` output (before LM head) | `f_` / `bf_` / `vf_` **after** `out_mix` |

### Target Tap Buffer (`tgt_feats_`)
- Add a lightweight optional tap buffer on `FastEngine` (allocated only when DSpark is attached):
  - `float * tgt_feats_ = nullptr;` sized `[max(TB_, VFAST)][4][D]` (~`36 KB` for `VFAST=8`, ~`18 MB` for `TB_=512` — or aliased onto a dedicated slice outside `pool_` during `prefill()`).
  - During `prefill()` (last chunk or all chunks that feed `inject()`), `step()`, and `verify()`, copy `[n][D]` from `xa_` before visits `5, 23, 41` and from `f_` after `out_mix` into `tgt_feats_[tok][0..3][D]`.
  - **Zero host round-trip:** `DsparkEngine::inject()` reads `tgt_feats_` directly in device/UMA memory!

---

## 4. Bit-Exact Forward Pass Equations

Let $D = 2304$, $H = 24$, $H_{kv} = 8$, $d_h = 128$, $R = 256$ (`markov_rank`), $V = 200058$, $\varepsilon = 10^{-6}$.

### Mode A — Context Injection (`build_inject_ctx_mix`)
For each committed token at position $p$ with tapped target vectors $T_0, T_1, T_2, T_3 \in \mathbb{R}^D$:
1. **Markov context state (`f_rnn[p]`):**
   $$f_{\text{rnn}}[p] = \text{RMSNorm}\!\left(\sum_{k=0}^{3} w^{\text{mix,mk}}_k \, T_k,\; \varepsilon\right) \odot w^{\text{norm,mk}}$$
   Stored in a ring buffer of `block_size + 1 = 9` slots indexed by `p % 9`.
2. **Per-layer injected KV (`il ∈ {0, 1, 2}`):**
   $$c_{il} = \text{RMSNorm}\!\left(\sum_{k=0}^{3} w^{\text{mix}}_{il,k} \, T_k,\; \varepsilon\right) \odot w^{\text{ctx\_norm}}_{il}$$
   $$K^{\text{ctx}}_{il}[p] = \text{RoPE}_{\text{NeoX}}\!\left(\text{RMSNorm}_{\text{head}}\!\left(W^{k,\text{ctx}}_{il} c_{il},\; \varepsilon\right) \odot w^{\text{k\_ctx\_norm}}_{il},\; p\right), \quad V^{\text{ctx}}_{il}[p] = W^{v,\text{ctx}}_{il} c_{il}$$
   Written to the 3-layer draft KV cache at `(il, p)` (layers `0, 2` use SWA ring; layer `1` uses full context).

### Mode B — Parallel Block-Diffusion Backbone (`1 + K` tokens, $K \le 7$)
Given anchor token `id_last` at `pos0` and block size `M = 1 + K`:
1. **Input embeddings `X[0..M-1]`:**
   - `X[0] = RMSNorm(tok_embd[id_last], eps) * dspark.tok_norm` (uses target `tok_embd`, DSpark's `token_embd_norm`).
   - `X[i] = mask_embd[i - 1]` for `i = 1..K` (raw F32 row, **not** multiplied by `tok_norm`).
2. **For each layer `il ∈ {0, 1, 2}`:**
   - `h = RMSNorm(X, eps) * blk.attn_norm`
   - `Q = RoPE_NeoX(RMSNorm_head(W_q h, eps) * attn_q_norm, pos0 + i)`
   - `K_blk = RoPE_NeoX(RMSNorm_head(W_k h, eps) * attn_k_norm, pos0 + i)`
   - `V_blk = W_v h`
   - **Non-causal block-uniform attention:** Each query at slot `i ∈ [0, M)` attends over:
     - All injected context KV keys `k ∈ [k_lo, pos0 - 1]`, where `k_lo = max(0, pos0 - 1025 + 1)` for SWA layers (`0, 2`) and `k_lo = 0` for layer `1` (note `block_uniform`: all slots in the block share the anchor's `k_lo`), **plus**
     - All `M` ephemeral block keys `K_blk[0..M-1], V_blk[0..M-1]` (bidirectional across the block, **not** written permanently to the draft KV cache).
     - Attention scale = `1 / sqrt(128) = 0.0883883476f`.
   - `X = X + W_o @ attn_out`
   - `fn = RMSNorm(X, eps) * blk.ffn_norm`
   - `X = X + W_down @ (SiLU(W_gate fn) * (W_up fn))`
3. **Hidden state & base logits:**
   - `hidden[0..M-1] = RMSNorm(X, eps) * dspark.output_norm`
   - `base_logits[1..K] = m_.output[:V] @ hidden[1..K]` (**uncapped**! Slot `0` logits are unused because `sample_from_anchor = False`).

### Mode C — Serial Markov RNN Head (`markov_head_type = "rnn_hidden"`)
Initialize `prev = id_last`, `state = zeros(R = 256)`, and read `f_rnn = f_rnn_ring[(pos0 - 1) % 9]` (for the token immediately preceding the block).

For each draft slot `i = 1 .. K`:
1. `w1_prev = markov_w1[prev]` (`[256]`, BF16 -> F32)
2. `joint_in = concat(state [256], w1_prev [256], hidden[i] [2304], f_rnn [2304])` (`[5120]`)
3. `j_out = joint_proj @ joint_in + joint_proj_bias` (`[768]`, chunked into `g [0..256)`, `c [256..512)`, `m [512..768)`)
4. `gate = sigmoid(g)`, `cand = tanh(c)`
5. `state = cand + gate * (state - cand)`  *(equivalent to `gate * state + (1 - gate) * cand`)*
6. `markov = tanh(m)` (`[256]`)
7. `bias = markov_w2 @ markov + markov_w2_bias` (`[200058]`)
8. **Softcap AFTER bias:**
   $$\text{logits}_i = 30 \cdot \tanh\!\left(\frac{\text{base\_logits}[i] + \text{bias}}{30}\right)$$
9. **Token pick & distribution:**
   - **Greedy (`temp <= 0`):** `prev = argmax(logits_i)`; emit `prev`.
   - **Probabilistic (`temp > 0`):** Draw `prev` from top-64 softmax of `logits_i / temp` (or full `filtered_probs`), and record draft probabilities $q_i$ for lossless target rejection sampling (`accept` with $\min(1, p_{\text{target}}(d_i)/q_i(d_i))$; on rejection sample from $\text{norm}(\max(0, p_{\text{target}} - q_i))$).

---

## 5. Arc 140V (`UD-Q4_K_M`, 106 GB/s) Bandwidth & Latency Budget

### 5.1 Draft Pass Latency Breakdown (106 GB/s LPDDR5X-8533)

| Stage | Weight Bytes Read | Est. Time (`K = 3`, 4 tok) | Est. Time (`K = 7`, 8 tok) | Notes |
|---|---:|---:|---:|---|
| **KV inject** (per accepted token, amortized) | ~14.4 MB (`3 × 2 × 2.4 MB`) | ~0.15 ms (for 1 tok) | ~0.15 ms (for 1 tok) | Only runs for newly committed tokens |
| **3-layer backbone** (`Q8_0R` batched) | ~265.5 MB | ~2.55 ms | ~2.70 ms | Weight-read dominated; compute is tiny for `M <= 8` |
| **Shared LM head** (`m_.output`, `Q8_0R` or `Q4_KR`) | ~230–280 MB | ~2.5–3.0 ms (`M=3`) | ~3.0–3.5 ms (`M=7`) | `output.weight` read once across slots `1..K` |
| **Markov RNN head (`K` serial steps):** `joint_proj` + `w2` (`BF16`) | `7.5 MB + 97.7 MB` per slot | ~3.1 ms (`3 × 1.0 ms`) | ~7.2 ms (`7 × 1.0 ms`) | `markov_w2` (`256 × 200058` BF16) is read `K` times! |
| **Total DSpark Draft Step (as-is BF16 `w2`)** | — | **~8.3–8.8 ms** | **~13.0–13.5 ms** | Plus ~0.3–0.6 ms inject for `a+1` committed tokens |

### 5.2 Key Lunar Lake Optimizations
1. **Repack `markov_w2` from `BF16` -> `Q8_0R` (or fuse `w2` inside `k_vargmax`):**
   - `markov_w2` is `[256, 200058]` BF16 (`97.7 MiB`), read **once per draft slot** (3× at `K=3`, 7× at `K=7`).
   - Repacking `markov_w2` to `Q8_0R` at load cuts each step's read from `97.7 MiB` to `51.8 MiB` (-47% bandwidth!), dropping per-slot Markov latency from `~1.0 ms` to `~0.52 ms` (**`K=3` total draft ~6.8–7.3 ms**, **`K=7` total draft ~9.7–10.2 ms**) with negligible quantization noise on a rank-256 bias vector.
2. **Reuse `k_mv_batch` (`VB = 4` / `8`) from Phase 1:** All 3 backbone layers (`Q/K/V/O` and `gate/up/down`) are `Q8_0` in `dspark-Inkling-2-Mini-Q8_0.gguf`, so repacking them to `Q8_0R` via [src/model.cpp](src/model.cpp) lets DSpark directly reuse Phase 1's `k_mv_batch<q::Q8_0R, VB>` and `m_.output`!
3. **Draft KV Cache Footprint:** Only **1 of 3 layers** (layer 1) is full-attention, and layers 0 & 2 use a 1025-token ring. At 8-bit KV (matching `kv_bits`), 128k context in the draft cache needs only:
   - Layer 1 (full, 128k): `~272 MiB`; Layers 0 & 2 (SWA 1025): `~4.4 MiB` total → **~277 MiB** (or **~145 MiB** at 4-bit KV).
   - Combined with ~470 MiB draft weights = **~0.6–0.75 GiB total**, leaving > 9 GiB free on a 32 GB Lunar Lake machine at 128k context.

### 5.3 Net Speedup Projection on Arc 140V (`UD-Q4_K_M`, 21.1 tok/s = 47.4 ms/tok)
Using the Phase 1 verify cost model (`verify(4) ≈ 90 ms`, `verify(8) ≈ 129 ms`) + DSpark draft + inject (`~7.5 ms` at `K=3`, `~10.5 ms` at `K=7`):
- **At `K = 3` (total round = `~97.5 ms`), mean `2.2–2.6` tokens/step (40–53% acceptance):** `22.6–26.7 tok/s` (**1.1×–1.27×** on general prose/reasoning).
- **At `K = 7` (total round = `~139.5 ms`), mean `3.5` tokens/step (published DSpark average) to `6.4` tokens/step (structured/code):**
  - Average workload (`3.5` tok/round): **`25.1 tok/s` (~1.19×)** short context, scaling higher at **16k–128k context** where target attention grows while DSpark stays cheap.
  - Structured / coding workload (`5.0–6.4` tok/round): **`35.8–45.9 tok/s` (~1.7×–2.17×)**.

---

## 6. Proposed ArcAngel C++/SYCL Module Design

### 6.1 New Files & Classes (`src/dspark.h`, `src/dspark.cpp`)

- **`DsparkModel`**:
  - Opens `dspark-Inkling-2-Mini-*.gguf` via existing `GgufFile`, validates `general.architecture == "dflash"`, `dflash.block_size == 8`, `target_layers == [5, 23, 41, 48]`, `dflash.has_markov_context == true`, and `embedding_length == target.hp.n_embd`.
  - Repacks `Q8_0` backbone tensors to `Q8_0R` (and optionally `markov_w2` `BF16 -> Q8_0R`, toggleable via `INK_DSPARK_W2_Q8=1`).
- **`DsparkEngine`**:
  - Holds `sycl::queue & q_`, `const Model & target_`, `const DsparkModel & dm_`, 3-layer KV cache (`kc_`, `vc_`, `ks_`, `vs_`), `f_rnn_ring_[9][D]`, and recorded `ExecGraph`s for `inject_1` (single token), `draft_4` (`K=3`), and `draft_8` (`K=7`).
  - Key methods:
    - `void inject(const float * tgt_feats_dev, int n, int pos0)` — runs Mode A for `n` newly committed target tokens (`prefill` tail or `a + 1` accepted tokens).
    - `int draft_greedy(int last_tok, int pos0, int k, int32_t * out_draft)` — runs Mode B + Mode C (`temp <= 0`) inside a single recorded graph (copies only `k * 4` bytes back to host!).
    - `void draft_prob(int last_tok, int pos0, int k, float temp, float * out_logits)` (or top-64 `(ids, probs)` packs) — runs Mode B + Mode C for `temp > 0`.
  - Snapshot/rollback integration: `f_rnn_ring_` is tiny (`9 * 2304 * 4 = 83 KB`), and SWA draft rings (`1025 + TB_`) never overwrite live keys on rejected target tokens as long as `inject()` is called **only after `commit(a + 1)`**!

### 6.2 C API & Server Lifecycle (`src/capi.cpp`, `server/inkling_server.py`)

```cpp
// Proposed C API additions in src/capi.cpp
int ink_dspark_open(void * hp, const char * dspark_gguf_path);
int ink_dspark_draft(void * hp, int32_t last_tok, int pos0, int k, int32_t * out_toks, float * out_logits);
```

Exact round lifecycle in `Server.generate()`:
1. `load_prompt(toks)` prefills target (capturing `tgt_feats_` for the prompt tokens) and calls `dspark.inject(pos0 .. len(toks))`.
2. Each decoding round from committed token `t` at `pos`:
   - `draft = dspark.draft(t, pos, K)` (`K = 3` or `7`).
   - `batch = [t] + draft`.
   - Run Phase 1 `eng.verify_argmax(batch, pos)` (or `eng.verify(batch, pos)` + lossless rejection sampling).
   - `eng.commit(a + 1)` (commits `t` + `a` accepted tokens).
   - `dspark.inject()` for the `a + 1` committed tokens (`pos .. pos + a`) using the `tgt_feats_` already saved during `verify()`!
   - Emit accepted tokens and repeat with `t = next_tok`, `pos += a + 1`.

### 6.3 Phase 2 Validation Plan

1. **Tensor-by-tensor CPU/GPU parity:** Dump `f_rnn`, `Kcur_injected`, `inp_noise_embd`, `l_out` (layers 0..2), `base_logits`, and Markov `logits[1..7]` from `llama.cpp` (`ref-dump` extended for `dflash` + `dspark`) and verify KL `<= 0.002` and 100% greedy top-1 match in `tests/test_dspark.cpp`.
2. **Server parity:** Verify greedy determinism (`--spec-dspark` vs. baseline) and lossless statistical equivalence under `--temperature > 0`.
3. **Hardware sweep on Arc 140V:** Benchmark `K ∈ {2, 3, 5, 7}` and `INK_DSPARK_W2_Q8=0|1` across short (`2k`) and long (`16k`, `64k`) contexts on both coding and reasoning traces.
