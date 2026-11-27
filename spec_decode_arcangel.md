# Speculative Decoding on ArcAngel (`lunar-lake-optimizations`)

Feasibility assessment, two-phase roadmap, and implementation-ready Phase 1 kernel/data-layout design for adding speculative decoding to [`alpha2beta/arcangel`](https://github.com/alpha2beta/arcangel/tree/lunar-lake-optimizations) targeting Intel Lunar Lake iGPUs (Arc 140V / `lnl_m`, 32 GB UMA, `UD-Q4_K_M`) as well as discrete Battlemage GPUs.

---

## 1. Executive Summary & Feasibility Verdict

**Speculative decoding is very feasible on `lunar-lake-optimizations`**, and **Lunar Lake (Arc 140V, 32 GB UMA, `UD-Q4_K_M`) is a stronger fit for it than the 12 GB Arc B580** for three reasons:

1. **Higher shared-weight fraction:** On Arc 140V with `UD-Q4_K_M` (`47.4 ms/tok`, `21.1 tok/s`), dense projections + LM head (`proj`, `wo`, `head`) account for **~18.0 ms (~38%)** of a decode step (vs. ~31% on B580 `Q2_K_XL`), and CPU expert offloading (`INK_OFFLOAD`) is unused on UMA.
2. **Plenty of UMA headroom for a real drafter:** `UD-Q4_K_M` + 128k 4-bit KV uses ~20 GiB of 30.9 GiB OS-visible RAM, leaving ~10 GiB available—easily fitting Inkling 2 Mini's native **DSpark** draft model (~0.4–0.65 GiB) without reducing context length.
3. **Rollback plumbing already exists:** `FastEngine::verify`, `commit`, `draft`, and sliding-window ring buffers oversized to `n_swa + TB_` are already in the codebase.

**The catch:** Wiring the existing `ink_verify` directly into [server/inkling_server.py](server/inkling_server.py) will **not** yield a speedup. Real gains require a **dedicated small-batch decode verify path** ($T \in \{2..8\}$) first, because the current `verify()` runs through the prompt-prefill path and costs ~7–9× a single decode step.

---

## 2. Existing Scaffolding & Why Current `verify()` Is Too Slow

### 2.1 What Is Already in the Repository

- **Verify + convolution-history rollback:** `FastEngine::verify(tokens, pos0)` and `FastEngine::commit(keep)` in [src/prefill.cpp](src/prefill.cpp#L1270-L1348) (up to `VMAX = 16` tokens), exposed via `ink_verify` / `ink_commit` in [src/capi.cpp](src/capi.cpp#L74-L97) and checked for KL parity and rollback correctness in [tests/test_verify.cpp](tests/test_verify.cpp).
- **Safe KV overwrite without KV rollback:** In [src/fast.cpp](src/fast.cpp#L960-L962), sliding-window KV rings are sized to `min(n_swa + TB_, n_ctx)` where `TB_ >= 128 >= VMAX = 16`. Speculatively writing up to 16 tokens never overwrites live keys in the active 1024-token window, so rejected KV slots need no restoration—only the 4-tap convolution history (`hist_`) does, and `commit(keep)` already handles it.
- **Early-exit prototype & simulators:** `FastEngine::draft(token, pos, n_pass)` in [src/fast.cpp](src/fast.cpp#L1917-L1948), plus trace/cost-model simulators [bench/draft_sim.py](bench/draft_sim.py), [bench/spec_sim.py](bench/spec_sim.py), and [bench/early_exit.py](bench/early_exit.py).

Baseline measurement command on target hardware:
```bash
build/test-verify MODEL.gguf 2000 8192
```

### 2.2 Five Bottlenecks in the Existing `prefill.cpp` `verify()` Path

1. **LM head runs `n` separate times** ([src/prefill.cpp](src/prefill.cpp#L1312)): `for (int r = 0; r < n; ++r) run_bmm("b_head", ..., 1)` re-reads `output.weight` (`4.64 ms` on Arc 140V) $n$ separate times instead of once (~32 ms wasted for `n = 8`).
2. **XMX expert tiles are massively under-filled** ([src/prefill.cpp](src/prefill.cpp#L169-L178), [L1079-L1102](src/prefill.cpp#L1079-L1102)): `k_gxmm` uses `GT = 64, GR = 128` tiles tuned for 512-token chunks. With 4–8 verify tokens, each chosen expert receives only 1–3 pairs, leaving >95% of every XMX tile zero-padded while still decoding weights to FP16 in SLM. Disabling XMX (`INK_VERIFY_XMX=0`) falls back to scalar FP32 `step_decode` instead of int8 DP4A decode kernels (`k_gate_up_r` / `k_down_q8`), which is even slower.
3. **Prefill attention starves the GPU at small `n`** ([src/prefill.cpp](src/prefill.cpp#L681-L698)): For $n \le 16$, `nqb = 1` and `nkv = 8`, launching only **8 work-groups total** (1 per Xe2 core on Arc 140V, no split-K parallelism across long context).
4. **No SYCL command graph:** `verify()` eagerly submits kernels across all 48 visits instead of replaying a recorded `command_graph`.
5. **Full host logits transfer + CPU softcap** ([src/prefill.cpp](src/prefill.cpp#L1321-L1324)): Copies `n × 804 KB` back to host and runs `std::tanh` on CPU (no batched GPU `k_argmax` for greedy mode).

---

## 3. Lunar Lake Performance Physics & Drafter Comparison

### 3.1 Arc 140V (`UD-Q4_K_M`) Decode Breakdown & MoE Batch Scaling

From [PERF.md](PERF.md#L232-L252) and [arcangel_140v.md](arcangel_140v.md#L28-L45), one short-context decode step takes **47.4 ms** (`21.1 tok/s`, ~103–106 GB/s LPDDR5X-8533):

- **Dense projections + LM head (`Q8_0R`, 100% shared across $T$ tokens):** `proj` (9.8 ms) + `wo` (3.6 ms) + `head` (4.6 ms) = **~18.0 ms (38%)**.
- **MoE experts (`NVFP4R` + `Q5_KR`, shared only when tokens pick the same expert):** `gate_up` (13.5 ms) + `down` (~7.5–8.5 ms) = **~21.5–22.5 ms (~46%)**.
- **Attention + small kernels (KV reads shared across $T$ tokens):** **~7 ms** at short context, growing to **~14 ms** at 16k+ context.

Because Inkling 2 Mini routes **top-8 of 128 experts**, the distinct-expert share $f(T)$ measured in [bench/spec_sim.py](bench/spec_sim.py#L16) governs verify cost:

| Batch size $T = 1 + K$ | Distinct experts / layer (vs 8) | Est. batched-decode verify time (140V, short ctx) | Cost in decode steps |
|---|---:|---:|---:|
| $T = 1$ (normal decode) | 8.0 (`1.00×`) | 47.4 ms | `1.00×` |
| $T = 2$ ($K = 1$) | 13.1 (`1.64×`) | ~64 ms | `1.35×` |
| $T = 4$ ($K = 3$) | 21.1 (`2.64×`) | ~90 ms | `1.89×` |
| $T = 5$ ($K = 4$) | 24.8 (`3.10×`) | ~102 ms | `2.15×` |
| $T = 8$ ($K = 7$) | 32.6 (`4.08×`) | ~129 ms | `2.72×` |

> **Subtle win for Lunar Lake:** On Arc 140V, `NVFP4R` experts run at ~58–61 GB/s (partly unpack-instruction-bound, like B580). Grouping pairs by expert and unpacking `fp4x4` / `ue4m3` **once** per shared expert across $M$ tokens amortizes both LPDDR5X bandwidth and unpack instructions. At **long context (16k–128k)**, reading quantized KV once across $T$ queries improves the verify ratio further.

### 3.2 Drafter Options Compared

| Approach | Extra Memory (32 GB UMA) | Expected Speedup on 140V (`UD-Q4_K_M`) | Implementation Effort | Verdict |
|---|---|---|---|---|
| **1. DSpark (native Inkling 2 Mini draft GGUF, e.g. `dspark-Inkling-2-Mini-Q8_0.gguf`)** | ~0.4–0.65 GiB (shares `tok_embd` & `output` head) | **~1.35×–1.6× typical** (~28–34 tok/s), **up to ~1.8×–2.0×** on code (~38–42 tok/s); stronger at 16k+ ctx | **High** — batched decode verify + hidden-state tap (4 target visits) + DSpark `dflash`/Markov-head loader & SYCL kernels + rejection sampling | **Best long-term payoff.** Only option that speeds up *all* generation (reasoning + prose + code). |
| **2. Prompt-Lookup / N-gram Copy Drafting** (simulated in [bench/spec_sim.py](bench/spec_sim.py)) | 0 GiB (CPU suffix hash map) | **~1.20×–1.25× overall** on coding traces (gated at `match >= 8`, `K = 4..8`); **~1.6×–2.3×** on file rewrites / quoted code; **1.0×** on fresh text | **Medium** — batched decode verify + ~60 lines in `inkling_server.py` | **Best Phase 1 milestone.** Validates the batched-decode verify engine with zero draft-model complexity. |
| **3. Untrained Early-Exit Self-Drafting** (1 of 3 loop passes, `ink_draft`) | 0 GiB | **~1.0×–1.2×** at best (~72–75% top-1 agreement, but drafting 24/48 visits + head costs ~26 ms per draft token) | **Medium-High** (requires reusing draft's prelude + pass-0 state inside verify to beat `1.0×`) | **Not worth it** without a trained early-exit / 2DSD head. |

---

## 4. Two-Phase Implementation Roadmap

- **Phase 1 — Fast Small-Batch Decode Verify + Copy Drafting:**
  1. Implement recorded batched-decode verify graphs (`VB ∈ {2, 4, 8}`) with dense batched GEMMs (`k_mv_batch`), parallel single-WG expert bucketing (`k_vgroup`), unpack-once DP4A grouped experts (`k_vgate_up_r`, `k_vdown_q8`), batched split-K attention (`k_vattn`, `k_vcombine`), fused conv/mix/ctrl glue kernels, and batched GPU `k_vargmax`.
  2. Expose `ink_verify_argmax` in [src/capi.cpp](src/capi.cpp) and wire gated n-gram copy drafting (`minmatch >= 8`, `K = 3` or `7`) with greedy + lossless probabilistic acceptance into [server/inkling_server.py](server/inkling_server.py).
  3. Validate numerical parity and speedups via `test-verify` and real request replay.
- **Phase 2 — Native DSpark Drafter Support:**
  1. Tap target hidden states at DSpark's `target_layers` visits (e.g., `[5, 23, 41, 48]`) into a small device buffer during `step()` and `verify()`.
  2. Add `src/dspark.*` to load `dspark-Inkling-2-Mini-*.gguf` (`dflash` architecture) reusing target `tok_embd` and `output`, executing the 2–3 layer parallel backbone + low-rank serial Markov head + confidence head (dynamic $K \le 7$).
  3. Plug DSpark into the Phase 1 verify/commit loop (optionally combined with copy drafting on low confidence).

---

## 5. Phase 1 Detailed Design: Kernel Signatures & Data Layouts

Aligned with the exact types, constants (`SG = 16`, `WG = 256`, `RPG = 16`, `D = 2304`, `HD = 128`, `INK_SWA_AWG = 128`, `INK_GUR = 2` on `lnl_m`), and weight layouts in [src/kernels.h](src/kernels.h), [src/quant.h](src/quant.h), and [src/fast.cpp](src/fast.cpp).

### 5.1 Execution Model & SYCL Graph Strategy

- **Fixed compile-time verify buckets:** `VB ∈ {2, 4, 8}` (`constexpr int VBUCKETS[3] = {2, 4, 8}`).
  - Any `verify(tokens, pos0)` with `n ∈ [1, 8]` selects the smallest bucket `VB >= n` (`n == 1` can dispatch directly to `step()`/`step_argmax()` when no rollback is needed, or bucket `VB = 2`).
  - Requests with `n ∈ [9, 16]` (for `test-verify` compatibility) or `INK_FAST_VERIFY=0` fall back to the existing prefill-path `verify()` in [src/prefill.cpp](src/prefill.cpp#L1270).
- **Lazy graph recording:** Each bucket `bi ∈ {0, 1, 2}` records its own `ExecGraph` (`vgraphs_[bi]`) and `vjob_cache_[bi]` on first use. Zero startup latency when speculative decoding is disabled.
- **Dynamic runtime inputs (device scalars):** Like `token_` and `pos_` in `FastEngine::step()`, each replay updates three small device buffers before `q_.ext_oneapi_graph(*vgraphs_[bi])`:
  - `int32_t * vtok_` (`[8]`): padded tail repeats `tokens[n - 1]` so `embed`/dense projections never read invalid token IDs.
  - `int32_t * vpos0_` (`[1]`): starting sequence position `pos0`.
  - `int32_t * vntok_` (`[1]`): true token count `n <= VB` (gates KV writes, attention masking, and expert pair counting so padded slots are inert).
- **UMA / Offload boundary:** On Lunar Lake (`lnl_m`), `cpu_` is always null. If `cpu_` is active (discrete B580 offload), Phase 1 falls back to the existing `verify()` until batched `CpuExperts` is added.

### 5.2 Data Layouts & Device Memory Footprint

All activation arrays use **token-major `[VB][stride]`** layout with every `stride` a multiple of 16 floats (64-byte aligned), guaranteeing coalesced 256-byte sub-group reads across lanes in dense GEMMs, DP4A expert kernels, and 4-tap convolutions.

#### New `FastEngine` Device Buffers (in [src/fast.h](src/fast.h), allocated lazily on first fast `verify()` call)

| Buffer | Type | Shape (at `VFAST = 8`) | Size | Notes |
|---|---|---|---:|---|
| `vtok_`, `vpos0_`, `vntok_` | `int32_t` | `[8]`, `[1]`, `[1]` | < 1 KB | Dynamic graph inputs |
| `vxa_`, `vxb_`, `vprelude_` | `float` | `3 × [8][D]` | 216 KB | Main residual stream |
| `vstore_` | `float` | `[n_vectors][8][D]` | ~864 KB | **FP32** (not f16) to match `step()` KL |
| `vin_`, `vinn_`, `vmsg_`, `vmsgn_`, `vctl_`, `vnin_` | `float` | `(3+3+4+4+3+3) × [8][D]` | 1.44 MB | Attention/loop-control activations (eliminates `bgain_`/`bforget_`) |
| `vg_` | `float` | `[4][8][64]` | 8 KB | Loop controller rank-64 bottleneck |
| `vproj_` | `float` | `[8][8704]` | 272 KB | Same per-token offsets as `proj_` (`Q:0, K:3072, V:4096, R:5120, sink:5504, gate:5528, kdyn:8600, vdyn:8632`) |
| `vqf_`, `vrf_`, `vatt_` | `float` | `[8][3072]`, `[8][384]`, `[8][3072]` | 204 KB | Post-processed Q, R, and attention output |
| `vf_`, `vfraw_`, `vfn_`, `vo_` | `float` | `4 × [8][D]` | 288 KB | FFN input + `wo`/expert output buffer |
| `vfq_`, `vfqd_`, `vfqs_` | `int8_t`, `float` | `[8][D]`, `2 × [8][72]` | 23 KB | Quantized FFN input for `k_vgate_up_r` (fits in L1) |
| `vact_` | `float` | `[64][1152]` | 288 KB | Indexed by `pair = tok * 8 + k` |
| `vaq_`, `vaqd_`, `vaqs_` | `int8_t`, `float` | `[64][1152]`, `2 × [64][36]` | 90 KB | Quantized expert activations for `k_vdown_q8` |
| `vrouter_`, `vids_`, `vwts_` | `float`, `int32_t` | `[8][256]`, `[64]`, `[64]` | 9 KB | Router logits, top-8 IDs & weights |
| `vpairs_`, `vtiles_` | `int32_t` | `[64]`, `[1 + 3 * 64]` | 1 KB | Grouped expert tile descriptors (`V_MMAX = 4` pairs/tile) |
| `vargmax_val_`, `vargmax_idx_` | `float`, `int32_t` | `[8][WG]`, `[8][WG]` | 16 KB | Batched 2-stage GPU argmax scratch |
| `vpart_` | `float` | `[8][24][max_vchunks_][130]` | 12.8 MB (32k) / 51 MB (128k) | Split-K attention partials; can **alias onto prompt `pool_`** (idle during `verify()`, 54–108 MB for `TB = 256/512`) |
| `vraw_`, `vlogits_`, `vhist_` | *(existing)* | `[48][16][6656]`, `[16][V]`, `[48]` | *(already allocated)* | Reused directly by `commit(keep)` |

**Total new memory:** **~3.7 MB** (aliasing `vpart_` onto `pool_`), or **~16.5 MB** at 32k context if allocated separately.

#### Core Descriptor Structs (add to [src/kernels.h](src/kernels.h))

```cpp
// Batched decode matrix-vector job (device-side table for graph replay, like MvJob)
struct VMvJob {
    WRef w;
    uint32_t type = 0;
    int cols = 0, rows = 0, row0 = 0;
    int split = 1 << 30;
    const float * x = nullptr;
    const float * x2 = nullptr;
    size_t xs = 0;       // token stride of x and x2 (in floats)
    float * y = nullptr;
    size_t ys = 0;       // token stride of y (in floats)
    float cap = 0.0f;    // > 0: softcap * tanh(v / softcap)
};

struct VMvGroup {
    uint32_t type = 0;
    VMvJob * jobs = nullptr;
    int n = 0;
    int rows = 0;
};

// Expert tile descriptor produced by k_vgroup (all device pointers, fixed max_tiles = VB * 8)
// tiles[0] = active tile count (<= VB * 8)
// tiles[1 + 3*t + 0] = expert id e
// tiles[1 + 3*t + 1] = first offset in pairs[]
// tiles[1 + 3*t + 2] = nb in [1, V_MMAX]  (V_MMAX = 4)
constexpr int V_MMAX = 4;

struct VGateUpArgs {
    WRef w;
    const int8_t * xq;     // [VB][cols]
    const float * xd, * xs;// [VB][cols/32]
    const float * f;       // [VB][cols] (fallback for non-q8 formats)
    int cols, rows_e, nff, nu;
    const float * scale;   // [n_expert] or nullptr
    const int32_t * pairs; // [VB * nu] -> pair = tok * nu + k
    const int32_t * tiles; // [1 + 3 * VB * nu]
    float * act;           // [VB * nu][nff]
    float beta, lb;
};

struct VDownQ8Args {
    WRef w;
    const int8_t * xq;     // [VB * nu][cols]
    const float * xd, * xs;// [VB * nu][cols/32]
    const float * act;     // [VB * nu][cols] (fallback for non-q8 formats)
    int cols, nu;
    const float * wts;     // [VB * nu]
    const float * scale;   // [n_expert] or nullptr
    const int32_t * pairs;
    const int32_t * tiles;
    float * out;           // [VB][D], zeroed before launch, accumulated via float atomics
};
```

### 5.3 Exact Kernel Signatures & Design Notes

#### Dense Batched Matrix-Vector: `k_mv_batch` & `k_mv_batch_wide`
Used for `proj`, `wo`, `head` (with `cap = hp_.softcap`), `ctl_down`, and `router`. Replaces the $n$-loop `b_head` and scalar `k_bmm`.

```cpp
// Sub-group per output row; reads each weight row ONCE across all VB tokens.
// Grid: nd_range<1>(((total_rows + RPG - 1) / RPG) * WG, WG), reqd_sub_group_size(SG)
template <uint32_t T, int VB>
sycl::event k_mv_batch(sycl::queue & q, const VMvJob * jobs, int njobs, int total_rows);

// Work-group per output row for small F32/F16/BF16 matrices (router 128, ctl_down 64, sink 24, conv_proj 32).
// Grid: nd_range<1>(total_rows * WG, WG), reqd_sub_group_size(SG)
template <uint32_t T, int VB>
sycl::event k_mv_batch_wide(sycl::queue & q, const VMvJob * jobs, int njobs, int total_rows);
```

- **Inner loop (`lane_dot_batch<T, VB>`):**
  - **`Q8_0R` and `Q6_KR`** (100% of `proj`, `wo`, `head` in `UD-Q4_K_M`): Each lane loads `sycl::int4 wv` **once** (16 bytes = 16 weights; 256 contiguous bytes per SG), loops `k = 0..3` unpacking 4 float weights, and loops `t = 0..VB-1` loading `sycl::float4` from `xs_t + 4 * k` into `s[t]`, scaling `acc[t] += s[t] * d` once per 16 values. Keeps `VB = 8` memory-bandwidth bound (~1.7 TFLOPS vs ~4.1 TFLOPS vector ceiling) with zero input-quantization error or launch overhead.
  - **`Q4_KR` / `Q5_KR`:** Load 16-byte `uint4` weight block once, accumulate `slo[VB]/shi[VB]` + `xl[VB]/xh[VB]` across `t = 0..VB-1`.
  - **Generic fallback:** Calls `step_decode<T>` once per step and updates `acc[0..VB-1]` inside the lambda.
- **Reduction & write:**
  ```cpp
  for (int t = 0; t < VB; ++t) {
      float v = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
      if (lane == t) b.y[(size_t) t * b.ys + r] = b.cap > 0.0f ? b.cap * sycl::tanh(v / b.cap) : v;
  }
  ```

#### Parallel Single-WG Expert Bucketer: `k_vgroup`
Replaces the 1-thread `single_task` `b_group` in [src/prefill.cpp](src/prefill.cpp#L1055) and caps each tile at `V_MMAX = 4` tokens so expert kernels never spill registers on `lnl_m`.

```cpp
// Grid: nd_range<1>(128, 128) (one thread per physical expert e in [0, ne); assumes ne <= 128, else 256)
// SLM: int32_t s_ids[VB * 8] (<= 256 bytes)
template <int VB>
sycl::event k_vgroup(sycl::queue & q, const int32_t * ids, const int32_t * ntokp,
                     int ne, int nu, int32_t * pairs, int32_t * tiles);
```

1. Threads `l < (*ntokp) * nu` load `s_ids[l] = ids[l]`; `group_barrier`.
2. Each thread `e` scans `s_ids[0..np)` and collects matching pair indices into `my_p[VB]` with count `cnt`.
3. `n_tiles_e = (cnt + V_MMAX - 1) / V_MMAX` (`0`, `1`, or `2`).
4. `pair_off = sycl::exclusive_scan_over_group(grp, cnt, sycl::plus<int>())`.
5. `tile_off = sycl::exclusive_scan_over_group(grp, n_tiles_e, sycl::plus<int>())`.
6. Thread `e` writes `pairs[pair_off + j] = my_p[j]` and emits `n_tiles_e` descriptors `(e, pair_off + s, min(V_MMAX, cnt - s))` into `tiles[1 + 3*(tile_off + k)..]`.
7. Thread `e == ne - 1` writes `tiles[0] = tile_off + n_tiles_e`.

#### Grouped Unpack-Once DP4A Experts: `k_vgate_up_r` & `k_vdown_q8`

Batched int8 input quantization:
```cpp
// Quantize [VB][D] -> vfq_[VB][D], vfqd_[VB][D/32], vfqs_[VB][D/32]
// Grid: nd_range<1>(((VB * (D / 32) + RPG - 1) / RPG) * WG, WG), reqd_sub_group_size(SG)
template <int VB>
sycl::event k_vquant_x(sycl::queue & q, const float * x, int8_t * xq, float * xd, float * xs);

// Quantize [VB * nu][nff] -> vaq_, vaqd_, vaqs_ (perm = (T == q::Q2_0R))
// Grid: nd_range<1>(((VB * nu * (cols / 32) + RPG - 1) / RPG) * WG, WG), reqd_sub_group_size(SG)
template <int VB>
sycl::event k_vquant_act(sycl::queue & q, const float * act, int nu, int cols, bool perm,
                         int8_t * xq, float * xd, float * xs);
```

Shared unpack-once device helpers:
```cpp
// Unpacked 32-value block in int8x4 DP4A form + up to 2 sub-block float scales + optional min/delta term
struct DecodedU32 {
    int32_t w[8];   // 8 x int8x4 (32 signed/unsigned int8 values)
    float s0, s1;   // scale for w[0..3] and w[4..7]
    float m0;       // Q5_KR: -dmin*mn; IQ1_SR: dl*delta; Q2_0R: -d; NVFP4R/IQ2_*: 0
};

// Decodes 32 weights ONCE from RowRef rr at 32-value block ib
template <uint32_t T>
inline DecodedU32 decode_unit_q8(const RowRef rr, int ib, const q::Tabs & tb);

// Dot one already-unpacked block against one token's int8 block (8 DP4A + 2 FMAs)
template <uint32_t T>
inline float dot_decoded_q8(const DecodedU32 & u, const sycl::int4 x0, const sycl::int4 x1,
                            float xdb, float xsb);
```

- **`NVFP4R`:** `decode_unit_q8` runs `fp4x4` (8×) + `ue4m3` (2×) once; `dot_decoded_q8` runs 8 `dot_acc` + `xdb * (u.s0 * is0 + u.s1 * is1)`.
- **`Q5_KR`:** `decode_unit_q8` runs `hi16x4` + mask/or (8×) + `d*sc`, `-dmin*mn` once; `dot_decoded_q8` runs 8 `dot_acc` + `u.s0 * xdb * isum + u.m0 * xsb`.
- **`IQ1_SR` / `IQ2_*`:** Codebook lookups + `signed4` run once; `dot_decoded_q8` runs 8 `dot_acc`.
- **`Q2_0R`:** 2-bit shifts/masks run once; `dot_decoded_q8` runs 8 `dot_acc` + `u.s0 * (isum - (int32_t) xsb)`.

Gate-Up + Fused SITU and Grouped Down:
```cpp
#ifndef INK_VGUR
#define INK_VGUR 2   // rows/sub-group (gate+up pair); 1 or 2 tunable on lnl_m
#endif

// Grid: max_tiles = VB * nu; nsg = max_tiles * (nff / INK_VGUR)
// nd_range<1>(((nsg + RPG - 1) / RPG) * WG, WG), reqd_sub_group_size(SG)
template <uint32_t T, int VB>
sycl::event k_vgate_up_r(sycl::queue & q, const VGateUpArgs a);

#ifndef INK_VDR
#define INK_VDR 2   // output rows per sub-group
#endif

// Grid: max_tiles = VB * nu; nsg = max_tiles * (D / INK_VDR)
// nd_range<1>(((nsg + RPG - 1) / RPG) * WG, WG), reqd_sub_group_size(SG)
template <uint32_t T, int VB>
sycl::event k_vdown_q8(sycl::queue & q, const VDownQ8Args a);

// Generic FP32 fallback for non-DP4A down formats (Q8_0R, Q6_KR, Q5_KR, Q4_KR, MXFP4)
template <uint32_t T, int VB>
sycl::event k_vdown_fp(sycl::queue & q, const VDownQ8Args a);
```

- **`k_vgate_up_r`:** Maps like `k_gate_up_r` ([src/fast.cpp](src/fast.cpp#L513-L539)); uniform early exit `if (tile >= a.tiles[0]) return;`. Decodes `rows[r]` once per `ib`, loops `m = 0..nb-1` (`<= 4`) reading `vfq_` from L1, XOR-reduces across `0..7` / `8..15`, applies SITU, writes `a.act[pr[m] * nff + i0 + r]`. Register budget at `GR = 2, V_MMAX = 4` is ~11 floats (no spill on `lnl_m`).
- **`k_vdown_q8`:** `ib = lane; ib < cols / 32; ib += SG`; decodes each output row once per `ib`, accumulates `acc[r][m]`, reduces over `sg`, scales by `a.wts[pr[m]] * es`, and `fetch_add` (relaxed, device scope) into `a.out[tok[m] * D + i0 + r]`.

#### Batched Split-K Decode Attention: `k_vattn` & `k_vcombine`

```cpp
struct VAttnArgs {
    const float * qf;      // [VB][nh][HD]
    const float * rf;      // [VB][nh][n_rel]
    const int8_t * kc, * vc;
    const sycl::half * ksc, * vsc;
    const float * rb;      // [extent][n_rel]
    const int32_t * pos0p; // device scalar
    const int32_t * ntokp; // device scalar (1..VB)
    float * part;          // [VB][nh][pstride][HD + 2]
    int nh, nkv, n_rel, extent, n_swa, ring, kvlen, ch, nc, pstride;
    bool swa, kv4;
};

// Grid: nd_range<1>(nkv * nc * AWG, AWG), reqd_sub_group_size(SG)
//   SWA:  AWG = INK_SWA_AWG (128), ch = swa_chunk (64),
//         nc = (min(n_swa + VB - 1, n_ctx) + ch - 1) / ch   <-- +VB-1 (17 chunks, not 16)
//   Full: AWG = 256, ch = 256, nc = max_vchunks_
template <int AWG, int VB>
sycl::event k_vattn(sycl::queue & q, const VAttnArgs a);

// Grid: nd_range<1>(VB * nh * (8 * HD), 8 * HD)
template <int VB>
sycl::event k_vcombine(sycl::queue & q, const float * part, const float * proj,
                       float * att, int nh, int nc, int pstride);
```

- **Critical SWA chunk fix:** At `pos0 >= 1024`, tokens `pos0 .. pos0 + VB - 1` span `n_swa + VB - 1 = 1031` keys (`lo = pos0 - 1023` to `p_last = pos0 + VB - 1`). Using `nc_swa = (min(n_swa + VB - 1, n_ctx) + 63) / 64` (17 chunks for `swa_chunk = 64`) ensures draft tokens can attend to themselves.
- **SLM layout (`R = VB * ratio`, `ratio = 3` → `R = 6, 12, 24`):** `sq8[R * 32]` (3.0 KB), `sqd[R * 4]` + `sqs[R * 4]` (0.75 KB), `sr[R * 16]` (1.5 KB), `sycl::half sp[R * ch]` (12.0 KB Full / 3.0 KB SWA). Total SLM at `VB = 8`: **17.25 KB (Full)** / **8.25 KB (SWA)**, with **no `sa` buffer**.
- **Score phase:** One thread per key loads the K block once into `kw[8]`, scores all `R` query rows via DP4A, applies causal/SWA mask (`u < ntok && k <= pos0 + u && (!swa || (pos0 + u) - k < n_swa)`) + relative bias, computes per-row `m[r]`/`l[r]`, and writes `(sycl::half) exp(sc - m)` into `sp`.
- **P·V phase (zero local atomics):** `for (int r = sgid; r < R; r += AWG / SG)` assigns one sub-group per row `r = (u, j)` reading 4 consecutive keys/step (`sub = lane / 4`, `d32 = (lane % 4) * 32`), XOR-reducing `acc[32]` across `sub = 0..3`, and writing directly to `part[(u * nh + h0 + j) * pstride + c]`.

#### Fused Visit Glue Kernels & Batched Argmax

```cpp
// Q/K/V/R post-processing + KV cache write (guarded by tok < *ntokp) + FUSED vraw_ save for K & V
// Grid: nd_range<1>(VB * (nh + 2*nkv + nh) * HD, HD)
template <int VB>
sycl::event k_vqkv_post(sycl::queue & q, const float * proj, const float * qclip, const float * rclip,
                        const float * kconv, const float * vconv, const float * hist,
                        int8_t * kc, int8_t * vc, sycl::half * ks, sycl::half * vs,
                        float * qf, float * rf, float * vraw_slot,
                        const int32_t * pos0p, const int32_t * ntokp,
                        int nh, int nkv, int n_rel, int ring, int kvlen, int hc,
                        float qk_clip, float rel_clip, float alpha, float floor_, bool full, bool kv4);

// 4-tap causal conv (reads hist + raw without mutating hist) + residual + FUSED vraw_ save
// Grid: range<2>(VB, D)
template <int VB>
sycl::event k_vconv_res(sycl::queue & q, const float * raw, const float * w, const float * hist,
                        const float * res, float * out, float * vraw_slot_off, int hc, bool drop);

// Depth mix in FP32 (single-barrier multi-score reduction from k_mix)
// Grid: nd_range<1>(VB * a.nq * WG, WG), reqd_sub_group_size(SG)
template <int VB>
sycl::event k_vmix(sycl::queue & q, const VMixArgs a);

// Fused loop-control up-projection + combine (reads gain/forget [D, 64] ONCE across VB tokens)
// Grid: nd_range<1>((a.n * D / RPG) * WG, WG), reqd_sub_group_size(SG)
template <int VB>
sycl::event k_vctl_up(sycl::queue & q, const VCtlArgs a);

// Batched 2-stage GPU argmax over vlogits_[0..VB)[n_vocab] -> out_idx[0..VB)
// Stage 1: nd_range<2>(VB, NG * WG, 1, WG); Stage 2: nd_range<2>(VB, WG, 1, WG)
template <int VB>
void k_vargmax(sycl::queue & q, const float * logits, int n_vocab,
               float * tmp_v, int32_t * tmp_i);
```

Fusing `vraw_` writes into `k_vqkv_post` and `k_vconv_res` removes 144 standalone `bhist` launches per verify call.

### 5.4 C API & Server Speculative Loop

#### C API ([src/capi.cpp](src/capi.cpp))
- `ink_verify(void * hp, const int32_t * toks, int n, int pos0, float * logits)` — unchanged signature; routes `n <= 8` (when `!cpu_` and `INK_FAST_VERIFY != 0`) to the recorded batched-decode graph.
- `ink_verify_argmax(void * hp, const int32_t * toks, int n, int pos0, int32_t * out_toks)` — **new**; runs the exact same verify graph + `k_vargmax`, copying `n * 4` bytes instead of `n * 804 KB` when `temperature <= 0`.
- `ink_commit(void * hp, int keep)` — unchanged.

#### Server Loop ([server/inkling_server.py](server/inkling_server.py))
- **CLI flags:** `--spec-copy`, `--spec-draft-k` (`3` or `7`, default `3` → exact `VB = 4` bucket), `--spec-min-match` (default `8`, per [PERF.md](PERF.md#L161)).
- **Index maintenance:** Rolling suffix map `(n, tuple(seq[end-n:end])) -> end` for `n ∈ (16, 8)` over `self.eng.tokens` (rebuilt lazily on prompt load / rollback; O(1) update per committed token).
- **Exact step / commit transition** (given `cur_tok` already emitted at top of loop, pending position `pos`):
  1. Look up suffix ending with `cur_tok`; if match length `>= min_match`, propose `draft = seq[j : j + K]` (trimmed to `n_ctx` and `max_new`).
  2. If `draft` is empty: run normal `step_argmax(cur_tok, pos)` / `step(cur_tok, pos)`.
  3. Otherwise, form `batch = [cur_tok] + draft` (`n = 1 + len(draft)`):
     - **Greedy (`temp <= 0`):** `pred = eng.verify_argmax(batch, pos)`. Find first `a < len(draft)` where `pred[a] != draft[a]` (else `a = len(draft)`); `next_tok = pred[a]`.
     - **Probabilistic (`temp > 0`, mathematically lossless):** `logits = eng.verify(batch, pos)`. For `a = 0 .. len(draft) - 1`: compute filtered softmax $p_a$, accept `draft[a]` if `u < p_a[draft[a]]`; on first rejection zero out `p_a[draft[a]]`, renormalize, sample `next_tok` from the residual distribution, and break. If all accepted (`a == len(draft)`), sample `next_tok` from $p_{\text{len(draft)}}$.
  4. **Commit & emit:** Call `eng.commit(a + 1)`, append `batch[:a + 1]` to `eng.tokens`, update suffix index, advance `pos += a + 1`, feed `draft[:a]` through `parser.feed()` / `emit()` (handling `stop_id` / `eos_id` and `n_gen`), and continue with `cur_tok = next_tok`.

### 5.5 Verification & Tuning Checklist

1. **Correctness gate (`test-verify`):** Run `build/test-verify MODEL.gguf 2000 8192` and `build/test-verify MODEL.gguf 1020 8192` (to stress SWA wraparound across 1024). Expect `MATCH` with worst KL `<= 0.002` (8-bit KV) / `<= 0.016` (4-bit KV) across `n ∈ {1, 2, 4, 7, 8}`, `commit(8)`, and `commit(0)`.
2. **Greedy parity:** Assert `ink_verify_argmax` matches `np.argmax` of `ink_verify` on 100% of positions.
3. **A/B kernel tuning on `lnl_m`:** Compare `INK_VGUR=1` vs `INK_VGUR=2` and `INK_VDR=1` vs `INK_VDR=2` using `test-verify`'s timing table (target `< 1.4×` for `n = 2`, `< 1.95×` for `n = 4`, `< 2.8×` for `n = 8`), and check `INK_FAST_VERIFY=0` vs `1` at `pos = 2000` and `pos = 16000`.
4. **End-to-end trace check:** Replay coding requests through `server/inkling_server.py` with `--spec-copy --spec-draft-k 3` and `--spec-draft-k 7`, verifying identical greedy outputs vs non-speculative generation.
