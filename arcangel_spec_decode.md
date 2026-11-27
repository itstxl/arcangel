# [Title]
Follow-up: Native DSpark & Copy-Draft Speculative Decoding in ArcAngel on Lunar Lake Arc 140V (up to 30 tok/s at 128k ctx + full HumanEval run)

---

Following up on my [previous post bringing ArcAngel to the Lunar Lake Arc 140V](https://www.reddit.com/r/IntelArc/comments/1y65206/) (~21 tok/s decode, 226 tok/s prefill, 128k context on 32 GB UMA): I've now implemented **both Phase 1 (batched-decode verify + zero-model prompt copy drafting) and Phase 2 (native GPU DSpark speculative decoding)** directly in SYCL on the same branch:

**https://github.com/alpha2beta/arcangel/tree/lunar-lake-optimizations** (latest commit `59826ac`)

Everything works end-to-end in `inkling_server.py` at **`n_ctx=131072` with 4-bit KV**, supports both **greedy (`temp=0`) and lossless probabilistic rejection sampling (`temp > 0`)**, and preserves multi-turn prompt caching via mirrored target/draft snapshots.

---

### Benchmark Summary (`Inkling-2-Mini-Uncensored-Q4_K_M`, Arc 140V, Core Ultra 7 258V, 4-bit KV)

#### 1. Raw Batched Verify Scaling (`test-verify`, `pos=2000`, `KV=4`)

| Path | Tokens Verified | Latency | Cost vs 1-Token Step |
|---|---:|---:|---:|
| `step_argmax` (baseline) | 1 | `46.19 ms` | `1.00x` |
| `verify<VB=2>` (`K=1`) | 2 | `64.03 ms` | `1.39x` |
| `verify<VB=4>` (`K=3`) | 4 | `89.57 ms` | `1.94x` |
| `verify<VB=8>` (`K=7`) | 8 | `144.00 ms` | `3.12x` |

#### 2. End-to-End Speculative Decoding (`bench_dspark.py` @ `n_ctx=8192` & Live `128k` Server)

| Workload | Mode | Throughput | Speedup | Acceptance (`tok/step`) | Breakdown (`draft` / `verify` / `inject`) |
|---|---|---:|---:|---:|---|
| Prose (`effort=high`, 256 tok) | Baseline (`step_argmax`) | `22.63 tok/s` | `1.00x` | — | `44.34 ms/step` |
| Prose (`effort=high`, 256 tok) | **DSpark `K=3`** | **`21.89 tok/s`** | `0.97x` | `42.0%` (`2.26`) | `13.85` / `89.32` / `0.09 ms` |
| Prose (`effort=high`, 256 tok) | DSpark `K=7` | `14.33 tok/s` | `0.63x` | `21.3%` (`2.48`) | `26.65` / `146.52` / `0.09 ms` |
| Code Edit (`effort=none`, 512 tok) | Baseline (`step_argmax`) | `19.36 tok/s` | `1.00x` | — | `51.72 ms/step` |
| Code Edit (`effort=none`, 512 tok) | **DSpark `K=3`** | **`23.40 tok/s`** | **`1.21x`** | `69.6%` (`3.08`) | `16.64` / `114.78` / `0.16 ms` |
| Code Edit (`effort=none`, 512 tok) | DSpark `K=7` | `20.08 tok/s` | `1.04x` | `48.5%` (`4.37`) | `32.12` / `185.39` / `0.15 ms` |
| Live `128k` Server — Code Edit (`temp=0.0`) | **DSpark `K=7`** | **`30.01 tok/s`** | — | `70.4%` (`5.93`) | `n_ctx=131072, kv_bits=4` |
| Live `128k` Server — Python Refactor (`temp=1.0`) | **`--spec-copy K=3`** | **`29.57 tok/s`** | — | `98.4%` (`496/504`) | `n_ctx=131072, kv_bits=4` |

#### 3. Full 164-Task HumanEval / HumanEval+ Soak Test (`--spec-dspark K=3`, `n_ctx=131072`, `temp=0.0`)

To stress-test stability and real-world coding throughput across 164 sequential requests (`147,347` generated tokens, `33,808` prompt tokens), I ran EvalPlus against the live 128k server (`start_arcangel_server.sh --spec-dspark dspark-Inkling-2-Mini-Q8_0.gguf --spec-draft-k 3`, `--reasoning-effort medium`, `EVALPLUS_MAX_NEW_TOKENS=2048`):

- **Throughput:** **`21.43 tok/s` weighted** (`21.89` mean / `22.06` median, up to **`26.67 tok/s`**) over `2:00:13` with zero crashes or errors.
- **Scores:** **91.5% HumanEval Base (`150/164`) / 88.4% HumanEval+ (`145/164`)**.
- **Important note on the Base score vs my earlier `light-think` run (`93.3%`):** `13` of the `14` base failures were `finish_reason=length` truncations hitting the `2048`-token EvalPlus cap because `--reasoning-effort medium` spends more tokens thinking than `low`. Among the `149` tasks that finished cleanly (`finish stop`), base pass rate was `148/149` (only `HumanEval/99` failed logic, plus `5` HumanEval+ edge-case failures).

---

### What Was Built

1. **Phase 1 — Batched-Decode Verify Engine (`src/verify.cpp`) & `--spec-copy` (`d204b2f`, `023da6f`)**
   - Added dedicated SYCL command-graph verify buckets for **`VB ∈ {2, 4, 8}`** (`K ∈ {1, 3, 7}` draft tokens + 1 anchor token).
   - Wrote unpack-once DP4A grouped MoE kernels (`Q8_0R` / `Q5_0R` / `Q4_KR` + `Q5_0` GGUF repacking) so shared experts across the verify batch aren't reloaded redundantly, plus a 3-barrier SLM softmax `k_vattn_r` and GPU `ink_verify_argmax`.
   - Wired `--spec-copy` (zero-extra-VRAM prompt n-gram copy drafting) into `server/inkling_server.py` with both greedy verification and lossless probabilistic rejection sampling for `temperature > 0`.

2. **Phase 2 — Native GPU DSpark Draft Engine (`src/dspark.h`, `src/dspark.cpp`, `59826ac`)**
   - Loads the official 58-tensor `dspark-Inkling-2-Mini-Q8_0.gguf` (`dflash` architecture, ~481 MB), repacks all 3 layers' `Q8_0` weights to `Q8_0R`, and quantizes/repacks the `BF16` `markov_w2` matrix to `Q8_0R` by default (`INK_DSPARK_W2_Q8=0` keeps `BF16`).
   - **Zero-extra-pass target taps:** `FastEngine::enable_taps()` captures the 4 hidden-state taps (`[5, 23, 41, 48]`) directly inside existing target `prefill()`, `step()`, and `verify()` graphs.
   - Implements the full 3-mode DSpark pipeline on GPU:
     - **Mode A (KV Inject, ~`0.09–0.16 ms`):** `f_rnn_ring_[9][D]` + NeoX RoPE + 3-layer interleaved `[SWA=1025, Full, SWA=1025]` KV cache.
     - **Mode B (Backbone):** Block-uniform split-K + ephemeral bidirectional attention + uncapped shared target LM head.
     - **Mode C (Markov Head):** Serial rank-256 Markov RNN head with recorded greedy SYCL `ExecGraph`s for `K=3`/`K=7` (plus per-slot softcapped logits for `temp > 0`), and mirrored `save()`/`restore()`/`stash()` snapshots.

---

### Why `K=3` Is the Sweet Spot on Arc 140V (and Why Battlemage Should Gain More)

Speculative decoding on a bandwidth-limited MoE iGPU (`~100 GB/s` LPDDR5X-8533) is a tight math problem:
- Because Inkling 2 Mini routes top-k experts per token, verifying `4` tokens (`VB=4`) touches more distinct experts than a single token (`89.57 ms` vs `46.19 ms`, ~`1.94x` single-step), and drafting `K=3` takes ~`14–17 ms`.
- That puts the breakeven point on Arc 140V at ~`2.3` accepted+bonus tokens/step. On high-entropy open-ended prose (`2.26 tok/step`), `K=3` breaks roughly even (`0.97x`); on structured code edits (`3.08 tok/step`), it hits **`1.21x` (`23.4 tok/s`)**, and on repetitive refactoring/editing tasks it reaches **`~30 tok/s`**.
- `K=7` needs `>3.7` tokens/step just to break even on 140V (`144 ms` verify + `27–32 ms` draft), so it only wins on highly predictable code edits.
- **On a discrete Arc B580 / B60** (456+ GB/s bandwidth and far more Xe2 cores for batched MoE + DSpark head), both `verify` and `draft` should scale much more favorably—I'd love to see what `K=3` and `K=7` hit on Battlemage!

---

### Verification & Numerical Parity

- **Verify parity (`test-verify`, `pos=2000`, `KV=4`):** `worst KL 0.00519`, `verify_argmax: 0 mismatches` vs single-token decode.
- **DSpark GPU parity (`test-dspark`):** Tap cosine similarity **`>= 0.997281`** across all 4 taps `[5, 23, 41, 48]`, `W2 BF16 vs Q8_0R` `7/7` draft agreement, and exact match on snapshot save/restore.
- **Cross-check vs official `llama.cpp` (`llama-cli --spec-type draft-dspark`, identical 40-tok prompt, `K=7`, `128` tok):**
  - `llama.cpp`: `80/312` (`25.6%`), `mean_len = 2.74`, per-slot `(0.652, 0.391, 0.283, 0.174, 0.109, 0.065, 0.065)`
  - ArcAngel: `82/306` (`26.8%`), `mean_len = 2.82`, per-slot `(0.644, 0.432, 0.295, 0.205, 0.136, 0.070, 0.071)`
- **Unit tests:** `4/4` CPU DSpark math suites (`tests/test_dspark_math_cpu.cpp`) + `7/7` server speculative loop & probabilistic rejection-sampling tests (`tests/test_spec_server.py`).

---

### Quick Try

Build (use `-DINK_AOT_TARGETS=bmg_g21` for B580/B60, or `lnl_m` for Lunar Lake):
```bash
git clone -b lunar-lake-optimizations https://github.com/alpha2beta/arcangel.git
cd arcangel
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DINK_AOT_TARGETS=lnl_m
cmake --build build --target inkling test-verify test-dspark -j 2
```

Run with **native DSpark (`K=3` recommended on 140V)**:
```bash
python3 server/inkling_server.py \
  --model /path/to/Inkling-2-Mini-Uncensored-Q4_K_M.gguf \
  --assets /path/to/assets \
  --n-ctx 131072 --chunk 512 --kv-bits 4 --stash-ram-gb 2 \
  --spec-dspark /path/to/dspark-Inkling-2-Mini-Q8_0.gguf \
  --spec-draft-k 3
```

Or zero-extra-VRAM **copy drafting** (great for editing/refactoring existing files):
```bash
python3 server/inkling_server.py \
  --model /path/to/Inkling-2-Mini-Uncensored-Q4_K_M.gguf \
  --assets /path/to/assets \
  --n-ctx 131072 --chunk 512 --kv-bits 4 --stash-ram-gb 2 \
  --spec-copy --spec-draft-k 3
```

Shout-out again to u/txlfxrd for the clean architecture in [itstxl/arcangel](https://github.com/itstxl/arcangel) — if anyone tries `--spec-dspark` with `K=3` or `K=7` on a discrete B580/B60, please drop your numbers in the comments!
