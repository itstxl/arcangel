# Performance notes

Measured on the Arc B580 (20 Xe2 cores, 128 KB shared local memory per core, SIMD16 sub-groups, ~456 GB/s),
oneAPI 2026.0, Unsloth UD-Q2_K_XL. Run-to-run noise is about ±10% on short runs: use >= 32 decode tokens or
>= 2k prompt tokens and repeat before trusting a difference. Correctness gate for every change:
`test-forward MODEL ref/p1 prefill` (and `graph` for decode) must report MATCH with KL in the 0.001-0.002 range.

## Where things stand (2026-11-27, Arc B580)

| | llama.cpp Vulkan + DSpark | engine, full VRAM (Q2_K_XL) | engine, offload (Q4_K_XL) |
|---|---|---|---|
| decode, short context | 33-38 tok/s | 67 tok/s | 44 tok/s |
| decode at 4k / 12k / 30k / 90k (4-bit KV) | ~18 tok/s at 30k | 66 / 64 / 60 / 48 tok/s | 41 at 30k |
| decode, copy-heavy edit with copy drafting | | ~100 tok/s | (not with offloading) |
| prompt 4k / 16k / 32k / 90k | ~225 tok/s at 4k | 949 / 899 / 850 / 659 tok/s | 609 / 590 at 4k / 16k |
| context | 32k (drafter) / 64k | 96k (4-bit KV) | 96k (4-bit KV) |

Other GPUs (contributor reports): Arc Pro B60 51 tok/s full VRAM (Q4_K_XL, 256k), 35 tok/s offload (1M);
Arc 140V (Lunar Lake) 21 tok/s, 226 tok/s prompt (Q4_K_M, 32k). See the README.

## Arc Pro B60: 128k coding-agent baseline

Measured at revision `ceb2403` with oneAPI 2026.2 (`bmg_g21` AOT), Inkling-2-Mini UD-Q4_K_XL, full-VRAM weights, and 8-bit KV.
The engine used 2,346 MiB for KV and 281 MiB for prompt buffers with 1,024-token chunks in a 131,072-token context.
One run of `bench/code_context.py`, restoring the empty engine snapshot after warmup, gave these rates (only engine time):

- Cold prefill, 130,048 tokens: **449.2 tok/s**.
- Uncached append, 1,024 tokens after a 129,024-token prefix: **309.6 tok/s**.
- Graph decode, 128 fixed source-code tokens after that prompt: **34.93 tok/s**.

The prompt corpus was the first 35 tracked `src/*.cpp` files from the llama.cpp checkout at `afe77e18fadc2e673c6a6299495a443e17378ce4`, wrapped as source reads.
Use the serving tokenizer verified by `server/tokenizer.sha256`.
The corpus SHA-256 was `494b1c2af5999b544ebaf1b2154ff81255b8dca137acd9b04ff88a448a06f652`.

```fish
set llama ~/llama.cpp
set rev afe77e18fadc2e673c6a6299495a443e17378ce4
begin
    printf 'Review the following source files and propose a focused change with tests.\n'
    for path in (git -C $llama ls-tree -r --name-only -z $rev src | string split0 | string match 'src/*.cpp' | head -n 35)
        printf '\n--- tool: read_file %s ---\n' "$path"
        git -C $llama show "$rev:$path"
    end
end > code-corpus.txt
```

### Packed XMX prefill

Stage `Kᵀ` and `V` in packed XMX operand layout and skip score masking only on full-attention tiles that need neither a causal mask nor relative bias.
KV dequantization still multiplies in float before casting to half, and neither the stored KV layout nor the decode kernel changes.

At 128k, the two-run medians were:

- Cold prefill: **449.2 → 488.0 tok/s (+8.6%)**.
- Uncached append: **309.8 → 347.1 tok/s (+12.1%)**.
- Graph decode: **34.91 → 34.92 tok/s (flat)**.

The repeat also copied out logits after the prefix, append, and final decode token; all three top-1 tokens matched the baseline engine and KL divergence was at most 0.00042.
Both versions used the same amount of VRAM.

On the same corpus with a 4,096-token prompt in an 8,192-token context, the two-run medians were 783.6 → 789.2 tok/s cold prefill, 757.0 → 764.7 tok/s for the append, and 57.78 → 57.82 tok/s graph decode.
Against llama.cpp CPU, the baseline and packed path both matched top-1 at all three checked steps in each of two short runs for each KV precision.
With 8-bit KV, the observed KL range was 0.0053–0.0513 before and 0.0065–0.0468 after.

### Decode bottleneck at 128k

In a separate synthetic profile at position 130,048, full attention took 11.57 ms/token, about 41% of summed kernel time.
Batching the per-head softmax reductions cut full attention to 11.31 ms/token but moved synthetic decode only from 28.88 to 29.07 tok/s; changing the work-group size and moving query quantization out of the attention kernel were slower.
These trials were not kept, and the measured graph decode rate on the populated coding cache remains below the 5% improvement target.

## Lessons

1. **On this GPU the prompt kernels were limited by shared local memory (SLM), not by arithmetic.** The
   expert kernel decoded only ~85 G values/s (~16 GB/s of weights, ~5 TFLOPS of XMX) regardless of format.
   Switching off decode and XMX (`INK_EXP_SKIP=3`) still left 45% of the time: the plumbing was the cost.
2. **SLM bank conflicts.** Any staging loop where consecutive threads write addresses a multiple of 256 bytes
   apart serialises. Map consecutive threads to consecutive addresses in the *destination* layout:
   - K transposed into SLM for the attention XMX: consecutive threads take consecutive keys. Attention at a
     16k prompt went 9.7 s -> 5.0 s.
   - Weight decode into the XMX B tile (`sB[k][row]`): consecutive threads take consecutive rows
     (`n = item % GR`), not consecutive steps of one row. Projections 1.45 -> 0.89 s, down experts 1.27 ->
     0.97 s at 4k.
3. **SLM occupancy.** 128 KB per Xe-core is shared by the resident work-groups. 56 KB per group means two
   groups per core, too few to hide memory latency. Fixes that paid off:
   - Alias the accumulator buffer onto the weight tile after the k loop (expert kernel 56 -> 40 KB). 2k prompt
     391 -> 462 tok/s.
   - Attention: keep the output accumulators and the Q fragments in registers (`joint_matrix_apply` with
     row/col coordinates rescales the accumulators in place); SLM 64 -> 29 KB.
4. **Do not zero tiles that get overwritten**, and stage activations with vector loads (float4 -> half8).
   Expert gate_up 3.38 -> 2.01 s at 2k.
5. **Give every thread work.** IQ1_S decodes 128 values per step; 128 rows x 1 step left half of the 256
   threads idle. Decoding per 32-value sub-block (`iq1s_part`) gives 512 items: gate_up 2.64 -> 2.04 s at 4k.
6. **Expert tiles should match the tokens per expert.** At 512-token chunks an expert gets ~32 (token,
   expert) pairs on average; 64-pair tiles (GT=64) decode each expert's weights once per chunk instead of
   twice: 713 -> 800 tok/s at 4k even though SLM per group grows to 48 KB.
7. Bigger prompt chunks amortise weight decode: 256 -> 512 is +10-15%, 512 -> 1024 another +7%, but 1024 does
   not fit beside a 64k cache (prompt buffers ~1.9 GB).

8. **Decode experts: the activations, not the weights, were the traffic.** With a sub-group per expert row,
   every (row, expert) re-read the whole activation vector: ~2 GB per token for gate_up against 0.57 GB of
   weights. Fixes, in order of effect (2026-11-24, short-context decode 52.5 -> 62 tok/s):
   - `k_gate_up_r`: 4 rows per sub-group, lanes 0-7 gate / 8-15 up (same activation block), each activation
     block loaded once for the 4 rows. 56 -> 60 tok/s. 2 or 8 rows were slower.
   - `k_down_q8`: down with int8 activations and DP4A (Q2_0: 2-bit fields with activations stored in a matching
     permuted order; NVFP4: SWAR nibble -> int8, no tables), 2 rows per sub-group. -> 61.5 tok/s. 1 or 4 rows
     slower.
   - Balance lanes: split rows into 32-value units (`unit_q8`), so an IQ1_S row (18 steps) no longer leaves
     14 of 16 lanes waiting on 2.
9. **Decode attention:** load V 4 bytes per thread (8 key groups in parallel) instead of 1 byte, K in 16-byte
   loads: 30k decode 42.8 -> 45.9 tok/s. Still ~5 ms per token at 30k for ~0.5 GB, so there is room left.
10. Prompt buffers: gate-up and SITU fused (f16 activations straight from the XMX epilogue) and the expert sum
   done with float atomics into the output: buffers 142k -> 119k floats per token, same speed, slightly lower
   KL. The engine picks the largest prompt chunk that fits (1024 / 512 / 256 / 128) at start-up.

11. **Long-context decode attention (2026-11-25, 90k depth: 30 -> 42.6 tok/s, full attention 14.4 -> 7.7 ms).**
    Measured by switching phases off (`INK_ATT_DBG`: 1 skip scores, 2 skip P.V, 4 skip the partial sums).
    - Scores with DP4A: the query quantised to int8 once per work-group (for the 4-bit cache in [even | odd]
      order per 8 values, matching the nibbles of a K word); 96 DP4A per key instead of 384 FMAs with scalar
      local-memory query loads.
    - Head-major cache layout `[head][token][128]`: a work-group's keys are one contiguous run (was a 64-byte
      row every 512 bytes). 33.2 -> 35.0 tok/s.
    - The combine kernel ran 24 work-groups looping over ~350 chunks: now 8 chunk groups x 128 dims per head.
      1.4 -> 0.1 ms.
    - The P.V partials: per-key-group SLM rows (24-32 KB) cut occupancy, a tree reduction (12 KB + barriers) was
      slower still; local float atomics into one ratio x 128 buffer won, and fewest collisions came from a
      sub-group per (query head, key group): 10.5 -> 7.7 ms.
    - 128-thread work-groups for the sliding-window layers (1.97 -> 1.2 ms), 256 for full attention.
12. **4-bit KV cache** (`--kv-bits 4`, q4_0-style, f16 scale per 32 values, low nibble = even dim): KL vs
    llama.cpp 0.006-0.016 on the hardest step (8-bit: 0.001-0.002), top-1 unchanged; needles at 10/55/95% of
    90k and 4/4 facts of a 60k multi-needle test found. 96k context needs less VRAM than 8-bit at 64k.
14. Depth mix (`k_mix`): runs on only 1-3 work-groups, so its n back-to-back work-group reductions were pure
    latency; all n scores now in one pass (sub-group sums, one local-memory exchange, one barrier): attn/ffn mix
    0.5 -> 0.4 ms per token. The remaining small kernels (~2.5 ms/token over 48 visits) are launch/latency
    bound: kernel fusion is the next lever there.
16. **Decode P.V at depth was latency-bound, not bandwidth-bound** (2026-11-24): at 90k the full-attention
    P.V read V at ~74 GB/s (scores read the same bytes at ~170). A lane read 4 bytes (8 dims) of one key per
    step. Now a sub-group step covers 4 consecutive keys, a lane per (key, 32 dims): 16-byte loads, 256
    contiguous bytes per load instruction, partial sums folded with two sub-group shuffles. Decode 42.3 ->
    45.7 tok/s at 90k, 53.7 -> 56.0 at 30k, +1.7% at 2k.
15. **Prompt attention was limited by the relative-position bias, not by the matrix units** (2026-11-25, EU
    stall sampling): the softmax step read 16 bias values per (row, key) from global memory with distance-
    dependent addresses, and the sliding-window layers bias every key. Each K tile now stages the ~47 distances
    it needs into local memory with coalesced loads: attention -15% (16k: 4.16 -> 3.53 s), 16k prompt
    20.2 -> 19.1 s. Tried and slower: 32-query blocks with 512-thread work-groups (no register cost, but the
    K/V staging was not the limit and bigger groups wait longer at barriers: 4.41 s), 64-key tiles (5.13 s),
    the bias as a small XMX product per tile (3.69 s, and f16 bias), the row's bias weights in a register
    array (spilled: ~10 s). The prompt GEMMs (`k_xmm`, `k_gxmm`) show the decode kernels' pattern: 63-67%
    active, 27-42% of instructions are moves (weights unpacked into the XMX layout).
13. **Prompt buffers 119k -> 72k floats per token**: one pool for buffers with disjoint lifetimes within a
    layer visit (controller temporaries / attention buffers / FFN inputs and expert activations / second
    output), and the depth-mix states in f16. 96k + 4-bit + 1024-token chunks = 1303 MiB; the engine picks
    1024 whenever free VRAM allows (needs ~1.82 GB free after the weights), else 512.

## Tried and rejected

| Change | Result |
|---|---|
| Dense XMX tiles XT=128 (XMT=4) | slower than XT=32 for the projections |
| Transposed expert kernel (weights as the A operand, contiguous 16-byte decode stores) | gate_up -4%, down much slower |
| Expert row tiles GR=64 (more groups per core) | slower than GR=128 |
| Decode attention v2 (sub-group per key), v3 (flash-decoding) | slower than v1 (thread per key, shared GQA heads) |
| `k_down8` decode kernel | no gain (left in) |
| Decode down, float activations, flattened 4 experts per sub-group | slower than k_down8 |
| Decode down q8 split over 2/4 sub-groups per row, or 2 units in flight per lane | slower |
| Staging codebook tables in SLM (`INK_STAGE=1`) | no difference |
| Multi-row dense matrix-vector (Q8_0R/Q6_KR, generic step decoder or specialised) | slower or equal: dense rows are ~365 GB/s already |
| Prompt chunk 1024 at 64k (8-bit KV) | +7% but did not fit before the buffer pool (fits at 96k with 4-bit KV now) |
| Decode attention: 512 / 1024 keys per work-group (several keys per thread) | slower at every depth |
| Decode attention: P.V unrolled 2-4 keys, or 2 keys per sub-group with 16 dims per lane | equal or slower |
| Decode attention: tree reduction of the P.V partials | slower than local atomics |
| SWA decode chunk 128 (vs 64) | slower at every depth |
| SWA decode work-group 64 threads | slower, and broke the output (it writes one dim per thread: now a static_assert) |
| Decode P.V: one sub-group per key group for all 3 query heads (V read once) | equal (the repeats hit the cache) |
| Decode P.V: two 4-key steps in flight per lane | slower (registers) |
| Decode gate-up (IQ2_S, 187 GB/s): codebook lookups replaced by arithmetic (timing only) | no change: not lookup-bound |
| Decode gate-up work-group 64 / 128 / 512 threads (256 default) | no change |
| Speculative decoding on the prompt path (`verify`, kept, off) | 115-150 ms per verify of 1-16 tokens (7-9 decode steps): every expert's weights are decoded; needs small-batch expert kernels |
| llama.cpp SYCL backend, Strata 0.2.19/0.2.20, arcint | slower than llama.cpp Vulkan or cannot run the formats |

## Decode expert kernels: compute-bound, not memory-bound (2026-11-24, unitrace EU stall sampling)

`unitrace --stall-sampling` (intel/pti-gpu, built from source; needs `sysctl dev.xe.observation_paranoid=0`):

| Kernel | Active | Send / scoreboard stalls |
|---|---|---|
| gate-up IQ2_S (`k_gate_up_r`) | 65% (+12% pipe) | 17% |
| down NVFP4 (`k_down_q8`) | 78% | 9% |
| dense Q8_0 matrix-vector (`k_mv`) | 32% | 61% |

So the experts are limited by instruction throughput, the dense layers by memory. In the gate-up kernel the
dot products (dp4a) are ~3% of the instructions: moves (27-31%), 64-bit per-lane address arithmetic, and the
shifts/masks that build codebook indices are the rest. (An earlier "arithmetic only" experiment read 0.43 ms
because the compiler folded the 4 identical rows into one: misleading.) Kept: codebook entries read as two
32-bit words and scalar 32/16-bit weight loads instead of uint2 / byte-wise loads (interleaved results cost
strided moves): experts 6.25 -> 5.98 ms per token (IQ1_S gate-up 0.63 -> 0.48). Tried and no better: full
unroll, hand-pipelined loads, codebook in SLM, work-group 64-512, and a row per lane with the activations as
uniform loads (`k_gate_up_l`: IQ2_S 1.48 -> 2.04 ms, slower for every format). What is left is the decode
itself: a lossless re-encoding of the codebook formats into something cheaper to expand (e.g. a 4-bit
sign+level code, ~1.6x the bytes) would trade memory for instructions; not tried.

## Speculative decoding (2026-11-24: measured, simulated, not built)

Copy drafting (prompt lookup) on 37 traced opencode requests (16k generated tokens, `ARCANGEL_TRACE` +
`bench/draft_sim.py`): 1.90 / 2.08 / 2.18 tokens per step with 4 / 8 / 16 drafted tokens; tool calls that
repeat code reach 2.5-7.

Why that is not a 2x: in a MoE only the experts the batched tokens share are read once. Distinct experts per
offloaded layer among T consecutive tokens (opencode routing): T=2 82%, 3 73%, 4 66%, 8 51% of the reads of
T separate steps. Only the dense layers and the head (~1/5 of a step) are shared fully. With those costs
(`bench/spec_sim.py`, B580):

| Drafting policy | offload setup | full VRAM |
|---|---|---|
| draft 4 whenever any 2-token match | 0.96x | 0.98x |
| draft 4 when the last 8+ tokens match | 1.16x | 1.17x |
| draft 8 when the last 8+ tokens match | 1.18x | 1.17x |

The gain is on copying stretches (file rewrites, quoted code: ~1.5-2x there), nothing elsewhere. Deferred in
favour of the decode expert kernels (every token). What exists: `FastEngine::verify` + `commit` (batch through
the prompt path, rollback of the convolution history; `test-verify` checks both), but the prompt path costs
7-9 decode steps per verify (XMX tiles; the non-XMX fallback is slower still, 168 ms for 1 token). Building it
needs a batched decode path: per-token buffers and positions, sequential convolution with rollback, expert
kernels grouped by expert (shared experts read once), a batched CPU-expert path, a graph per batch size, and
the drafting / acceptance loop in the server.

### Early-exit self-drafting (2026-11-25, after 2DSD)

2DSD (Kim et al. 2026, "Two-Dimensional Speculative Decoding for Looped Language Models") drafts with a
looped model's own early loop passes plus a trained drafter, and reports 4-5x on Ouro on an H100. Its trained
parts (drafter, shared-KV fine-tune) do not exist for Inkling. The untrained part was measured
(`FastEngine::draft` / `ink_draft`, `bench/early_exit.py`; 6 traced opencode requests, 1.5k tokens):

| Draft | top-1 = full model | acceptance (sampling, p_full of the draft) |
|---|---|---|
| prelude + 1 of 3 loop passes + coda (24 of 48 visits) | 61-75% (75% overall) | 72% |
| prelude + 2 passes + coda (36 visits) | 68-77% (76%) | 72% |

High for a model never trained to exit early, and a second pass adds almost nothing. But with this MoE's
batch costs (shared experts only, see above) a draft-then-verify cycle barely pays: full VRAM 1.18x at best
(one draft token, verify reusing the draft's prelude and first pass), offload 1.0x (the CPU-expert layers are
all in the prelude and coda, so the draft carries their full cost). Plain verify is slower in both setups.
Not built.

## Offload setup (2026-11-24, UD-Q4_K_XL on the B580)

- Placement: the 12 single-pass layers' experts (8.3 GiB) in pinned RAM: half of the expert bytes, a quarter
  of the per-token expert reads (the loop core runs 3x). GPU weights 8.1 GiB, less than Q2_K_XL's 8.5.
- GPU reading the experts over PCIe: ~6-9 GB/s, decode ~11-17 tok/s. CPU computing them (AVX2 int8): 0.53
  GB per token at ~37 GB/s (DDR4 dual channel, 6 threads; 3 threads slower, 12 no faster): 32 tok/s.
- Hot-expert cache: routing in the offloaded layers is skewed (top 8 / 16 / 32 of 128 experts per layer
  serve 29 / 44 / 64% of activations on opencode traffic; a recency cache does worse than frequency). The
  cached chosen experts run on the GPU concurrently with the CPU: CPU 13.9 -> 9.9 ms per token at 8 slots,
  36-37 tok/s. Auto size = free VRAM above 768 MiB (13 slots at 96k).
- Pitfalls found: q.memcpy between two host buffers goes through driver staging (minutes for 8 GiB: use
  memcpy); with vm.swappiness 150 the model file's page cache made the kernel swap the experts out, so the
  GPU's reads stalled (prompt reading > 5 min for 4k tokens): the loader drops copied file pages when
  offloading; a Q8_0 embedding was repacked but read in the file layout (garbage, then a GPU hang).
- Prompt reading: the GPU read each chunk's offloaded experts over PCIe (8.6 GB/s): 368-385 tok/s at 4k. Now
  each offloaded expert tensor is copied into the cache's VRAM by the copy engine (14.1 GB/s) before its
  kernel: 470 tok/s at 4k, 452 at 16k; the cache is refilled after the prompt (~60 ms).

## KV cache

Selectable: `--kv-bits 8` (int8) or `4` (default in the service, 96k context); layout head-major
`[kv head][token][128]`, one f16 scale per 32 values. The 8-bit notes:
int8 with one f16 scale per 32 values (q8_0-style), every slot. Accuracy vs llama.cpp unchanged (KL within
noise of the f16 cache); decode at 30k improved 38 -> 41 tok/s (half the bytes per key). 64k fits; 32k with an
f16 cache was the limit before. f16 source backup: `src.bak-f16kv/`.

## Decode profile (reference points, 2026-11-24)

Short context, 16 ms per token: experts ~5.9 ms (gate_up IQ2_S 1.5, NVFP4 down 1.3), projections + wo ~3.5
ms (~365 GB/s), head 1.0 ms (Q4_K, 264 GB/s), ~3 ms of small per-visit kernels (mixes, controllers, router,
norm; 48 visits each), attention 1.5 ms at 0 rising to ~6.5 ms at 30k. The whole token reads ~2.7 GB of
weights: 5.9 ms at full bandwidth. Profile with `bench-decode MODEL 32 32768 profile START` (profile timings
of small kernels are distorted; trust graph-mode wall time).

Note: `swa_chunk` / `INK_SWA_CHUNK` has no effect on graph-mode runs (the graph is recorded at construction).

## VRAM headroom

The desktop (Hyprland, quickshell, Chromium) holds ~1 GB of VRAM and it moves: after the weights load, free
VRAM has measured 1.8-2.2 GB. 64k context needs ~1.3-1.5 GB plus the 512 MiB margin, so start-up at 64k can
fail when the desktop is busy (the planner refuses rather than spill).

## Lunar Lake iGPU (Arc 140V / lnl_m) and UD-Q4_K_M optimizations (2026-11-26)

Measured on an Intel Core Ultra 7 258V (Arc 140V iGPU, 8 Xe2 cores, 128-bit LPDDR5X-8533, 30.88 GiB OS-visible UMA), oneAPI 2026.0, Unsloth `UD-Q4_K_M` (32k context). Sustained GPU read bandwidth on `Q8_0R` dense kernels is **103-106 GB/s** (`proj:Q8_0R` 9.78 ms, `head:Q8_0R` 4.64 ms, `wo:Q8_0R` 3.62 ms). `UD-Q4_K_M` reads **3.95 GB** of active weights per decoded token (~37.2 ms / 26.9 tok/s bandwidth ceiling).

All kernel/numeric changes were verified against pre-edit `libinkling.so` logits on a 16-token prefill + 8 greedy decode steps (`prefill` KL `0.000000`, decode steps KL `0.000016-0.001493`, 9/9 top-1 token matches, and `ink_step_argmax` matching `np.argmax`).

### Summary of changes

- **UMA memory handling (`src/device.cpp`, `src/model.cpp`):** On `host_unified_memory` devices, Level Zero's `ext_intel_free_memory` reports raw Linux `MemFree` (ignoring reclaimable file page cache and the TTM shrinker pool), causing `Model::check_fits` to reject loads falsely; skipping `free_memory` on UMA falls back to `total_bytes`. Enabling `g_release_on` on UMA drops mmap'd GGUF page cache (`MADV_DONTNEED` / `POSIX_FADV_DONTNEED`) after GPU upload so unified memory does not hold a duplicate ~16 GB copy. CPU expert offloading (`INK_OFFLOAD`) is not used on UMA because CPU and GPU share the same LPDDR5X bus and package TDP.
- **`gate_up:NVFP4R` DP4A + `INK_GUR=2` (`src/fast.cpp`, `CMakeLists.txt`):** Added `q::NVFP4R` to `has_q8()` and `k_gate_up_r` using `fp4x4()` + `dot_acc` (`dp4a`) with 4-row activation reuse (`18.08 -> 12.60 ms` at `INK_GUR=4`, `-> 12.13 ms` at `INK_GUR=2`). `CMakeLists.txt` sets `-DINK_GUR=2` only when `INK_AOT_TARGETS STREQUAL "lnl_m"`, keeping `INK_GUR=4` on discrete Battlemage.
- **`down:NVFP4R` (`8.66 -> 7.51 ms`, `src/quant.h`, `src/fast.cpp`):** Replaced `sycl::ldexp` in `q::ue4m3()` with bit-identical IEEE-754 bit-cast math, simplified `fp4x4()` byte-wise negation, shared `nvfp4_q8_dot()`, and hoisted `int4` activation loads out of the row loop in `k_down_q8()`. (`INK_DR=4` tested slower at `8.75 ms`, so `INK_DR=2` was kept.)
- **`Q5_K` -> `Q5_KR` repacking + DP4A (`2.64 -> 1.43 ms`, `1.33 ms` at `INK_GUR=2`, `src/quant.h`, `src/model.cpp`, `src/kernels.h`, `src/fast.cpp`):** Added planar `Q5_KR` repacking (16 B low nibbles, 8 B `[qh u32, sc, m]`, 4 B `d`/`dmin`), `lane_dot`/`step_decode`, and `unit_q8x` DP4A; also sped up 2048-token prefill from `174.8` to `198.2 tok/s` (`chunk=256`).
- **Server latency (`14.94 -> 20.81 tok/s` live greedy, `src/fast.*`, `src/capi.cpp`, `server/inkling_server.py`):** Added 2-stage GPU `k_argmax` (`step_argmax` / `ink_step_argmax`) transferring 4 bytes instead of 804 KB of logits on greedy (`temperature <= 0`) decoding, replaced O(N^2) full-block `OutputParser._flush()` detokenization with O(1) incremental anchor-window decoding, and skipped the extra final `step()` when hitting `max_new` or `n_ctx`.

| Benchmark (`UD-Q4_K_M`, 32k ctx, Arc 140V) | Before | After |
|---|---|---|
| Decode (`graph`, 8-bit KV, pos 0) | 17.90 tok/s (55.87 ms) | **21.08 tok/s (47.44 ms)** |
| Decode (`graph`, 4-bit KV, pos 0) | 17.91 tok/s (55.84 ms) | **21.10 tok/s (47.40 ms)** |
| Decode (`graph`, 8-bit / 4-bit KV, pos 16000, `INK_GUR=4`) | 16.04 / 16.31 tok/s | **18.33 / 18.56 tok/s** |
| Prefill (2048 tokens, `chunk=256` / `512`, `INK_GUR=4`) | 174.8 tok/s (`chunk=256`) | **198.2 / 225.8 tok/s** |
| Live server (`--chunk 512 --kv-bits 8`, greedy 512 tok) | 14.94 tok/s (mean) | **20.81 tok/s** (24.61 s) |
