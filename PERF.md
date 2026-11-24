# Performance notes

Measured on the Arc B580 (20 Xe2 cores, 128 KB shared local memory per core, SIMD16 sub-groups, ~456 GB/s),
oneAPI 2026.0, Unsloth UD-Q2_K_XL. Run-to-run noise is about ±10% on short runs: use >= 32 decode tokens or
>= 2k prompt tokens and repeat before trusting a difference. Correctness gate for every change:
`test-forward MODEL ref/p1 prefill` (and `graph` for decode) must report MATCH with KL in the 0.001-0.002 range.

## Where things stand (2026-11-24)

| | llama.cpp Vulkan + DSpark | engine |
|---|---|---|
| decode, short context | 33-38 tok/s | 65 tok/s |
| decode at 4k / 30k / 90k (4-bit KV) | ~18 tok/s at 30k | 61 / 54 / 42.6 tok/s |
| prompt 4k (1024-token chunks) | ~225 tok/s | 900 tok/s |
| prompt 16k | | 755 tok/s |
| prompt 50k | | ~620 tok/s (82 s) |
| context | 32k (drafter) / 64k | 96k (4-bit KV) |

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
15. Prompt attention at long context: 27% of a 32k prompt (11.3 s), ~4-5 TFLOPS on the XMX units; larger query
    blocks (QB=32) would halve the per-block K/V staging but need ~130 registers (large-GRF mode). Not done.
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
