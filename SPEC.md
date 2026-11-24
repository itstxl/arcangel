# Inkling 2 Mini: what one token goes through

Source of truth: llama.cpp b12631 + Unsloth's Inkling pins, `~/.local/src/llama.cpp-inkling`,
files `src/models/inkling2.cpp` (layer), `src/models/inkling2-loop.cpp` (loop), `src/llama-graph.cpp`
(`build_moe_ffn`, `build_lora_mm_id`). Model file: Unsloth `Inkling-2-Mini-UD-Q2_K_XL.gguf` (mainline
`inkling2_loop` layout). Written 2026-11-23. Notation: d = 2304 (n_embd), T = tokens in the batch.

## Hyperparameters (from the GGUF)

| Key | Value |
| --- | --- |
| Physical layers | 24; loop core = layers [6, 18), 3 passes -> 48 layer visits (6 + 36 + 6) |
| Hidden size d | 2304 |
| Heads | 24 query, 8 KV (GQA 3), head dim 128, no RoPE |
| Full attention | layers 5, 11, 17, 23 (every 6th); others sliding window 1024 |
| Full-attention cache slots | 8 of the 48 visits (5, 11x3, 17x3, 23); the other 40 hold <= 1024 tokens |
| Relative bias | n_relative = 16 per head; extent 1024 (full) and 1024 (= n_swa, SWA); clip 8.0 |
| QK clip | 22.627 (L2 norm cap per head row) |
| Log-position scale | tau = 1 + 0.1*ln(max(1, (pos+1)/8192)), multiplies Q and R on full layers only |
| Short convolution | 4 taps (3 past inputs kept per channel, per cache slot) |
| Block size for depth mixing | 6 (attn_res.block_size) |
| Experts | 128, top 8, expert FFN 1152 (gate_up fused = 2304 rows), sigmoid gating, + selection bias, weights normalised (eps 1e-20) then x 8.0 |
| Activation (SITU) | beta*tanh(g/beta)*sigmoid(g) * lb*tanh(u/lb), beta = 4, lb = 25 |
| Loop controllers | gate rank 64; four controls per core layer (Q, K, V, FFN) |
| Vocab | 200058 real, 201024 padded; final logit softcap 30; no BOS |
| RMS eps | 1e-6 everywhere (the model's f_norm_rms_eps is 1e-6 too) |

## Weight formats in this file (each needs a native decode routine)

| Tensor | Formats | Size |
| --- | --- | --- |
| ffn_gate_up_exps | IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S (+ per-expert F32 `.scale`) | 4.15 GB |
| ffn_down_exps | Q2_0, NVFP4, MXFP4 (+ per-expert F32 `.scale`) | 3.19 GB |
| attention q/k/v/o, gate, rel | Q8_0, Q6_K, Q5_K, Q4_K, IQ3_XXS | ~0.8 GB |
| token_embd / output | IQ4_XS / Q4_K | 0.23 / 0.28 GB |
| sink_proj, k/v_conv_proj | BF16 | small |
| norms, clips, conv taps, rel_bias, res queries, loop controls, router | F32 | 0.14 GB |

The IQ types are codebook (grid) formats: repacking them to a plain linear 2-bit format is lossy, so the
engine decodes each format natively (port of ggml's reference dequantisers in `ggml-quants.c` and the
grids in `ggml-common.h`).

## Forward pass, per token

0. `x = rms_norm(token_embd[tok]) * token_embd_norm`. `tau` computed from the position.
   `blocks = []` (block outputs for depth mixing).
1. For each visit (prelude layers 0-5, then core 6-17 three times, then coda 18-23) run LAYER below.
   Each visit has its own cache slot (KV + conv history); core visits share physical weights.
   - Pass 0 of the core runs LAYER plainly.
   - Pass p > 0 restarts from the prelude output and prelude blocks; the previous pass's
     blocks after its first core block plus its final output form the "bank"; each core layer gets
     `messages = MIX(bank, bank_out, loop_msg_query[layer])` (4 rows: Q, K, V, FFN).
2. Output: `h = rms_norm(MIX(blocks, x, output_res_query)) * output_norm`; logits = output . h;
   `30*tanh(logits/30)`; drop the padded vocabulary.

MIX(blocks, cur, query): stack rms_norm(eps 1e-6, no weight) of every block (newest first) and of cur;
scores = states . query / sqrt(d), softmax over the stack; result = weighted sum of the normalised
states. `query` may have several rows (several outputs).

CONTROL(cur, msg): `gate = ctrl_down . [rms(cur); rms(msg)]` (rank 64);
`cur*(1 - tanh(ctrl_forget . gate)) + tanh(ctrl_gain . gate) * msg`.

LAYER(il, ip):
1. Attention inputs: if layer > 0, `mix = MIX(blocks, x, attn_res_query)` (1 row on SWA layers,
   3 rows Q/K/V on full layers); else x.
2. Pass > 0: each input = CONTROL(input, message[Q|K|V]).
3. Norms: SWA layers `attn_norm` on the input(s); full layers `attn_{q,k,v}_inp_norm` separately.
4. `Q = wq.in_q` [24x128]; `K = CONV(wk.in_k, attn_k_conv, dynamic = full ? k_conv_proj.in_k : none)`;
   `V = CONV(wv.in_v, attn_v_conv, dyn = full ? v_conv_proj.in_v : none)`; `R = attn_rel.in_q` [24x16].
5. `Q = CLIP(Q*attn_q_clip, 22.63)`, `K = CLIP(K, 22.63)`, `V = rms_norm(V, 1e-6)` per head,
   `R = CLIP(R*attn_rel_clip, 8.0)`. Full layers: `Q *= tau`, `R *= tau`.
   CLIP(v, t): `v * clamp(t / (|v|_2 + 1e-6), 0, 1)` per head row.
6. Sink per head: `s = attn_sink_proj . in_q` [24].
7. Banded attention: bias table `B = attn_rel_bias^T . R` -> [extent] per head and query;
   score(q, k) = Q.K/sqrt(128) + B[dist] if 0 <= dist < extent, dist = q_pos - k_pos; causal mask;
   SWA layers also mask dist >= 1024; softmax with sink s (denominator gets + exp(s)).
8. `out = attn * sigmoid(attn_gate . in_q)`; `out = CONV(wo.out, attn_post_conv)`.
9. Residual: if ip % 6 == 0, `blocks.push(x_in)` (x_in = this layer's input) and `x = out`;
   else `x = x_in + out`.
10. FFN input: `f = MIX(blocks, x, ffn_res_query)`; pass > 0: `f = CONTROL(f, message[FFN])`;
    `f = rms_norm(f) * ffn_norm`.
11. Router: `p = sigmoid(ffn_gate_inp . f)` (F32); pick top 8 by `p + exp_probs_b`; weights = p of
    those, divided by their sum (+1e-20), times 8.0.
12. Each chosen expert e: `[g; u] = gate_up[e] . f * gate_up_scale[e]`; `a = SITU(g, u)`;
    `y_e = down[e] . a * down_scale[e]`. `y = sum_e w_e * y_e`.
13. `y = CONV(y, ffn_post_conv)`; `x = x + y` (x here is the post-attention residual from step 9).

CONV(x, w, dyn): causal depthwise conv over the last 4 inputs of each channel (3 kept in the slot's
history): `out = sum_j w[j] * x[t-3+j]`; full layers add per-head dynamic taps
`sum_lag dyn[lag, head] * x[t - lag]`; result `(x + out) / sqrt(4*(dyn ? 2 : 1) + 1)`.
History channel offsets in the state: K at 0, V at 1024, attn out at 2048, FFN out at 2048 + 2304.

## Memory per token of context (int8 KV)

8 growing slots x 8 KV heads x 128 x (K + V) bytes + int8 scales: ~17 KiB/token (measured 544 MiB at
32k). 40 SWA slots: fixed ~128 MiB at 1024 tokens. Conv history: 48 slots x 3 x 6656 floats ~ 3.8 MiB.

## Open points to confirm against llama.cpp tensor dumps in Phase 1

- Exact order of rows in the 3-row attention MIX query and in the 4-row message query.
- Whether `dist` for SWA layers uses the same banded formula (extent = n_swa = 1024).
- The exact bank composition on pass 2 (blocks after the first core block).
- MXFP4 / NVFP4 / Q2_0 decode details (port from ggml).
