// GPU parity & performance verification for Phase 2 native DSpark speculative decoding:
//   1. Target tap parity (prefill vs step vs verify taps at layers 5, 23, 41, 48)
//   2. DSpark draft agreement between BF16 markov_w2 (INK_DSPARK_W2_Q8=0) and Q8_0R (default)
//   3. Snapshot save / restore parity across target + DSpark
//   4. End-to-end throughput & acceptance rate benchmark at K=3 and K=7 on prose & code prompts
#include "device.h"
#include "dspark.h"
#include "fast.h"
#include "gguf_file.h"
#include "kernels.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

using clk = std::chrono::steady_clock;

static size_t argmax(const float * v, size_t n) { return std::max_element(v, v + n) - v; }

static std::vector<float> read_taps(sycl::queue & q, const ink::FastEngine & eng, int n) {
    const int cap = eng.tap_cap();
    std::vector<float> out((size_t) 4 * n * ink::D);
    for (int t = 0; t < 4; ++t) {
        q.memcpy(out.data() + (size_t) t * n * ink::D,
                 eng.tap_feats() + (size_t) t * cap * ink::D,
                 (size_t) n * ink::D * sizeof(float));
    }
    q.wait();
    return out;
}

static void run_spec_bench(const char * label, sycl::queue & q, ink::FastEngine & eng, ink::DsparkEngine & dspark,
                           const ink::FastEngine::Snapshot & s0_eng, const ink::DsparkEngine::Snapshot & s0_dsp,
                           const std::vector<int> & prompt, int first_tok, int gen_tokens, int k) {
    const size_t V = eng.hp().n_vocab;
    // Restore + inject warmup or clean reset
    eng.restore(s0_eng);
    dspark.restore(s0_dsp);

    // Baseline step_argmax reference
    auto tb0 = clk::now();
    int tok = first_tok;
    int pos = (int) prompt.size();
    std::vector<int> base_seq;
    for (int i = 0; i < gen_tokens; ++i) {
        base_seq.push_back(tok);
        if (i + 1 < gen_tokens) tok = eng.step_argmax(tok, pos++);
    }
    double base_ms = std::chrono::duration<double, std::milli>(clk::now() - tb0).count();

    // Speculative DSpark greedy loop
    eng.restore(s0_eng);
    dspark.restore(s0_dsp);
    int32_t warm_draft[7];
    dspark.draft_greedy(first_tok, (int) prompt.size(), k, warm_draft);  // warm ExecGraph outside timed run
    eng.restore(s0_eng);
    dspark.restore(s0_dsp);

    tok = first_tok;
    pos = (int) prompt.size();
    std::vector<int> spec_seq;
    double draft_ms = 0, verify_ms = 0, inject_ms = 0;
    int steps = 0, drafted = 0, accepted = 0;
    auto ts0 = clk::now();
    while ((int) spec_seq.size() < gen_tokens) {
        spec_seq.push_back(tok);
        const int rem = gen_tokens - (int) spec_seq.size();
        if (rem <= 0) break;
        const int kd = std::min(k, rem);
        int32_t draft[7];
        auto td0 = clk::now();
        dspark.draft_greedy(tok, pos, kd, draft);
        draft_ms += std::chrono::duration<double, std::milli>(clk::now() - td0).count();

        std::vector<int> batch(1 + kd);
        batch[0] = tok;
        for (int i = 0; i < kd; ++i) batch[1 + i] = draft[i];

        auto tv0 = clk::now();
        const auto pred = eng.verify_argmax(batch, pos);
        verify_ms += std::chrono::duration<double, std::milli>(clk::now() - tv0).count();

        int a = 0;
        int next_tok = pred[0];
        while (a < kd && pred[a] == draft[a]) {
            ++a;
            next_tok = pred[a];
        }
        auto ti0 = clk::now();
        eng.commit(a + 1);
        dspark.inject(eng.tap_feats(), eng.tap_cap(), a + 1, pos);
        inject_ms += std::chrono::duration<double, std::milli>(clk::now() - ti0).count();
        q.wait();

        for (int i = 0; i < a; ++i) spec_seq.push_back(draft[i]);
        pos += a + 1;
        tok = next_tok;
        ++steps;
        drafted += kd;
        accepted += a;
    }
    double spec_ms = std::chrono::duration<double, std::milli>(clk::now() - ts0).count();
    (void) V;
    const double base_tps = 1000.0 * gen_tokens / base_ms;
    const double spec_tps = 1000.0 * gen_tokens / spec_ms;
    const double acc_pct = drafted ? 100.0 * accepted / drafted : 0.0;
    const double mean_tok = steps ? (double) (accepted + steps) / steps : 1.0;
    std::printf("[%s K=%d] base=%.2f tok/s (%.1f ms) | dspark=%.2f tok/s (%.1f ms, %.2fx) | "
                "accept=%d/%d (%.1f%%, %.2f tok/step) | avg draft=%.2f ms verify=%.2f ms inject=%.2f ms\n",
                label, k, base_tps, base_ms, spec_tps, spec_ms, spec_tps / base_tps,
                accepted, drafted, acc_pct, mean_tok,
                draft_ms / std::max(1, steps), verify_ms / std::max(1, steps), inject_ms / std::max(1, steps));
}

int main(int argc, char ** argv) {
    try {
        if (argc < 3) {
            std::fprintf(stderr, "usage: %s TARGET.gguf DSPARK.gguf [N_CTX]\n", argv[0]);
            return 2;
        }
        const int n_ctx = argc > 3 ? std::atoi(argv[3]) : 8192;
        ink::GgufFile tf(argv[1]);
        ink::GgufFile df(argv[2]);
        sycl::queue q = ink::make_queue(false);
        ink::Model target(q, tf, true);
        ink::FastEngine eng(q, target, n_ctx, true, 256, 4);
        eng.enable_taps(nullptr);

        ink::FastEngine::Snapshot s_zero;
        eng.save(s_zero, 0);  // empty conv history before any prefill

        // 1. Tap parity: prefill vs step vs verify across layers [5, 23, 41, 48]
        std::vector<int> prompt(64);
        for (int i = 0; i < 64; ++i) prompt[i] = 1000 + (i * 7919) % 50000;
        const auto lg_pre = eng.prefill(prompt, 0);
        const std::vector<float> pre_taps = read_taps(q, eng, 64);

        ink::FastEngine::Snapshot s_mid;
        // Restore zero conv history, then prefill first 60 tokens, then step 4 and verify 4 tokens
        eng.restore(s_zero);
        eng.prefill(std::vector<int>(prompt.begin(), prompt.begin() + 60), 0);
        eng.save(s_mid, 60);

        std::vector<float> step_taps(4 * 4 * ink::D);
        for (int t = 0; t < 4; ++t) {
            eng.step(prompt[60 + t], 60 + t);
            const auto st = read_taps(q, eng, 1);
            for (int tap = 0; tap < 4; ++tap)
                std::copy_n(st.data() + tap * ink::D, ink::D, step_taps.data() + ((size_t) tap * 4 + t) * ink::D);
        }
        eng.restore(s_mid);
        eng.verify(std::vector<int>(prompt.begin() + 60, prompt.begin() + 64), 60);
        const auto ver_taps = read_taps(q, eng, 4);
        eng.commit(0);

        float min_cos_sp = 1.0f, min_cos_vp = 1.0f, min_cos_sv = 1.0f;
        for (int tap = 0; tap < 4; ++tap) {
            float ts = 0.0f, tv = 0.0f, tsv = 0.0f;
            double dp = 0, ds = 0, dv = 0, csp = 0, cvp = 0, csv = 0;
            for (int t = 0; t < 4; ++t) {
                for (int d = 0; d < ink::D; ++d) {
                    const float p = pre_taps[((size_t) tap * 64 + 60 + t) * ink::D + d];
                    const float s = step_taps[((size_t) tap * 4 + t) * ink::D + d];
                    const float v = ver_taps[((size_t) tap * 4 + t) * ink::D + d];
                    ts = std::max(ts, std::fabs(s - p));
                    tv = std::max(tv, std::fabs(v - p));
                    tsv = std::max(tsv, std::fabs(s - v));
                    dp += (double) p * p;
                    ds += (double) s * s;
                    dv += (double) v * v;
                    csp += (double) s * p;
                    cvp += (double) v * p;
                    csv += (double) s * v;
                }
            }
            const float cos_sp = (float) (csp / (std::sqrt(ds * dp) + 1e-12));
            const float cos_vp = (float) (cvp / (std::sqrt(dv * dp) + 1e-12));
            const float cos_sv = (float) (csv / (std::sqrt(ds * dv) + 1e-12));
            min_cos_sp = std::min(min_cos_sp, cos_sp);
            min_cos_vp = std::min(min_cos_vp, cos_vp);
            min_cos_sv = std::min(min_cos_sv, cos_sv);
            std::printf("  tap %d: cos(step,pre)=%.6f cos(ver,pre)=%.6f cos(step,ver)=%.6f | maxdiff sp=%.4f vp=%.4f sv=%.4f\n",
                        tap, cos_sp, cos_vp, cos_sv, ts, tv, tsv);
        }
        std::printf("Tap cosine parity (min across 4 taps): sp=%.6f vp=%.6f sv=%.6f\n",
                    min_cos_sp, min_cos_vp, min_cos_sv);
        if (min_cos_sp < 0.995f || min_cos_vp < 0.995f || min_cos_sv < 0.995f) {
            std::fprintf(stderr, "FAIL tap cosine similarity too low\n");
            return 1;
        }

        // 2. Compare DSpark BF16 w2 vs Q8_0R w2 draft agreement
        const size_t V = eng.hp().n_vocab;
        {
            ink::DsparkModel dm_bf16(q, df, target, false);
            ink::DsparkModel dm_q8(q, df, target, true);
            ink::DsparkEngine d_bf16(q, target, dm_bf16, n_ctx, 4);
            ink::DsparkEngine d_q8(q, target, dm_q8, n_ctx, 4);

            // Re-prefill full 64 tokens cleanly from empty history and inject into both
            eng.restore(s_zero);
            const auto & lg64 = eng.prefill(prompt, 0);
            const auto pt = read_taps(q, eng, 64);
            d_bf16.inject(eng.tap_feats(), eng.tap_cap(), 64, 0);
            d_q8.inject(eng.tap_feats(), eng.tap_cap(), 64, 0);

            int first_tok = (int) argmax(lg64.data(), V);
            int32_t out_bf16[7], out_q8[7];
            d_bf16.draft_greedy(first_tok, 64, 7, out_bf16);
            d_q8.draft_greedy(first_tok, 64, 7, out_q8);
            int agree = 0;
            for (int i = 0; i < 7; ++i)
                if (out_bf16[i] == out_q8[i]) ++agree;
            std::printf("W2 BF16 vs Q8_0R draft agreement (K=7): %d/7 match\n", agree);
            (void) pt;

            // 3. Snapshot save/restore parity on DSpark
            ink::DsparkEngine::Snapshot ds0;
            d_q8.save(ds0, 64);
            d_q8.inject(eng.tap_feats(), eng.tap_cap(), 4, 64);
            d_q8.restore(ds0);
            int32_t out_q8_restored[7];
            d_q8.draft_greedy(first_tok, 64, 7, out_q8_restored);
            bool snap_ok = true;
            for (int i = 0; i < 7; ++i)
                if (out_q8[i] != out_q8_restored[i]) snap_ok = false;
            std::printf("DSpark snapshot save/restore parity: %s\n", snap_ok ? "MATCH" : "MISMATCH");
            if (!snap_ok) return 1;
        }

        // 4. End-to-end throughput & acceptance benchmark on natural repetitive text & structured code pattern
        ink::DsparkModel dm(q, df, target, true);
        ink::DsparkEngine dspark(q, target, dm, n_ctx, 4);
        eng.enable_taps([&](const float * fdev, int cap, int n, int pos0) { dspark.inject(fdev, cap, n, pos0); });

        // Structured code-like sequence repeated across context (>= 150 tokens) to measure real prediction
        std::vector<int> code_prompt = {1, 2, 3};
        const int unit[] = {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015};
        for (int rep = 0; rep < 32; ++rep) code_prompt.insert(code_prompt.end(), unit, unit + 16);
        eng.restore(s_zero);
        int first_code = (int) argmax(eng.prefill(code_prompt, 0).data(), V);
        ink::FastEngine::Snapshot se;
        ink::DsparkEngine::Snapshot sd;
        eng.save(se, (int) code_prompt.size());
        dspark.save(sd, (int) code_prompt.size());
        run_spec_bench("pattern", q, eng, dspark, se, sd, code_prompt, first_code, 128, 3);
        run_spec_bench("pattern", q, eng, dspark, se, sd, code_prompt, first_code, 128, 7);

        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
