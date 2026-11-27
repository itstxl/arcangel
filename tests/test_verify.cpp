// test-verify MODEL.gguf [PROMPT_LEN] [N_CTX]
//
// Speculative-decoding building block: FastEngine::verify(n tokens) + commit(keep) must give the same logits
// as stepping those tokens one at a time, and decoding after commit(keep) must continue as if only `keep`
// tokens had been fed. Also times verify for n = 1, 2, 4, 8, 16 against one decode step.
#include "device.h"
#include "fast.h"
#include "gguf_file.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using clk = std::chrono::steady_clock;

static double kl(const std::vector<float> & a, const float * b, size_t n) {  // KL(softmax a || softmax b)
    double ma = -1e30, mb = -1e30;
    for (size_t i = 0; i < n; ++i) ma = std::max(ma, (double) a[i]), mb = std::max(mb, (double) b[i]);
    double za = 0, zb = 0;
    for (size_t i = 0; i < n; ++i) za += std::exp(a[i] - ma), zb += std::exp(b[i] - mb);
    double k = 0;
    for (size_t i = 0; i < n; ++i) {
        const double pa = std::exp(a[i] - ma) / za;
        if (pa > 1e-12) k += pa * ((a[i] - ma - std::log(za)) - (b[i] - mb - std::log(zb)));
    }
    return k;
}
static size_t argmax(const float * v, size_t n) { return std::max_element(v, v + n) - v; }

int main(int argc, char ** argv) {
    try {
        if (argc < 2) {
            std::fprintf(stderr, "usage: %s MODEL.gguf [PROMPT_LEN] [N_CTX]\n", argv[0]);
            return 2;
        }
        const int L = argc > 2 ? std::atoi(argv[2]) : 2000;
        const int n_ctx = argc > 3 ? std::atoi(argv[3]) : 8192;
        const bool vprof = std::getenv("INK_VPROF") && std::string(std::getenv("INK_VPROF")) == "1";
        ink::GgufFile f(argv[1]);
        sycl::queue q = ink::make_queue(vprof);
        ink::Model model(q, f, true);
        ink::FastEngine eng(q, model, n_ctx, !vprof, 256, std::getenv("INK_KV") ? std::atoi(std::getenv("INK_KV")) : 4);
        const size_t V = eng.hp().n_vocab;

        // a prompt of ordinary text-like tokens, then greedy decoding to get "drafts" the model agrees with
        std::vector<int> prompt(L);
        for (int i = 0; i < L; ++i) prompt[i] = 1000 + (i * 7919) % 50000;
        std::vector<float> lg = eng.prefill(prompt, 0);
        ink::FastEngine::Snapshot s0;
        eng.save(s0, L);

        const int N = 12;
        std::vector<int> seq;  // greedy continuation: seq[0] = argmax after the prompt, ...
        std::vector<std::vector<float>> ref;  // ref[i] = logits after feeding seq[0..i]
        int t = (int) argmax(lg.data(), V);
        for (int i = 0; i < N; ++i) {
            seq.push_back(t);
            ref.push_back(eng.step(t, L + i));
            t = (int) argmax(ref.back().data(), V);
        }
        // make the last few drafts wrong, as a rejected draft would be
        std::vector<int> draft = seq;
        for (int i = 8; i < N; ++i) draft[i] = (draft[i] + 17) % 50000;

        for (int nn : {4, 7, 8, 12}) {  // per-position detail for the bmm (n < 8) and XMX (n >= 8) head paths
            eng.restore(s0);
            std::vector<int> d(draft.begin(), draft.begin() + nn);
            const auto & v2 = eng.verify(d, L);
            std::printf("n=%2d KL per position:", nn);
            for (int i = 0; i < std::min(nn, 8); ++i) std::printf(" %.4f", kl(ref[i], v2.data() + (size_t) i * V, V));
            std::printf("\n");
            eng.commit(0);
        }
        int fails = 0;
        eng.restore(s0);
        const auto & vl = eng.verify(draft, L);
        double worst = 0;
        for (int i = 0; i < 8; ++i) {
            const double k = kl(ref[i], vl.data() + (size_t) i * V, V);
            const bool top = argmax(ref[i].data(), V) == argmax(vl.data() + (size_t) i * V, V);
            worst = std::max(worst, k);
            if (!top || k > 0.02) ++fails;
        }
        std::printf("verify vs one-at-a-time, positions 0-7: worst KL %.5f %s\n", worst, fails ? "MISMATCH" : "MATCH");

        // verify_argmax must match argmax(verify) on 100% of positions across n in {1, 2, 4, 7, 8}
        int fails_argmax = 0;
        for (int nn : {1, 2, 4, 7, 8}) {
            eng.restore(s0);
            std::vector<int> d(draft.begin(), draft.begin() + nn);
            const auto vref = eng.verify(d, L);
            eng.commit(0);
            eng.restore(s0);
            const auto va = eng.verify_argmax(d, L);
            eng.commit(0);
            for (int i = 0; i < nn; ++i) {
                if ((size_t) va[i] != argmax(vref.data() + (size_t) i * V, V)) ++fails_argmax;
            }
        }
        std::printf("verify_argmax parity (n=1,2,4,7,8): %s (%d mismatches)\n",
                    fails_argmax ? "MISMATCH" : "MATCH", fails_argmax);

        if (vprof) {
            eng.profile = true;
            eng.restore(s0);
            eng.step(seq[0], L);
            eng.reset_profile();
            for (int r = 0; r < 3; ++r) {
                eng.restore(s0);
                eng.step(seq[0], L);
            }
            std::printf("\n=== PROFILE decode step ===\n");
            eng.print_profile();
            for (int n : {2, 4, 8}) {
                std::vector<int> d(seq.begin(), seq.begin() + n);
                eng.restore(s0);
                eng.verify(d, L);
                eng.commit(0);
                eng.reset_profile();
                for (int r = 0; r < 3; ++r) {
                    eng.restore(s0);
                    eng.verify(d, L);
                    eng.commit(0);
                }
                std::printf("\n=== PROFILE verify(%d) ===\n", n);
                eng.print_profile();
            }
            return 0;
        }

        // keep the 8 good tokens, then decode on: must match stepping seq one at a time
        eng.commit(8);
        double worst2 = 0;
        int fails2 = 0;
        for (int i = 8; i < N; ++i) {
            const std::vector<float> & l = eng.step(seq[i], L + i);
            const double k = kl(ref[i], l.data(), V);
            worst2 = std::max(worst2, k);
            if (argmax(ref[i].data(), V) != argmax(l.data(), V) || k > 0.02) ++fails2;
        }
        std::printf("decode after commit(8), positions 8-11: worst KL %.5f %s\n", worst2, fails2 ? "MISMATCH" : "MATCH");

        // commit(0) after a fully rejected verify: decoding must continue exactly from the snapshot state
        eng.restore(s0);
        std::vector<int> bad(6, 12345);
        eng.verify(bad, L);
        eng.commit(0);
        double worst3 = 0;
        int fails3 = 0;
        for (int i = 0; i < 4; ++i) {
            const std::vector<float> & l = eng.step(seq[i], L + i);
            const double k = kl(ref[i], l.data(), V);
            worst3 = std::max(worst3, k);
            if (argmax(ref[i].data(), V) != argmax(l.data(), V) || k > 0.02) ++fails3;
        }
        std::printf("decode after a rejected verify (commit 0):   worst KL %.5f %s\n", worst3, fails3 ? "MISMATCH" : "MATCH");

        // timing at this depth
        auto time_it = [&](auto && fn, int reps) {
            fn();
            const auto t0 = clk::now();
            for (int r = 0; r < reps; ++r) fn();
            return std::chrono::duration<double, std::milli>(clk::now() - t0).count() / reps;
        };
        eng.restore(s0);
        const double step_ms = time_it([&] { eng.step(seq[0], L); }, 20);
        std::printf("decode step: %.2f ms\n", step_ms);
        for (int n : {1, 2, 4, 8, 12, 16}) {
            std::vector<int> d(n);
            for (int i = 0; i < n; ++i) d[i] = seq[i % N];
            const double ms = time_it([&] { eng.verify(d, L); eng.commit(0); }, 10);
            std::printf("verify %2d tokens: %6.2f ms (%.2f decode steps)\n", n, ms, ms / step_ms);
        }
        return fails || fails2 || fails3 || fails_argmax ? 1 : 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
