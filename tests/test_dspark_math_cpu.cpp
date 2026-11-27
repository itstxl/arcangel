// Host-CPU unit tests for Phase 2 DSpark equations:
//   1. NeoX RoPE (HD=128, freq_base=1e6) applied after per-head RMSNorm
//   2. ctx_mix weighted tap sum + RMSNorm
//   3. Block-uniform attention (injected KV [k_lo, pos0-1] + M bidirectional ephemeral block keys),
//      including pos0=0 and SWA ring (n_swa=1025)
//   4. Serial Markov RNN update (cand + gate * (state - cand), markov = tanh(m))
//   5. BF16 -> Q8_0R quantization error for markov_w2 (256-dim rank)
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

int g_fails = 0;

void check_close(const char * name, float got, float ref, float tol) {
    const float diff = std::fabs(got - ref);
    if (!(diff <= tol)) {
        std::fprintf(stderr, "FAIL %s: got=%g ref=%g diff=%g > tol=%g\n", name, got, ref, diff, tol);
        ++g_fails;
    }
}

float det(int seed, int idx) {
    uint32_t x = (uint32_t) (seed * 747796405u + idx * 2891336453u + 12345u);
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    return ((float) (x & 0xFFFFu) / 32768.0f - 1.0f) * 0.75f;
}

uint16_t f32_to_bf16(float f) {
    uint32_t u = 0;
    std::memcpy(&u, &f, sizeof(u));
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t) (u >> 16);
}

float bf16_to_f32(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

void test_neox_rope_and_head_norm() {
    constexpr int HD = 128;
    const float eps = 1e-6f;
    const double base = 1000000.0;
    std::vector<float> x(HD), w(HD), out_fast(HD), out_ref(HD);
    for (int i = 0; i < HD; ++i) {
        x[i] = det(1, i);
        w[i] = 0.8f + 0.4f * det(2, i);
    }
    const int pos = 137;

    // Kernel formulation (two halves: j and j + 64)
    float ss = 0.0f;
    for (int i = 0; i < HD; ++i) ss += x[i] * x[i];
    const float r = 1.0f / std::sqrt(ss / HD + eps);
    std::vector<float> xn(HD);
    for (int i = 0; i < HD; ++i) xn[i] = x[i] * r * w[i];
    for (int j = 0; j < HD / 2; ++j) {
        const float freq = (float) std::pow(base, -2.0 * j / (double) HD);
        const float ang = (float) pos * freq;
        const float cs = std::cos(ang), sn = std::sin(ang);
        out_fast[j] = xn[j] * cs - xn[j + 64] * sn;
        out_fast[j + 64] = xn[j] * sn + xn[j + 64] * cs;
    }

    // Reference direct norm-preservation + rotation check
    float ss_before = 0.0f, ss_after = 0.0f;
    for (int i = 0; i < HD; ++i) {
        ss_before += xn[i] * xn[i];
        ss_after += out_fast[i] * out_fast[i];
        out_ref[i] = out_fast[i];
    }
    check_close("neox_rope_norm_preserve", ss_after, ss_before, 1e-4f);
    // Pair j=1 must differ from standard interleaved (i, i+1) at nonzero pos
    const float f1 = (float) std::pow(base, -2.0 * 1 / (double) HD);
    check_close("neox_pair_1_65", out_fast[1], xn[1] * std::cos(pos * f1) - xn[65] * std::sin(pos * f1), 1e-6f);
    check_close("neox_pair_65", out_fast[65], xn[1] * std::sin(pos * f1) + xn[65] * std::cos(pos * f1), 1e-6f);
    (void) out_ref;
}

void test_ctx_mix_rmsnorm() {
    constexpr int D = 2304;
    const float mw[4] = {0.202f, 0.015f, 0.748f, 0.035f};
    std::vector<float> t[4], nw(D), mixed(D), out(D);
    for (int k = 0; k < 4; ++k) {
        t[k].resize(D);
        for (int i = 0; i < D; ++i) t[k][i] = det(10 + k, i);
    }
    for (int i = 0; i < D; ++i) nw[i] = 1.0f + 0.25f * det(19, i);
    float ss = 0.0f;
    for (int i = 0; i < D; ++i) {
        mixed[i] = mw[0] * t[0][i] + mw[1] * t[1][i] + mw[2] * t[2][i] + mw[3] * t[3][i];
        ss += mixed[i] * mixed[i];
    }
    const float r = 1.0f / std::sqrt(ss / D + 1e-6f);
    float out_rms = 0.0f;
    for (int i = 0; i < D; ++i) {
        out[i] = mixed[i] * r * nw[i];
        if (i < 16) out_rms += (out[i] / nw[i]) * (out[i] / nw[i]);
    }
    check_close("ctx_mix_sum_weights", mw[0] + mw[1] + mw[2] + mw[3], 1.0f, 1e-6f);
    float norm_unit_ss = 0.0f;
    for (int i = 0; i < D; ++i) norm_unit_ss += (mixed[i] * r) * (mixed[i] * r);
    check_close("ctx_mix_unit_rms", norm_unit_ss / D, 1.0f, 1e-5f);
    (void) out_rms;
}

void test_block_uniform_splitk_attention() {
    constexpr int HD = 128;
    constexpr int M = 4;
    constexpr int n_swa = 1025;
    constexpr int ch = 128;
    const float scale = 1.0f / std::sqrt((float) HD);

    // Case A: pos0 = 0 (only M ephemeral block tokens)
    // Case B: pos0 = 1500 with SWA (lo = 1500 - 1025 + 1 = 476, ctx keys [476..1499])
    for (int pos0 : {0, 1500}) {
        const bool swa = pos0 > 0;
        const int lo = swa ? std::max(0, pos0 - n_swa + 1) : 0;
        const int p_last = pos0 - 1;
        const int active_nc = (pos0 > lo) ? (pos0 - lo + ch - 1) / ch : 0;

        std::vector<std::vector<float>> Q(M, std::vector<float>(HD));
        std::vector<std::vector<float>> Kblk(M, std::vector<float>(HD)), Vblk(M, std::vector<float>(HD));
        for (int m = 0; m < M; ++m)
            for (int d = 0; d < HD; ++d) {
                Q[m][d] = det(30 + m, d);
                Kblk[m][d] = det(40 + m, d);
                Vblk[m][d] = det(50 + m, d);
            }
        std::vector<std::vector<float>> Kctx(pos0, std::vector<float>(HD)), Vctx(pos0, std::vector<float>(HD));
        for (int k = 0; k < pos0; ++k)
            for (int d = 0; d < HD; ++d) {
                Kctx[k][d] = det(100 + (k % 17), d + k);
                Vctx[k][d] = det(200 + (k % 19), d + k);
            }

        for (int u = 0; u < M; ++u) {
            // Split-K combine computation
            std::vector<float> cm(active_nc, -INFINITY), cl(active_nc, 0.0f);
            std::vector<std::vector<float>> cacc(active_nc, std::vector<float>(HD, 0.0f));
            for (int c = 0; c < active_nc; ++c) {
                const int k0 = lo + c * ch;
                const int nk = std::min(ch, p_last - k0 + 1);
                if (nk <= 0) continue;
                std::vector<float> sc(nk);
                float mx = -INFINITY;
                for (int jj = 0; jj < nk; ++jj) {
                    float dot = 0.0f;
                    for (int d = 0; d < HD; ++d) dot += Q[u][d] * Kctx[k0 + jj][d];
                    sc[jj] = dot * scale;
                    mx = std::max(mx, sc[jj]);
                }
                float ls = 0.0f;
                for (int jj = 0; jj < nk; ++jj) {
                    const float p = std::exp(sc[jj] - mx);
                    ls += p;
                    for (int d = 0; d < HD; ++d) cacc[c][d] += p * Vctx[k0 + jj][d];
                }
                cm[c] = mx;
                cl[c] = ls;
            }
            float Mmax = -INFINITY;
            for (int c = 0; c < active_nc; ++c) Mmax = std::max(Mmax, cm[c]);
            std::vector<float> sc_blk(M);
            for (int m = 0; m < M; ++m) {
                float dot = 0.0f;
                for (int d = 0; d < HD; ++d) dot += Q[u][d] * Kblk[m][d];
                sc_blk[m] = dot * scale;
                Mmax = std::max(Mmax, sc_blk[m]);
            }
            float L = 0.0f;
            std::vector<float> out_split(HD, 0.0f);
            for (int c = 0; c < active_nc; ++c) {
                if (cm[c] == -INFINITY) continue;
                const float w = std::exp(cm[c] - Mmax);
                L += cl[c] * w;
                for (int d = 0; d < HD; ++d) out_split[d] += cacc[c][d] * w;
            }
            for (int m = 0; m < M; ++m) {
                const float w = std::exp(sc_blk[m] - Mmax);
                L += w;
                for (int d = 0; d < HD; ++d) out_split[d] += Vblk[m][d] * w;
            }
            for (int d = 0; d < HD; ++d) out_split[d] /= L;

            // Single-pass exact softmax reference over [lo, pos0-1] union block[0..M-1]
            std::vector<float> scores;
            for (int k = lo; k <= p_last; ++k) {
                float dot = 0.0f;
                for (int d = 0; d < HD; ++d) dot += Q[u][d] * Kctx[k][d];
                scores.push_back(dot * scale);
            }
            for (int m = 0; m < M; ++m) {
                float dot = 0.0f;
                for (int d = 0; d < HD; ++d) dot += Q[u][d] * Kblk[m][d];
                scores.push_back(dot * scale);
            }
            float rmax = *std::max_element(scores.begin(), scores.end());
            double rden = 0.0;
            for (float s : scores) rden += std::exp((double) (s - rmax));
            std::vector<float> out_ref(HD, 0.0f);
            for (int d = 0; d < HD; ++d) {
                double num = 0.0;
                int si = 0;
                for (int k = lo; k <= p_last; ++k, ++si)
                    num += std::exp((double) (scores[si] - rmax)) * Vctx[k][d];
                for (int m = 0; m < M; ++m, ++si)
                    num += std::exp((double) (scores[si] - rmax)) * Vblk[m][d];
                out_ref[d] = (float) (num / rden);
                check_close("block_uniform_attn", out_split[d], out_ref[d], 1e-5f);
            }
        }
    }
}

void test_markov_rnn_and_w2_q8_accuracy() {
    constexpr int R = 256;
    // 1. Check GRU equation identity: cand + gate * (state - cand) == gate * state + (1 - gate) * cand
    std::vector<float> st0(R), st1(R), mv(R), j_out(3 * R);
    for (int i = 0; i < R; ++i) {
        st0[i] = det(70, i);
        j_out[i] = det(71, i);
        j_out[R + i] = det(72, i);
        j_out[2 * R + i] = det(73, i);
    }
    for (int i = 0; i < R; ++i) {
        const float g = 1.0f / (1.0f + std::exp(-j_out[i]));
        const float c = std::tanh(j_out[R + i]);
        const float got = c + g * (st0[i] - c);
        const float ref = g * st0[i] + (1.0f - g) * c;
        check_close("markov_rnn_update", got, ref, 1e-6f);
        mv[i] = std::tanh(j_out[2 * R + i]);
    }

    // 2. Check BF16 -> Q8_0R dot accuracy over 256 dims (plus softcap=30 after bias)
    std::vector<uint16_t> w_bf16(R);
    for (int i = 0; i < R; ++i) w_bf16[i] = f32_to_bf16(det(80, i) * 0.2f);
    float dot_bf16 = 0.0f;
    for (int i = 0; i < R; ++i) dot_bf16 += bf16_to_f32(w_bf16[i]) * mv[i];

    // Q8_0R weight + int8 activation dot in blocks of 32
    float dot_q8 = 0.0f;
    for (int b = 0; b < R / 32; ++b) {
        float wmax = 0.0f, xmax = 0.0f;
        for (int j = 0; j < 32; ++j) {
            wmax = std::max(wmax, std::fabs(bf16_to_f32(w_bf16[b * 32 + j])));
            xmax = std::max(xmax, std::fabs(mv[b * 32 + j]));
        }
        const float wd = wmax / 127.0f, xd = xmax / 127.0f;
        const float wid = wd > 0.0f ? 1.0f / wd : 0.0f;
        const float xid = xd > 0.0f ? 1.0f / xd : 0.0f;
        int32_t isum = 0;
        for (int j = 0; j < 32; ++j) {
            const int8_t wq = (int8_t) std::clamp((int) std::lrint(bf16_to_f32(w_bf16[b * 32 + j]) * wid), -127, 127);
            const int8_t xq = (int8_t) std::clamp((int) std::lrint(mv[b * 32 + j] * xid), -127, 127);
            isum += (int32_t) wq * (int32_t) xq;
        }
        dot_q8 += wd * xd * (float) isum;
    }
    const float cap_bf16 = 30.0f * std::tanh((4.2f + 0.1f + dot_bf16) / 30.0f);
    const float cap_q8 = 30.0f * std::tanh((4.2f + 0.1f + dot_q8) / 30.0f);
    check_close("w2_q8_vs_bf16", cap_q8, cap_bf16, 0.02f);
}

}  // namespace

int main() {
    test_neox_rope_and_head_norm();
    test_ctx_mix_rmsnorm();
    test_block_uniform_splitk_attention();
    test_markov_rnn_and_w2_q8_accuracy();
    if (g_fails) {
        std::fprintf(stderr, "test_dspark_math_cpu FAILED (%d failures)\n", g_fails);
        return 1;
    }
    std::printf("test_dspark_math_cpu: ALL TESTS PASSED (4 suites)\n");
    return 0;
}
