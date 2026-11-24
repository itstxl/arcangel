#include "fast.h"

#include "kernels.h"
#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>
#include <stdexcept>
#include <type_traits>

#include <sycl/ext/oneapi/dot_product.hpp>

namespace sycl_exp = sycl::ext::oneapi::experimental;

namespace ink {

// The input quantised to int8 in blocks of 32 (scale xd, block sum xs), as llama.cpp does for these formats.
template <uint32_t T> constexpr bool has_q8() { return T == q::IQ2_SR || T == q::IQ2_XXSR || T == q::IQ2_XSR || T == q::IQ1_SR; }

// 4 bits -> 4 bytes of 0xFF where the bit is set
inline uint32_t byte_mask4(uint32_t n) { return ((n * 0x00204081u) & 0x01010101u) * 0xFFu; }
// apply 8 sign bits to 8 grid bytes (each <= 0x2b, so the negation stays inside its byte)
inline int32_t signed4(uint32_t g, uint32_t m) { return (int32_t) ((g ^ m) + (m & 0x01010101u)); }

// One 32-value block (ib) of a codebook row . int8 x. Rows split into these units balance over the 16 lanes far
// better than whole 64/128-value steps (an IQ1_S row of 2304 has 18 steps: 2 lanes would do double work).
template <uint32_t T>
inline float unit_q8x(const RowRef rr, int ib, const sycl::int4 x0, const sycl::int4 x1, float xdb, float xsb, const q::Tabs & tb) {
    using sycl::ext::oneapi::dot_acc;
    const uint8_t * arow = rr.a;
    if constexpr (T == q::IQ1_SR) {
        const uint32_t w = *(const uint32_t *) (rr.p + 4 * ib);
        const uint32_t qh = q::rd16(arow + 2 * ib);
        const float dl = q::f16(rr.a2 + (ib / 8) * 2) * (float) (2 * ((qh >> 12) & 7) + 1);
        const float delta = qh & 0x8000 ? -0.125f : 0.125f;
        int32_t isum = 0;
        for (int l = 0; l < 4; ++l) {
            const uint64_t grid = tb.iq1s[((w >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)];
            const int32_t xa = l < 2 ? x0[2 * l] : x1[2 * l - 4], xb = l < 2 ? x0[2 * l + 1] : x1[2 * l - 3];
            isum = dot_acc((int32_t) (uint32_t) grid, xa, isum);
            isum = dot_acc((int32_t) (uint32_t) (grid >> 32), xb, isum);
        }
        return dl * (xdb * (float) isum + delta * xsb);
    } else {
        const sycl::uint2 wv = *(const sycl::uint2 *) (rr.p + 8 * ib);
        int32_t part[2] = {0, 0};  // the two 16-value halves (IQ2_S / IQ2_XS scale them separately)
        for (int l = 0; l < 4; ++l) {
            uint64_t grid;
            uint32_t signs;
            if constexpr (T == q::IQ2_SR) {
                const uint32_t qh = arow[2 * ib];
                grid = tb.iq2s[((wv[0] >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300)];
                signs = (wv[1] >> (8 * l)) & 0xFF;
            } else if constexpr (T == q::IQ2_XXSR) {
                grid = tb.iq2xxs[(wv[0] >> (8 * l)) & 0xFF];
                signs = tb.ksigns[(wv[1] >> (7 * l)) & 127];
            } else {  // IQ2_XSR
                const uint32_t v = (wv[l / 2] >> (16 * (l % 2))) & 0xFFFF;
                grid = tb.iq2xs[v & 511];
                signs = tb.ksigns[v >> 9];
            }
            const int32_t xa = l < 2 ? x0[2 * l] : x1[2 * l - 4], xb = l < 2 ? x0[2 * l + 1] : x1[2 * l - 3];
            int32_t t = dot_acc(signed4((uint32_t) grid, byte_mask4(signs & 0xF)), xa, 0);
            t = dot_acc(signed4((uint32_t) (grid >> 32), byte_mask4(signs >> 4)), xb, t);
            part[l / 2] += t;
        }
        float db0, db1;
        if constexpr (T == q::IQ2_XXSR) {
            db0 = db1 = q::f16(arow + (ib / 8) * 2) * (0.5f + (float) (wv[1] >> 28)) * 0.25f;
        } else {
            const uint32_t sc = T == q::IQ2_SR ? arow[2 * ib + 1] : arow[ib];
            const float d = q::f16(rr.a2 + (ib / 8) * 2);
            db0 = d * (0.5f + (float) (sc & 0xf)) * 0.25f;
            db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
        }
        return xdb * (db0 * (float) part[0] + db1 * (float) part[1]);
    }
}

template <uint32_t T>
inline float unit_q8(const RowRef rr, int ib, const int8_t * xq, const float * xd, const float * xsum, const q::Tabs & tb) {
    const sycl::int4 * xv = (const sycl::int4 *) (xq + 32 * ib);
    return unit_q8x<T>(rr, ib, xv[0], xv[1], xd[ib], xsum[ib], tb);
}

template <uint32_t T>
inline float lane_dot_q8(const RowRef rr, int cols, const int8_t * xq, const float * xd, const float * xsum, int lane, const q::Tabs & tb) {
    float acc = 0.0f;
    for (int ib = lane; ib < cols / 32; ib += SG) acc += unit_q8<T>(rr, ib, xq, xd, xsum, tb);
    return acc;
}


// ---------------------------------------------------------------------------------------------------
// kernels

namespace {

// Depth mix of normalised states (+ the current input, RMS-normalised here) with nq learned queries.
// Per output row: raw, weighted-RMS-normed (normw) and/or unweighted-normed (outn) results.
struct MixArgs {
    const float * st[12] = {};
    int ns = 0;
    const float * cur = nullptr;
    const float * qw = nullptr;  // F32 [D, nq]; unused when there is a single state
    int nq = 1;
    const float * normw[4] = {};
    float * out[4] = {};
    float * outn[4] = {};
    float * push = nullptr;  // store the normalised cur here
    float eps = 1e-6f;
};

sycl::event k_mix(sycl::queue & q, const MixArgs a) {
    return q.submit([&](sycl::handler & cgh) {
    sycl::local_accessor<float, 1> red(RPG * 13, cgh);
    cgh.parallel_for(sycl::nd_range<1>((size_t) a.nq * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto grp = it.get_group();
        const auto sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];
        const int j = (int) it.get_group(0), l = (int) it.get_local_id(0);
        float c[PER];
        if (a.cur) {
            float ss = 0.0f;
            for (int k = 0; k < PER; ++k) {
                c[k] = a.cur[l + k * WG];
                ss += c[k] * c[k];
            }
            ss = sycl::reduce_over_group(grp, ss, sycl::plus<float>());
            const float cs = 1.0f / sycl::sqrt(ss / D + 1e-6f);
            for (int k = 0; k < PER; ++k) c[k] *= cs;
            if (a.push && j == 0)
                for (int k = 0; k < PER; ++k) a.push[l + k * WG] = c[k];
        } else {
            for (int k = 0; k < PER; ++k) c[k] = 0.0f;
        }
        const int n = a.ns + (a.cur ? 1 : 0);
        float sc[13];
        float mx = 0.0f;
        if (n == 1) {
            sc[0] = 0.0f;
        } else {
            mx = -INFINITY;
            const float * qr = a.qw + (size_t) j * D;
            const float inv = 1.0f / sycl::sqrt((float) D);
            // all n scores in one pass: sub-group sums, one exchange through local memory, one barrier (was n
            // work-group reductions in a row; the kernel runs on 1-3 work-groups, so it is all latency)
            float qv[PER];
            for (int k = 0; k < PER; ++k) qv[k] = qr[l + k * WG];
            for (int s = 0; s < n; ++s) {
                float acc = 0.0f;
                for (int k = 0; k < PER; ++k) acc += (s < a.ns ? a.st[s][l + k * WG] : c[k]) * qv[k];
                acc = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
                if (lane == 0) red[sgid * 13 + s] = acc;
            }
            sycl::group_barrier(grp);
            for (int s = 0; s < n; ++s) {
                float v = 0.0f;
                for (int g2 = 0; g2 < RPG; ++g2) v += red[g2 * 13 + s];
                sc[s] = v * inv;
                mx = sycl::fmax(mx, sc[s]);
            }
        }
        float sum = 0.0f;
        for (int s = 0; s < n; ++s) {
            sc[s] = sycl::exp(sc[s] - mx);
            sum += sc[s];
        }
        if (!a.out[j] && !a.outn[j]) return;
        float o[PER];
        float ss = 0.0f;
        for (int k = 0; k < PER; ++k) {
            const int i = l + k * WG;
            float acc = 0.0f;
            for (int s = 0; s < n; ++s) acc += sc[s] * (s < a.ns ? a.st[s][i] : c[k]);
            o[k] = acc / sum;
            ss += o[k] * o[k];
        }
        ss = sycl::reduce_over_group(grp, ss, sycl::plus<float>());
        if (a.outn[j]) {
            const float r = 1.0f / sycl::sqrt(ss / D + 1e-6f);
            for (int k = 0; k < PER; ++k) a.outn[j][l + k * WG] = o[k] * r;
        }
        if (a.out[j]) {
            const float r = a.normw[j] ? 1.0f / sycl::sqrt(ss / D + a.eps) : 1.0f;
            for (int k = 0; k < PER; ++k) {
                const int i = l + k * WG;
                a.out[j][i] = o[k] * r * (a.normw[j] ? a.normw[j][i] : 1.0f);
            }
        }
    });
    });
}

// Matrix-vector jobs of one format in one launch; one sub-group per output row.
template <uint32_t T, bool STAGE>
sycl::event k_mv(sycl::queue & q, const MvJob * jobs, int njobs, int total_rows) {
    const size_t groups = (size_t) (total_rows + RPG - 1) / RPG;
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint64_t, 1> gl(grid64<T>(), cgh);
        sycl::local_accessor<uint8_t, 1> ksl(128, cgh);
        cgh.parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            q::Tabs tb;
            if constexpr (STAGE) tb = stage<T>(it, &gl[0], &ksl[0]);
            const auto sg = it.get_sub_group();
            const int row = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
            if (row >= total_rows) return;
            int j = 0;
            while (j + 1 < njobs && row >= jobs[j + 1].row0) ++j;
            const MvJob & b = jobs[j];
            const int r = row - b.row0;
            float v = lane_dot<T>(row_of(b.w, (size_t) r), b.cols, b.x, b.x2, b.split, (int) sg.get_local_id()[0], tb);
            v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
            if (sg.get_local_id()[0] == 0) b.y[r] = b.cap > 0.0f ? b.cap * sycl::tanh(v / b.cap) : v;
        });
    });
}

// Few-row matrices in an elementwise format (router, loop-control down projections): one work-group per
// row, so a 64-row matrix still spreads over 64 work-groups.
template <uint32_t T> sycl::event k_mv_wide(sycl::queue & q, const MvJob * jobs, int njobs, int total_rows) {
    return q.parallel_for(sycl::nd_range<1>((size_t) total_rows * WG, WG), [=](sycl::nd_item<1> it) {
        const int row = (int) it.get_group(0), l = (int) it.get_local_id(0);
        int j = 0;
        while (j + 1 < njobs && row >= jobs[j + 1].row0) ++j;
        const MvJob & b = jobs[j];
        const int r = row - b.row0;
        const uint8_t * w = b.w.p + (size_t) r * b.w.rb;
        float acc = 0.0f;
        for (int c = l; c < b.cols; c += WG) {
            const uint8_t * p = w + (size_t) c * q::shape(T).bytes;
            float v;
            if constexpr (T == q::F32) v = sycl::bit_cast<float>(q::rd32(p));
            else if constexpr (T == q::F16) v = q::f16(p);
            else v = q::bf16(p);
            acc += v * (c < b.split ? b.x[c] : b.x2[c - b.split]);
        }
        acc = sycl::reduce_over_group(it.get_group(), acc, sycl::plus<float>());
        if (l == 0) b.y[r] = b.cap > 0.0f ? b.cap * sycl::tanh(acc / b.cap) : acc;
    });
}

// Loop control, second half: out = cur - tanh(forget . g) * cur + tanh(gain . g) * msg (up to 4 controls).
struct CtlArgs {
    const float * gain[4] = {}, * forget[4] = {}, * g[4] = {}, * cur[4] = {}, * msg[4] = {};
    float * out[4] = {};
    int n = 0, rank = 64;
};

sycl::event k_ctl_up(sycl::queue & q, const CtlArgs a) {
    return q.parallel_for(sycl::nd_range<1>((size_t) a.n * D / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int row = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
        const int c = row / D, i = row % D, lane = (int) sg.get_local_id()[0];
        const float * gw = a.gain[c] + (size_t) i * a.rank;
        const float * fw = a.forget[c] + (size_t) i * a.rank;
        const float * gc = a.g[c];
        float ga = 0.0f, fo = 0.0f;
        for (int k = lane; k < a.rank; k += SG) {
            ga += gw[k] * gc[k];
            fo += fw[k] * gc[k];
        }
        ga = sycl::reduce_over_group(sg, ga, sycl::plus<float>());
        fo = sycl::reduce_over_group(sg, fo, sycl::plus<float>());
        if (lane == 0) {
            const float x = a.cur[c][i];
            a.out[c][i] = x - sycl::tanh(fo) * x + sycl::tanh(ga) * a.msg[c][i];
        }
    });
}

// Weighted RMS norm of up to 3 vectors.
struct NormArgs {
    const float * x[3] = {}, * w[3] = {};
    float * y[3] = {};
    int n = 0;
    float eps = 1e-6f;
};

sycl::event k_norm(sycl::queue & q, const NormArgs a) {
    return q.parallel_for(sycl::nd_range<1>((size_t) a.n * WG, WG), [=](sycl::nd_item<1> it) {
        const int j = (int) it.get_group(0), l = (int) it.get_local_id(0);
        float v[PER];
        float ss = 0.0f;
        for (int k = 0; k < PER; ++k) {
            v[k] = a.x[j][l + k * WG];
            ss += v[k] * v[k];
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.0f / sycl::sqrt(ss / D + a.eps);
        for (int k = 0; k < PER; ++k) a.y[j][l + k * WG] = v[k] * r * a.w[j][l + k * WG];
    });
}

// Router second half: sigmoid, top-k by probability + selection bias, normalised and scaled weights.
sycl::event k_topk(sycl::queue & q, const float * logits, const float * pb, int ne, int nu, float ws, int32_t * ids, float * wts) {
    // one work-item per expert; nu rounds of (max, lowest index at the max) over the work-group
    return q.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
        const auto grp = it.get_group();
        const int t = (int) it.get_local_id(0);
        const float p = t < ne ? sigmoidf(logits[t]) : 0.0f;
        float sel = t < ne ? p + (pb ? pb[t] : 0.0f) : -INFINITY;
        int rank = -1;
        float sum = 0.0f;
        for (int k = 0; k < nu; ++k) {
            const float m = sycl::reduce_over_group(grp, sel, sycl::maximum<float>());
            const int idx = sycl::reduce_over_group(grp, sel == m ? t : 1 << 30, sycl::minimum<int>());
            if (t == idx) {
                rank = k;
                sel = -INFINITY;
            }
            sum += sycl::reduce_over_group(grp, t == idx ? p : 0.0f, sycl::plus<float>());
        }
        if (rank >= 0) {
            ids[rank] = t;
            wts[rank] = p / (sum + 1e-20f) * ws;
        }
    });
}

// Quantise a D-vector to int8 in blocks of 32 (llama.cpp q8_1 style): scale, values, block sum.
sycl::event k_quant_x(sycl::queue & q, const float * x, int8_t * xq, float * xd, float * xs) {
    constexpr int nblk = D / 32;
    return q.parallel_for(sycl::nd_range<1>((size_t) (nblk + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int b = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
        if (b >= nblk) return;
        const int lane = (int) sg.get_local_id()[0];
        const float v0 = x[32 * b + 2 * lane], v1 = x[32 * b + 2 * lane + 1];
        const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
        const float sum = sycl::reduce_over_group(sg, v0 + v1, sycl::plus<float>());
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        xq[32 * b + 2 * lane] = (int8_t) sycl::round(v0 * id);
        xq[32 * b + 2 * lane + 1] = (int8_t) sycl::round(v1 * id);
        if (lane == 0) {
            xd[b] = d;
            xs[b] = sum;
        }
    });
}

// Expert gate-up for the chosen experts: rows i and n_ff + i of expert ids[k], SITU activation.
struct GateUpArgs {
    WRef w;
    const int8_t * xq;
    const float * xd, * xs;
    int cols, rows_e, nff, nu;
    const float * f, * scale;
    const int32_t * ids;
    float * act;
    float beta, lb;
};

template <uint32_t T, bool STAGE> sycl::event k_gate_up(sycl::queue & q, const GateUpArgs a) {
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint64_t, 1> gl(grid64<T>(), cgh);
        sycl::local_accessor<uint8_t, 1> ksl(128, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) a.nu * a.nff / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            q::Tabs tb;
            if constexpr (STAGE) tb = stage<T>(it, &gl[0], &ksl[0]);
            const auto sg = it.get_sub_group();
            const int row = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
            const int k = row / a.nff, i = row % a.nff, lane = (int) sg.get_local_id()[0];
            const int e = a.ids[k];
            const size_t r0 = (size_t) e * a.rows_e;
            float gg, uu;
            if constexpr (has_q8<T>()) {
                // gate and up rows as one run of 2 x cols/32 units (2 x 72 = 9 per lane at cols 2304)
                const RowRef rg = row_of(a.w, r0 + i), ru = row_of(a.w, r0 + a.nff + i);
                const int nb = a.cols / 32;
                gg = uu = 0.0f;
                for (int u = lane; u < 2 * nb; u += SG) {
                    const bool up = u >= nb;
                    const float v = unit_q8<T>(up ? ru : rg, up ? u - nb : u, a.xq, a.xd, a.xs, tb);
                    if (up) uu += v;
                    else gg += v;
                }
            } else {
                gg = lane_dot<T>(row_of(a.w, r0 + i), a.cols, a.f, a.f, a.cols, lane, tb);
                uu = lane_dot<T>(row_of(a.w, r0 + a.nff + i), a.cols, a.f, a.f, a.cols, lane, tb);
            }
            gg = sycl::reduce_over_group(sg, gg, sycl::plus<float>());
            uu = sycl::reduce_over_group(sg, uu, sycl::plus<float>());
            if (lane == 0) {
                const float s = a.scale ? a.scale[e] : 1.0f;
                gg *= s;
                uu *= s;
                const float g = a.beta * sycl::tanh(gg / a.beta) * sigmoidf(gg);
                a.act[row] = g * (a.lb > 0.0f ? a.lb * sycl::tanh(uu / a.lb) : uu);
            }
        });
    });
}

// Expert gate-up for the codebook formats, GR rows per sub-group: lanes 0-7 take the gate rows i..i+GR-1,
// lanes 8-15 the matching up rows, each lane a balanced run of 32-value blocks (cols/32/8 = 9 at 2304). Each
// activation block is loaded once and used for all GR rows (and by the gate and up lane of the same block):
// the activations, not the weights, were most of the memory traffic with a row per sub-group.
#ifndef INK_GUR
#define INK_GUR 4
#endif
template <uint32_t T> sycl::event k_gate_up_r(sycl::queue & q, const GateUpArgs a) {
    constexpr int GR = INK_GUR;
    const size_t nsg = (size_t) a.nu * a.nff / GR;
    return q.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const q::Tabs tb;
        const auto sg = it.get_sub_group();
        const size_t gid = it.get_group(0) * RPG + sg.get_group_id()[0];
        if (gid >= nsg) return;
        const int lane = (int) sg.get_local_id()[0], half = lane / 8, l8 = lane % 8;
        const int k = (int) (gid * GR / a.nff), i0 = (int) (gid * GR % a.nff);
        const int e = a.ids[k];
        const size_t r0 = (size_t) e * a.rows_e + (half ? a.nff : 0) + i0;
        RowRef rows[GR];
        for (int r = 0; r < GR; ++r) rows[r] = row_of(a.w, r0 + r);
        float acc[GR];
        for (int r = 0; r < GR; ++r) acc[r] = 0.0f;
        for (int ib = l8; ib < a.cols / 32; ib += 8) {
            const sycl::int4 * xv = (const sycl::int4 *) (a.xq + 32 * ib);
            const sycl::int4 x0 = xv[0], x1 = xv[1];
            const float xdb = a.xd[ib], xsb = a.xs[ib];
            for (int r = 0; r < GR; ++r) acc[r] += unit_q8x<T>(rows[r], ib, x0, x1, xdb, xsb, tb);
        }
        // sum over the 8 lanes of each half
        for (int r = 0; r < GR; ++r)
            for (int off = 4; off > 0; off /= 2) acc[r] += sycl::permute_group_by_xor(sg, acc[r], off);
        const float s = a.scale ? a.scale[e] : 1.0f;
        for (int r = 0; r < GR; ++r) {
            const float gg = sycl::select_from_group(sg, acc[r], 0) * s, uu = sycl::select_from_group(sg, acc[r], 8) * s;
            if (lane == r) {
                const float g = a.beta * sycl::tanh(gg / a.beta) * sigmoidf(gg);
                a.act[(size_t) k * a.nff + i0 + r] = g * (a.lb > 0.0f ? a.lb * sycl::tanh(uu / a.lb) : uu);
            }
        }
    });
}

// A [D x cols] projection whose output goes through a 4-tap convolution and a residual:
// out[i] = (drop ? 0 : res[i]) + (y + conv(hist, y)) / sqrt(5). With experts (nu > 0) y is the weighted
// sum over the chosen experts of down[e] . act_k.
struct ProjConvArgs {
    WRef w;
    int cols;
    const float * x;       // input (cols), or the experts' activations (nu x cols)
    int nu = 0;            // > 0: expert down projection
    const int32_t * ids = nullptr;
    const float * wts = nullptr, * scale = nullptr;
    const float * conv;    // [4, D]
    float * hist;          // this slot's history at the channel offset
    const float * res;
    float * out;
    bool drop = false;
};

template <uint32_t T, bool STAGE> sycl::event k_proj_conv(sycl::queue & q, const ProjConvArgs a) {
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint64_t, 1> gl(grid64<T>(), cgh);
        sycl::local_accessor<uint8_t, 1> ksl(128, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) D / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            q::Tabs tb;
            if constexpr (STAGE) tb = stage<T>(it, &gl[0], &ksl[0]);
            const auto sg = it.get_sub_group();
            const int i = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
            const int lane = (int) sg.get_local_id()[0];
            float y = 0.0f;
            if (a.nu == 0) {
                y = lane_dot<T>(row_of(a.w, (size_t) i), a.cols, a.x, a.x, a.cols, lane, tb);
            } else {
                for (int k = 0; k < a.nu; ++k) {
                    const int e = a.ids[k];
                    const float * xk = a.x + (size_t) k * a.cols;
                    y += lane_dot<T>(row_of(a.w, (size_t) e * D + i), a.cols, xk, xk, a.cols, lane, tb) * a.wts[k] *
                         (a.scale ? a.scale[e] : 1.0f);
                }
            }
            y = sycl::reduce_over_group(sg, y, sycl::plus<float>());
            if (lane == 0) {
                float * h = a.hist + (size_t) i * 3;
                const float xx[4] = {h[0], h[1], h[2], y};
                float cv = 0.0f;
                for (int j = 0; j < 4; ++j) cv += a.conv[j + 4 * i] * xx[j];
                h[0] = xx[1];
                h[1] = xx[2];
                h[2] = xx[3];
                a.out[i] = (a.drop ? 0.0f : a.res[i]) + (y + cv) * 0.44721359549995793f;  // 1/sqrt(5)
            }
        });
    });
}

// Expert down, one sub-group per (row, chosen expert): 2 rows x 8 experts per work-group, the 8 partials
// summed in local memory, then the convolution and the residual as in k_proj_conv.
template <uint32_t T, bool STAGE> sycl::event k_down8(sycl::queue & q, const ProjConvArgs a) {
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint64_t, 1> gl(grid64<T>(), cgh);
        sycl::local_accessor<uint8_t, 1> ksl(128, cgh);
        sycl::local_accessor<float, 1> part(RPG, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) D / 2 * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            q::Tabs tb;
            if constexpr (STAGE) tb = stage<T>(it, &gl[0], &ksl[0]);
            const auto sg = it.get_sub_group();
            const int s = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];
            const int k = s % 8, i = (int) it.get_group(0) * 2 + s / 8;
            const int e = a.ids[k];
            const float * xk = a.x + (size_t) k * a.cols;
            float y = lane_dot<T>(row_of(a.w, (size_t) e * D + i), a.cols, xk, xk, a.cols, lane, tb);
            y = sycl::reduce_over_group(sg, y, sycl::plus<float>());
            if (lane == 0) part[s] = y * a.wts[k] * (a.scale ? a.scale[e] : 1.0f);
            sycl::group_barrier(it.get_group());
            if (k == 0 && lane == 0) {
                float yy = 0.0f;
                for (int j = 0; j < 8; ++j) yy += part[s + j];
                float * h = a.hist + (size_t) i * 3;
                const float xx[4] = {h[0], h[1], h[2], yy};
                float cv = 0.0f;
                for (int j = 0; j < 4; ++j) cv += a.conv[j + 4 * i] * xx[j];
                h[0] = xx[1];
                h[1] = xx[2];
                h[2] = xx[3];
                a.out[i] = a.res[i] + (yy + cv) * 0.44721359549995793f;
            }
        });
    });
}

// ---- expert down with int8 activations (DP4A), for NVFP4R and Q2_0R

template <uint32_t T> constexpr bool has_down_q8() { return T == q::NVFP4R || T == q::Q2_0R; }

// The chosen experts' activations (nu x cols) to int8, one scale per 32 values, plus each block's integer sum.
// perm (Q2_0R): within each 16-value group store value 4j + r at byte 4r + j, so that the 2-bit field r of
// each weight byte lines up with word r of the group (see down_unit).
sycl::event k_quant_act(sycl::queue & q, const float * x, int n, bool perm, int8_t * xq, float * xd, float * xs) {
    const int nblk = n / 32;
    return q.parallel_for(sycl::nd_range<1>((size_t) (nblk + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int b = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
        if (b >= nblk) return;
        const int lane = (int) sg.get_local_id()[0];
        const float v0 = x[32 * b + 2 * lane], v1 = x[32 * b + 2 * lane + 1];
        const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        const int q0 = (int) sycl::round(v0 * id), q1 = (int) sycl::round(v1 * id);
        const int qs = sycl::reduce_over_group(sg, q0 + q1, sycl::plus<int>());
        auto pos = [&](int v) { return perm ? (v / 16) * 16 + (v % 4) * 4 + (v % 16) / 4 : v; };
        xq[32 * b + pos(2 * lane)] = (int8_t) q0;
        xq[32 * b + pos(2 * lane + 1)] = (int8_t) q1;
        if (lane == 0) {
            xd[b] = d;
            xs[b] = (float) qs;
        }
    });
}

// 4 NVFP4 nibbles (one per byte, low 4 bits) -> 4 int8 values 2 * E2M1 = sign x {0, 1, 2, 3, 4, 6, 8, 12}
inline int32_t fp4x4(uint32_t n) {
    const uint32_t m = n & 0x07070707u, s = (n >> 3) & 0x01010101u;
    const uint32_t gt4 = ((m + 0x03030303u) >> 3) & 0x01010101u;  // m > 4
    const uint32_t eq7 = ((m + 0x01010101u) >> 3) & 0x01010101u;  // m == 7
    const uint32_t mask = gt4 * 0xFFu;
    const uint32_t v = m + ((m & mask) - (0x04040404u & mask)) + 2 * eq7;  // m, but 5 6 7 -> 6 8 12
    const uint32_t nz = ((v + 0x7F7F7F7Fu) >> 7) & 0x01010101u;   // v != 0 (keeps -0 from carrying)
    const uint32_t sn = s & nz;
    return (int32_t) ((v ^ (sn * 0xFFu)) + sn);
}

// one 32-value block ub of an expert row . int8 x, as a float (before the activation and expert scales)
template <uint32_t T> inline float down_unit(const RowRef rr, int ub, const int8_t * xq, float xqs) {
    using sycl::ext::oneapi::dot_acc;
    const sycl::int4 * xv = (const sycl::int4 *) (xq + 32 * ub);
    const sycl::int4 x0 = xv[0], x1 = xv[1];
    const int32_t X[8] = {x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]};
    if constexpr (T == q::NVFP4R) {
        const sycl::uint4 wv = *(const sycl::uint4 *) (rr.p + 16 * ub);  // two 16-value sub-blocks of 8 bytes
        float acc = 0.0f;
        for (int t = 0; t < 2; ++t) {
            const uint32_t w0 = wv[2 * t], w1 = wv[2 * t + 1];
            int32_t is = dot_acc(fp4x4(w0), X[4 * t], 0);              // values 0-3: low nibbles of bytes 0-3
            is = dot_acc(fp4x4(w1), X[4 * t + 1], is);                 // 4-7
            is = dot_acc(fp4x4(w0 >> 4), X[4 * t + 2], is);            // 8-11: high nibbles
            is = dot_acc(fp4x4(w1 >> 4), X[4 * t + 3], is);            // 12-15
            acc += q::ue4m3(rr.a[2 * ub + t]) * (float) is;
        }
        return acc;
    } else {  // Q2_0R: 16-value words, field k at bits 2k = value k; x permuted by k_quant_act
        const sycl::uint2 wv = *(const sycl::uint2 *) (rr.p + 8 * ub);
        int32_t is = 0;
        for (int w = 0; w < 2; ++w)
            for (int r = 0; r < 4; ++r) is = dot_acc((int32_t) ((wv[w] >> (2 * r)) & 0x03030303u), X[4 * w + r], is);
        return q::f16(rr.a + 2 * (ub / 2)) * (float) (is - (int32_t) xqs);  // stored 0..3 = value + 1
    }
}

struct DownQ8Args {
    ProjConvArgs p;
    const int8_t * xq;
    const float * xd, * xs;
};

// RD output rows per sub-group: one balanced run over all nu x cols/32 blocks of the chosen experts, each
// activation block loaded once and used for the RD rows (as in k_gate_up_r). Measured 2026-11-24: splitting a
// row over 2 or 4 sub-groups, or two units in flight per lane, were slower than one row per sub-group.
#ifndef INK_DR
#define INK_DR 2
#endif
template <uint32_t T> sycl::event k_down_q8(sycl::queue & q, const DownQ8Args a8) {
    constexpr int RD = INK_DR;
    return q.parallel_for(sycl::nd_range<1>((size_t) D / RD / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const ProjConvArgs & a = a8.p;
        const auto sg = it.get_sub_group();
        const int i0 = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]) * RD;
        const int lane = (int) sg.get_local_id()[0];
        const int nb = a.cols / 32;
        float y[RD];
        for (int r = 0; r < RD; ++r) y[r] = 0.0f;
        for (int u = lane; u < a.nu * nb; u += SG) {
            const int k = u / nb, ub = u % nb, e = a.ids[k];
            const int blk = k * nb + ub;
            const float f = a8.xd[blk] * a.wts[k] * (a.scale ? a.scale[e] : 1.0f), xqs = a8.xs[blk];
            const int8_t * xk = a8.xq + (size_t) k * a.cols;
            for (int r = 0; r < RD; ++r) y[r] += down_unit<T>(row_of(a.w, (size_t) e * D + i0 + r), ub, xk, xqs) * f;
        }
        for (int r = 0; r < RD; ++r) {
            const float yy = sycl::reduce_over_group(sg, y[r], sycl::plus<float>());
            if (lane == r) {
                const int i = i0 + r;
                float * h = a.hist + (size_t) i * 3;
                const float xx[4] = {h[0], h[1], h[2], yy};
                float cv = 0.0f;
                for (int j = 0; j < 4; ++j) cv += a.conv[j + 4 * i] * xx[j];
                h[0] = xx[1];
                h[1] = xx[2];
                h[2] = xx[3];
                a.out[i] = a.res[i] + (yy + cv) * 0.44721359549995793f;
            }
        }
    });
}

}  // namespace

// ---------------------------------------------------------------------------------------------------
// setup

template <typename T> T * FastEngine::dev(size_t n) {
    T * p = sycl::malloc_device<T>(n, q_);
    if (!p) throw std::runtime_error("device allocation failed");
    allocs_.push_back(p);
    return p;
}

// Job tables are built once (prepare_jobs) on a separate queue, so recording the graph never waits;
// every later record() asks for them in the same order and gets the cached ones.
MvJob * FastEngine::jobs(const std::vector<MvJob> & js) {
    if (job_next_ < job_cache_.size()) return job_cache_[job_next_++];
    sycl::queue setup(q_.get_context(), q_.get_device());
    MvJob * d = sycl::malloc_device<MvJob>(js.size(), setup);
    setup.memcpy(d, js.data(), js.size() * sizeof(MvJob)).wait();
    allocs_.push_back(d);
    job_cache_.push_back(d);
    ++job_next_;
    return d;
}

std::vector<MvGroup> FastEngine::plan_mv(const std::vector<MvJob> & js) {
    std::vector<MvGroup> plan;
    std::vector<uint32_t> order;
    for (const MvJob & j : js)
        if (std::find(order.begin(), order.end(), j.type) == order.end()) order.push_back(j.type);
    for (uint32_t t : order) {
        std::vector<MvJob> g;
        int rows = 0;
        for (MvJob j : js) {
            if (j.type != t) continue;
            j.row0 = rows;
            rows += j.rows;
            g.push_back(j);
        }
        plan.push_back({t, jobs(g), (int) g.size(), rows});
    }
    return plan;
}

namespace {
const char * type_name(uint32_t t) {
    switch (t) {
        case q::F32: return "F32"; case q::F16: return "F16"; case q::BF16: return "BF16"; case q::Q8_0: return "Q8_0";
        case q::Q4_K: return "Q4_K"; case q::Q5_K: return "Q5_K"; case q::Q6_K: return "Q6_K"; case q::IQ2_XXS: return "IQ2_XXS";
        case q::IQ2_XS: return "IQ2_XS"; case q::IQ3_XXS: return "IQ3_XXS"; case q::IQ1_S: return "IQ1_S"; case q::IQ2_S: return "IQ2_S";
        case q::IQ4_XS: return "IQ4_XS"; case q::MXFP4: return "MXFP4"; case q::NVFP4: return "NVFP4"; case q::Q2_0: return "Q2_0";
        case q::Q8_0R: return "Q8_0R";
        case q::Q6_KR: return "Q6_KR";
        case q::Q4_KR: return "Q4_KR"; case q::IQ2_XXSR: return "IQ2_XXSR"; case q::IQ2_XSR: return "IQ2_XSR";
        case q::IQ1_SR: return "IQ1_SR"; case q::IQ2_SR: return "IQ2_SR"; case q::NVFP4R: return "NVFP4R";
        case q::Q2_0R: return "Q2_0R";
        default: return "?";
    }
}
// "label:FORMAT", interned so the profiler can keep the pointer
const char * tlabel(const char * label, uint32_t t) {
    static std::vector<std::unique_ptr<std::string>> pool;
    const std::string s = std::string(label) + ":" + type_name(t);
    for (const auto & p : pool) if (*p == s) return p->c_str();
    pool.push_back(std::make_unique<std::string>(s));
    return pool.back()->c_str();
}
}  // namespace

void FastEngine::run_mv(const char * label0, const std::vector<MvGroup> & plan) {
    for (const MvGroup & g : plan) {
        const char * label = profile ? tlabel(label0, g.type) : label0;
        dispatch_type(g.type, [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            if constexpr (q::shape(T).elems == 1) {
                if (g.rows <= 512) {
                    P(label, k_mv_wide<T>(q_, g.jobs, g.n, g.rows));
                    return;
                }
            }
            P(label, stage_tabs ? k_mv<T, true>(q_, g.jobs, g.n, g.rows) : k_mv<T, false>(q_, g.jobs, g.n, g.rows));
        });
    }
}

FastEngine::FastEngine(sycl::queue & q, const Model & m, int n_ctx, bool use_graph, int prompt_chunk, int kv_bits)
    : q_(q), m_(m), hp_(m.hp), n_ctx_(n_ctx), TB_(prompt_chunk) {
    if (const char * e = getenv("INK_KV")) kv_bits = atoi(e);
    if (kv_bits != 8 && kv_bits != 4) throw std::runtime_error("fast engine: KV cache bits must be 8 or 4");
    kv4_ = kv_bits == 4;
    const Hparams & h = hp_;
    if (h.n_embd != D || h.head_dim != HD || h.n_head % h.n_head_kv || h.n_head / h.n_head_kv > 4 || h.n_rel > 16 ||
        h.n_expert > 256 || h.n_used > 16)
        throw std::runtime_error("fast engine: unexpected model shape");
    const int kvw = h.n_head_kv * HD;
    max_chunks_ = std::max((n_ctx + FCH - 1) / FCH, (std::min(h.n_swa, n_ctx) + 63) / 64);

    uint64_t need = 0;
    for (int s = 0; s < h.n_slot(); ++s) {
        const int ip = s < h.n_layer ? s : h.loop_start + (s - h.n_layer) % h.n_core();
        // sliding-window rings hold the window plus one prompt chunk, so a chunk never overwrites keys that
        // its own earlier tokens still need
        const int len = h.is_swa[ip] ? std::min(h.n_swa + TB_, n_ctx) : n_ctx;
        kv_len_.push_back(len);
        need += 2ull * len * (kv_val_bytes(kvw) + kvw / 32 * sizeof(sycl::half)) + (uint64_t) h.conv_channels() * 3 * sizeof(float);
    }
    need += 64ull << 20;  // decode activations
    const size_t TD = (size_t) TB_ * D;
    const size_t nvec = (size_t) std::max(1, m.schedule.n_vectors);
    const size_t bfloats = TD * (3 + nvec / 2 + 23) + (size_t) TB_ * (4 * 64 + 256) + (size_t) TB_ * h.n_used;  // see the pool
    need += bfloats * sizeof(float) + (size_t) TB_ * (2 + 2 * h.n_used) * sizeof(int32_t) + 64 * 1024;
    if (getenv("INK_MEM")) {
        uint64_t kv = 0;
        for (int s2 = 0; s2 < h.n_slot(); ++s2) kv += 2ull * kv_len_[s2] * (kv_val_bytes(kvw) + kvw / 32 * sizeof(sycl::half));
        std::fprintf(stderr, "memory: KV %.0f MiB (%d slots), prompt buffers %.0f MiB (%zu floats/token, n_vectors %zu), total %.0f MiB\n",
                     kv / 1048576.0, h.n_slot(), bfloats * 4 / 1048576.0, bfloats / TB_, nvec, need / 1048576.0);
    }
    Model::check_fits(q_, need, "the fast engine's caches and prompt buffers");

    for (int s = 0; s < h.n_slot(); ++s) {
        kc_.push_back(dev<int8_t>((size_t) kv_len_[s] * kv_val_bytes(kvw)));
        vc_.push_back(dev<int8_t>((size_t) kv_len_[s] * kv_val_bytes(kvw)));
        ks_.push_back(dev<sycl::half>((size_t) kv_len_[s] * kvw / 32));
        vs_.push_back(dev<sycl::half>((size_t) kv_len_[s] * kvw / 32));
        hist_.push_back(dev<float>((size_t) h.conv_channels() * 3));
        q_.fill(hist_.back(), 0.0f, (size_t) h.conv_channels() * 3);
    }
    token_ = dev<int32_t>(1);
    pos_ = dev<int32_t>(1);
    xa_ = dev<float>(D);
    xb_ = dev<float>(D);
    prelude_ = dev<float>(D);
    store_ = dev<float>((size_t) std::max(1, m.schedule.n_vectors) * D);
    in_ = dev<float>(3 * D);
    inn_ = dev<float>(3 * D);
    msg_ = dev<float>(4 * D);
    msgn_ = dev<float>(4 * D);
    ctl_ = dev<float>(3 * D);
    nin_ = dev<float>(3 * D);
    g_ = dev<float>(4 * 64);
    proj_ = dev<float>(8704);
    qf_ = dev<float>((size_t) h.n_head * HD);
    rf_ = dev<float>((size_t) h.n_head * h.n_rel);
    part_ = dev<float>((size_t) h.n_head * max_chunks_ * (HD + 2));
    att_ = dev<float>((size_t) h.n_head * HD);
    f_ = dev<float>(D);
    fraw_ = dev<float>(D);
    fn_ = dev<float>(D);
    act_ = dev<float>((size_t) h.n_used * h.n_ff);
    fq_ = dev<int8_t>(D);
    aq_ = dev<int8_t>((size_t) 8 * h.n_ff);
    aqd_ = dev<float>((size_t) 8 * h.n_ff / 32);
    aqs_ = dev<float>((size_t) 8 * h.n_ff / 32);
    fqd_ = dev<float>(D / 32);
    fqs_ = dev<float>(D / 32);
    router_logits_ = dev<float>(256);
    btok_ = dev<int32_t>(TB_);
    bids_ = dev<int32_t>((size_t) TB_ * h.n_used);
    bpairs_ = dev<int32_t>((size_t) TB_ * h.n_used);
    btiles_ = dev<int32_t>(1 + 3 * ((size_t) TB_ * h.n_used / 8 + h.n_expert + 1));
    bxa_ = dev<float>(TD);
    bxb_ = dev<float>(TD);
    bprelude_ = dev<float>(TD);
    bstore_ = dev<sycl::half>(TD * nvec);  // f16: RMS-normalised states
    bg_ = dev<float>((size_t) 4 * TB_ * 64);
    // One pool for buffers whose lifetimes within a layer visit do not overlap (prefill.cpp order):
    //   [0, 23 TD): the loop controllers' temporaries, live from the attention-input mix to the projection input
    //     (bnin_), except msg / msgn slot 3 ([9, 10) and [13, 14) TD), written then and read by the FFN controller
    //   attention phase (projections .. wo): bproj_, bq_, br_, batt_ from 0 (< 6.7 TD, clear of slot 3)
    //   FFN phase (after wo): bfraw_, bfn_, bf_ at 0-3 TD, the experts' f16 activations at 3-5 TD
    //   bo_ at 21 TD (controller slot ctl 1, free from wo to the end of the visit)
    //   bnin_ at 14 TD (gain / forget are dead between the controller step and the next FFN controller)
    if (TB_ % 16) throw std::runtime_error("fast engine: the prompt chunk must be a multiple of 16");
    float * pool = dev<float>(23 * TD);
    bin_ = pool;
    binn_ = pool + 3 * TD;
    bmsg_ = pool + 6 * TD;
    bmsgn_ = pool + 10 * TD;
    bgain_ = pool + 14 * TD;
    bforget_ = pool + 17 * TD;
    bctl_ = pool + 20 * TD;
    bnin_ = pool + 14 * TD;  // over gain / forget: written by the norm after the controllers, read by the projections
    bproj_ = pool;
    bq_ = bproj_ + (size_t) TB_ * 8704;
    br_ = bq_ + (size_t) TB_ * h.n_head * HD;
    batt_ = br_ + (size_t) TB_ * h.n_head * h.n_rel;  // whole 16-token blocks (the XMX attention stores them)
    if (batt_ + (size_t) TB_ * h.n_head * HD > bmsg_ + 3 * TD) throw std::runtime_error("fast engine: prompt pool layout");
    bfraw_ = pool;
    bfn_ = pool + TD;
    bf_ = pool + 2 * TD;
    bact_ = reinterpret_cast<sycl::half *>(pool + 3 * TD);
    if ((size_t) TB_ * h.n_used * h.n_ff > 2 * 2 * TD) throw std::runtime_error("fast engine: prompt pool layout (experts)");
    bo_ = pool + 21 * TD;
    blogits_ = dev<float>((size_t) TB_ * 256);
    bwts_ = dev<float>((size_t) TB_ * h.n_used);
    logits_dev_ = dev<float>((size_t) m.output.ne[1]);
    wts_ = dev<float>(64);
    ids_ = dev<int32_t>(64);
    q_.wait();

    if (const char * e = getenv("INK_DOWNQ8")) down_q8 = std::string(e) != "0";
    if (const char * e = getenv("INK_GUR")) gate_up_r = std::string(e) != "0";
    prepare_jobs();
    if (use_graph) build_graph();
}

void FastEngine::build_graph() {
    try {
        sycl_exp::command_graph<sycl_exp::graph_state::modifiable> g(q_.get_context(), q_.get_device());
        g.begin_recording(q_);
        record();
        g.end_recording(q_);
        graph_ = std::make_unique<sycl_exp::command_graph<sycl_exp::graph_state::executable>>(g.finalize());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fast engine: command graph unavailable (%s), submitting directly\n", e.what());
        graph_.reset();
    }
}

FastEngine::~FastEngine() {
    q_.wait();
    for (void * p : allocs_) sycl::free(p, q_);
}

// Build every job table without submitting work.
void FastEngine::prepare_jobs() {
    jobs_only_ = true;
    record();
    jobs_only_ = false;
}

// ---------------------------------------------------------------------------------------------------
// one layer visit

void FastEngine::visit(const Visit & v) {
    const Hparams & h = hp_;
    const LayerW & L = m_.layers[v.ip];
    const bool swa = h.is_swa[v.ip];
    const bool loop = !v.bank.empty();
    const int nh = h.n_head, nkv = h.n_head_kv, kvw = nkv * HD;
    const LoopW * lw = loop ? &m_.loops[v.ip - h.loop_start] : nullptr;
    auto store = [&](int id) { return store_ + (size_t) id * D; };
    const W * norms[3] = {swa ? &L.attn_norm : &L.q_inp_norm, swa ? &L.attn_norm : &L.k_inp_norm, swa ? &L.attn_norm : &L.v_inp_norm};

    // the inputs the projections read: the norms' outputs (SWA layers without loop control share one)
    const float * nin[3];
    for (int j = 0; j < 3; ++j) nin[j] = nin_ + (size_t) (swa && !loop ? 0 : j) * D;
    const float * inr[3], * inz[3];
    for (int j = 0; j < 3; ++j) {
        inr[j] = in_ + (size_t) (swa ? 0 : j) * D;
        inz[j] = inn_ + (size_t) (swa ? 0 : j) * D;
    }
    // proj_ layout: Q 0, K 3072, V 4096, R 5120, sink 5504, gate 5528, kdyn 8600, vdyn 8632
    float * Q = proj_, * K = proj_ + 3072, * V = proj_ + 4096, * R = proj_ + 5120, * sink = proj_ + 5504,
          * gate = proj_ + 5528, * kdyn = proj_ + 8600, * vdyn = proj_ + 8632;
    auto job = [](const W & w, const float * x, const float * x2, int split, float * y) {
        return MvJob{wref(w), w.ktype, (int) w.ne[0], (int) w.ne[1], 0, split, x, x2, y, 0.0f};
    };

    // job tables (built in prepare_jobs, reused afterwards in the same order)
    std::vector<MvGroup> ctl_plan, ffn_ctl_plan;
    if (loop) {
        std::vector<MvJob> js;
        for (int c = 0; c < 3; ++c) js.push_back(job(lw->down[c], inz[c], msgn_ + (size_t) c * D, D, g_ + c * 64));
        ctl_plan = plan_mv(js);
        ffn_ctl_plan = plan_mv({job(lw->down[3], fn_, msgn_ + 3 * (size_t) D, D, g_ + 3 * 64)});
    }
    std::vector<MvJob> pj = {job(L.wq, nin[0], nin[0], 1 << 30, Q),    job(L.wk, nin[1], nin[1], 1 << 30, K),
                             job(L.wv, nin[2], nin[2], 1 << 30, V),    job(L.rel, nin[0], nin[0], 1 << 30, R),
                             job(L.sink_proj, nin[0], nin[0], 1 << 30, sink), job(L.gate, nin[0], nin[0], 1 << 30, gate)};
    if (!swa) {
        pj.push_back(job(L.k_conv_proj, nin[1], nin[1], 1 << 30, kdyn));
        pj.push_back(job(L.v_conv_proj, nin[2], nin[2], 1 << 30, vdyn));
    }
    const std::vector<MvGroup> proj_plan = plan_mv(pj);
    const std::vector<MvGroup> router_plan = plan_mv({job(L.gate_inp, f_, f_, 1 << 30, router_logits_)});
    if (jobs_only_) return;

    if (v.restart >= 0) {  // a new pass: keep the previous pass's output for the messages, restart from the prelude
        MixArgs a;
        a.cur = xa_;
        a.push = store(v.restart);
        P("restart", k_mix(q_, a));
        P("restart", q_.memcpy(xa_, prelude_, D * sizeof(float)));
    }

    // 1-3: attention inputs (depth mix, loop control, input norms)
    {
        MixArgs a;
        a.qw = L.attn_res_query.ok() ? L.attn_res_query.f32() : nullptr;
        if (a.qw) {
            for (size_t s = 0; s < v.blocks.size(); ++s) a.st[s] = store(v.blocks[s]);
            a.ns = (int) v.blocks.size();
        }
        a.nq = a.qw ? (int) L.attn_res_query.ne[1] : 1;
        a.cur = xa_;
        a.eps = h.eps;
        if (v.push >= 0) a.push = store(v.push);
        for (int j = 0; j < a.nq; ++j) {
            if (loop) {
                a.out[j] = in_ + (size_t) j * D;
                a.outn[j] = inn_ + (size_t) j * D;
            } else {
                a.out[j] = nin_ + (size_t) j * D;
                a.normw[j] = norms[j]->f32();
            }
        }
        P("attn_mix", k_mix(q_, a));
    }
    if (loop) {
        MixArgs b;  // messages from the previous pass's bank
        for (size_t s = 0; s < v.bank.size(); ++s) b.st[s] = store(v.bank[s]);
        b.ns = (int) v.bank.size();
        b.qw = lw->msg_query.f32();
        b.nq = 4;
        for (int j = 0; j < 4; ++j) {
            b.out[j] = msg_ + (size_t) j * D;
            b.outn[j] = msgn_ + (size_t) j * D;
        }
        P("msg_mix", k_mix(q_, b));
        run_mv("ctl_down", ctl_plan);
        CtlArgs c;
        c.n = 3;
        c.rank = (int) lw->gain[0].ne[0];
        for (int k = 0; k < 3; ++k) {
            c.gain[k] = lw->gain[k].f32();
            c.forget[k] = lw->forget[k].f32();
            c.g[k] = g_ + k * 64;
            c.cur[k] = inr[k];
            c.msg[k] = msg_ + (size_t) k * D;
            c.out[k] = ctl_ + (size_t) k * D;
        }
        P("ctl_up", k_ctl_up(q_, c));
        NormArgs n;
        n.n = 3;
        n.eps = h.eps;
        for (int k = 0; k < 3; ++k) {
            n.x[k] = ctl_ + (size_t) k * D;
            n.w[k] = norms[k]->f32();
            n.y[k] = nin_ + (size_t) k * D;
        }
        P("norm", k_norm(q_, n));
    }

    // 4: projections, grouped by weight format
    run_mv("proj", proj_plan);

    // 5: Q/K/V/R post-processing and the cache write
    {
        const float * qclip = L.q_clip.f32(), * rclip = L.rel_clip.f32(), * kconv = L.k_conv.f32(), * vconv = L.v_conv.f32();
        float * hist = hist_[v.slot];
        int8_t * kc = kc_[v.slot], * vc = vc_[v.slot];
        sycl::half * ks = ks_[v.slot], * vs = vs_[v.slot];
        const bool kv4 = kv4_;
        float * qf = qf_, * rf = rf_;
        const int32_t * posp = pos_;
        const float qk_clip = h.qk_clip, rel_clip = h.rel_clip, alpha = h.log_alpha, floor_ = h.log_floor;
        const int n_rel = h.n_rel, ring = swa ? kv_len_[v.slot] : 0;
        const int kvlen = kv_len_[v.slot];  // head-major cache: [head][kvlen][HD]
        const bool full = !swa;
        const float * kd = full ? kdyn : nullptr, * vd = full ? vdyn : nullptr;
        const float cnorm = 1.0f / std::sqrt((float) (4 * (full ? 2 : 1) + 1));
        P("qkv_post", q_.parallel_for(sycl::nd_range<1>((size_t) (nh + 2 * nkv + nh) * HD, HD), [=](sycl::nd_item<1> it) {
            const auto grp = it.get_group();
            const int g = (int) it.get_group(0), t = (int) it.get_local_id(0);
            const int pos = *posp;
            const float tau = full ? 1.0f + alpha * sycl::log(sycl::fmax(1.0f, (float) (pos + 1) / floor_)) : 1.0f;
            if (g < nh) {  // Q head: scale, clip, log-position scale
                const int i = g * HD + t;
                float x = Q[i] * qclip[i];
                const float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                x *= sycl::clamp(qk_clip / (sycl::sqrt(ss) + 1e-6f), 0.0f, 1.0f);
                qf[i] = x * tau;
            } else if (g < nh + 2 * nkv) {  // K or V head: convolution, then clip (K) or RMS norm (V), into the cache
                const bool is_v = g >= nh + nkv;
                const int kh = g - nh - (is_v ? nkv : 0);
                const int c = kh * HD + t;
                const int ch = (is_v ? kvw : 0) + c;
                const float * w = is_v ? vconv : kconv;
                const float * dyn = is_v ? vd : kd;
                const float xx[4] = {hist[ch * 3 + 0], hist[ch * 3 + 1], hist[ch * 3 + 2], is_v ? V[c] : K[c]};
                float o = 0.0f;
                for (int j = 0; j < 4; ++j) o += w[j + 4 * c] * xx[j];
                if (dyn)
                    for (int lag = 0; lag < 4; ++lag) o += dyn[lag + 4 * kh] * xx[3 - lag];
                hist[ch * 3 + 0] = xx[1];
                hist[ch * 3 + 1] = xx[2];
                hist[ch * 3 + 2] = xx[3];
                float x = (xx[3] + o) * cnorm;
                const float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                if (is_v) x *= 1.0f / sycl::sqrt(ss / HD + 1e-6f);
                else x *= sycl::clamp(qk_clip / (sycl::sqrt(ss) + 1e-6f), 0.0f, 1.0f);
                const int idx = ring ? pos % ring : pos;
                kv_put(grp, it.get_sub_group(), t, x, kv4, is_v ? vc : kc, is_v ? vs : ks, ((size_t) kh * kvlen + idx) * HD + t);
            } else {  // R head (n_rel values)
                const int rh = g - nh - 2 * nkv;
                const int i = rh * n_rel + t;
                float x = t < n_rel ? R[i] * rclip[i] : 0.0f;
                const float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                x *= sycl::clamp(rel_clip / (sycl::sqrt(ss) + 1e-6f), 0.0f, 1.0f);
                if (t < n_rel) rf[i] = x * tau;
            }
        }));
    }

    // 6: attention, split over chunks of keys; then combine the chunks with the sink and the output gate
    // sliding-window layers: 64-key chunks (16 over the window) so they still fill the GPU; full layers: 256
    const int chunk = swa ? swa_chunk : FCH;
    const int n_chunks = swa ? (std::min(h.n_swa, n_ctx_) + chunk - 1) / chunk : max_chunks_;
    {
        const float * qf = qf_, * rf = rf_;
        const int8_t * kc = kc_[v.slot], * vc = vc_[v.slot];
        const sycl::half * ksc = ks_[v.slot], * vsc = vs_[v.slot];
        const bool kv4 = kv4_;
        const float * rb = L.rel_bias.f32();
        const int extent = (int) L.rel_bias.ne[1], n_rel = h.n_rel, ratio = nh / nkv, n_swa = h.n_swa;
        const int ring = swa ? kv_len_[v.slot] : 0;
        const int kvlen = kv_len_[v.slot];  // head-major cache: [head][kvlen][HD]
        const int32_t * posp = pos_;
        float * part = part_;
        const int nc = n_chunks, pstride = max_chunks_;
        const float scale = 1.0f / std::sqrt((float) HD);
        // One work-group per (KV head, chunk of keys); the KV head's `ratio` query heads share each K and V read.
        // Scores: a thread per key (queries in local memory); then V summed with a thread per dim. Measured best
        // of three designs on the B580 (2026-11-23): sub-group-per-key scoring and flash-decoding were slower.
        const int8_t * kch = kc, * vch = vc;
        static const int att_dbg_env = getenv("INK_ATT_DBG") ? atoi(getenv("INK_ATT_DBG")) : 0;  // 1: skip scores, 2: skip P.V
        const int att_dbg = att_dbg_env;
        const int ch = chunk;
        // one launcher, two work-group sizes: 128 threads for the sliding-window layers (<= 1024 keys, 64-key
        // chunks), 256 for full attention (measured 2026-11-24: 128 is better for SWA and short context, 256 at depth)
        auto run_attn = [&](auto awg_c) {
        constexpr int AWG = decltype(awg_c)::value, FCH = AWG, KPT = 1;
        static_assert(AWG >= HD && AWG % SG == 0, "decode attention writes one output dim per thread (measured: 64 broke it)");
        P(swa ? "attn_swa" : "attn_full", q_.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> sq(4 * HD, cgh);
            sycl::local_accessor<float, 1> sr(4 * 16, cgh);
            sycl::local_accessor<float, 1> sp(4 * FCH, cgh);
            sycl::local_accessor<float, 1> sa(4 * HD, cgh);  // ratio x HD output partials (local float atomics)
            sycl::local_accessor<int32_t, 1> sq8(4 * HD / 4, cgh);  // the queries as int8 (4 per word), see below
            sycl::local_accessor<float, 1> sqd(4 * HD / 32, cgh), sqs(4 * HD / 32, cgh);
            cgh.parallel_for(sycl::nd_range<1>((size_t) nkv * nc * AWG, AWG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto grp = it.get_group();
                const int g = (int) it.get_group(0), t = (int) it.get_local_id(0);
                const int kvh = g / nc, c = g % nc, h0 = kvh * ratio;
                const int pos = *posp;
                const int lo = swa ? sycl::max(0, pos - n_swa + 1) : 0;
                const int k0 = lo + c * ch;
                if (k0 > pos) {  // empty chunk (uniform over the work-group)
                    for (int j = 0; j < ratio; ++j) {
                        float * out = part + ((size_t) (h0 + j) * pstride + c) * (HD + 2);
                        if (t == 0) {
                            out[0] = -INFINITY;
                            out[1] = 0.0f;
                        }
                        if (t < HD) out[2 + t] = 0.0f;
                    }
                    return;
                }
                for (int i = t; i < ratio * HD; i += AWG) sq[i] = qf[h0 * HD + i];
                for (int i = t; i < ratio * n_rel; i += AWG) sr[i] = rf[h0 * n_rel + i];
                for (int i = t; i < ratio * HD; i += AWG) sa[i] = 0.0f;
                sycl::group_barrier(grp);
                // queries to int8, a scale and an integer sum per 32 values (one thread per block). For the 4-bit
                // cache each 8-value group is stored as [even dims][odd dims], matching the low and high nibbles
                // of a 4-byte K word.
                {
                    const auto sgq = it.get_sub_group();
                    const int sgid = (int) sgq.get_group_id()[0], lane = (int) sgq.get_local_id()[0];
                    for (int qb = sgid; qb < ratio * (HD / 32); qb += AWG / SG) {  // a sub-group per 32-value block
                        const int j = qb / (HD / 32), b = qb % (HD / 32);
                        const float v0 = sq[j * HD + b * 32 + 2 * lane], v1 = sq[j * HD + b * 32 + 2 * lane + 1];
                        const float amax = sycl::reduce_over_group(sgq, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
                        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
                        const int q0 = (int) sycl::rint(v0 * id), q1 = (int) sycl::rint(v1 * id);
                        const int qs = sycl::reduce_over_group(sgq, q0 + q1, sycl::plus<int>());
                        int8_t * dst = (int8_t *) &sq8[0] + j * HD + b * 32;
                        auto at = [&](int v) { const int w = v / 8, e = v % 8; return kv4 ? (2 * w + (e & 1)) * 4 + e / 2 : v; };
                        dst[at(2 * lane)] = (int8_t) q0;
                        dst[at(2 * lane + 1)] = (int8_t) q1;
                        if (lane == 0) {
                            sqd[qb] = d;
                            sqs[qb] = (float) qs;
                        }
                    }
                }
                sycl::group_barrier(grp);
                // scores: up to KPT keys per thread (key t + q * AWG of the chunk)
                float sc[KPT][4];
                bool valid[KPT];
                for (int q4 = 0; q4 < KPT; ++q4) {
                    const int kl = t + q4 * AWG, k = k0 + kl;
                    valid[q4] = kl < ch && k <= pos;
                    for (int j = 0; j < 4; ++j) sc[q4][j] = -INFINITY;
                    if (!valid[q4]) continue;
                    if (att_dbg & 1) {
                        for (int j = 0; j < ratio; ++j) sc[q4][j] = 0.0f;
                        continue;
                    }
                    const int idx = ring ? k % ring : k;
                    const size_t kbase = ((size_t) kvh * kvlen + idx) * HD;
                    // K . q with DP4A: per 32-value block an int32 dot, then the K and q scales
                    using sycl::ext::oneapi::dot_acc;
                    float dot[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                    for (int b = 0; b < HD / 32; ++b) {
                        const float ds = (float) ksc[kbase / 32 + b];
                        int32_t kw[8];
                        if (kv4) {
                            const sycl::uint4 w4 = *(const sycl::uint4 *) ((const uint8_t *) kch + (kbase + 32 * b) / 2);
                            for (int w = 0; w < 4; ++w) {
                                kw[2 * w] = (int32_t) (w4[w] & 0x0F0F0F0Fu);
                                kw[2 * w + 1] = (int32_t) ((w4[w] >> 4) & 0x0F0F0F0Fu);
                            }
                        } else {
                            const sycl::int4 * p4 = (const sycl::int4 *) (kch + kbase + 32 * b);
                            const sycl::int4 a0 = p4[0], a1 = p4[1];
                            for (int w = 0; w < 4; ++w) {
                                kw[w] = a0[w];
                                kw[4 + w] = a1[w];
                            }
                        }
                        for (int j = 0; j < ratio; ++j) {
                            const int32_t * qw = &sq8[(j * HD + 32 * b) / 4];
                            int32_t is = 0;
                            for (int w = 0; w < 8; ++w) is = dot_acc(kw[w], qw[w], is);
                            const float val = kv4 ? (float) is - 8.0f * sqs[j * (HD / 32) + b] : (float) is;  // nibble - 8
                            dot[j] += ds * sqd[j * (HD / 32) + b] * val;
                        }
                    }
                    for (int j = 0; j < ratio; ++j) sc[q4][j] = dot[j] * scale;
                    const int dist = pos - k;
                    if (dist < extent) {
                        for (int r = 0; r < n_rel; ++r) {
                            const float bb = rb[r + n_rel * dist];
                            for (int j = 0; j < ratio; ++j) sc[q4][j] += bb * sr[j * n_rel + r];
                        }
                    }
                }
                float m[4], l[4];
                for (int j = 0; j < ratio; ++j) {
                    float mx = -INFINITY;
                    for (int q4 = 0; q4 < KPT; ++q4) mx = sycl::fmax(mx, sc[q4][j]);
                    m[j] = sycl::reduce_over_group(grp, mx, sycl::maximum<float>());
                    float ps = 0.0f;
                    for (int q4 = 0; q4 < KPT; ++q4) {
                        const int kl = t + q4 * AWG;
                        const float pr = valid[q4] ? sycl::exp(sc[q4][j] - m[j]) : 0.0f;
                        ps += pr;
                        if (kl < ch) sp[j * FCH + kl] = pr;
                    }
                    l[j] = sycl::reduce_over_group(grp, ps, sycl::plus<float>());
                }
                sycl::group_barrier(grp);
                // P . V: a sub-group per (query head, key group), each step 4 consecutive keys with a lane per (key,
                // 32 dims): 16-byte loads, 256 contiguous bytes per sub-group (measured 2026-11-24: 8 dims per lane
                // with one 64-byte key per step ran at ~74 GB/s; this is +8% decode at 90k, +4% at 30k)
                {
                    const auto sgv = it.get_sub_group();
                    const int sgid = (int) sgv.get_group_id()[0], lane = (int) sgv.get_local_id()[0];
                    // variant: a sub-group per (query head, key group) reads 4 consecutive keys per step, a lane per
                    // (key of the 4, 32 dims): 16-byte loads (4-bit cache), 256 contiguous bytes per sub-group
                    const int KGH = (AWG / SG) / ratio;
                    const int j = sgid % ratio, kg = sgid / ratio, sub = lane / 4, d32 = (lane % 4) * 32;
                    if (kg < KGH && !(att_dbg & 2)) {
                        float acc[32] = {};
                        const int nk = sycl::min(ch, pos - k0 + 1);
                        for (int b4 = kg * 4; b4 < nk; b4 += KGH * 4) {
                            const int jj = b4 + sub;
                            if (jj < nk) {
                                const int kk = k0 + jj;
                                const int idx = ring ? kk % ring : kk;
                                const size_t vi = ((size_t) kvh * kvlen + idx) * HD + d32;
                                float v32[32];
                                if (kv4) {
                                    const sycl::uint4 w = *(const sycl::uint4 *) ((const uint8_t *) vch + vi / 2);
                                    for (int q = 0; q < 4; ++q)
                                        for (int b = 0; b < 4; ++b) {
                                            const uint32_t by = (w[q] >> (8 * b)) & 0xFF;
                                            v32[8 * q + 2 * b] = (float) ((int) (by & 15) - 8);
                                            v32[8 * q + 2 * b + 1] = (float) ((int) (by >> 4) - 8);
                                        }
                                } else {
                                    const sycl::int4 * p4 = (const sycl::int4 *) (vch + vi);
                                    for (int h2 = 0; h2 < 2; ++h2) {
                                        const sycl::int4 w = p4[h2];
                                        for (int q = 0; q < 4; ++q)
                                            for (int b = 0; b < 4; ++b) v32[16 * h2 + 4 * q + b] = (float) (int8_t) ((w[q] >> (8 * b)) & 0xFF);
                                    }
                                }
                                const float w = sp[j * FCH + jj] * (float) vsc[vi / 32];
                                for (int e = 0; e < 32; ++e) acc[e] += w * v32[e];
                            }
                        }
                        for (int e = 0; e < 32; ++e) {
                            float v = acc[e];
                            v += sycl::permute_group_by_xor(sgv, v, 4);
                            v += sycl::permute_group_by_xor(sgv, v, 8);
                            acc[e] = v;
                        }
                        if (sub == 0)
                            for (int e = 0; e < 32; ++e)
                                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::work_group, sycl::access::address_space::local_space>(
                                    sa[j * HD + d32 + e]).fetch_add(acc[e]);
                    }
                }
                sycl::group_barrier(grp);
                for (int j = 0; j < ratio; ++j) {
                    float * out = part + ((size_t) (h0 + j) * pstride + c) * (HD + 2);
                    if (t < HD) out[2 + t] = sa[j * HD + t];
                    if (t == 0) {
                        out[0] = m[j];
                        out[1] = l[j];
                    }
                }
            });
        }));
        };
        if (swa) run_attn(std::integral_constant<int, INK_SWA_AWG>{});
        else run_attn(std::integral_constant<int, 256>{});
        float * att = att_;
        const float * sk = sink, * gt = gate;
        // combine the chunks: a work-group per head, 8 groups of chunks x HD dims (the old one-thread-per-dim loop
        // over ~350 chunks at 90k left the GPU idle), the sink in the denominator, then the output gate
        constexpr int CG = 8;
        P(swa ? "combine_swa" : "combine_full", q_.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> sacc(CG * HD, cgh), sls(CG, cgh);
            cgh.parallel_for(sycl::nd_range<1>((size_t) nh * CG * HD, CG * HD), [=](sycl::nd_item<1> it) {
                const auto grp = it.get_group();
                const int hh = (int) it.get_group(0), t = (int) it.get_local_id(0), d = t % HD, gq = t / HD;
                const float * base = part + (size_t) hh * pstride * (HD + 2);
                float ml = -INFINITY;
                for (int c = t; c < nc; c += CG * HD) ml = sycl::fmax(ml, base[c * (HD + 2)]);
                const float M = sycl::fmax(sk[hh], sycl::reduce_over_group(grp, ml, sycl::maximum<float>()));
                float ls = 0.0f, acc = 0.0f;
                for (int c = gq; c < nc; c += CG) {
                    const float mc = base[c * (HD + 2)];
                    if (mc == -INFINITY) continue;
                    const float w = sycl::exp(mc - M);
                    ls += base[c * (HD + 2) + 1] * w;
                    acc += base[c * (HD + 2) + 2 + d] * w;
                }
                sacc[gq * HD + d] = acc;
                if (d == 0) sls[gq] = ls;
                sycl::group_barrier(grp);
                if (t < HD) {
                    float o = 0.0f, L = sycl::exp(sk[hh] - M);
                    for (int g2 = 0; g2 < CG; ++g2) {
                        o += sacc[g2 * HD + t];
                        L += sls[g2];
                    }
                    att[hh * HD + t] = o / L * sigmoidf(gt[hh * HD + t]);
                }
            });
        }));
    }

    // 7-9: output projection, its convolution, and the residual (a block start drops the old residual)
    {
        ProjConvArgs a;
        a.w = wref(L.wo);
        a.cols = (int) L.wo.ne[0];
        a.x = att_;
        a.conv = L.post_conv.f32();
        a.hist = hist_[v.slot] + (size_t) 2 * kvw * 3;
        a.res = xa_;
        a.out = xb_;
        a.drop = v.push >= 0;
        dispatch_type(L.wo.ktype, [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            P(profile ? tlabel("wo", T) : "wo", stage_tabs ? k_proj_conv<T, true>(q_, a) : k_proj_conv<T, false>(q_, a));
        });
    }

    // 10: FFN input: depth mix (blocks + this visit's new block), loop control, norm
    {
        MixArgs a;
        for (size_t s = 0; s < v.blocks.size(); ++s) a.st[s] = store(v.blocks[s]);
        a.ns = (int) v.blocks.size();
        if (v.push >= 0) a.st[a.ns++] = store(v.push);
        a.cur = xb_;
        a.qw = L.ffn_res_query.f32();
        a.nq = 1;
        a.eps = h.eps;
        if (loop) {
            a.out[0] = fraw_;
            a.outn[0] = fn_;
        } else {
            a.out[0] = f_;
            a.normw[0] = L.ffn_norm.f32();
        }
        P("ffn_mix", k_mix(q_, a));
        if (loop) {
            run_mv("ffn_ctl", ffn_ctl_plan);
            CtlArgs c;
            c.n = 1;
            c.rank = (int) lw->gain[3].ne[0];
            c.gain[0] = lw->gain[3].f32();
            c.forget[0] = lw->forget[3].f32();
            c.g[0] = g_ + 3 * 64;
            c.cur[0] = fraw_;
            c.msg[0] = msg_ + 3 * (size_t) D;
            c.out[0] = ctl_;
            P("ffn_ctl", k_ctl_up(q_, c));
            NormArgs n;
            n.n = 1;
            n.eps = h.eps;
            n.x[0] = ctl_;
            n.w[0] = L.ffn_norm.f32();
            n.y[0] = f_;
            P("ffn_ctl", k_norm(q_, n));
        }
    }

    // 11: router on the GPU
    run_mv("router", router_plan);
    P("router", k_topk(q_, router_logits_, L.probs_b.ok() ? L.probs_b.f32() : nullptr, h.n_expert, h.n_used, h.w_scale, ids_, wts_));

    // 12a: gate-up of the chosen experts with SITU
    {
        bool q8 = false;
        dispatch_type(L.gate_up.ktype, [&](auto tag) { q8 = has_q8<decltype(tag)::value>(); });
        if (q8) P("quant_x", k_quant_x(q_, f_, fq_, fqd_, fqs_));
        GateUpArgs a{wref(L.gate_up), fq_, fqd_, fqs_, (int) L.gate_up.ne[0], (int) L.gate_up.ne[1], h.n_ff, h.n_used, f_,
                     L.gate_up_s.ok() ? L.gate_up_s.f32() : nullptr, ids_, act_, h.situ_beta, h.situ_lb};
        dispatch_type(L.gate_up.ktype, [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            const char * lab = profile ? tlabel("gate_up", T) : "gate_up";
            if constexpr (has_q8<T>()) {
                if (gate_up_r && a.nff % INK_GUR == 0) {
                    P(lab, k_gate_up_r<T>(q_, a));
                    return;
                }
            }
            P(lab, stage_tabs ? k_gate_up<T, true>(q_, a) : k_gate_up<T, false>(q_, a));
        });
    }

    // 12b-13: down of the chosen experts, weighted sum, convolution, residual
    {
        ProjConvArgs a;
        a.w = wref(L.down);
        a.cols = (int) L.down.ne[0];
        a.x = act_;
        a.nu = h.n_used;
        a.ids = ids_;
        a.wts = wts_;
        a.scale = L.down_s.ok() ? L.down_s.f32() : nullptr;
        a.conv = L.ffn_post_conv.f32();
        a.hist = hist_[v.slot] + (size_t) (2 * kvw + D) * 3;
        a.res = xb_;
        a.out = xa_;
        dispatch_type(L.down.ktype, [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            const char * lab = profile ? tlabel("down", T) : "down";
            if constexpr (has_down_q8<T>()) {
                if (down_q8 && a.cols % 32 == 0 && a.nu * a.cols <= 8 * h.n_ff) {
                    P("quant_act", k_quant_act(q_, act_, a.nu * a.cols, T == q::Q2_0R, aq_, aqd_, aqs_));
                    P(lab, k_down_q8<T>(q_, DownQ8Args{a, aq_, aqd_, aqs_}));
                    return;
                }
            }
            if (a.nu == 8) P(lab, stage_tabs ? k_down8<T, true>(q_, a) : k_down8<T, false>(q_, a));
            else P(lab, stage_tabs ? k_proj_conv<T, true>(q_, a) : k_proj_conv<T, false>(q_, a));
        });
    }
}

// ---------------------------------------------------------------------------------------------------
// the whole token

void FastEngine::record() {
    job_next_ = 0;
    const Hparams & h = hp_;
    const bool run = !jobs_only_;
    if (run) {  // embedding row, RMS norm with token_embd_norm
        const W & e = m_.tok_embd;
        const uint8_t * ep = e.p;
        const size_t rb = e.row_bytes;
        const uint32_t type = e.type;
        const q::BlockShape bs = q::shape(type);
        const int32_t * tok = token_;
        const float * nw = m_.tok_norm.f32();
        float * x = xa_;
        const float eps = h.eps;
        P("embed", q_.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) {
            const int t = (int) it.get_local_id(0);
            const uint8_t * row = ep + (size_t) *tok * rb;
            float v[32];
            float ss = 0.0f;
            const bool mine = t < D / 32;
            if (mine) {
                const int c0 = t * 32;
                q::decode32(type, row + (size_t) (c0 / bs.elems) * bs.bytes, (c0 % bs.elems) / 32, v);
                for (int j = 0; j < 32; ++j) ss += v[j] * v[j];
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float r = 1.0f / sycl::sqrt(ss / D + eps);
            if (mine)
                for (int j = 0; j < 32; ++j) x[t * 32 + j] = v[j] * r * nw[t * 32 + j];
        }));
    }
    const auto & vs = m_.schedule.visits;
    for (size_t i = 0; i < vs.size(); ++i) {
        visit(vs[i]);
        if (run && (int) i + 1 == m_.schedule.prelude_end) P("prelude", q_.memcpy(prelude_, xa_, D * sizeof(float)));
        if (run && !graph_ && hook) hook("l_out-" + std::to_string(vs[i].slot), xa_, D);
    }
    // output: depth mix with the final blocks, norm, LM head with softcap
    const W & o = m_.output;
    const std::vector<MvGroup> head = plan_mv({MvJob{wref(o), o.ktype, (int) o.ne[0], (int) o.ne[1], 0, 1 << 30, f_, f_, logits_dev_, h.softcap}});
    if (!run) return;
    MixArgs a;
    for (size_t s = 0; s < m_.schedule.out_blocks.size(); ++s) a.st[s] = store_ + (size_t) m_.schedule.out_blocks[s] * D;
    a.ns = (int) m_.schedule.out_blocks.size();
    a.cur = xa_;
    a.qw = m_.output_res_query.f32();
    a.nq = 1;
    a.eps = h.eps;
    a.out[0] = f_;
    a.normw[0] = m_.output_norm.f32();
    P("out_mix", k_mix(q_, a));
    run_mv("head", head);
}

const std::vector<float> & FastEngine::step(int token, int pos) {
    if (pos >= n_ctx_) throw std::runtime_error("context is full");
    q_.memcpy(token_, &token, sizeof(int32_t));
    q_.memcpy(pos_, &pos, sizeof(int32_t));
    if (graph_) q_.ext_oneapi_graph(*graph_);
    else record();
    logits_.resize(hp_.n_vocab);
    q_.memcpy(logits_.data(), logits_dev_, hp_.n_vocab * sizeof(float)).wait();
    if (profile && !prof_events_.empty()) {
        for (auto & [label, e] : prof_events_) {
            const double ms = (e.get_profiling_info<sycl::info::event_profiling::command_end>() -
                               e.get_profiling_info<sycl::info::event_profiling::command_start>()) * 1e-6;
            auto it = std::find_if(prof_ms_.begin(), prof_ms_.end(), [&](auto & p) { return p.first == label; });
            if (it == prof_ms_.end()) prof_ms_.emplace_back(label, ms);
            else it->second += ms;
        }
        prof_events_.clear();
        ++prof_tokens_;
    }
    return logits_;
}

// one slot's cache for tokens [tok0, tok0 + ntok) <-> host bytes laid out K, K scales, V, V scales
void FastEngine::kv_copy(int sl, size_t tok0, size_t ntok, uint8_t * host, bool to_host) {
    // head-major cache: each head's tokens [tok0, tok0 + ntok) are contiguous; host bytes are K values, K scales,
    // V values, V scales, each head after head
    const size_t len = kv_len_[sl], nh = hp_.n_head_kv;
    const size_t vb = kv_val_bytes(ntok * HD), sb = ntok * HD / 32 * sizeof(sycl::half);
    size_t o = 0;
    for (int kind = 0; kind < 4; ++kind)
        for (size_t hh = 0; hh < nh; ++hh) {
            const size_t first = (hh * len + tok0) * HD;  // value index
            void * dev = kind == 0 ? (void *) (kc_[sl] + kv_val_bytes(first)) : kind == 1 ? (void *) (ks_[sl] + first / 32)
                       : kind == 2 ? (void *) (vc_[sl] + kv_val_bytes(first)) : (void *) (vs_[sl] + first / 32);
            const size_t n = kind % 2 ? sb : vb;
            if (to_host) q_.memcpy(host + o, dev, n);
            else q_.memcpy(dev, host + o, n);
            o += n;
        }
}


size_t FastEngine::kv_bytes_(size_t ntok, size_t kvw) const {
    return 2 * (ntok * kv_val_bytes(kvw) + ntok * kvw / 32 * sizeof(sycl::half));
}

void FastEngine::save(Snapshot & s, int pos) {
    const Hparams & h = hp_;
    const size_t hn = (size_t) h.conv_channels() * 3, kvw = (size_t) h.n_head_kv * HD;
    s.hist.resize(hn * h.n_slot());
    size_t swa = 0;
    for (int sl = 0; sl < h.n_slot(); ++sl)
        if (is_swa_slot(sl)) swa += kv_bytes_(kv_len_[sl], kvw);
    s.swa.resize(swa);
    size_t o = 0;
    for (int sl = 0; sl < h.n_slot(); ++sl) {
        q_.memcpy(s.hist.data() + hn * sl, hist_[sl], hn * sizeof(float));
        if (is_swa_slot(sl)) {
            kv_copy(sl, 0, kv_len_[sl], s.swa.data() + o, true);
            o += kv_bytes_(kv_len_[sl], kvw);
        }
    }
    q_.wait();
    s.pos = pos;
}

void FastEngine::restore(const Snapshot & s) {
    const Hparams & h = hp_;
    const size_t hn = (size_t) h.conv_channels() * 3, kvw = (size_t) h.n_head_kv * HD;
    if (s.hist.size() != hn * h.n_slot()) throw std::runtime_error("restore: snapshot from another engine");
    size_t o = 0, f = 0;
    for (int sl = 0; sl < h.n_slot(); ++sl) {
        q_.memcpy(hist_[sl], s.hist.data() + hn * sl, hn * sizeof(float));
        if (is_swa_slot(sl)) {
            kv_copy(sl, 0, kv_len_[sl], const_cast<uint8_t *>(s.swa.data()) + o, false);
            o += kv_bytes_(kv_len_[sl], kvw);
        } else if (!s.full.empty()) {
            kv_copy(sl, 0, s.pos, const_cast<uint8_t *>(s.full.data()) + f, false);
            f += kv_bytes_(s.pos, kvw);
        }
    }
    q_.wait();
}

void FastEngine::stash(Snapshot & dst, const Snapshot & src) {
    const Hparams & h = hp_;
    const size_t kvw = (size_t) h.n_head_kv * HD;
    dst.pos = src.pos;
    dst.hist = src.hist;
    dst.swa = src.swa;
    size_t full = 0;
    for (int sl = 0; sl < h.n_slot(); ++sl)
        if (!is_swa_slot(sl)) full += kv_bytes_(src.pos, kvw);
    dst.full.resize(full);
    size_t o = 0;
    for (int sl = 0; sl < h.n_slot(); ++sl) {
        if (is_swa_slot(sl)) continue;
        kv_copy(sl, 0, src.pos, dst.full.data() + o, true);
        o += kv_bytes_(src.pos, kvw);
    }
    q_.wait();
}

bool FastEngine::is_swa_slot(int sl) const {
    const Hparams & h = hp_;
    const int ip = sl < h.n_layer ? sl : h.loop_start + (sl - h.n_layer) % h.n_core();
    return h.is_swa[ip];
}

void FastEngine::P(const char * label, sycl::event e) {
    if (profile) prof_events_.emplace_back(label, std::move(e));
}

void FastEngine::print_profile() const {
    double total = 0;
    for (const auto & [l, ms] : prof_ms_) total += ms;
    std::printf("kernel time per token (%d tokens): %.2f ms\n", prof_tokens_, total / std::max(1, prof_tokens_));
    std::vector<std::pair<std::string, double>> v = prof_ms_;
    std::sort(v.begin(), v.end(), [](auto & a, auto & b) { return a.second > b.second; });
    for (const auto & [l, ms] : v) std::printf("  %-10s %7.3f ms  %5.1f%%\n", l.c_str(), ms / std::max(1, prof_tokens_), 100.0 * ms / total);
}

}  // namespace ink
