// The prompt path: a chunk of up to TB tokens through all 48 layer visits at once. Weights are read once per
// B tokens (batched matrix-vector kernels), the experts as a grouped GEMM over the tokens routed to each
// expert, attention as a flash-attention kernel over blocks of QB query tokens, and the 4-tap convolutions
// across time within the chunk. Shares the caches (KV, convolution history) with the decode path, so decode
// continues where the prompt ends.
#include "fast.h"

#include "kernels.h"
#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <type_traits>

namespace ink {

namespace {

constexpr int B = 8;    // tokens per sub-group in the batched matrix-vector kernel
constexpr int QB = 16;  // query tokens per attention work-group
constexpr int KT = 32;  // keys per attention tile
constexpr int MAXR = 4 * QB;  // attention rows per work-group (ratio x QB)

template <uint32_t T> sycl::event k_bmm(sycl::queue & q, const BJobs J, int total_rows, int ntok) {
    const int tg = (ntok + B - 1) / B;
    const size_t nsg = (size_t) total_rows * tg;
    const size_t groups = (nsg + RPG - 1) / RPG;
    return q.parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const size_t gid = it.get_group(0) * RPG + sg.get_group_id()[0];
        if (gid >= nsg) return;
        const int lane = (int) sg.get_local_id()[0];
        const int row = (int) (gid / tg), t0 = (int) (gid % tg) * B;
        int jj = 0;
        while (jj + 1 < J.n && row >= J.j[jj + 1].row0) ++jj;
        const BJob & b = J.j[jj];
        const int r = row - b.row0;
        const int nb = sycl::min(B, ntok - t0);
        const float * xr[B];
        const float * x2r[B];
        for (int k = 0; k < B; ++k) {
            const size_t tt = (size_t) (t0 + sycl::min(k, nb - 1));
            xr[k] = b.x + tt * b.xs;
            x2r[k] = b.x2 + tt * b.xs - b.split;
        }
        float acc[B];
        for (int k = 0; k < B; ++k) acc[k] = 0.0f;
        const RowRef rr = row_of(b.w, (size_t) r);
        const q::Tabs tb;
        constexpr int V = step_vals<T>();
        const int split = b.split;
        for (int s = lane; s < b.cols / V; s += SG)
            step_decode<T>(rr, s, [&](int c, float v) {
                if (c < split)
                    for (int k = 0; k < B; ++k) acc[k] += v * xr[k][c];
                else
                    for (int k = 0; k < B; ++k) acc[k] += v * x2r[k][c];
            }, tb);
        for (int k = 0; k < B; ++k) {
            const float v = sycl::reduce_over_group(sg, acc[k], sycl::plus<float>());
            if (lane == k && k < nb) b.y[(size_t) (t0 + k) * b.ys + r] = v;
        }
    });
}


// Tiled matrix multiply on the XMX units: Y[t][r] = X[t] . W[r] for a group of jobs of one weight format.
// Work-group tile: XT tokens x XR rows; each sub-group owns one 8 x 16 output tile. The K dimension goes in
// steps of XK: the token tile and the decoded weight tile are staged in local memory as fp16, then the XMX
// units multiply with fp32 accumulation. Jobs start on XR-row boundaries (run_bmm pads), so a work-group
// tile never mixes inputs.
constexpr int XT = 32, XR = 64, XK = 128, XMT = 1;  // measured: XT=128 (XMT=4) was slower for the projections
namespace xmx = sycl::ext::oneapi::experimental::matrix;

template <uint32_t T> sycl::event k_xmm(sycl::queue & q, const BJobs J, int total_rows, int ntok) {
    const int ttiles = (ntok + XT - 1) / XT, rtiles = total_rows / XR;
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> sA(XT * XK, cgh);
        // the weight tile; after the k loop it holds the accumulators (as float)
        constexpr int SB = XK * XR > 2 * RPG * XMT * 128 ? XK * XR : 2 * RPG * XMT * 128;
        sycl::local_accessor<sycl::half, 1> sB(SB, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) rtiles * ttiles * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int g = (int) it.get_group(0), t = (int) it.get_local_id(0);
            const int sgid = (int) sg.get_group_id()[0];
            const int rt = g / ttiles, tt = g % ttiles;
            const int row_base = rt * XR, tok_base = tt * XT;
            int jj = 0;
            while (jj + 1 < J.n && row_base >= J.j[jj + 1].row0) ++jj;
            const BJob & b = J.j[jj];
            const int r_base = row_base - b.row0;
            const bool ragged = b.cols % XK != 0;
            const bool vec_ok = b.xs % 4 == 0 && b.split % 8 == 0 && b.cols % 8 == 0;
            const int sm = sgid / 4, sn = sgid % 4;  // tokens [sm*32, sm*32+32), rows [sn*16, sn*16+16)
            xmx::joint_matrix<sycl::sub_group, float, xmx::use::accumulator, 8, 16> acc[XMT];
            for (int m = 0; m < XMT; ++m) xmx::joint_matrix_fill(sg, acc[m], 0.0f);
            const q::Tabs tb;
            constexpr int V = step_vals<T>();
            auto pA = sA.template get_multi_ptr<sycl::access::decorated::no>();
            auto pB = sB.template get_multi_ptr<sycl::access::decorated::no>();
            for (int k0 = 0; k0 < b.cols; k0 += XK) {
                if (vec_ok) {  // activations, 8 consecutive values per item
                    for (int i = t; i < XT * XK / 8; i += WG) {
                        const int u = i / (XK / 8), c = k0 + (i % (XK / 8)) * 8, tok = tok_base + u;
                        sycl::vec<sycl::half, 8> v8(0.0f);
                        if (tok < ntok && c < b.cols) {
                            const float * xr = c < b.split ? b.x + (size_t) tok * b.xs + c : b.x2 + (size_t) tok * b.xs + c - b.split;
                            const sycl::float4 a0 = *(const sycl::float4 *) xr, a1 = *(const sycl::float4 *) (xr + 4);
                            for (int e = 0; e < 4; ++e) {
                                v8[e] = (sycl::half) a0[e];
                                v8[4 + e] = (sycl::half) a1[e];
                            }
                        }
                        *(sycl::vec<sycl::half, 8> *) &sA[u * XK + (i % (XK / 8)) * 8] = v8;
                    }
                } else {
                    for (int i = t; i < XT * XK; i += WG) {
                        const int u = i / XK, c = k0 + i % XK, tok = tok_base + u;
                        float v = 0.0f;
                        if (tok < ntok && c < b.cols) v = c < b.split ? b.x[(size_t) tok * b.xs + c] : b.x2[(size_t) tok * b.xs + c - b.split];
                        sA[i] = (sycl::half) v;
                    }
                }
                if (ragged) {  // a partial k-step leaves stale weights to clear (rows past the job are never stored)
                    for (int i = t; i < XK * XR; i += WG) sB[i] = (sycl::half) 0.0f;
                    sycl::group_barrier(it.get_group());
                }
                for (int item = t; item < XR * (XK / V); item += WG) {
                    const int n = item % XR, st = item / XR;  // consecutive threads: consecutive rows (conflict-free)
                    const int r = r_base + n;
                    if (r < b.rows && k0 + st * V < b.cols)
                        step_decode<T>(row_of(b.w, (size_t) r), k0 / V + st, [&](int c, float v) { sB[(c - k0) * XR + n] = (sycl::half) v; }, tb);
                }
                sycl::group_barrier(it.get_group());
                for (int kk = 0; kk < XK; kk += 16) {
                    xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::b, 16, 16, xmx::layout::row_major> mb;
                    xmx::joint_matrix_load(sg, mb, pB + kk * XR + sn * 16, XR);
                    for (int m = 0; m < XMT; ++m) {
                        xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::a, 8, 16, xmx::layout::row_major> ma;
                        xmx::joint_matrix_load(sg, ma, pA + (sm * XMT + m) * 8 * XK + kk, XK);
                        xmx::joint_matrix_mad(sg, acc[m], ma, mb, acc[m]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            float * sCg = reinterpret_cast<float *>(&sB[0]);
            auto pC = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sCg);
            for (int m = 0; m < XMT; ++m) xmx::joint_matrix_store(sg, acc[m], pC + (sgid * XMT + m) * 128, 16, xmx::layout::row_major);
            sycl::group_barrier(it.get_group());
            for (int i = t; i < XT * XR; i += WG) {
                const int u = i / XR, n = i % XR;
                const int tok = tok_base + u, r = r_base + n;
                const int sgo = (u / (8 * XMT)) * 4 + n / 16, m = (u / 8) % XMT;
                if (tok < ntok && r < b.rows) b.y[(size_t) tok * b.ys + r] = sCg[(sgo * XMT + m) * 128 + (u % 8) * 16 + n % 16];
            }
        });
    });
}


// Grouped XMX for the experts: a work-group = one tile of up to GT (token, expert) pairs routed to the same
// expert x GR rows of that expert's matrix. y[pair][row] = W_e[row] . x[pair / pdiv] * escale[e] * pw[pair].
#ifndef INK_GR
#define INK_GR 128
#endif
#ifndef INK_GT
#define INK_GT 64
#endif
constexpr int GT = INK_GT, GR = INK_GR;
constexpr int GNS = GR / 16;              // sub-groups across the rows
constexpr int GMT = GT / 8 / (RPG / GNS);  // 8-pair tiles per sub-group
#ifndef INK_GK
#define INK_GK 128
#endif
constexpr int GK = INK_GK;  // k-step of the expert kernel (>= the largest step_vals, 128)  // GT = 2 sub-group rows x GMT tiles x 8 pairs

// Epilogue of k_gxmm. GX_STORE: y[pair][row]. GX_SITU: the gate-up matrix (gate rows [0, nff), up rows
// [nff, 2 nff)); a work-group takes GR/2 gate rows and the matching up rows and writes act[pair][i] =
// SITU(gate, up) in f16. GX_ADD: y[pair / tdiv][row] += value (float atomics), the per-token sum of the experts.
enum GxMode { GX_STORE, GX_SITU, GX_ADD };
struct GxOut {
    float * y = nullptr;
    size_t ys = 0;
    sycl::half * act = nullptr;
    int nff = 0, tdiv = 1;
    float beta = 0.0f, lb = 0.0f;
};

template <uint32_t T, int MODE, typename XT>
sycl::event k_gxmm(sycl::queue & q, const WRef w, int cols, int rows_e, const XT * x, size_t xs, int pdiv, const int32_t * pairs,
                   const int32_t * tiles, int max_tiles, const float * escale, const float * pw, const GxOut o, int dbg = 0) {
    const int rtiles = rows_e / GR;
    const bool ragged = cols % GK != 0;
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<sycl::half, 1> sA(GT * GK, cgh);
        // the weight tile; after the k loop it holds the accumulators (as float)
        constexpr int SB = GK * GR > 2 * RPG * GMT * 128 ? GK * GR : 2 * RPG * GMT * 128;
        sycl::local_accessor<sycl::half, 1> sB(SB, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) max_tiles * rtiles * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int g = (int) it.get_group(0), t = (int) it.get_local_id(0);
            const int sgid = (int) sg.get_group_id()[0];
            const int tile = g / rtiles, rt = g % rtiles;
            if (tile >= tiles[0]) return;  // uniform over the work-group
            const int e = tiles[1 + 3 * tile], first = tiles[2 + 3 * tile], nb = tiles[3 + 3 * tile];
            // the matrix row behind local row n
            auto grow = [&](int n) { return MODE == GX_SITU ? (n < GR / 2 ? rt * (GR / 2) + n : o.nff + rt * (GR / 2) + n - GR / 2) : rt * GR + n; };
            const int sm = sgid / GNS, sn = sgid % GNS;  // pairs [sm*8*GMT, ...), rows [sn*16, sn*16+16)
            xmx::joint_matrix<sycl::sub_group, float, xmx::use::accumulator, 8, 16> acc[GMT];
            for (int m = 0; m < GMT; ++m) xmx::joint_matrix_fill(sg, acc[m], 0.0f);
            const q::Tabs tb;
            constexpr int V = step_vals<T>();
            auto pA = sA.template get_multi_ptr<sycl::access::decorated::no>();
            auto pB = sB.template get_multi_ptr<sycl::access::decorated::no>();
            for (int k0 = 0; k0 < cols; k0 += GK) {
                // activations: 8 consecutive values per item (cols is a multiple of 8)
                for (int i = t; i < GT * GK / 8; i += WG) {
                    const int u = i / (GK / 8), c = k0 + (i % (GK / 8)) * 8;
                    sycl::vec<sycl::half, 8> v8(0.0f);
                    if (u < nb && c < cols) {
                        const XT * xr = x + (size_t) (pairs[first + u] / pdiv) * xs + c;
                        if constexpr (std::is_same_v<XT, sycl::half>) {
                            v8 = *(const sycl::vec<sycl::half, 8> *) xr;
                        } else {
                            const sycl::float4 a0 = *(const sycl::float4 *) xr, a1 = *(const sycl::float4 *) (xr + 4);
                            for (int e = 0; e < 4; ++e) {
                                v8[e] = (sycl::half) a0[e];
                                v8[4 + e] = (sycl::half) a1[e];
                            }
                        }
                    }
                    *(sycl::vec<sycl::half, 8> *) &sA[u * GK + (i % (GK / 8)) * 8] = v8;
                }
                if (ragged) {  // only a partial last k-step leaves stale weights to clear
                    for (int i = t; i < GK * GR; i += WG) sB[i] = (sycl::half) 0.0f;
                    sycl::group_barrier(it.get_group());
                }
                if constexpr (T == q::IQ1_SR) {  // a 32-value sub-block per item
                    for (int item = t; item < GR * (GK / 32); item += WG) {
                        const int n = item % GR, st = item / GR;
                        if (k0 + st * 32 < cols && !(dbg & 1))
                            iq1s_part(row_of(w, (size_t) e * rows_e + grow(n)), k0 / 32 + st,
                                      [&](int c, float v) { sB[(c - k0) * GR + n] = (sycl::half) v; }, tb);
                    }
                } else {
                    for (int item = t; item < GR * (GK / V); item += WG) {
                        const int n = item % GR, st = item / GR;  // consecutive threads: consecutive rows (conflict-free)
                        if (k0 + st * V < cols && !(dbg & 1))
                            step_decode<T>(row_of(w, (size_t) e * rows_e + grow(n)), k0 / V + st,
                                           [&](int c, float v) { sB[(c - k0) * GR + n] = (sycl::half) v; }, tb);
                    }
                }
                sycl::group_barrier(it.get_group());
                for (int kk = 0; kk < GK && !(dbg & 2); kk += 16) {
                    xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::b, 16, 16, xmx::layout::row_major> mb;
                    xmx::joint_matrix_load(sg, mb, pB + kk * GR + sn * 16, GR);
                    for (int m = 0; m < GMT; ++m) {
                        xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::a, 8, 16, xmx::layout::row_major> ma;
                        xmx::joint_matrix_load(sg, ma, pA + (sm * GMT + m) * 8 * GK + kk, GK);
                        xmx::joint_matrix_mad(sg, acc[m], ma, mb, acc[m]);
                    }
                }
                sycl::group_barrier(it.get_group());
            }
            float * sCg = reinterpret_cast<float *>(&sB[0]);
            auto pC = sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(sCg);
            for (int m = 0; m < GMT; ++m) xmx::joint_matrix_store(sg, acc[m], pC + (sgid * GMT + m) * 128, 16, xmx::layout::row_major);
            sycl::group_barrier(it.get_group());
            const float es = escale ? escale[e] : 1.0f;
            auto cval = [&](int u, int n) {
                const int sgo = (u / (8 * GMT)) * GNS + n / 16, m = (u / 8) % GMT;
                return sCg[(sgo * GMT + m) * 128 + (u % 8) * 16 + n % 16];
            };
            if constexpr (MODE == GX_SITU) {
                for (int i = t; i < GT * (GR / 2); i += WG) {
                    const int u = i / (GR / 2), n = i % (GR / 2);
                    if (u >= nb) continue;
                    const int pr = pairs[first + u];
                    const float gg = cval(u, n) * es, uu = cval(u, n + GR / 2) * es;
                    const float a = o.beta * sycl::tanh(gg / o.beta) * sigmoidf(gg);
                    o.act[(size_t) pr * o.nff + rt * (GR / 2) + n] = (sycl::half) (a * (o.lb > 0.0f ? o.lb * sycl::tanh(uu / o.lb) : uu));
                }
            } else {
                for (int i = t; i < GT * GR; i += WG) {
                    const int u = i / GR, n = i % GR;
                    if (u >= nb) continue;
                    const int pr = pairs[first + u];
                    const float v = cval(u, n) * es * (pw ? pw[pr] : 1.0f);
                    if constexpr (MODE == GX_ADD) {
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(
                            o.y[(size_t) (pr / o.tdiv) * o.ys + rt * GR + n]).fetch_add(v);
                    } else {
                        o.y[(size_t) pr * o.ys + rt * GR + n] = v;
                    }
                }
            }
        });
    });
}

// Depth mix, batched: work-group (token, output row). All per-token arrays are token-major with stride D.
struct BMix {
    const sycl::half * st[12] = {};  // stored states (RMS-normalised, f16)
    int ns = 0;
    const float * cur = nullptr;
    const float * qw = nullptr;
    int nq = 1;
    const float * normw[4] = {};
    float * out[4] = {};
    float * outn[4] = {};
    sycl::half * push = nullptr;
    float eps = 1e-6f;
};

sycl::event k_bmix(sycl::queue & q, const BMix a, int ntok) {
    return q.parallel_for(sycl::nd_range<1>((size_t) ntok * a.nq * WG, WG), [=](sycl::nd_item<1> it) {
        const auto grp = it.get_group();
        const int g = (int) it.get_group(0), l = (int) it.get_local_id(0);
        const int j = g % a.nq;
        const size_t to = (size_t) (g / a.nq) * D;
        float c[PER];
        if (a.cur) {
            float ss = 0.0f;
            for (int k = 0; k < PER; ++k) {
                c[k] = a.cur[to + l + k * WG];
                ss += c[k] * c[k];
            }
            ss = sycl::reduce_over_group(grp, ss, sycl::plus<float>());
            const float cs = 1.0f / sycl::sqrt(ss / D + 1e-6f);
            for (int k = 0; k < PER; ++k) c[k] *= cs;
            if (a.push && j == 0)
                for (int k = 0; k < PER; ++k) a.push[to + l + k * WG] = (sycl::half) c[k];
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
            for (int s = 0; s < n; ++s) {
                float acc = 0.0f;
                for (int k = 0; k < PER; ++k) {
                    const int i = l + k * WG;
                    acc += (s < a.ns ? (float) a.st[s][to + i] : c[k]) * qr[i];
                }
                sc[s] = sycl::reduce_over_group(grp, acc, sycl::plus<float>()) * inv;
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
            for (int s = 0; s < n; ++s) acc += sc[s] * (s < a.ns ? (float) a.st[s][to + i] : c[k]);
            o[k] = acc / sum;
            ss += o[k] * o[k];
        }
        ss = sycl::reduce_over_group(grp, ss, sycl::plus<float>());
        if (a.outn[j]) {
            const float r = 1.0f / sycl::sqrt(ss / D + 1e-6f);
            for (int k = 0; k < PER; ++k) a.outn[j][to + l + k * WG] = o[k] * r;
        }
        if (a.out[j]) {
            const float r = a.normw[j] ? 1.0f / sycl::sqrt(ss / D + a.eps) : 1.0f;
            for (int k = 0; k < PER; ++k) {
                const int i = l + k * WG;
                a.out[j][to + i] = o[k] * r * (a.normw[j] ? a.normw[j][i] : 1.0f);
            }
        }
    });
}

// Weighted RMS norm of up to 3 token-major arrays.
struct BNorm {
    const float * x[3] = {}, * w[3] = {};
    float * y[3] = {};
    int n = 0;
    float eps = 1e-6f;
};

sycl::event k_bnorm(sycl::queue & q, const BNorm a, int ntok) {
    return q.parallel_for(sycl::nd_range<1>((size_t) a.n * ntok * WG, WG), [=](sycl::nd_item<1> it) {
        const int g = (int) it.get_group(0), l = (int) it.get_local_id(0);
        const int j = g % a.n;
        const size_t to = (size_t) (g / a.n) * D;
        float v[PER];
        float ss = 0.0f;
        for (int k = 0; k < PER; ++k) {
            v[k] = a.x[j][to + l + k * WG];
            ss += v[k] * v[k];
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.0f / sycl::sqrt(ss / D + a.eps);
        for (int k = 0; k < PER; ++k) a.y[j][to + l + k * WG] = v[k] * r * a.w[j][l + k * WG];
    });
}

// Loop control combine: out = cur - tanh(forget) * cur + tanh(gain) * msg, elementwise over token-major arrays.
struct BCtl {
    const float * cur[4] = {}, * msg[4] = {}, * gain[4] = {}, * forget[4] = {};
    float * out[4] = {};
    int n = 0;
};

sycl::event k_bctl(sycl::queue & q, const BCtl a, int ntok) {
    return q.parallel_for(sycl::range<1>((size_t) a.n * ntok * D), [=](sycl::id<1> id) {
        const size_t i = id[0] % ((size_t) ntok * D);
        const int c = (int) (id[0] / ((size_t) ntok * D));
        const float x = a.cur[c][i];
        a.out[c][i] = x - sycl::tanh(a.forget[c][i]) * x + sycl::tanh(a.gain[c][i]) * a.msg[c][i];
    });
}

// The value of channel c at token t for a 4-tap causal convolution: the chunk's raw inputs, or the slot's
// history (the 3 inputs before the chunk) for t < 0.
inline float conv_in(const float * raw, size_t rs, const float * hist, int c, int t) {
    return t >= 0 ? raw[(size_t) t * rs + c] : hist[c * 3 + 3 + t];
}

// History update after a chunk of n tokens: the last 3 inputs of each channel.
sycl::event k_bhist(sycl::queue & q, const float * raw, size_t rs, float * hist, int channels, int ntok) {
    return q.parallel_for(sycl::range<1>(channels), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        float nv[3];
        for (int j = 0; j < 3; ++j) nv[j] = conv_in(raw, rs, hist, c, ntok - 3 + j);
        for (int j = 0; j < 3; ++j) hist[c * 3 + j] = nv[j];
    });
}

// Post convolution + residual over a chunk: out[t][i] = (drop ? 0 : res[t][i]) + (y + conv) / sqrt(5).
sycl::event k_bconv_res(sycl::queue & q, const float * raw, const float * w, const float * hist, const float * res, float * out,
                        bool drop, int ntok) {
    return q.parallel_for(sycl::range<2>((size_t) ntok, D), [=](sycl::id<2> id) {
        const int t = (int) id[0], i = (int) id[1];
        float cv = 0.0f;
        for (int j = 0; j < 4; ++j) cv += w[j + 4 * i] * conv_in(raw, D, hist, i, t - 3 + j);
        const float y = raw[(size_t) t * D + i];
        out[(size_t) t * D + i] = (drop ? 0.0f : res[(size_t) t * D + i]) + (y + cv) * 0.44721359549995793f;
    });
}

}  // namespace

// ---------------------------------------------------------------------------------------------------

void FastEngine::run_bmm(const char * label, const std::vector<BJob> & js, int ntok) {
    std::vector<uint32_t> order;
    for (const BJob & j : js)
        if (std::find(order.begin(), order.end(), j.type) == order.end()) order.push_back(j.type);
    for (uint32_t t : order) {
        BJobs g;
        int rows = 0;
        const bool xm = use_xmx && ntok >= 8;
        for (BJob j : js) {
            if (j.type != t) continue;
            if (g.n == 8) throw std::runtime_error("run_bmm: too many jobs of one format");
            j.row0 = rows;
            rows += xm ? (j.rows + XR - 1) / XR * XR : j.rows;  // XMX tiles: every job starts on a tile boundary
            g.j[g.n++] = j;
        }
        dispatch_type(t, [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            P(label, xm ? k_xmm<T>(q_, g, rows, ntok) : k_bmm<T>(q_, g, rows, ntok));
        });
    }
}

static BJob bjob(const W & w, const float * x, size_t xs, float * y, size_t ys, const float * x2 = nullptr, int split = 1 << 30) {
    BJob j;
    j.w = wref(w);
    j.type = w.ktype;
    j.cols = (int) w.ne[0];
    j.rows = (int) w.ne[1];
    j.x = x;
    j.x2 = x2 ? x2 : x;
    j.split = split;
    j.xs = xs;
    j.y = y;
    j.ys = ys;
    return j;
}

void FastEngine::bvisit(const Visit & v, int n, int pos0) {
    const Hparams & h = hp_;
    const LayerW & L = m_.layers[v.ip];
    const bool swa = h.is_swa[v.ip];
    const bool loop = !v.bank.empty();
    const int nh = h.n_head, nkv = h.n_head_kv, kvw = nkv * HD, ratio = nh / nkv;
    const LoopW * lw = loop ? &m_.loops[v.ip - h.loop_start] : nullptr;
    const size_t TD = (size_t) TB_ * D;  // one [TB][D] array
    auto store = [&](int id) { return bstore_ + (size_t) id * TD; };
    const W * norms[3] = {swa ? &L.attn_norm : &L.q_inp_norm, swa ? &L.attn_norm : &L.k_inp_norm, swa ? &L.attn_norm : &L.v_inp_norm};

    if (v.restart >= 0) {
        BMix a;
        a.cur = bxa_;
        a.push = store(v.restart);
        P("b_restart", k_bmix(q_, a, n));
        P("b_restart", q_.memcpy(bxa_, bprelude_, (size_t) n * D * sizeof(float)));
    }

    // attention inputs
    {
        BMix a;
        a.qw = L.attn_res_query.ok() ? L.attn_res_query.f32() : nullptr;
        if (a.qw) {
            for (size_t s = 0; s < v.blocks.size(); ++s) a.st[s] = store(v.blocks[s]);
            a.ns = (int) v.blocks.size();
        }
        a.nq = a.qw ? (int) L.attn_res_query.ne[1] : 1;
        a.cur = bxa_;
        a.eps = h.eps;
        if (v.push >= 0) a.push = store(v.push);
        for (int j = 0; j < a.nq; ++j) {
            if (loop) {
                a.out[j] = bin_ + j * TD;
                a.outn[j] = binn_ + j * TD;
            } else {
                a.out[j] = bnin_ + j * TD;
                a.normw[j] = norms[j]->f32();
            }
        }
        P("b_mix", k_bmix(q_, a, n));
    }
    const float * nin[3];
    for (int j = 0; j < 3; ++j) nin[j] = bnin_ + (swa && !loop ? 0 : j) * TD;
    if (loop) {
        const float * inr[3], * inz[3];
        for (int j = 0; j < 3; ++j) {
            inr[j] = bin_ + (swa ? 0 : j) * TD;
            inz[j] = binn_ + (swa ? 0 : j) * TD;
        }
        BMix b;
        for (size_t s = 0; s < v.bank.size(); ++s) b.st[s] = store(v.bank[s]);
        b.ns = (int) v.bank.size();
        b.qw = lw->msg_query.f32();
        b.nq = 4;
        for (int j = 0; j < 4; ++j) {
            b.out[j] = bmsg_ + j * TD;
            b.outn[j] = bmsgn_ + j * TD;
        }
        P("b_mix", k_bmix(q_, b, n));
        std::vector<BJob> js;
        for (int c = 0; c < 3; ++c) js.push_back(bjob(lw->down[c], inz[c], D, bg_ + (size_t) c * TB_ * 64, 64, bmsgn_ + c * TD, D));
        run_bmm("b_ctl", js, n);
        js.clear();
        for (int c = 0; c < 3; ++c) {
            js.push_back(bjob(lw->gain[c], bg_ + (size_t) c * TB_ * 64, 64, bgain_ + c * TD, D));
            js.push_back(bjob(lw->forget[c], bg_ + (size_t) c * TB_ * 64, 64, bforget_ + c * TD, D));
        }
        run_bmm("b_ctl", js, n);
        BCtl c;
        c.n = 3;
        for (int k = 0; k < 3; ++k) {
            c.cur[k] = inr[k];
            c.msg[k] = bmsg_ + k * TD;
            c.gain[k] = bgain_ + k * TD;
            c.forget[k] = bforget_ + k * TD;
            c.out[k] = bctl_ + k * TD;
        }
        P("b_ctl", k_bctl(q_, c, n));
        BNorm nm;
        nm.n = 3;
        nm.eps = h.eps;
        for (int k = 0; k < 3; ++k) {
            nm.x[k] = bctl_ + k * TD;
            nm.w[k] = norms[k]->f32();
            nm.y[k] = bnin_ + k * TD;
        }
        P("b_norm", k_bnorm(q_, nm, n));
        for (int j = 0; j < 3; ++j) nin[j] = bnin_ + j * TD;
    }

    // projections: bproj_ is [TB][PJ] with the decode layout (Q 0, K 3072, V 4096, R 5120, sink 5504, gate 5528,
    // kdyn 8600, vdyn 8632)
    constexpr size_t PJ = 8704;
    float * Q = bproj_, * K = bproj_ + 3072, * V = bproj_ + 4096, * R = bproj_ + 5120, * sink = bproj_ + 5504,
          * gate = bproj_ + 5528, * kdyn = bproj_ + 8600, * vdyn = bproj_ + 8632;
    {
        std::vector<BJob> js = {bjob(L.wq, nin[0], D, Q, PJ), bjob(L.wk, nin[1], D, K, PJ), bjob(L.wv, nin[2], D, V, PJ),
                                bjob(L.rel, nin[0], D, R, PJ), bjob(L.sink_proj, nin[0], D, sink, PJ), bjob(L.gate, nin[0], D, gate, PJ)};
        if (!swa) {
            js.push_back(bjob(L.k_conv_proj, nin[1], D, kdyn, PJ));
            js.push_back(bjob(L.v_conv_proj, nin[2], D, vdyn, PJ));
        }
        run_bmm("b_proj", js, n);
    }

    // Q/K/V/R post-processing over the chunk; K and V convolve across time and go to the cache
    {
        const float * qclip = L.q_clip.f32(), * rclip = L.rel_clip.f32(), * kconv = L.k_conv.f32(), * vconv = L.v_conv.f32();
        const float * hist = hist_[v.slot];
        int8_t * kc = kc_[v.slot], * vc = vc_[v.slot];
        sycl::half * ks = ks_[v.slot], * vs = vs_[v.slot];
        const bool kv4 = kv4_;
        float * qf = bq_, * rf = br_;
        const float qk_clip = h.qk_clip, rel_clip = h.rel_clip, alpha = h.log_alpha, floor_ = h.log_floor;
        const int n_rel = h.n_rel, ring = swa ? kv_len_[v.slot] : 0;
        const int kvlen = kv_len_[v.slot];  // head-major cache: [head][kvlen][HD]
        const bool full = !swa;
        const float cnorm = 1.0f / std::sqrt((float) (4 * (full ? 2 : 1) + 1));
        const float * proj = bproj_;
        const int groups = nh + 2 * nkv + nh;
        P("b_qkv", q_.parallel_for(sycl::nd_range<1>((size_t) n * groups * HD, HD), [=](sycl::nd_item<1> it) {
            const auto grp = it.get_group();
            const int gg = (int) it.get_group(0), t = (int) it.get_local_id(0);
            const int tok = gg / groups, g = gg % groups;
            const int pos = pos0 + tok;
            const float * pr = proj + (size_t) tok * PJ;
            const float tau = full ? 1.0f + alpha * sycl::log(sycl::fmax(1.0f, (float) (pos + 1) / floor_)) : 1.0f;
            if (g < nh) {
                const int i = g * HD + t;
                float x = pr[i] * qclip[i];
                const float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                x *= sycl::clamp(qk_clip / (sycl::sqrt(ss) + 1e-6f), 0.0f, 1.0f);
                qf[(size_t) tok * nh * HD + i] = x * tau;
            } else if (g < nh + 2 * nkv) {
                const bool is_v = g >= nh + nkv;
                const int kh = g - nh - (is_v ? nkv : 0);
                const int c = kh * HD + t;
                const int ch = (is_v ? kvw : 0) + c;
                const float * w = is_v ? vconv : kconv;
                const int off = is_v ? 4096 : 3072;  // raw K or V within a token's projections
                float xx[4];
                for (int j = 0; j < 4; ++j) {
                    const int tt = tok - 3 + j;
                    xx[j] = tt >= 0 ? proj[(size_t) tt * PJ + off + c] : hist[ch * 3 + 3 + tt];
                }
                float o = 0.0f;
                for (int j = 0; j < 4; ++j) o += w[j + 4 * c] * xx[j];
                if (full) {
                    const float * dyn = pr + (is_v ? 8632 : 8600);
                    for (int lag = 0; lag < 4; ++lag) o += dyn[lag + 4 * kh] * xx[3 - lag];
                }
                float x = (xx[3] + o) * cnorm;
                const float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                if (is_v) x *= 1.0f / sycl::sqrt(ss / HD + 1e-6f);
                else x *= sycl::clamp(qk_clip / (sycl::sqrt(ss) + 1e-6f), 0.0f, 1.0f);
                const int idx = ring ? pos % ring : pos;
                kv_put(grp, it.get_sub_group(), t, x, kv4, is_v ? vc : kc, is_v ? vs : ks, ((size_t) kh * kvlen + idx) * HD + t);
            } else {
                const int rh = g - nh - 2 * nkv;
                const int i = rh * n_rel + t;
                float x = t < n_rel ? pr[5120 + i] * rclip[i] : 0.0f;
                const float ss = sycl::reduce_over_group(grp, x * x, sycl::plus<float>());
                x *= sycl::clamp(rel_clip / (sycl::sqrt(ss) + 1e-6f), 0.0f, 1.0f);
                if (t < n_rel) rf[(size_t) tok * nh * n_rel + i] = x * tau;
            }
        }));
        // convolution history of K and V: the chunk's last 3 raw inputs (K at channel 0, V at kvw)
        bhist("b_qkv", bproj_ + 3072, PJ, v.slot, 0, 2 * kvw, n);
    }

    // attention: flash attention over blocks of QB query tokens; the KV head's query heads share each K/V tile
    {
        const float * qf = bq_, * rf = br_, * pr = bproj_;
        const int8_t * kc = kc_[v.slot], * vc = vc_[v.slot];
        const sycl::half * ksc = ks_[v.slot], * vsc = vs_[v.slot];
        const bool kv4 = kv4_;
        const float * rb = L.rel_bias.f32();
        const int extent = (int) L.rel_bias.ne[1], n_rel = h.n_rel, n_swa = h.n_swa;
        const int ring = swa ? kv_len_[v.slot] : 0;
        const int kvlen = kv_len_[v.slot];  // head-major cache: [head][kvlen][HD]
        float * att = batt_;
        const float scale = 1.0f / std::sqrt((float) HD);
        const int nqb = (n + QB - 1) / QB;
        const int rows = ratio * QB;
        if (use_xmx && rows % 8 == 0) {
        // XMX version: S = Q K^T and O += P V on the matrix units, softmax and bias on the vector units. The
        // queries (as A fragments) and the output accumulators stay in registers for the whole key loop; the
        // output is rescaled in place with joint_matrix_apply, so shared memory holds only the K/V tile and the
        // scores (about 29 KB, four work-groups per Xe-core).
        const int mt = rows / 8;  // 8-row tiles
        constexpr int PVT = MAXR / 8 * (HD / 16) / RPG;  // output tiles per sub-group (at most)
        P("b_attn", q_.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<sycl::half, 1> sKV(2 * HD * KT > MAXR * HD ? 2 * HD * KT : MAXR * HD, cgh);  // Q staging, then [K^T | V]
            sycl::local_accessor<float, 1> sS(rows * KT, cgh);
            sycl::local_accessor<sycl::half, 1> sP(rows * KT, cgh);
            sycl::local_accessor<float, 1> sR(rows * 16, cgh);
            sycl::local_accessor<float, 1> sM(rows, cgh), sL(rows, cgh), sA(rows, cgh);
            cgh.parallel_for(sycl::nd_range<1>((size_t) nkv * nqb * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                namespace xi = sycl::ext::intel::experimental::matrix;
                const auto grp = it.get_group();
                const auto sg = it.get_sub_group();
                const int g = (int) it.get_group(0), t = (int) it.get_local_id(0);
                const int sgid = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];
                const int kvh = g / nqb, qb = g % nqb;
                const int tq0 = qb * QB, nu = sycl::min(QB, n - tq0);
                const int h0 = kvh * ratio;
                auto pKV = sKV.template get_multi_ptr<sycl::access::decorated::no>();
                auto pS = sS.template get_multi_ptr<sycl::access::decorated::no>();
                auto pP = sP.template get_multi_ptr<sycl::access::decorated::no>();
                const auto pKT = pKV, pV = pKV + HD * KT;
                // stage Q (scaled, f16), then this sub-group's Q tile into registers
                for (int i = t; i < rows * HD; i += WG) {
                    const int r = i / HD, d = i % HD, j = r / QB, u = r % QB;
                    sKV[i] = (sycl::half) (u < nu ? qf[(size_t) (tq0 + u) * nh * HD + (h0 + j) * HD + d] * scale : 0.0f);
                }
                for (int i = t; i < rows * n_rel; i += WG) {
                    const int r = i / n_rel, k = i % n_rel, j = r / QB, u = r % QB;
                    sR[r * 16 + k] = u < nu ? rf[(size_t) (tq0 + u) * nh * n_rel + (h0 + j) * n_rel + k] : 0.0f;
                }
                for (int r = t; r < rows; r += WG) {
                    sM[r] = -INFINITY;
                    sL[r] = 0.0f;
                }
                sycl::group_barrier(grp);
                const bool qk = sgid < mt * (KT / 16);
                const int qm = sgid / (KT / 16), qn = sgid % (KT / 16);
                xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::a, 8, 16, xmx::layout::row_major> qa[HD / 16];
                if (qk)
                    for (int kk = 0; kk < HD / 16; ++kk) xmx::joint_matrix_load(sg, qa[kk], pKV + qm * 8 * HD + kk * 16, HD);
                xmx::joint_matrix<sycl::sub_group, float, xmx::use::accumulator, 8, 16> acc[PVT];
                for (int i = 0; i < PVT; ++i) xmx::joint_matrix_fill(sg, acc[i], 0.0f);
                sycl::group_barrier(grp);
                const int pfirst = pos0 + tq0, plast = pos0 + tq0 + nu - 1;
                const int klo = swa ? sycl::max(0, pfirst - n_swa + 1) : 0;
                for (int kt = klo; kt <= plast; kt += KT) {
                    // K transposed: consecutive threads take consecutive keys (conflict-free local stores)
                    for (int i = t; i < KT * HD / 8; i += WG) {
                        const int j = i % KT, d8 = (i / KT) * 8;
                        const int k = kt + j;
                        sycl::vec<sycl::half, 8> kv(0.0f);
                        if (k <= plast) {
                            const size_t o = ((size_t) kvh * kvlen + (ring ? k % ring : k)) * HD + d8;
                            float k8[8];
                            kv_get<8>(kc, kv4, o, k8);
                            const float kd = (float) ksc[o / 32];
                            for (int e = 0; e < 8; ++e) kv[e] = (sycl::half) (k8[e] * kd);
                        }
                        for (int e = 0; e < 8; ++e) sKV[(d8 + e) * KT + j] = kv[e];
                    }
                    // V row-major: consecutive threads take consecutive 8-value pieces of a key
                    for (int i = t; i < KT * HD / 8; i += WG) {
                        const int j = i / (HD / 8), d8 = (i % (HD / 8)) * 8;
                        const int k = kt + j;
                        sycl::vec<sycl::half, 8> vv(0.0f);
                        if (k <= plast) {
                            const size_t o = ((size_t) kvh * kvlen + (ring ? k % ring : k)) * HD + d8;
                            float v8[8];
                            kv_get<8>(vc, kv4, o, v8);
                            const float vd = (float) vsc[o / 32];
                            for (int e = 0; e < 8; ++e) vv[e] = (sycl::half) (v8[e] * vd);
                        }
                        *(sycl::vec<sycl::half, 8> *) &sKV[HD * KT + j * HD + d8] = vv;
                    }
                    sycl::group_barrier(grp);
                    if (qk) {  // S tile (qm, qn): 8 rows x 16 keys
                        xmx::joint_matrix<sycl::sub_group, float, xmx::use::accumulator, 8, 16> sc;
                        xmx::joint_matrix_fill(sg, sc, 0.0f);
                        for (int kk = 0; kk < HD / 16; ++kk) {
                            xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::b, 16, 16, xmx::layout::row_major> mb;
                            xmx::joint_matrix_load(sg, mb, pKT + kk * 16 * KT + qn * 16, KT);
                            xmx::joint_matrix_mad(sg, sc, qa[kk], mb, sc);
                        }
                        xmx::joint_matrix_store(sg, sc, pS + qm * 8 * KT + qn * 16, KT, xmx::layout::row_major);
                    }
                    sycl::group_barrier(grp);
                    // per row (a sub-group, two keys per lane): bias, mask, online softmax
                    for (int r = sgid; r < rows; r += RPG) {
                        const int u = r % QB, p = pos0 + tq0 + u;
                        float sv[2];
                        for (int e2 = 0; e2 < 2; ++e2) {
                            const int j = 2 * lane + e2, k = kt + j;
                            float sx = -INFINITY;
                            if (u < nu && k <= p && (!swa || p - k < n_swa)) {
                                sx = sS[r * KT + j];
                                const int dist = p - k;
                                if (dist < extent)
                                    for (int e = 0; e < n_rel; ++e) sx += rb[e + n_rel * dist] * sR[r * 16 + e];
                            }
                            sv[e2] = sx;
                        }
                        const float mx = sycl::reduce_over_group(sg, sycl::fmax(sv[0], sv[1]), sycl::maximum<float>());
                        const float mo = sM[r];
                        const float mn = sycl::fmax(mo, mx);
                        float p0 = 0.0f, p1 = 0.0f, al = 1.0f;
                        if (mn != -INFINITY) {
                            p0 = sv[0] == -INFINITY ? 0.0f : sycl::exp(sv[0] - mn);
                            p1 = sv[1] == -INFINITY ? 0.0f : sycl::exp(sv[1] - mn);
                            al = mo == -INFINITY ? 0.0f : sycl::exp(mo - mn);
                        }
                        const float rs = sycl::reduce_over_group(sg, p0 + p1, sycl::plus<float>());
                        sP[r * KT + 2 * lane] = (sycl::half) p0;
                        sP[r * KT + 2 * lane + 1] = (sycl::half) p1;
                        if (lane == 0) {
                            sL[r] = sL[r] * al + rs;
                            sM[r] = mn;
                            sA[r] = al;
                        }
                    }
                    sycl::group_barrier(grp);
                    // O = O * alpha + P V, in registers: tiles tl = sgid + i * RPG of (rows / 8) x (HD / 16)
                    for (int i = 0; i < PVT; ++i) {
                        const int tl = sgid + i * RPG;
                        if (tl >= mt * (HD / 16)) break;
                        const int m = tl / (HD / 16), nn = tl % (HD / 16);
                        xi::joint_matrix_apply(sg, acc[i], [&](float & x, size_t row, size_t) { x *= sA[m * 8 + (int) row]; });
                        for (int kk = 0; kk < KT; kk += 16) {
                            xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::a, 8, 16, xmx::layout::row_major> ma;
                            xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::b, 16, 16, xmx::layout::row_major> mb;
                            xmx::joint_matrix_load(sg, ma, pP + m * 8 * KT + kk, KT);
                            xmx::joint_matrix_load(sg, mb, pV + kk * HD + nn * 16, HD);
                            xmx::joint_matrix_mad(sg, acc[i], ma, mb, acc[i]);
                        }
                    }
                    sycl::group_barrier(grp);
                }
                // finish: the sink joins the denominator, then the output gate, straight from the accumulators
                for (int r = t; r < rows; r += WG) {
                    const int j = r / QB, u = r % QB;
                    const float m = sM[r];
                    float den = INFINITY;
                    if (u < nu && m != -INFINITY) den = sL[r] + sycl::exp(pr[(size_t) (tq0 + u) * PJ + 5504 + h0 + j] - m);
                    sA[r] = 1.0f / den;
                }
                sycl::group_barrier(grp);
                for (int i = 0; i < PVT; ++i) {
                    const int tl = sgid + i * RPG;
                    if (tl >= mt * (HD / 16)) break;
                    const int m = tl / (HD / 16), nn = tl % (HD / 16);
                    const int j = (m * 8) / QB, u0 = (m * 8) % QB, head = h0 + j;
                    xi::joint_matrix_apply(sg, acc[i], [&](float & x, size_t row, size_t col) {
                        const int u = u0 + (int) row, d = nn * 16 + (int) col;
                        x = u < nu ? x * sA[m * 8 + (int) row] * sigmoidf(pr[(size_t) (tq0 + u) * PJ + 5528 + head * HD + d]) : 0.0f;
                    });
                    // rows past the chunk's last token land in unused rows of the TB-token buffer
                    auto dst = sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(
                        att + (size_t) (tq0 + u0) * nh * HD + head * HD + nn * 16);
                    xmx::joint_matrix_store(sg, acc[i], dst, (size_t) nh * HD, xmx::layout::row_major);
                }
            });
        }));
        } else
        P("b_attn", q_.submit([&](sycl::handler & cgh) {
            sycl::local_accessor<float, 1> sQ(MAXR * HD, cgh);
            sycl::local_accessor<float, 1> sR(MAXR * 16, cgh);
            sycl::local_accessor<float, 1> sK(KT * HD, cgh);
            sycl::local_accessor<float, 1> sV(KT * HD, cgh);
            sycl::local_accessor<float, 1> sS(MAXR * KT, cgh);
            sycl::local_accessor<float, 1> sM(MAXR, cgh), sL(MAXR, cgh), sA(MAXR, cgh);
            cgh.parallel_for(sycl::nd_range<1>((size_t) nkv * nqb * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto grp = it.get_group();
                const auto sg = it.get_sub_group();
                const int g = (int) it.get_group(0), t = (int) it.get_local_id(0);
                const int sgid = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];
                const int kvh = g / nqb, qb = g % nqb;
                const int tq0 = qb * QB, nu = sycl::min(QB, n - tq0);
                const int h0 = kvh * ratio;
                // stage the queries: row r = head j * QB + token u
                for (int i = t; i < rows * HD; i += WG) {
                    const int r = i / HD, d = i % HD, j = r / QB, u = r % QB;
                    sQ[i] = u < nu ? qf[(size_t) (tq0 + u) * nh * HD + (h0 + j) * HD + d] * scale : 0.0f;
                }
                for (int i = t; i < rows * n_rel; i += WG) {
                    const int r = i / n_rel, k = i % n_rel, j = r / QB, u = r % QB;
                    sR[r * 16 + k] = u < nu ? rf[(size_t) (tq0 + u) * nh * n_rel + (h0 + j) * n_rel + k] : 0.0f;
                }
                for (int r = t; r < rows; r += WG) {
                    sM[r] = -INFINITY;
                    sL[r] = 0.0f;
                }
                const int pfirst = pos0 + tq0, plast = pos0 + tq0 + nu - 1;
                const int klo = swa ? sycl::max(0, pfirst - n_swa + 1) : 0;
                const int d = t % HD, rg = t / HD;  // PV: this thread's dim and row parity
                float acc[MAXR / 2];
                for (int i = 0; i < MAXR / 2; ++i) acc[i] = 0.0f;
                sycl::group_barrier(grp);
                for (int kt = klo; kt <= plast; kt += KT) {
                    // K and V tiles to local memory as float
                    for (int i = t; i < KT * HD / 8; i += WG) {
                        const int j = i / (HD / 8), d8 = (i % (HD / 8)) * 8;
                        const int k = kt + j;
                        if (k <= plast) {
                            const int idx = ring ? k % ring : k;
                            const size_t o = ((size_t) kvh * kvlen + idx) * HD + d8;
                            float kv[8], vv[8];
                            kv_get<8>(kc, kv4, o, kv);
                            kv_get<8>(vc, kv4, o, vv);
                            const float kd = (float) ksc[o / 32], vd = (float) vsc[o / 32];
                            for (int e = 0; e < 8; ++e) {
                                sK[j * HD + d8 + e] = kv[e] * kd;
                                sV[j * HD + d8 + e] = vv[e] * vd;
                            }
                        } else {
                            for (int e = 0; e < 8; ++e) {
                                sK[j * HD + d8 + e] = 0.0f;
                                sV[j * HD + d8 + e] = 0.0f;
                            }
                        }
                    }
                    sycl::group_barrier(grp);
                    // scores with the relative bias and the causal / sliding-window mask
                    for (int i = t; i < rows * KT; i += WG) {
                        const int r = i / KT, j = i % KT, u = r % QB;
                        const int k = kt + j, p = pos0 + tq0 + u;
                        float s = -INFINITY;
                        if (u < nu && k <= p && (!swa || p - k < n_swa)) {
                            s = 0.0f;
                            for (int e = 0; e < HD; ++e) s += sQ[r * HD + e] * sK[j * HD + e];
                            const int dist = p - k;
                            if (dist < extent)
                                for (int e = 0; e < n_rel; ++e) s += rb[e + n_rel * dist] * sR[r * 16 + e];
                        }
                        sS[r * KT + j] = s;
                    }
                    sycl::group_barrier(grp);
                    // online softmax per row: a sub-group per row, two keys per lane
                    for (int r = sgid; r < rows; r += RPG) {
                        const float s0 = sS[r * KT + 2 * lane], s1 = sS[r * KT + 2 * lane + 1];
                        const float mx = sycl::reduce_over_group(sg, sycl::fmax(s0, s1), sycl::maximum<float>());
                        const float mo = sM[r];
                        const float mn = sycl::fmax(mo, mx);
                        float p0 = 0.0f, p1 = 0.0f, al = 1.0f;
                        if (mn != -INFINITY) {
                            p0 = s0 == -INFINITY ? 0.0f : sycl::exp(s0 - mn);
                            p1 = s1 == -INFINITY ? 0.0f : sycl::exp(s1 - mn);
                            al = mo == -INFINITY ? 0.0f : sycl::exp(mo - mn);
                        }
                        const float rs = sycl::reduce_over_group(sg, p0 + p1, sycl::plus<float>());
                        sS[r * KT + 2 * lane] = p0;
                        sS[r * KT + 2 * lane + 1] = p1;
                        if (lane == 0) {
                            sL[r] = sL[r] * al + rs;
                            sM[r] = mn;
                            sA[r] = al;
                        }
                    }
                    sycl::group_barrier(grp);
                    // acc[r] = acc[r] * alpha + P[r] . V
                    for (int i = 0; 2 * i + rg < rows; ++i) {
                        const int r = 2 * i + rg;
                        float a = acc[i] * sA[r];
                        for (int j = 0; j < KT; ++j) a += sS[r * KT + j] * sV[j * HD + d];
                        acc[i] = a;
                    }
                    sycl::group_barrier(grp);
                }
                // finish: the sink joins the denominator, then the output gate
                for (int i = 0; 2 * i + rg < rows; ++i) {
                    const int r = 2 * i + rg, j = r / QB, u = r % QB;
                    if (u >= nu) continue;
                    const int tok = tq0 + u, head = h0 + j;
                    const float * ptok = pr + (size_t) tok * PJ;
                    const float m = sM[r], sk = ptok[5504 + head];
                    const float o = m == -INFINITY ? 0.0f : acc[i] / (sL[r] + sycl::exp(sk - m));
                    att[(size_t) tok * nh * HD + head * HD + d] = o * sigmoidf(ptok[5528 + head * HD + d]);
                }
            });
        }));
    }

    // output projection, its convolution across time, residual
    run_bmm("b_wo", {bjob(L.wo, batt_, (size_t) nh * HD, bo_, D)}, n);
    P("b_conv", k_bconv_res(q_, bo_, L.post_conv.f32(), hist_[v.slot] + (size_t) 2 * kvw * 3, bxa_, bxb_, v.push >= 0, n));
    bhist("b_conv", bo_, D, v.slot, 2 * kvw, D, n);

    // FFN input
    {
        BMix a;
        for (size_t s = 0; s < v.blocks.size(); ++s) a.st[s] = store(v.blocks[s]);
        a.ns = (int) v.blocks.size();
        if (v.push >= 0) a.st[a.ns++] = store(v.push);
        a.cur = bxb_;
        a.qw = L.ffn_res_query.f32();
        a.nq = 1;
        a.eps = h.eps;
        if (loop) {
            a.out[0] = bfraw_;
            a.outn[0] = bfn_;
        } else {
            a.out[0] = bf_;
            a.normw[0] = L.ffn_norm.f32();
        }
        P("b_mix", k_bmix(q_, a, n));
        if (loop) {
            run_bmm("b_ctl", {bjob(lw->down[3], bfn_, D, bg_ + (size_t) 3 * TB_ * 64, 64, bmsgn_ + 3 * TD, D)}, n);
            run_bmm("b_ctl", {bjob(lw->gain[3], bg_ + (size_t) 3 * TB_ * 64, 64, bgain_, D),
                              bjob(lw->forget[3], bg_ + (size_t) 3 * TB_ * 64, 64, bforget_, D)}, n);
            BCtl c;
            c.n = 1;
            c.cur[0] = bfraw_;
            c.msg[0] = bmsg_ + 3 * TD;
            c.gain[0] = bgain_;
            c.forget[0] = bforget_;
            c.out[0] = bctl_;
            P("b_ctl", k_bctl(q_, c, n));
            BNorm nm;
            nm.n = 1;
            nm.eps = h.eps;
            nm.x[0] = bctl_;
            nm.w[0] = L.ffn_norm.f32();
            nm.y[0] = bf_;
            P("b_norm", k_bnorm(q_, nm, n));
        }
    }

    // router per token, then group the (token, expert) pairs by expert into tiles of at most B tokens
    run_bmm("b_router", {bjob(L.gate_inp, bf_, D, blogits_, 256)}, n);
    {
        const float * lg = blogits_, * pb = L.probs_b.ok() ? L.probs_b.f32() : nullptr;
        const int ne = h.n_expert, nu = h.n_used;
        const float ws = h.w_scale;
        int32_t * ids = bids_;
        float * wts = bwts_;
        P("b_router", q_.parallel_for(sycl::nd_range<1>((size_t) n * 256, 256), [=](sycl::nd_item<1> it) {
            const auto grp = it.get_group();
            const int tok = (int) it.get_group(0), e = (int) it.get_local_id(0);
            const float p = e < ne ? sigmoidf(lg[(size_t) tok * 256 + e]) : 0.0f;
            float sel = e < ne ? p + (pb ? pb[e] : 0.0f) : -INFINITY;
            int rank = -1;
            float sum = 0.0f;
            for (int k = 0; k < nu; ++k) {
                const float m = sycl::reduce_over_group(grp, sel, sycl::maximum<float>());
                const int idx = sycl::reduce_over_group(grp, sel == m ? e : 1 << 30, sycl::minimum<int>());
                if (e == idx) {
                    rank = k;
                    sel = -INFINITY;
                }
                sum += sycl::reduce_over_group(grp, e == idx ? p : 0.0f, sycl::plus<float>());
            }
            if (rank >= 0) {
                ids[tok * nu + rank] = e;
                wts[tok * nu + rank] = p / (sum + 1e-20f) * ws;
            }
        }));
        int32_t * pairs = bpairs_, * tiles = btiles_;
        const int tsz = use_xmx ? GT : B;
        P("b_group", q_.single_task([=]() {
            int cnt[256], off[257];
            for (int e = 0; e < ne; ++e) cnt[e] = 0;
            for (int i = 0; i < n * nu; ++i) ++cnt[ids[i]];
            off[0] = 0;
            for (int e = 0; e < ne; ++e) off[e + 1] = off[e] + cnt[e];
            for (int e = 0; e < ne; ++e) cnt[e] = 0;
            for (int i = 0; i < n * nu; ++i) {
                const int e = ids[i];
                pairs[off[e] + cnt[e]++] = i;  // pair index = token * nu + k
            }
            int nt = 0;  // tiles: (expert, first pair, count)
            for (int e = 0; e < ne; ++e)
                for (int s = off[e]; s < off[e + 1]; s += tsz) {
                    tiles[1 + 3 * nt + 0] = e;
                    tiles[1 + 3 * nt + 1] = s;
                    tiles[1 + 3 * nt + 2] = sycl::min(tsz, off[e + 1] - s);
                    ++nt;
                }
            tiles[0] = nt;
        }));
    }
    const int max_tiles = n * h.n_used / (use_xmx ? GT : B) + h.n_expert;
    static const int gx_dbg = getenv("INK_EXP_SKIP") ? atoi(getenv("INK_EXP_SKIP")) : 0;  // 1: skip decode, 2: skip XMX  // A/B switch for the transposed expert kernel
    if (use_xmx) {
        // gate-up on XMX into a scratch [pair][2 n_ff], then SITU; down on XMX weighted per pair; per-token sum
        const W wg = stage(L.gate_up, v.ip);  // offloaded: streamed to VRAM by the copy engine first
        const int nff = h.n_ff, nu = h.n_used;
        if ((int) wg.ne[1] != 2 * nff || nff % (GR / 2)) throw std::runtime_error("prefill: gate-up shape does not fit the fused kernel");
        GxOut og;
        og.act = bact_;
        og.nff = nff;
        og.beta = h.situ_beta;
        og.lb = h.situ_lb;
        dispatch_type(wg.ktype, [&](auto tag) {
            P("b_gate_up", k_gxmm<decltype(tag)::value, GX_SITU, float>(q_, wref(wg), (int) wg.ne[0], (int) wg.ne[1], bf_, D, nu, bpairs_, btiles_,
                                                                        max_tiles, L.gate_up_s.ok() ? L.gate_up_s.f32() : nullptr, nullptr, og, gx_dbg));
        });
        const W wd = stage(L.down, v.ip);
        GxOut od;
        od.y = bo_;
        od.ys = D;
        od.tdiv = nu;
        P("b_down", q_.memset(bo_, 0, (size_t) n * D * sizeof(float)));
        dispatch_type(wd.ktype, [&](auto tag) {
            P("b_down", k_gxmm<decltype(tag)::value, GX_ADD, sycl::half>(q_, wref(wd), (int) wd.ne[0], D, bact_, nff, 1, bpairs_, btiles_, max_tiles,
                                                                         L.down_s.ok() ? L.down_s.f32() : nullptr, bwts_, od, gx_dbg));
        });
        P("b_conv", k_bconv_res(q_, bo_, L.ffn_post_conv.f32(), hist_[v.slot] + (size_t) (2 * kvw + D) * 3, bxb_, bxa_, false, n));
        bhist("b_conv", bo_, D, v.slot, 2 * kvw + D, D, n);
        return;
    }

    // experts: gate-up with SITU for every pair, grouped by expert
    {
        const W w = stage(L.gate_up, v.ip);
        const WRef wr = wref(w);
        const int cols = (int) w.ne[0], rows_e = (int) w.ne[1], nff = h.n_ff, nu = h.n_used;
        const float * f = bf_, * gs = L.gate_up_s.ok() ? L.gate_up_s.f32() : nullptr;
        const int32_t * pairs = bpairs_, * tiles = btiles_;
        sycl::half * act = bact_;
        const float beta = h.situ_beta, lb = h.situ_lb;
        dispatch_type(w.ktype, [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            const size_t nsg = (size_t) max_tiles * nff;
            P("b_gate_up", q_.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto sg = it.get_sub_group();
                const size_t gid = it.get_group(0) * RPG + sg.get_group_id()[0];
                if (gid >= nsg) return;
                const int tile = (int) (gid / nff), i = (int) (gid % nff), lane = (int) sg.get_local_id()[0];
                if (tile >= tiles[0]) return;
                const int e = tiles[1 + 3 * tile], first = tiles[2 + 3 * tile], nb = tiles[3 + 3 * tile];
                const float * xr[B];
                int pr[B];
                for (int k = 0; k < B; ++k) {
                    pr[k] = pairs[first + sycl::min(k, nb - 1)];
                    xr[k] = f + (size_t) (pr[k] / nu) * D;
                }
                const size_t r0 = (size_t) e * rows_e;
                const RowRef rg = row_of(wr, r0 + i), ru = row_of(wr, r0 + nff + i);
                const q::Tabs tb;
                constexpr int V = step_vals<T>();
                float ag[B], au[B];
                for (int k = 0; k < B; ++k) ag[k] = au[k] = 0.0f;
                for (int s = lane; s < cols / V; s += SG) {
                    step_decode<T>(rg, s, [&](int c, float v) { for (int k = 0; k < B; ++k) ag[k] += v * xr[k][c]; }, tb);
                    step_decode<T>(ru, s, [&](int c, float v) { for (int k = 0; k < B; ++k) au[k] += v * xr[k][c]; }, tb);
                }
                const float sc = gs ? gs[e] : 1.0f;
                for (int k = 0; k < B; ++k) {
                    const float gg = sycl::reduce_over_group(sg, ag[k], sycl::plus<float>()) * sc;
                    const float uu = sycl::reduce_over_group(sg, au[k], sycl::plus<float>()) * sc;
                    if (lane == k && k < nb) {
                        const float a = beta * sycl::tanh(gg / beta) * sigmoidf(gg);
                        act[(size_t) pr[k] * nff + i] = (sycl::half) (a * (lb > 0.0f ? lb * sycl::tanh(uu / lb) : uu));
                    }
                }
            }));
        });
    }

    // experts: down for every pair (weighted), then the per-token sum
    {
        const W w = stage(L.down, v.ip);
        const WRef wr = wref(w);
        const int cols = (int) w.ne[0], nu = h.n_used;
        const sycl::half * act = bact_;
        const float * ds = L.down_s.ok() ? L.down_s.f32() : nullptr, * wts = bwts_;
        const int32_t * pairs = bpairs_, * tiles = btiles_;
        float * by = bo_;
        P("b_down", q_.memset(bo_, 0, (size_t) n * D * sizeof(float)));
        dispatch_type(w.ktype, [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            const size_t nsg = (size_t) max_tiles * D;
            P("b_down", q_.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
                const auto sg = it.get_sub_group();
                const size_t gid = it.get_group(0) * RPG + sg.get_group_id()[0];
                if (gid >= nsg) return;
                const int tile = (int) (gid / D), i = (int) (gid % D), lane = (int) sg.get_local_id()[0];
                if (tile >= tiles[0]) return;
                const int e = tiles[1 + 3 * tile], first = tiles[2 + 3 * tile], nb = tiles[3 + 3 * tile];
                const sycl::half * xr[B];
                int pr[B];
                for (int k = 0; k < B; ++k) {
                    pr[k] = pairs[first + sycl::min(k, nb - 1)];
                    xr[k] = act + (size_t) pr[k] * cols;
                }
                const RowRef rr = row_of(wr, (size_t) e * D + i);
                const q::Tabs tb;
                constexpr int V = step_vals<T>();
                float a[B];
                for (int k = 0; k < B; ++k) a[k] = 0.0f;
                for (int s = lane; s < cols / V; s += SG)
                    step_decode<T>(rr, s, [&](int c, float v) { for (int k = 0; k < B; ++k) a[k] += v * (float) xr[k][c]; }, tb);
                const float sc = ds ? ds[e] : 1.0f;
                for (int k = 0; k < B; ++k) {
                    const float y = sycl::reduce_over_group(sg, a[k], sycl::plus<float>());
                    if (lane == k && k < nb)
                        sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(
                            by[(size_t) (pr[k] / nu) * D + i]).fetch_add(y * wts[pr[k]] * sc);
                }
            }));
        });
    }
    P("b_conv", k_bconv_res(q_, bo_, L.ffn_post_conv.f32(), hist_[v.slot] + (size_t) (2 * kvw + D) * 3, bxb_, bxa_, false, n));
    bhist("b_conv", bo_, D, v.slot, 2 * kvw + D, D, n);
}

// Prompt path, offloaded layer: copy the whole expert tensor into VRAM with the copy engine (14 GB/s on a
// B580 against ~8.6 GB/s for kernels reading pinned RAM directly), into the hot-expert cache's memory, which
// is refilled after the prompt. Falls back to reading RAM when the tensor does not fit.
W FastEngine::stage(const W & w, int ip) {
    if (ip >= (int) m_.offloaded.size() || !m_.offloaded[ip] || !stage_pool_) return w;
    const size_t rows = (size_t) w.ne[1] * w.ne[2];
    const size_t b0 = rows * w.row_bytes, b1 = w.aux ? rows * w.aux_row_bytes : 0, b2 = w.aux2 ? rows * w.aux2_row_bytes : 0;
    if (b0 + b1 + b2 > stage_bytes_) return w;
    W s = w;
    uint8_t * d = stage_pool_;
    P("b_stage", q_.memcpy(d, w.p, b0));
    s.p = d;
    if (b1) {
        P("b_stage", q_.memcpy(d + b0, w.aux, b1));
        s.aux = d + b0;
    }
    if (b2) {
        P("b_stage", q_.memcpy(d + b0 + b1, w.aux2, b2));
        s.aux2 = d + b0 + b1;
    }
    staged_ = true;
    return s;
}

void FastEngine::embed(int n) {
    const Hparams & h = hp_;
    // embedding rows, RMS norm with token_embd_norm
    const W & e = m_.tok_embd;
    const uint8_t * ep = e.p;
    const size_t rb = e.row_bytes;
    const uint32_t type = e.type;
    const q::BlockShape bs = q::shape(type);
    const int32_t * tok = btok_;
    const float * nw = m_.tok_norm.f32();
    float * x = bxa_;
    const float eps = h.eps;
    P("b_embed", q_.parallel_for(sycl::nd_range<1>((size_t) n * WG, WG), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_local_id(0), tk = (int) it.get_group(0);
        const uint8_t * row = ep + (size_t) tok[tk] * rb;
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
            for (int j = 0; j < 32; ++j) x[(size_t) tk * D + t * 32 + j] = v[j] * r * nw[t * 32 + j];
    }));
}

void FastEngine::bhist(const char * label, const float * raw, size_t rs, int slot, int off, int channels, int n) {
    if (!verify_) {
        P(label, k_bhist(q_, raw, rs, hist_[slot] + (size_t) off * 3, channels, n));
        return;
    }
    const int HC = hp_.conv_channels();
    float * dst = vraw_ + (size_t) slot * VMAX * HC + off;
    P(label, q_.parallel_for(sycl::range<2>((size_t) n, (size_t) channels), [=](sycl::item<2> it) {
        const size_t t = it[0], c = it[1];
        dst[t * HC + c] = raw[t * rs + c];
    }));
}

const std::vector<float> & FastEngine::verify(const std::vector<int> & tokens, int pos0) {
    const Hparams & h = hp_;
    const int n = (int) tokens.size();
    if (n < 1 || n > VMAX || n > TB_) throw std::runtime_error("verify: 1 to 16 tokens");
    if (pos0 + n > n_ctx_) throw std::runtime_error("verify: context is full");
    if (!vraw_) {
        vraw_ = dev<float>((size_t) h.n_slot() * VMAX * h.conv_channels());
        vlogits_ = dev<float>((size_t) VMAX * h.n_vocab);
        vhist_ = sycl::malloc_device<float *>((size_t) h.n_slot(), q_);
        allocs_.push_back(vhist_);
        std::vector<float *> hp(hist_.begin(), hist_.end());
        q_.memcpy(vhist_, hp.data(), hp.size() * sizeof(float *)).wait();
    }
    verify_ = true;
    vn_ = n;
    try {
        q_.memcpy(btok_, tokens.data(), n * sizeof(int32_t));
        embed(n);
        const auto & vs = m_.schedule.visits;
        for (size_t i = 0; i < vs.size(); ++i) {
            bvisit(vs[i], n, pos0);
            if ((int) i + 1 == m_.schedule.prelude_end) P("b_prelude", q_.memcpy(bprelude_, bxa_, (size_t) n * D * sizeof(float)));
        }
        // logits after every token: output mix, norm, LM head, row by row (as prefill does for its last token)
        const size_t TD = (size_t) TB_ * D;
        for (int r = 0; r < n; ++r) {
            const size_t off = (size_t) r * D;
            BMix a;
            for (size_t s = 0; s < m_.schedule.out_blocks.size(); ++s) a.st[s] = bstore_ + (size_t) m_.schedule.out_blocks[s] * TD + off;
            a.ns = (int) m_.schedule.out_blocks.size();
            a.cur = bxa_ + off;
            a.qw = m_.output_res_query.f32();
            a.nq = 1;
            a.eps = h.eps;
            a.out[0] = bf_ + off;
            a.normw[0] = m_.output_norm.f32();
            P("b_out", k_bmix(q_, a, 1));
        }
        for (int r = 0; r < n; ++r) run_bmm("b_head", {bjob(m_.output, bf_ + (size_t) r * D, D, vlogits_ + (size_t) r * h.n_vocab, 0)}, 1);
    } catch (...) {
        verify_ = false;
        throw;
    }
    verify_ = false;
    reload_cache();
    vlogits_host_.resize((size_t) n * h.n_vocab);
    q_.memcpy(vlogits_host_.data(), vlogits_, vlogits_host_.size() * sizeof(float)).wait();
    if (h.softcap > 0.0f)
        for (float & v : vlogits_host_) v = h.softcap * std::tanh(v / h.softcap);
    return vlogits_host_;
}

void FastEngine::commit(int keep) {
    const Hparams & h = hp_;
    if (keep < 0 || keep > vn_) throw std::runtime_error("commit: keep out of range");
    if (keep == 0) return;
    // every slot's history becomes the last 3 inputs of (old history ++ the kept tokens' inputs)
    const int HC = h.conv_channels(), ns = h.n_slot();
    float ** hdev = vhist_;
    const float * vr = vraw_;
    q_.parallel_for(sycl::range<2>((size_t) ns, (size_t) HC), [=](sycl::item<2> it) {
        const int s = (int) it[0], c = (int) it[1];
        float * hist = hdev[s] + (size_t) c * 3;
        const float * raw = vr + (size_t) s * VMAX * HC + c;
        float nv[3];
        for (int j = 0; j < 3; ++j) {
            const int t = keep - 3 + j;
            nv[j] = t >= 0 ? raw[(size_t) t * HC] : hist[3 + t];
        }
        for (int j = 0; j < 3; ++j) hist[j] = nv[j];
    }).wait();
    vn_ = 0;
}

const std::vector<float> & FastEngine::prefill(const std::vector<int> & tokens, int pos0) {
    if (tokens.empty()) throw std::runtime_error("prefill: no tokens");
    if (pos0 + (int) tokens.size() > n_ctx_) throw std::runtime_error("prefill: context is full");
    const Hparams & h = hp_;
    for (size_t c0 = 0; c0 < tokens.size(); c0 += TB_) {
        const int n = (int) std::min<size_t>(TB_, tokens.size() - c0);
        const int p0 = pos0 + (int) c0;
        q_.memcpy(btok_, tokens.data() + c0, n * sizeof(int32_t));
        embed(n);
        const auto & vs = m_.schedule.visits;
        for (size_t i = 0; i < vs.size(); ++i) {
            bvisit(vs[i], n, p0);
            if ((int) i + 1 == m_.schedule.prelude_end) P("b_prelude", q_.memcpy(bprelude_, bxa_, (size_t) n * D * sizeof(float)));
            if (bhook) {
                q_.wait();
                bhook("l_out-" + std::to_string(vs[i].slot), bxa_, n);
            }
        }
        // logits for the chunk's last token only: the output mix over its blocks, norm, LM head
        if (c0 + n == tokens.size()) {
            const size_t TD = (size_t) TB_ * D, last = (size_t) (n - 1) * D;
            BMix a;
            for (size_t s = 0; s < m_.schedule.out_blocks.size(); ++s) a.st[s] = bstore_ + (size_t) m_.schedule.out_blocks[s] * TD + last;
            a.ns = (int) m_.schedule.out_blocks.size();
            a.cur = bxa_ + last;
            a.qw = m_.output_res_query.f32();
            a.nq = 1;
            a.eps = h.eps;
            a.out[0] = f_;
            a.normw[0] = m_.output_norm.f32();
            P("b_out", k_bmix(q_, a, 1));
            const W & o = m_.output;
            BJob j = bjob(o, f_, D, logits_dev_, 0);
            run_bmm("b_head", {j}, 1);
        }
    }
    logits_.resize(h.n_vocab);
    q_.memcpy(logits_.data(), logits_dev_, h.n_vocab * sizeof(float)).wait();
    if (h.softcap > 0.0f)
        for (float & v : logits_) v = h.softcap * std::tanh(v / h.softcap);
    reload_cache();
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

}  // namespace ink
