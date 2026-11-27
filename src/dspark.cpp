// Native DSpark speculative drafter (3-layer interleaved SWA/Full block-diffusion backbone +
// context KV injection over target taps [5, 23, 41, 48] + serial rank-256 Markov RNN head).
#include "dspark.h"

#include "device.h"
#include "kernels.h"
#include "quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace sycl_exp = sycl::ext::oneapi::experimental;

namespace ink {

namespace {

inline float siluf(float x) { return x / (1.0f + sycl::exp(-x)); }

inline uint16_t f32_to_f16_host(float v) {
    sycl::half h(v);
    uint16_t u = 0;
    std::memcpy(&u, &h, sizeof(u));
    return u;
}

inline float bf16_to_f32_host(uint16_t b) {
    uint32_t u = (uint32_t) b << 16;
    float f = 0.0f;
    std::memcpy(&f, &u, sizeof(f));
    return f;
}

W load_dspark_tensor(sycl::queue & q, const GgufFile & f, const std::string & name,
                     uint64_t & total, std::vector<void *> & allocs, bool repack_q8) {
    const GgufTensor * t = f.tensor(name);
    if (!t) throw std::runtime_error("dspark model missing tensor: " + name);
    W w;
    w.type = w.ktype = t->type;
    for (int i = 0; i < 4; ++i) w.ne[i] = t->ne[i];
    const q::BlockShape bs = q::shape(t->type);
    w.row_bytes = (size_t) (t->ne[0] / bs.elems) * bs.bytes;
    const int64_t cols = t->ne[0], rows = t->n_elements() / cols;

    if (repack_q8 && t->type == q::Q8_0 && cols % 32 == 0) {
        uint8_t * dev = sycl::malloc_device<uint8_t>(t->n_bytes, q);
        if (!dev) throw std::runtime_error("device allocation failed for " + name);
        allocs.push_back(dev);
        total += t->n_bytes;
        std::vector<uint8_t> host(t->n_bytes);
        const int64_t nb = cols / 32;
        uint8_t * vals = host.data();
        uint8_t * scl = host.data() + (size_t) rows * cols;
        for (int64_t r = 0; r < rows; ++r) {
            const uint8_t * src = t->data + (size_t) r * w.row_bytes;
            for (int64_t b = 0; b < nb; ++b) {
                std::memcpy(scl + ((size_t) r * nb + b) * 2, src + b * 34, 2);
                std::memcpy(vals + (size_t) r * cols + b * 32, src + b * 34 + 2, 32);
            }
        }
        q.memcpy(dev, host.data(), t->n_bytes).wait();
        w.ktype = q::Q8_0R;
        w.p = dev;
        w.row_bytes = (size_t) cols;
        w.aux = dev + (size_t) rows * cols;
        w.aux_row_bytes = (size_t) nb * 2;
        return w;
    }

    uint8_t * dev = sycl::malloc_device<uint8_t>(t->n_bytes, q);
    if (!dev) throw std::runtime_error("device allocation failed for " + name);
    allocs.push_back(dev);
    total += t->n_bytes;
    q.memcpy(dev, t->data, t->n_bytes).wait();
    w.p = dev;
    return w;
}

W load_w2_quantized_q8r(sycl::queue & q, const GgufFile & f, const std::string & name,
                        uint64_t & total, std::vector<void *> & allocs) {
    const GgufTensor * t = f.tensor(name);
    if (!t) throw std::runtime_error("dspark model missing tensor: " + name);
    if (t->type != q::BF16 || t->ne[0] % 32 != 0)
        return load_dspark_tensor(q, f, name, total, allocs, false);

    const int64_t cols = t->ne[0], rows = t->n_elements() / cols, nb = cols / 32;
    const size_t vbytes = (size_t) rows * cols;
    const size_t sbytes = (size_t) rows * nb * 2;
    uint8_t * dev = sycl::malloc_device<uint8_t>(vbytes + sbytes, q);
    if (!dev) throw std::runtime_error("device allocation failed for " + name);
    allocs.push_back(dev);
    total += vbytes + sbytes;

    std::vector<uint8_t> host(vbytes + sbytes);
    int8_t * vals = (int8_t *) host.data();
    uint16_t * scl = (uint16_t *) (host.data() + vbytes);
    const uint16_t * src = (const uint16_t *) t->data;
    for (int64_t r = 0; r < rows; ++r) {
        const uint16_t * srow = src + (size_t) r * cols;
        int8_t * vrow = vals + (size_t) r * cols;
        uint16_t * scrow = scl + (size_t) r * nb;
        for (int64_t b = 0; b < nb; ++b) {
            float fv[32];
            float amax = 0.0f;
            for (int j = 0; j < 32; ++j) {
                fv[j] = bf16_to_f32_host(srow[b * 32 + j]);
                amax = std::max(amax, std::fabs(fv[j]));
            }
            const float d = amax / 127.0f;
            const float id = d > 0.0f ? 1.0f / d : 0.0f;
            scrow[b] = f32_to_f16_host(d);
            for (int j = 0; j < 32; ++j) {
                int qi = (int) std::lrint(fv[j] * id);
                vrow[b * 32 + j] = (int8_t) std::clamp(qi, -127, 127);
            }
        }
    }
    q.memcpy(dev, host.data(), host.size()).wait();

    W w;
    w.type = t->type;
    w.ktype = q::Q8_0R;
    for (int i = 0; i < 4; ++i) w.ne[i] = t->ne[i];
    w.p = dev;
    w.row_bytes = (size_t) cols;
    w.aux = dev + vbytes;
    w.aux_row_bytes = (size_t) nb * 2;
    return w;
}

// ---------------------------------------------------------------------------------------------------
// GPU kernels

// Zero initialize scalar + RNN state before Mode C serial steps.
void k_dspark_rnn_init(sycl::queue & q, const int32_t * last_tok, const int32_t * pos0p, const float * f_rnn_ring,
                       float * rnn_state, float * f_rnn_cur, int32_t * prev_tok, int rank, int ring_len) {
    q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        const int pos0 = *pos0p;
        const int p_prev = pos0 > 0 ? (pos0 - 1) : 0;
        f_rnn_cur[i] = f_rnn_ring[(size_t) (p_prev % ring_len) * D + i];
        if (i < rank) rnn_state[i] = 0.0f;
        if (i == 0) *prev_tok = *last_tok;
    });
}

// Mode A step 1: compute weighted sum across 4 target taps + RMSNorm + optional KV norm weight.
// feats is tap-major [4][tap_cap][D]; output c[tok_off..][D] or ring[(pos0+t)%ring][D].
void k_dspark_ctx_mix(sycl::queue & q, const float * feats, int tap_cap, int tok_off, int n,
                      const int32_t * pos0p, const float * mix_w, const float * norm_w,
                      float * out, int out_ring, float eps) {
    q.parallel_for(sycl::nd_range<1>((size_t) n * WG, WG), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_group(0), l = (int) it.get_local_id(0);
        const float * t0 = feats + ((size_t) 0 * tap_cap + tok_off + t) * D;
        const float * t1 = feats + ((size_t) 1 * tap_cap + tok_off + t) * D;
        const float * t2 = feats + ((size_t) 2 * tap_cap + tok_off + t) * D;
        const float * t3 = feats + ((size_t) 3 * tap_cap + tok_off + t) * D;
        const float mw0 = mix_w[0], mw1 = mix_w[1], mw2 = mix_w[2], mw3 = mix_w[3];
        float v[PER];
        float ss = 0.0f;
        for (int k = 0; k < PER; ++k) {
            const int i = l + k * WG;
            v[k] = mw0 * t0[i] + mw1 * t1[i] + mw2 * t2[i] + mw3 * t3[i];
            ss += v[k] * v[k];
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.0f / sycl::sqrt(ss / D + eps);
        const int dst_row = out_ring > 0 ? ((*pos0p + tok_off + t) % out_ring) : t;
        float * dst = out + (size_t) dst_row * D;
        for (int k = 0; k < PER; ++k) {
            const int i = l + k * WG;
            dst[i] = v[k] * r * norm_w[i];
        }
    });
}

// Mode B anchor + mask initialization: X[0] = RMSNorm(tok_embd[*last_tok]) * tok_norm; X[1..M-1] = mask_embd[0..K-1].
void k_dspark_embed(sycl::queue & q, WRef embd, uint32_t embd_type, const int32_t * last_tokp, const float * tok_norm,
                    const float * mask_embd, const int32_t * mp, float * x, float eps) {
    q.parallel_for(sycl::nd_range<1>((size_t) 8 * WG, WG), [=](sycl::nd_item<1> it) {
        const int m = (int) it.get_group(0), l = (int) it.get_local_id(0);
        if (m >= *mp) return;
        if (m == 0) {
            const uint8_t * row = embd.p + (size_t) (*last_tokp) * embd.rb;
            const q::BlockShape bs = q::shape(embd_type);
            float v[32];
            float ss = 0.0f;
            const bool mine = l < D / 32;
            if (mine) {
                const int c0 = l * 32;
                q::decode32(embd_type, row + (size_t) (c0 / bs.elems) * bs.bytes, (c0 % bs.elems) / 32, v);
                for (int j = 0; j < 32; ++j) ss += v[j] * v[j];
            }
            ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
            const float r = 1.0f / sycl::sqrt(ss / D + eps);
            if (mine)
                for (int j = 0; j < 32; ++j) x[l * 32 + j] = v[j] * r * tok_norm[l * 32 + j];
        } else {
            const float * src = mask_embd + (size_t) (m - 1) * D;
            float * dst = x + (size_t) m * D;
            for (int k = 0; k < PER; ++k) {
                const int i = l + k * WG;
                dst[i] = src[i];
            }
        }
    });
}

// Elementwise residual add or final output RMSNorm.
template <int M_MAX>
void k_dspark_rmsnorm(sycl::queue & q, const float * x, const float * w, float * y, const int32_t * mp,
                      int x_off_tok, int y_rows, float eps) {
    q.parallel_for(sycl::nd_range<1>((size_t) y_rows * WG, WG), [=](sycl::nd_item<1> it) {
        const int m = (int) it.get_group(0), l = (int) it.get_local_id(0);
        if (m + x_off_tok >= *mp) return;
        const float * sx = x + (size_t) (m + x_off_tok) * D;
        float * sy = y + (size_t) m * D;
        float v[PER];
        float ss = 0.0f;
        for (int k = 0; k < PER; ++k) {
            v[k] = sx[l + k * WG];
            ss += v[k] * v[k];
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.0f / sycl::sqrt(ss / D + eps);
        for (int k = 0; k < PER; ++k) {
            const int i = l + k * WG;
            sy[i] = v[k] * r * w[i];
        }
    });
}

void k_dspark_add(sycl::queue & q, const float * a, float * x, const int32_t * mp, int n_elems_per_tok) {
    q.parallel_for(sycl::range<2>(8, (size_t) n_elems_per_tok), [=](sycl::id<2> id) {
        const int m = (int) id[0], i = (int) id[1];
        if (m >= *mp) return;
        const size_t o = (size_t) m * n_elems_per_tok + i;
        x[o] += a[o];
    });
}

// Symmetric int8 quantization in blocks of 32 for Q8_0R DP4A (supports up to max_tok rows).
void k_dspark_quant_q8(sycl::queue & q, const float * x, size_t x_stride, int cols, int max_tok,
                       const int32_t * ntokp, int8_t * xq, float * xd) {
    const int nblk = cols / 32;
    const size_t nsg = (size_t) max_tok * nblk;
    q.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const size_t gid = it.get_group(0) * RPG + sg.get_group_id()[0];
        if (gid >= nsg) return;
        const int tok = (int) (gid / nblk), b = (int) (gid % nblk), lane = (int) sg.get_local_id()[0];
        if (ntokp && tok >= *ntokp) return;
        const float * row = x + (size_t) tok * x_stride + (size_t) b * 32;
        const float v0 = row[2 * lane], v1 = row[2 * lane + 1];
        const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        int8_t * dst = xq + ((size_t) tok * cols + (size_t) b * 32);
        dst[2 * lane] = (int8_t) sycl::rint(v0 * id);
        dst[2 * lane + 1] = (int8_t) sycl::rint(v1 * id);
        if (lane == 0) xd[(size_t) tok * nblk + b] = d;
    });
}

// Quantize hidden[0..K-1][D] for target LM head (also stores xsum for Q4_KR / Q5_KR / Q2_0R).
void k_dspark_quant_lm(sycl::queue & q, const float * x, int max_tok, const int32_t * ntokp,
                       int8_t * xq, float * xd, float * xs) {
    constexpr int nblk = D / 32;
    const size_t nsg = (size_t) max_tok * nblk;
    q.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const size_t gid = it.get_group(0) * RPG + sg.get_group_id()[0];
        if (gid >= nsg) return;
        const int tok = (int) (gid / nblk), b = (int) (gid % nblk), lane = (int) sg.get_local_id()[0];
        if (tok >= *ntokp) return;
        const float * row = x + (size_t) tok * D + (size_t) b * 32;
        const float v0 = row[2 * lane], v1 = row[2 * lane + 1];
        const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
        const float sum = sycl::reduce_over_group(sg, v0 + v1, sycl::plus<float>());
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        int8_t * dst = xq + ((size_t) tok * D + (size_t) b * 32);
        dst[2 * lane] = (int8_t) sycl::rint(v0 * id);
        dst[2 * lane + 1] = (int8_t) sycl::rint(v1 * id);
        if (lane == 0) {
            xd[(size_t) tok * nblk + b] = d;
            xs[(size_t) tok * nblk + b] = sum;
        }
    });
}

// Batched unpack-once Q8_0R matmul: y[tok][y_off + row] = W[row] . x[tok], tok in [0, M).
template <int M>
void k_dspark_q8r_mv(sycl::queue & q, WRef w, const int8_t * xq, const float * xd, int cols, int rows,
                     const int32_t * ntokp, float * y, size_t y_stride, int y_off) {
    const int nblk = cols / 32;
    const size_t nsg = (size_t) rows;
    q.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int r = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
        if (r >= rows) return;
        const int lane = (int) sg.get_local_id()[0];
        const int ntok = ntokp ? *ntokp : M;
        const uint8_t * wp_base = w.p + (size_t) r * w.rb;
        const uint8_t * ws_base = w.a + (size_t) r * w.arb;
        float acc[M]{};
        for (int ib = lane; ib < nblk; ib += SG) {
            const sycl::int4 * wp = (const sycl::int4 *) (wp_base + (size_t) ib * 32);
            const sycl::int4 w0 = wp[0], w1 = wp[1];
            const float ws = q::f16(ws_base + (size_t) ib * 2);
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                if (m < ntok) {
                    const sycl::int4 * xv = (const sycl::int4 *) (xq + (size_t) m * cols + (size_t) ib * 32);
                    const sycl::int4 x0 = xv[0], x1 = xv[1];
                    int32_t isum = 0;
                    for (int i = 0; i < 4; ++i) isum = dp4a(w0[i], x0[i], isum);
                    for (int i = 0; i < 4; ++i) isum = dp4a(w1[i], x1[i], isum);
                    acc[m] += ws * xd[(size_t) m * nblk + ib] * (float) isum;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            if (m < ntok) {
                const float v = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
                if (lane == 0) y[(size_t) m * y_stride + y_off + r] = v;
            }
        }
    });
}

// Fused Q8_0R SwiGLU gate+up: act[tok][row] = silu(W_gate[row].x) * (W_up[row].x).
template <int M>
void k_dspark_ffn_gu(sycl::queue & q, WRef wg, WRef wu, const int8_t * xq, const float * xd, int cols, int nff,
                     const int32_t * ntokp, float * act) {
    const int nblk = cols / 32;
    const size_t nsg = (size_t) nff;
    q.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int r = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
        if (r >= nff) return;
        const int lane = (int) sg.get_local_id()[0];
        const int ntok = *ntokp;
        const uint8_t * g_p = wg.p + (size_t) r * wg.rb;
        const uint8_t * g_s = wg.a + (size_t) r * wg.arb;
        const uint8_t * u_p = wu.p + (size_t) r * wu.rb;
        const uint8_t * u_s = wu.a + (size_t) r * wu.arb;
        float ag[M]{}, au[M]{};
        for (int ib = lane; ib < nblk; ib += SG) {
            const sycl::int4 * gp = (const sycl::int4 *) (g_p + (size_t) ib * 32);
            const sycl::int4 * up = (const sycl::int4 *) (u_p + (size_t) ib * 32);
            const sycl::int4 g0 = gp[0], g1 = gp[1];
            const sycl::int4 u0 = up[0], u1 = up[1];
            const float gs = q::f16(g_s + (size_t) ib * 2);
            const float us = q::f16(u_s + (size_t) ib * 2);
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                if (m < ntok) {
                    const sycl::int4 * xv = (const sycl::int4 *) (xq + (size_t) m * cols + (size_t) ib * 32);
                    const sycl::int4 x0 = xv[0], x1 = xv[1];
                    int32_t ig = 0, iu = 0;
                    for (int i = 0; i < 4; ++i) {
                        ig = dp4a(g0[i], x0[i], ig);
                        iu = dp4a(u0[i], x0[i], iu);
                    }
                    for (int i = 0; i < 4; ++i) {
                        ig = dp4a(g1[i], x1[i], ig);
                        iu = dp4a(u1[i], x1[i], iu);
                    }
                    const float xd_m = xd[(size_t) m * nblk + ib];
                    ag[m] += gs * xd_m * (float) ig;
                    au[m] += us * xd_m * (float) iu;
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            if (m < ntok) {
                const float vg = sycl::reduce_over_group(sg, ag[m], sycl::plus<float>());
                const float vu = sycl::reduce_over_group(sg, au[m], sycl::plus<float>());
                if (lane == 0) act[(size_t) m * nff + r] = siluf(vg) * vu;
            }
        }
    });
}

// Mode A KV store: K (RMSNorm + NeoX RoPE) & V (raw) written to layer's KV cache via kv_put.
// Workgroup of HD=128 holds one (tok, kvh, is_v) head.
void k_dspark_kv_put_ctx(sycl::queue & q, const float * kv_in, int tok_off, int n,
                         const float * k_norm_w, const float * inv_freq, const int32_t * pos0p,
                         int8_t * kc, int8_t * vc, sycl::half * ks, sycl::half * vs,
                         int nkv, int ring, int kvlen, float eps, bool kv4) {
    const int kvw = nkv * HD;
    const int groups = 2 * nkv;
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sx(HD, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) n * groups * HD, HD), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto grp = it.get_group();
            const int gg = (int) it.get_group(0), t = (int) it.get_local_id(0);
            const int tok = gg / groups, g = gg % groups;
            const bool is_v = g >= nkv;
            const int kh = g % nkv;
            const float * raw = kv_in + (size_t) tok * (2 * kvw) + (is_v ? kvw : 0) + kh * HD;
            float v = raw[t];
            if (!is_v) {
                const float ss = sycl::reduce_over_group(grp, v * v, sycl::plus<float>());
                const float r = 1.0f / sycl::sqrt(ss / HD + eps);
                sx[t] = v * r * k_norm_w[t];
                sycl::group_barrier(grp);
                const int j = t & 63;
                const float ang = (float) (*pos0p + tok_off + tok) * inv_freq[j];
                const float cs = sycl::cos(ang), sn = sycl::sin(ang);
                const float x0 = sx[j], x1 = sx[j + 64];
                v = (t < 64) ? (x0 * cs - x1 * sn) : (x0 * sn + x1 * cs);
            }
            const int p = *pos0p + tok_off + tok;
            const int idx = ring ? (p % ring) : p;
            kv_put(grp, it.get_sub_group(), t, v, kv4, is_v ? vc : kc, is_v ? vs : ks,
                   ((size_t) kh * kvlen + idx) * HD + t);
        });
    });
}

// Mode B Q/K/V post-process: per-head RMSNorm + NeoX RoPE on Q (into qf) and K (into kblk), copy V into vblk.
void k_dspark_qkv_post(sycl::queue & q, const float * qkv, const float * qnw, const float * knw,
                       const float * inv_freq, const int32_t * pos0p, const int32_t * mp,
                       float * qf, float * kblk, float * vblk, int nh, int nkv, float eps) {
    constexpr size_t PJ = 5120;  // 3072 (Q) + 1024 (K) + 1024 (V)
    const int groups = nh + 2 * nkv;
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sx(HD, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) 8 * groups * HD, HD), [=](sycl::nd_item<1> it) {
            const auto grp = it.get_group();
            const int gg = (int) it.get_group(0), t = (int) it.get_local_id(0);
            const int u = gg / groups, g = gg % groups;
            if (u >= *mp) return;
            const float * pr = qkv + (size_t) u * PJ;
            if (g < nh) {
                const int h = g;
                float v = pr[h * HD + t];
                const float ss = sycl::reduce_over_group(grp, v * v, sycl::plus<float>());
                const float r = 1.0f / sycl::sqrt(ss / HD + eps);
                sx[t] = v * r * qnw[t];
                sycl::group_barrier(grp);
                const int j = t & 63;
                const float ang = (float) (*pos0p + u) * inv_freq[j];
                const float cs = sycl::cos(ang), sn = sycl::sin(ang);
                const float x0 = sx[j], x1 = sx[j + 64];
                qf[((size_t) u * nh + h) * HD + t] = (t < 64) ? (x0 * cs - x1 * sn) : (x0 * sn + x1 * cs);
            } else if (g < nh + nkv) {
                const int kh = g - nh;
                float v = pr[3072 + kh * HD + t];
                const float ss = sycl::reduce_over_group(grp, v * v, sycl::plus<float>());
                const float r = 1.0f / sycl::sqrt(ss / HD + eps);
                sx[t] = v * r * knw[t];
                sycl::group_barrier(grp);
                const int j = t & 63;
                const float ang = (float) (*pos0p + u) * inv_freq[j];
                const float cs = sycl::cos(ang), sn = sycl::sin(ang);
                const float x0 = sx[j], x1 = sx[j + 64];
                kblk[((size_t) u * nkv + kh) * HD + t] = (t < 64) ? (x0 * cs - x1 * sn) : (x0 * sn + x1 * cs);
            } else {
                const int kh = g - nh - nkv;
                vblk[((size_t) u * nkv + kh) * HD + t] = pr[4096 + kh * HD + t];
            }
        });
    });
}

// Mode B split-K context attention over injected KV [k_lo, pos0 - 1] (shared across all M query slots).
template <int AWG, int M, int RATIO>
void k_dspark_attn(sycl::queue & q, const float * qf, const int8_t * kc, const int8_t * vc,
                   const sycl::half * ks, const sycl::half * vs, const int32_t * pos0p, const int32_t * mp,
                   float * part, int nh, int nkv, int n_swa, int ring, int kvlen, int ch, int nc,
                   int pstride, bool swa, bool kv4) {
    constexpr int R = M * RATIO;
    constexpr int NSG = AWG / SG;
    constexpr int SPU = NSG / M;
    constexpr int DPL = HD / (SG * SPU);
    static_assert(SPU >= 1 && (SG * SPU * DPL == HD), "k_dspark_attn geometry mismatch");
    static_assert(NSG <= SG, "k_dspark_attn softmax reduction expects NSG <= SG");
    const float scale = 1.0f / std::sqrt((float) HD);

    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int32_t, 1> sq8((size_t) R * (HD / 4), cgh);
        sycl::local_accessor<float, 1> sqd((size_t) R * (HD / 32), cgh), sqs((size_t) R * (HD / 32), cgh);
        sycl::local_accessor<sycl::half, 1> sp((size_t) R * ch, cgh);
        sycl::local_accessor<float, 1> sm(R, cgh), sred((size_t) 2 * NSG * R, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) nkv * nc * AWG, AWG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto grp = it.get_group();
            const int g = (int) it.get_group(0), t = (int) it.get_local_id(0);
            const int kvh = g / nc, c = g % nc, h0 = kvh * RATIO;
            const int pos0 = *pos0p, ntok = *mp, p_last = pos0 - 1;
            const int lo = swa ? sycl::max(0, pos0 - n_swa + 1) : 0;
            const int k0 = lo + c * ch;
            if (k0 > p_last || p_last < 0) {
                for (int u = 0; u < ntok; ++u)
                    #pragma unroll
                    for (int j = 0; j < RATIO; ++j) {
                        float * out = part + ((size_t) (u * nh + h0 + j) * pstride + c) * (HD + 2);
                        if (t == 0) {
                            out[0] = -INFINITY;
                            out[1] = 0.0f;
                        }
                        if (t < HD) out[2 + t] = 0.0f;
                    }
                return;
            }
            const int R_act = ntok * RATIO;
            {
                const auto sgq = it.get_sub_group();
                const int sgid = (int) sgq.get_group_id()[0], lane = (int) sgq.get_local_id()[0];
                for (int qb = sgid; qb < R_act * (HD / 32); qb += NSG) {
                    const int r = qb / (HD / 32), b = qb % (HD / 32), u = r / RATIO, j = r % RATIO;
                    const float * qrow = qf + ((size_t) u * nh + h0 + j) * HD + b * 32;
                    const float v0 = qrow[2 * lane], v1 = qrow[2 * lane + 1];
                    const float amax = sycl::reduce_over_group(sgq, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
                    const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
                    const int q0 = (int) sycl::rint(v0 * id), q1 = (int) sycl::rint(v1 * id);
                    const int qs = sycl::reduce_over_group(sgq, q0 + q1, sycl::plus<int>());
                    int8_t * dst = (int8_t *) &sq8[0] + r * HD + b * 32;
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
            {
                float sc[R];
                #pragma unroll
                for (int r = 0; r < R; ++r) sc[r] = -INFINITY;
                const int k = k0 + t;
                if (t < ch && k <= p_last) {
                    const int idx = ring ? (k % ring) : k;
                    const size_t kbase = ((size_t) kvh * kvlen + idx) * HD;
                    float dot[R]{};
                    for (int b = 0; b < HD / 32; ++b) {
                        const float ds = (float) ks[kbase / 32 + b];
                        int32_t kw[8];
                        if (kv4) {
                            const sycl::uint4 w4 = *(const sycl::uint4 *) ((const uint8_t *) kc + (kbase + 32 * b) / 2);
                            for (int w = 0; w < 4; ++w) {
                                kw[2 * w] = (int32_t) (w4[w] & 0x0F0F0F0Fu);
                                kw[2 * w + 1] = (int32_t) ((w4[w] >> 4) & 0x0F0F0F0Fu);
                            }
                        } else {
                            const sycl::int4 * p4 = (const sycl::int4 *) (kc + kbase + 32 * b);
                            const sycl::int4 a0 = p4[0], a1 = p4[1];
                            for (int w = 0; w < 4; ++w) {
                                kw[w] = a0[w];
                                kw[4 + w] = a1[w];
                            }
                        }
                        #pragma unroll
                        for (int r = 0; r < R; ++r) {
                            const int32_t * qw = &sq8[(r * HD + 32 * b) / 4];
                            int32_t is = 0;
                            for (int w = 0; w < 8; ++w) is = dp4a(kw[w], qw[w], is);
                            const float val = kv4 ? (float) is - 8.0f * sqs[r * (HD / 32) + b] : (float) is;
                            dot[r] += ds * sqd[r * (HD / 32) + b] * val;
                        }
                    }
                    #pragma unroll
                    for (int r = 0; r < R; ++r)
                        if (r < R_act) sc[r] = dot[r] * scale;
                }
                const auto sgr = it.get_sub_group();
                const int r_sgid = (int) sgr.get_group_id()[0], r_lane = (int) sgr.get_local_id()[0];
                for (int r = 0; r < R_act; ++r) {
                    const float s_mx = sycl::reduce_over_group(sgr, sc[r], sycl::maximum<float>());
                    if (r_lane == 0) sred[r * NSG + r_sgid] = s_mx;
                }
                sycl::group_barrier(grp);
                for (int r = 0; r < R_act; ++r) {
                    const float mx = sycl::reduce_over_group(sgr, r_lane < NSG ? sred[r * NSG + r_lane] : -INFINITY, sycl::maximum<float>());
                    const float pr = (t < ch && sc[r] != -INFINITY && mx != -INFINITY) ? sycl::exp(sc[r] - mx) : 0.0f;
                    if (t < ch) sp[r * ch + t] = (sycl::half) pr;
                    const float s_ls = sycl::reduce_over_group(sgr, pr, sycl::plus<float>());
                    if (r_lane == 0) sred[(R + r) * NSG + r_sgid] = s_ls;
                    if (t == 0) sm[r] = mx;
                }
            }
            sycl::group_barrier(grp);
            {
                const auto sgv = it.get_sub_group();
                const int sgid = (int) sgv.get_group_id()[0], lane = (int) sgv.get_local_id()[0];
                const int u = sgid / SPU, d_sg = sgid % SPU, d_base = (d_sg * SG + lane) * DPL;
                if (u < ntok) {
                    const int nk = sycl::clamp(p_last - k0 + 1, 0, ch);
                    float acc[RATIO][DPL]{};
                    for (int jj = 0; jj < nk; ++jj) {
                        const int kk = k0 + jj;
                        const int idx = ring ? (kk % ring) : kk;
                        const size_t vi = ((size_t) kvh * kvlen + idx) * HD + d_base;
                        float vvals[DPL];
                        if constexpr (DPL == 1) {
                            if (kv4) {
                                const uint8_t by = ((const uint8_t *) vc)[vi / 2];
                                vvals[0] = (float) ((int) ((d_base & 1) ? (by >> 4) : (by & 15)) - 8);
                            } else {
                                vvals[0] = (float) vc[vi];
                            }
                        } else if constexpr (DPL == 2) {
                            if (kv4) {
                                const uint8_t by = ((const uint8_t *) vc)[vi / 2];
                                vvals[0] = (float) ((int) (by & 15) - 8);
                                vvals[1] = (float) ((int) (by >> 4) - 8);
                            } else {
                                const sycl::vec<int8_t, 2> v2 = *(const sycl::vec<int8_t, 2> *) (vc + vi);
                                vvals[0] = (float) v2[0];
                                vvals[1] = (float) v2[1];
                            }
                        } else {
                            kv_get<DPL>(vc, kv4, vi, vvals);
                        }
                        const float vd = (float) vs[vi / 32];
                        #pragma unroll
                        for (int j = 0; j < RATIO; ++j) {
                            const float wj = (float) sp[(u * RATIO + j) * ch + jj] * vd;
                            #pragma unroll
                            for (int e = 0; e < DPL; ++e) acc[j][e] += wj * vvals[e];
                        }
                    }
                    #pragma unroll
                    for (int j = 0; j < RATIO; ++j) {
                        const float ls = d_sg == 0
                            ? sycl::reduce_over_group(sgv, lane < NSG ? sred[(R + u * RATIO + j) * NSG + lane] : 0.0f, sycl::plus<float>())
                            : 0.0f;
                        float * out = part + ((size_t) (u * nh + h0 + j) * pstride + c) * (HD + 2);
                        #pragma unroll
                        for (int e = 0; e < DPL; ++e) out[2 + d_base + e] = acc[j][e];
                        if (d_sg == 0 && lane == 0) {
                            out[0] = sm[u * RATIO + j];
                            out[1] = ls;
                        }
                    }
                }
            }
        });
    });
}

// Mode B combine: merges split-K context chunks with the M bidirectional ephemeral block tokens (K_blk, V_blk).
template <int M>
void k_dspark_combine(sycl::queue & q, const float * qf, const float * kblk, const float * vblk,
                      const float * part, float * att, const int32_t * pos0p, const int32_t * mp,
                      int nh, int nkv, int ratio, int n_swa, int ch, int nc, int pstride, bool swa) {
    const float scale = 1.0f / std::sqrt((float) HD);
    constexpr int NSG = HD / SG;  // 8
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sdot((size_t) M * NSG, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) M * nh * HD, HD), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto grp = it.get_group();
            const auto sg = it.get_sub_group();
            const int gh = (int) it.get_group(0), d = (int) it.get_local_id(0);
            const int u = gh / nh, h = gh % nh, kvh = h / ratio;
            const int ntok = *mp;
            if (u >= ntok) return;
            const int sgid = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];
            const float qd = qf[((size_t) u * nh + h) * HD + d];
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const float kd = (m < ntok) ? kblk[((size_t) m * nkv + kvh) * HD + d] : 0.0f;
                const float ss = sycl::reduce_over_group(sg, qd * kd, sycl::plus<float>());
                if (lane == 0) sdot[m * NSG + sgid] = ss;
            }
            sycl::group_barrier(grp);
            float sc_blk[M];
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                const float dot = sycl::reduce_over_group(sg, lane < NSG ? sdot[m * NSG + lane] : 0.0f, sycl::plus<float>());
                sc_blk[m] = (m < ntok) ? dot * scale : -INFINITY;
            }
            const int pos0 = *pos0p;
            const int lo = swa ? sycl::max(0, pos0 - n_swa + 1) : 0;
            const int active_nc = (pos0 > lo) ? sycl::min(nc, (pos0 - lo + ch - 1) / ch) : 0;
            const float * base = part + ((size_t) (u * nh + h) * pstride) * (HD + 2);
            float Mmax = -INFINITY;
            for (int c = 0; c < active_nc; ++c) Mmax = sycl::fmax(Mmax, base[(size_t) c * (HD + 2)]);
            #pragma unroll
            for (int m = 0; m < M; ++m) Mmax = sycl::fmax(Mmax, sc_blk[m]);

            float L = 0.0f, acc = 0.0f;
            for (int c = 0; c < active_nc; ++c) {
                const float mc = base[(size_t) c * (HD + 2)];
                if (mc == -INFINITY) continue;
                const float w = sycl::exp(mc - Mmax);
                L += base[(size_t) c * (HD + 2) + 1] * w;
                acc += base[(size_t) c * (HD + 2) + 2 + d] * w;
            }
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                if (m < ntok) {
                    const float w = sycl::exp(sc_blk[m] - Mmax);
                    L += w;
                    acc += vblk[((size_t) m * nkv + kvh) * HD + d] * w;
                }
            }
            att[((size_t) u * nh + h) * HD + d] = L > 0.0f ? acc / L : 0.0f;
        });
    });
}

// Target shared LM head for K slots (uncapped, supports DP4A Q8_0R/Q6_KR/Q4_KR + generic decode32f fallback).
template <uint32_t T, int K_MAX, bool STAGE = false>
sycl::event k_dspark_lm_head(sycl::queue & q, WRef w, const int8_t * xq, const float * xd, const float * xsum,
                              const float * xf, int cols, int rows, const int32_t * kp, float * y) {
    constexpr int nblk = D / 32;
    const size_t nsg = (size_t) rows;
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint64_t, 1> gl(STAGE ? grid64<T>() : 1, cgh);
        sycl::local_accessor<uint8_t, 1> ksl(STAGE ? 128 : 1, cgh);
        cgh.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            q::Tabs tb;
            if constexpr (STAGE) tb = stage<T>(it, &gl[0], &ksl[0]);
            const auto sg = it.get_sub_group();
            const int r = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
            if (r >= rows) return;
            const int lane = (int) sg.get_local_id()[0];
            const int ntok = *kp;
            const RowRef rr = row_of(w, r);
            float acc[K_MAX]{};
            if constexpr (T == q::Q8_0R) {
                for (int ib = lane; ib < nblk; ib += SG) {
                    const sycl::int4 * wp = (const sycl::int4 *) (rr.p + (size_t) ib * 32);
                    const sycl::int4 w0 = wp[0], w1 = wp[1];
                    const float ws = q::f16(rr.a + (size_t) ib * 2);
                    #pragma unroll
                    for (int m = 0; m < K_MAX; ++m) {
                        if (m < ntok) {
                            const sycl::int4 * xv = (const sycl::int4 *) (xq + (size_t) m * cols + (size_t) ib * 32);
                            const sycl::int4 x0 = xv[0], x1 = xv[1];
                            int32_t isum = 0;
                            for (int i = 0; i < 4; ++i) isum = dp4a(w0[i], x0[i], isum);
                            for (int i = 0; i < 4; ++i) isum = dp4a(w1[i], x1[i], isum);
                            acc[m] += ws * xd[(size_t) m * nblk + ib] * (float) isum;
                        }
                    }
                }
            } else if constexpr (T == q::Q6_KR) {
                for (int ib = lane; ib < nblk; ib += SG) {
                    const sycl::int4 * wp = (const sycl::int4 *) (rr.p + (size_t) ib * 32);
                    const sycl::int4 w0 = wp[0], w1 = wp[1];
                    const float d = q::f16(rr.a2 + (size_t) (ib / 8) * 2);
                    const float s0 = d * (float) (int8_t) rr.a[2 * ib];
                    const float s1 = d * (float) (int8_t) rr.a[2 * ib + 1];
                    #pragma unroll
                    for (int m = 0; m < K_MAX; ++m) {
                        if (m < ntok) {
                            const sycl::int4 * xv = (const sycl::int4 *) (xq + (size_t) m * cols + (size_t) ib * 32);
                            const sycl::int4 x0 = xv[0], x1 = xv[1];
                            int32_t is0 = 0, is1 = 0;
                            for (int i = 0; i < 4; ++i) is0 = dp4a(w0[i], x0[i], is0);
                            for (int i = 0; i < 4; ++i) is1 = dp4a(w1[i], x1[i], is1);
                            acc[m] += xd[(size_t) m * nblk + ib] * (s0 * (float) is0 + s1 * (float) is1);
                        }
                    }
                }
            } else if constexpr (T == q::Q4_KR) {
                for (int ib = lane; ib < nblk; ib += SG) {
                    const int g = ib / 2, hi = ib & 1;
                    const uint8_t * qp = rr.p + (size_t) g * 32;
                    int32_t w[8];
                    const int sh = hi ? 4 : 0;
                    for (int j = 0; j < 2; ++j) {
                        const sycl::uint4 w4 = *(const sycl::uint4 *) (qp + 16 * j);
                        for (int k = 0; k < 4; ++k) w[4 * j + k] = (int32_t) ((w4[k] >> sh) & 0x0F0F0F0Fu);
                    }
                    const uint16_t dd = q::rd16(rr.a2 + (ib / 8) * 4), dm = q::rd16(rr.a2 + (ib / 8) * 4 + 2);
                    const float s0 = q::f16((const uint8_t *) &dd) * (float) rr.a[2 * ib];
                    const float m0 = -q::f16((const uint8_t *) &dm) * (float) rr.a[2 * ib + 1];
                    #pragma unroll
                    for (int m = 0; m < K_MAX; ++m) {
                        if (m < ntok) {
                            const int blk = m * nblk + ib;
                            const sycl::int4 * xv = (const sycl::int4 *) (xq + (size_t) m * cols + (size_t) ib * 32);
                            const sycl::int4 x0 = xv[0], x1 = xv[1];
                            int32_t isum = 0;
                            for (int i = 0; i < 4; ++i) isum = dp4a(w[i], x0[i], isum);
                            for (int i = 0; i < 4; ++i) isum = dp4a(w[4 + i], x1[i], isum);
                            acc[m] += s0 * xd[blk] * (float) isum + m0 * xsum[blk];
                        }
                    }
                }
            } else if constexpr (q::shape(T).elems == 1) {
                for (int c = lane * 16; c < cols; c += SG * 16) {
                    float wv[16];
                    for (int j = 0; j < 16; ++j) {
                        const uint8_t * p = rr.p + (size_t) (c + j) * q::shape(T).bytes;
                        if constexpr (T == q::F32) wv[j] = sycl::bit_cast<float>(q::rd32(p));
                        else if constexpr (T == q::F16) wv[j] = q::f16(p);
                        else wv[j] = q::bf16(p);
                    }
                    #pragma unroll
                    for (int m = 0; m < K_MAX; ++m) {
                        if (m < ntok) {
                            const float * xb = xf + (size_t) m * cols + c;
                            for (int j = 0; j < 16; ++j) acc[m] += wv[j] * xb[j];
                        }
                    }
                }
            } else {
                constexpr q::BlockShape bs = q::shape(T);
                for (int c0 = lane * 32; c0 < cols; c0 += SG * 32) {
                    float wv[32];
                    q::decode32f(T, rr.p + (size_t) (c0 / bs.elems) * bs.bytes, (c0 % bs.elems) / 32,
                                 [&](int j, float val) { wv[j] = val; }, tb);
                    #pragma unroll
                    for (int m = 0; m < K_MAX; ++m) {
                        if (m < ntok) {
                            const float * xb = xf + (size_t) m * cols + c0;
                            for (int j = 0; j < 32; ++j) acc[m] += wv[j] * xb[j];
                        }
                    }
                }
            }
            #pragma unroll
            for (int m = 0; m < K_MAX; ++m) {
                if (m < ntok) {
                    const float yv = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
                    if (lane == 0) y[(size_t) m * rows + r] = yv;
                }
            }
        });
    });
}

// Mode C step 1: joint_in[5120] = [state(256), w1[prev](256), hidden[slot](2304), f_rnn(2304)]
void k_dspark_joint_in(sycl::queue & q, const float * state, const uint8_t * w1_p, size_t w1_rb,
                       const int32_t * prev_tokp, const float * hidden_slot, const float * f_rnn,
                       float * joint_in) {
    constexpr int TOTAL = 5120;
    q.parallel_for(sycl::range<1>(TOTAL), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        if (i < 256) {
            joint_in[i] = state[i];
        } else if (i < 512) {
            const uint8_t * row = w1_p + (size_t) (*prev_tokp) * w1_rb;
            joint_in[i] = q::bf16(row + (size_t) (i - 256) * 2);
        } else if (i < 2816) {
            joint_in[i] = hidden_slot[i - 512];
        } else {
            joint_in[i] = f_rnn[i - 2816];
        }
    });
}

// Mode C step 2: joint_out[768] = joint_proj_w(BF16) @ joint_in + joint_proj_b
void k_dspark_joint_proj(sycl::queue & q, WRef jw, const float * jb, const float * joint_in, float * j_out) {
    constexpr int COLS = 5120, ROWS = 768;
    q.parallel_for(sycl::nd_range<1>((size_t) (ROWS + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int r = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
        if (r >= ROWS) return;
        const int lane = (int) sg.get_local_id()[0];
        const uint8_t * wp = jw.p + (size_t) r * jw.rb;
        float acc = 0.0f;
        for (int c = lane * 16; c < COLS; c += SG * 16) {
            for (int j = 0; j < 16; ++j) acc += q::bf16(wp + (size_t) (c + j) * 2) * joint_in[c + j];
        }
        const float y = sycl::reduce_over_group(sg, acc, sycl::plus<float>());
        if (lane == 0) j_out[r] = y + jb[r];
    });
}

// Mode C step 3: GRU-style Markov RNN update + quantize markov[256] for Q8_0R w2
void k_dspark_rnn_update(sycl::queue & q, const float * j_out, float * rnn_state, float * markov,
                         int8_t * markov_q, float * markov_qd, int rank) {
    const int nb = rank / 32;
    q.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const int r = (int) it.get_local_id(0);
        if (r < rank) {
            const float g = sigmoidf(j_out[r]);
            const float c = sycl::tanh(j_out[rank + r]);
            const float st = rnn_state[r];
            rnn_state[r] = c + g * (st - c);
            markov[r] = sycl::tanh(j_out[2 * rank + r]);
        }
        sycl::group_barrier(it.get_group());
        const auto sg = it.get_sub_group();
        const int sgid = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];
        if (sgid < nb) {
            const float * row = markov + sgid * 32;
            const float v0 = row[2 * lane], v1 = row[2 * lane + 1];
            const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
            const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
            markov_q[sgid * 32 + 2 * lane] = (int8_t) sycl::rint(v0 * id);
            markov_q[sgid * 32 + 2 * lane + 1] = (int8_t) sycl::rint(v1 * id);
            if (lane == 0) markov_qd[sgid] = d;
        }
    });
}

// Mode C step 4 (greedy): fused w2 @ markov + w2_bias + base_logits[slot] + softcap(30) + wg_argmax.
// Two rows per SG=16 (8 lanes per row across the 256-dim Markov vector = 8 x 32-value blocks).
void k_dspark_w2_argmax(sycl::queue & q, WRef w2, bool use_q8, const float * w2_b, const float * base_logits_slot,
                        const float * markov, const int8_t * markov_q, const float * markov_qd,
                        int n_vocab, float softcap, float * tmp_v, int32_t * tmp_i,
                        int32_t * out_slot, int32_t * prev_tok) {
    constexpr int NG = WG;  // 256 workgroups for final argmax reduction
    (void) softcap;
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sv(RPG, cgh);
        sycl::local_accessor<int32_t, 1> si(RPG, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) NG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto sg = it.get_sub_group();
            const int lane = (int) sg.get_local_id()[0];
            const int sub = lane / 8;  // 0 or 1: two rows handled per subgroup
            const int ib = lane & 7;   // 0..7 blocks of 32 values
            float bv = -INFINITY;
            int32_t bi = 1 << 30;
            const int r_base_step = (NG * (WG / SG)) * 2;
            const int sg_global = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);

            int32_t mx0[4], mx1[4];
            float mxd = 0.0f;
            if (use_q8) {
                const sycl::int4 * xv = (const sycl::int4 *) (markov_q + ib * 32);
                const sycl::int4 a0 = xv[0], a1 = xv[1];
                for (int i = 0; i < 4; ++i) {
                    mx0[i] = a0[i];
                    mx1[i] = a1[i];
                }
                mxd = markov_qd[ib];
            }

            for (int r0 = sg_global * 2; r0 < n_vocab; r0 += r_base_step) {
                const int r = r0 + sub;
                float part = 0.0f;
                if (r < n_vocab) {
                    if (use_q8) {
                        const sycl::int4 * wp = (const sycl::int4 *) (w2.p + (size_t) r * w2.rb + (size_t) ib * 32);
                        const sycl::int4 w0 = wp[0], w1 = wp[1];
                        int32_t isum = 0;
                        for (int i = 0; i < 4; ++i) isum = dp4a(w0[i], mx0[i], isum);
                        for (int i = 0; i < 4; ++i) isum = dp4a(w1[i], mx1[i], isum);
                        const float ws = q::f16(w2.a + (size_t) r * w2.arb + (size_t) ib * 2);
                        part = ws * mxd * (float) isum;
                    } else {
                        const uint8_t * wp = w2.p + (size_t) r * w2.rb + (size_t) ib * 64;
                        for (int j = 0; j < 32; ++j) part += q::bf16(wp + j * 2) * markov[ib * 32 + j];
                    }
                }
                // Sum within each 8-lane half of the 16-lane subgroup
                for (int off = 4; off > 0; off /= 2) part += sycl::permute_group_by_xor(sg, part, off);
                if (r < n_vocab && ib == 0) {
                    const float logit = base_logits_slot[r] + w2_b[r] + part;  // monotonic softcap omitted for argmax
                    if (logit > bv || (logit == bv && r < bi)) {
                        bv = logit;
                        bi = r;
                    }
                }
            }
            wg_argmax(it, &sv[0], &si[0], bv, bi);
            if (it.get_local_id(0) == 0) {
                tmp_v[it.get_group(0)] = bv;
                tmp_i[it.get_group(0)] = bi;
            }
        });
    });
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sv(RPG, cgh);
        sycl::local_accessor<int32_t, 1> si(RPG, cgh);
        cgh.parallel_for(sycl::nd_range<1>(WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int l = (int) it.get_local_id(0);
            float bv = tmp_v[l];
            int32_t bi = tmp_i[l];
            wg_argmax(it, &sv[0], &si[0], bv, bi);
            if (l == 0) {
                *out_slot = bi;
                *prev_tok = bi;
            }
        });
    });
}

// Mode C step 4 (probabilistic): writes full softcapped logits[n_vocab] for host sampling.
void k_dspark_w2_logits(sycl::queue & q, WRef w2, bool use_q8, const float * w2_b, const float * base_logits_slot,
                        const float * markov, const int8_t * markov_q, const float * markov_qd,
                        int n_vocab, float softcap, float * out_logits) {
    // 2 rows per subgroup of 16 -> ceil(n_vocab / 2) subgroups
    const size_t nsg = ((size_t) n_vocab + 1) / 2;
    q.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int sg_g = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
        const int lane = (int) sg.get_local_id()[0];
        const int r = sg_g * 2 + (lane / 8);
        if (r >= n_vocab) return;
        const int ib = lane & 7;
        float part = 0.0f;
        if (use_q8) {
            const sycl::int4 * wp = (const sycl::int4 *) (w2.p + (size_t) r * w2.rb + (size_t) ib * 32);
            const sycl::int4 * xv = (const sycl::int4 *) (markov_q + ib * 32);
            const sycl::int4 w0 = wp[0], w1 = wp[1], a0 = xv[0], a1 = xv[1];
            int32_t isum = 0;
            for (int i = 0; i < 4; ++i) isum = dp4a(w0[i], a0[i], isum);
            for (int i = 0; i < 4; ++i) isum = dp4a(w1[i], a1[i], isum);
            const float ws = q::f16(w2.a + (size_t) r * w2.arb + (size_t) ib * 2);
            part = ws * markov_qd[ib] * (float) isum;
        } else {
            const uint8_t * wp = w2.p + (size_t) r * w2.rb + (size_t) ib * 64;
            for (int j = 0; j < 32; ++j) part += q::bf16(wp + j * 2) * markov[ib * 32 + j];
        }
        for (int off = 4; off > 0; off /= 2) part += sycl::permute_group_by_xor(sg, part, off);
        if (ib == 0) {
            float logit = base_logits_slot[r] + w2_b[r] + part;
            if (softcap > 0.0f) logit = softcap * sycl::tanh(logit / softcap);
            out_logits[r] = logit;
        }
    });
}

}  // namespace

// ---------------------------------------------------------------------------------------------------
// DsparkModel

DsparkModel::DsparkModel(sycl::queue & q, const GgufFile & f, const Model & target, bool w2_q8) : q_(q) {
    const std::string arch = f.get("general.architecture").s;
    if (arch != "dflash") throw std::runtime_error("dspark model must have general.architecture == dflash, got " + arch);
    const std::string a = arch + ".";
    auto gi = [&](const std::string & k) { return (int) f.get(a + k).as_int(); };
    auto gf = [&](const std::string & k) { return (float) f.get(a + k).as_float(); };

    hp.n_embd = gi("embedding_length");
    hp.n_layer = gi("block_count");
    hp.n_ff = gi("feed_forward_length");
    hp.n_head = gi("attention.head_count");
    hp.n_head_kv = gi("attention.head_count_kv");
    hp.head_dim = gi("attention.key_length");
    hp.rope_dim = gi("rope.dimension_count");
    hp.rope_freq_base = gf("rope.freq_base");
    hp.eps = gf("attention.layer_norm_rms_epsilon");
    hp.n_swa = gi("attention.sliding_window");
    hp.softcap = gf("final_logit_softcapping");
    hp.block_size = gi("block_size");
    hp.sample_from_anchor = f.find(a + "sample_from_anchor") ? f.get(a + "sample_from_anchor").b : false;
    hp.has_markov_context = f.find(a + "has_markov_context") ? f.get(a + "has_markov_context").b : true;
    for (const auto & v : f.get(a + "attention.sliding_window_pattern").arr) hp.is_swa.push_back(v.b);
    for (const auto & v : f.get(a + "target_layers").arr) hp.target_layers.push_back((int) v.as_int());

    if (hp.n_embd != target.hp.n_embd || hp.n_embd != D) throw std::runtime_error("dspark embedding_length mismatch");
    if (hp.head_dim != HD || hp.rope_dim != HD) throw std::runtime_error("dspark head_dim/rope_dim must be 128");
    if (hp.n_layer != 3 || hp.is_swa.size() != 3) throw std::runtime_error("dspark expects 3 layers");
    if (hp.block_size != 8 || hp.sample_from_anchor || !hp.has_markov_context)
        throw std::runtime_error("dspark unsupported block_size/anchor/markov configuration");
    if (hp.target_layers != std::vector<int>({5, 23, 41, 48}))
        throw std::runtime_error("dspark expects target_layers == [5, 23, 41, 48]");

    const GgufTensor * w1t = f.tensor("markov_w1.weight");
    if (!w1t) throw std::runtime_error("dspark missing markov_w1.weight");
    hp.markov_rank = (int) w1t->ne[0];
    hp.n_vocab = (int) w1t->ne[1];
    if (hp.n_vocab != target.hp.n_vocab)
        throw std::runtime_error("dspark vocab (" + std::to_string(hp.n_vocab) + ") does not match the target's (" +
                                 std::to_string(target.hp.n_vocab) + "): an older DSpark file? Use the current "
                                 "dspark-Inkling-2-Mini-Q8_0.gguf from unsloth/Inkling-2-Mini-GGUF (the _Old one has no Markov head)");
    if (hp.markov_rank != 256) throw std::runtime_error("dspark expects markov_rank == 256");

    uint64_t total_need = 0;
    for (const auto & t : f.tensors()) total_need += t.n_bytes;
    Model::check_fits(q_, total_need, "dspark weights");

    const char * env_w2 = getenv("INK_DSPARK_W2_Q8");
    const bool quantize_w2 = env_w2 ? (std::string(env_w2) != "0") : w2_q8;

    auto L = [&](const std::string & n, bool rep) { return load_dspark_tensor(q_, f, n, weight_bytes, allocs_, rep); };
    tok_norm = L("token_embd_norm.weight", false);
    mask_embd = L("mask_embd.weight", false);
    output_norm = L("output_norm.weight", false);
    markov_ctx_mix = L("markov_ctx_mix.weight", false);
    markov_ctx_norm = L("markov_ctx_norm.weight", false);
    joint_proj_w = L("joint_proj.weight", false);
    joint_proj_b = L("joint_proj.bias", false);
    markov_w1 = L("markov_w1.weight", false);
    markov_w2_w = quantize_w2 ? load_w2_quantized_q8r(q_, f, "markov_w2.weight", weight_bytes, allocs_)
                              : L("markov_w2.weight", false);
    markov_w2_b = L("markov_w2.bias", false);

    layers.resize(hp.n_layer);
    for (int i = 0; i < hp.n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        DsparkLayerW & l = layers[i];
        l.ctx_mix = L(p + "ctx_mix.weight", false);
        l.ctx_norm = L(p + "ctx_norm.weight", false);
        l.attn_k_ctx = L(p + "attn_k_ctx.weight", true);
        l.attn_v_ctx = L(p + "attn_v_ctx.weight", true);
        l.attn_k_ctx_norm = L(p + "attn_k_ctx_norm.weight", false);
        l.attn_norm = L(p + "attn_norm.weight", false);
        l.attn_q = L(p + "attn_q.weight", true);
        l.attn_k = L(p + "attn_k.weight", true);
        l.attn_v = L(p + "attn_v.weight", true);
        l.attn_q_norm = L(p + "attn_q_norm.weight", false);
        l.attn_k_norm = L(p + "attn_k_norm.weight", false);
        l.attn_output = L(p + "attn_output.weight", true);
        l.ffn_norm = L(p + "ffn_norm.weight", false);
        l.ffn_gate = L(p + "ffn_gate.weight", true);
        l.ffn_up = L(p + "ffn_up.weight", true);
        l.ffn_down = L(p + "ffn_down.weight", true);
    }
    q_.wait();
}

DsparkModel::~DsparkModel() {
    q_.wait();
    for (void * p : allocs_) sycl::free(p, q_);
}

// ---------------------------------------------------------------------------------------------------
// DsparkEngine

template <typename T> T * DsparkEngine::dev(size_t n) {
    T * p = sycl::malloc_device<T>(n, q_);
    if (!p) throw std::runtime_error("dspark: device allocation failed");
    allocs_.push_back(p);
    return p;
}

DsparkEngine::DsparkEngine(sycl::queue & q, const Model & target, const DsparkModel & dm,
                           int n_ctx, int kv_bits, int inject_chunk)
    : q_(q), target_(target), dm_(dm), hp_(dm.hp), n_ctx_(n_ctx), kv4_(kv_bits == 4),
      inject_chunk_(std::clamp(inject_chunk, 1, 8)) {
    const int nkv = hp_.n_head_kv, nh = hp_.n_head, kvw = nkv * HD;
    max_chunks_ = std::max((n_ctx_ + FCH - 1) / FCH, (hp_.n_swa + SCH - 1) / SCH);

    uint64_t kv_bytes = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        const int len = hp_.is_swa[il] ? std::min(hp_.n_swa + 512, n_ctx_) : n_ctx_;
        kv_len_.push_back(len);
        kv_bytes += 2ull * len * (kv_val_bytes(kvw) + kvw / 32 * sizeof(sycl::half));
    }
    const size_t scratch_bytes =
        ((size_t) M_MAX * nh * max_chunks_ * (HD + 2) + (size_t) (M_MAX - 1) * hp_.n_vocab + (size_t) hp_.n_vocab) * sizeof(float);
    Model::check_fits(q_, kv_bytes + scratch_bytes + (32ull << 20), "dspark state");

    for (int il = 0; il < hp_.n_layer; ++il) {
        kc_.push_back(dev<int8_t>((size_t) kv_len_[il] * kv_val_bytes(kvw)));
        vc_.push_back(dev<int8_t>((size_t) kv_len_[il] * kv_val_bytes(kvw)));
        ks_.push_back(dev<sycl::half>((size_t) kv_len_[il] * kvw / 32));
        vs_.push_back(dev<sycl::half>((size_t) kv_len_[il] * kvw / 32));
    }

    inv_freq_ = dev<float>(HD / 2);
    std::vector<float> freqs(HD / 2);
    for (int j = 0; j < HD / 2; ++j) freqs[j] = (float) std::pow((double) hp_.rope_freq_base, -2.0 * j / (double) HD);
    q_.memcpy(inv_freq_, freqs.data(), freqs.size() * sizeof(float));

    f_rnn_ring_ = dev<float>((size_t) RING_RNN * D);
    q_.memset(f_rnn_ring_, 0, (size_t) RING_RNN * D * sizeof(float));

    d_last_tok_ = dev<int32_t>(1);
    d_pos0_ = dev<int32_t>(1);
    d_m_ = dev<int32_t>(1);
    d_k_ = dev<int32_t>(1);
    d_prev_tok_ = dev<int32_t>(1);

    x_ = dev<float>((size_t) M_MAX * D);
    h_ = dev<float>((size_t) M_MAX * D);
    hq_ = dev<int8_t>((size_t) M_MAX * D);
    hqd_ = dev<float>((size_t) M_MAX * (D / 32));
    qkv_ = dev<float>((size_t) M_MAX * 5120);
    qf_ = dev<float>((size_t) M_MAX * nh * HD);
    kblk_ = dev<float>((size_t) M_MAX * nkv * HD);
    vblk_ = dev<float>((size_t) M_MAX * nkv * HD);
    part_ = dev<float>((size_t) M_MAX * nh * max_chunks_ * (HD + 2));
    att_ = dev<float>((size_t) M_MAX * nh * HD);
    attq_ = dev<int8_t>((size_t) M_MAX * nh * HD);
    attqd_ = dev<float>((size_t) M_MAX * (nh * HD / 32));
    act_ = dev<float>((size_t) M_MAX * hp_.n_ff);
    actq_ = dev<int8_t>((size_t) M_MAX * hp_.n_ff);
    actqd_ = dev<float>((size_t) M_MAX * (hp_.n_ff / 32));
    hidden_ = dev<float>((size_t) (M_MAX - 1) * D);
    base_logits_ = dev<float>((size_t) (M_MAX - 1) * hp_.n_vocab);

    rnn_state_ = dev<float>(hp_.markov_rank);
    joint_in_ = dev<float>(5120);
    joint_out_ = dev<float>(3 * hp_.markov_rank);
    markov_ = dev<float>(hp_.markov_rank);
    markov_q_ = dev<int8_t>(hp_.markov_rank);
    markov_qd_ = dev<float>(hp_.markov_rank / 32);
    slot_logits_ = dev<float>(hp_.n_vocab);
    argmax_val_ = dev<float>(WG);
    argmax_idx_ = dev<int32_t>(WG);
    draft_out_ = dev<int32_t>(M_MAX - 1);
    f_rnn_cur_ = dev<float>(D);

    inj_c_ = dev<float>((size_t) 8 * D);
    inj_q_ = dev<int8_t>((size_t) 8 * D);
    inj_qd_ = dev<float>((size_t) 8 * (D / 32));
    inj_kv_ = dev<float>((size_t) 8 * 2 * kvw);
    q_.wait();
}

DsparkEngine::~DsparkEngine() {
    q_.wait();
    for (void * p : allocs_) sycl::free(p, q_);
}

template <int NB>
void DsparkEngine::inject_tile(const float * tgt_feats_dev, int tap_cap, int tok_off, int n, int pos0) {
    const int nkv = hp_.n_head_kv, kvw = nkv * HD;
    k_dspark_ctx_mix(q_, tgt_feats_dev, tap_cap, tok_off, n, d_pos0_, dm_.markov_ctx_mix.f32(),
                     dm_.markov_ctx_norm.f32(), f_rnn_ring_, RING_RNN, hp_.eps);
    for (int il = 0; il < hp_.n_layer; ++il) {
        const DsparkLayerW & L = dm_.layers[il];
        const bool swa = hp_.is_swa[il];
        k_dspark_ctx_mix(q_, tgt_feats_dev, tap_cap, tok_off, n, d_pos0_, L.ctx_mix.f32(),
                         L.ctx_norm.f32(), inj_c_, 0, hp_.eps);
        k_dspark_quant_q8(q_, inj_c_, D, D, n, nullptr, inj_q_, inj_qd_);
        k_dspark_q8r_mv<NB>(q_, wref(L.attn_k_ctx), inj_q_, inj_qd_, D, kvw, nullptr, inj_kv_, 2 * kvw, 0);
        k_dspark_q8r_mv<NB>(q_, wref(L.attn_v_ctx), inj_q_, inj_qd_, D, kvw, nullptr, inj_kv_, 2 * kvw, kvw);
        k_dspark_kv_put_ctx(q_, inj_kv_, tok_off, n, L.attn_k_ctx_norm.f32(), inv_freq_, d_pos0_,
                            kc_[il], vc_[il], ks_[il], vs_[il], nkv, swa ? kv_len_[il] : 0, kv_len_[il], hp_.eps, kv4_);
    }
    (void) pos0;
}

void DsparkEngine::inject(const float * tgt_feats_dev, int tap_cap, int n, int pos0) {
    if (n <= 0) return;
    if (pos0 + n > n_ctx_) throw std::runtime_error("dspark inject: context is full");
    q_.memcpy(d_pos0_, &pos0, sizeof(int32_t));
    for (int off = 0; off < n; off += inject_chunk_) {
        const int cnt = std::min(inject_chunk_, n - off);
        if (cnt == 1) inject_tile<1>(tgt_feats_dev, tap_cap, off, cnt, pos0);
        else if (cnt <= 4) inject_tile<4>(tgt_feats_dev, tap_cap, off, cnt, pos0);
        else inject_tile<8>(tgt_feats_dev, tap_cap, off, cnt, pos0);
    }
}

template <int M>
void DsparkEngine::record_backbone() {
    const int nh = hp_.n_head, nkv = hp_.n_head_kv, ratio = nh / nkv, kvw = nkv * HD;
    k_dspark_embed(q_, wref(target_.tok_embd), target_.tok_embd.type, d_last_tok_, dm_.tok_norm.f32(), dm_.mask_embd.f32(), d_m_, x_, hp_.eps);

    for (int il = 0; il < hp_.n_layer; ++il) {
        const DsparkLayerW & L = dm_.layers[il];
        const bool swa = hp_.is_swa[il];
        k_dspark_rmsnorm<M>(q_, x_, L.attn_norm.f32(), h_, d_m_, 0, M, hp_.eps);
        k_dspark_quant_q8(q_, h_, D, D, M, d_m_, hq_, hqd_);
        k_dspark_q8r_mv<M>(q_, wref(L.attn_q), hq_, hqd_, D, nh * HD, d_m_, qkv_, 5120, 0);
        k_dspark_q8r_mv<M>(q_, wref(L.attn_k), hq_, hqd_, D, kvw, d_m_, qkv_, 5120, 3072);
        k_dspark_q8r_mv<M>(q_, wref(L.attn_v), hq_, hqd_, D, kvw, d_m_, qkv_, 5120, 4096);
        k_dspark_qkv_post(q_, qkv_, L.attn_q_norm.f32(), L.attn_k_norm.f32(), inv_freq_, d_pos0_, d_m_,
                          qf_, kblk_, vblk_, nh, nkv, hp_.eps);

        const int ch = swa ? SCH : FCH;
        const int nc = swa ? (hp_.n_swa + ch - 1) / ch : max_chunks_;
        if (swa) {
            k_dspark_attn<SCH, M, 3>(q_, qf_, kc_[il], vc_[il], ks_[il], vs_[il], d_pos0_, d_m_,
                                     part_, nh, nkv, hp_.n_swa, kv_len_[il], kv_len_[il], ch, nc, max_chunks_, true, kv4_);
        } else {
            k_dspark_attn<256, M, 3>(q_, qf_, kc_[il], vc_[il], ks_[il], vs_[il], d_pos0_, d_m_,
                                     part_, nh, nkv, hp_.n_swa, 0, kv_len_[il], ch, nc, max_chunks_, false, kv4_);
        }
        k_dspark_combine<M>(q_, qf_, kblk_, vblk_, part_, att_, d_pos0_, d_m_, nh, nkv, ratio, hp_.n_swa, ch, nc, max_chunks_, swa);
        k_dspark_quant_q8(q_, att_, nh * HD, nh * HD, M, d_m_, attq_, attqd_);
        k_dspark_q8r_mv<M>(q_, wref(L.attn_output), attq_, attqd_, nh * HD, D, d_m_, h_, D, 0);
        k_dspark_add(q_, h_, x_, d_m_, D);

        k_dspark_rmsnorm<M>(q_, x_, L.ffn_norm.f32(), h_, d_m_, 0, M, hp_.eps);
        k_dspark_quant_q8(q_, h_, D, D, M, d_m_, hq_, hqd_);
        k_dspark_ffn_gu<M>(q_, wref(L.ffn_gate), wref(L.ffn_up), hq_, hqd_, D, hp_.n_ff, d_m_, act_);
        k_dspark_quant_q8(q_, act_, hp_.n_ff, hp_.n_ff, M, d_m_, actq_, actqd_);
        k_dspark_q8r_mv<M>(q_, wref(L.ffn_down), actq_, actqd_, hp_.n_ff, D, d_m_, h_, D, 0);
        k_dspark_add(q_, h_, x_, d_m_, D);
    }

    // Extract slots 1..K (skipping anchor slot 0), apply output_norm, project through target.output[:n_vocab] (uncapped)
    constexpr int K = M - 1;
    k_dspark_rmsnorm<K>(q_, x_, dm_.output_norm.f32(), hidden_, d_m_, 1, K, hp_.eps);
    const W & o = target_.output;
    k_dspark_quant_lm(q_, hidden_, K, d_k_, hq_, hqd_, x_);  // reuse x_[0..K*(D/32)) temporarily as xsum
    dispatch_type(o.ktype, [&](auto tag) {
        constexpr uint32_t T = decltype(tag)::value;
        k_dspark_lm_head<T, K, false>(q_, wref(o), hq_, hqd_, x_, hidden_, D, hp_.n_vocab, d_k_, base_logits_);
    });
}

void DsparkEngine::run_markov_step(int slot, bool greedy_update) {
    k_dspark_joint_in(q_, rnn_state_, dm_.markov_w1.p, dm_.markov_w1.row_bytes, d_prev_tok_,
                      hidden_ + (size_t) slot * D, f_rnn_cur_, joint_in_);
    k_dspark_joint_proj(q_, wref(dm_.joint_proj_w), dm_.joint_proj_b.f32(), joint_in_, joint_out_);
    k_dspark_rnn_update(q_, joint_out_, rnn_state_, markov_, markov_q_, markov_qd_, hp_.markov_rank);
    const float * base_slot = base_logits_ + (size_t) slot * hp_.n_vocab;
    const bool w2_q8 = dm_.markov_w2_w.ktype == q::Q8_0R;
    if (greedy_update) {
        k_dspark_w2_argmax(q_, wref(dm_.markov_w2_w), w2_q8, dm_.markov_w2_b.f32(), base_slot,
                           markov_, markov_q_, markov_qd_, hp_.n_vocab, hp_.softcap,
                           argmax_val_, argmax_idx_, draft_out_ + slot, d_prev_tok_);
    } else {
        k_dspark_w2_logits(q_, wref(dm_.markov_w2_w), w2_q8, dm_.markov_w2_b.f32(), base_slot,
                           markov_, markov_q_, markov_qd_, hp_.n_vocab, hp_.softcap, slot_logits_);
    }
}

template <int K>
void DsparkEngine::build_greedy_graph(int bi) {
    try {
        ModGraph rec(q_.get_context(), q_.get_device());
        rec.begin_recording(q_);
        record_backbone<K + 1>();
        k_dspark_rnn_init(q_, d_last_tok_, d_pos0_, f_rnn_ring_, rnn_state_, f_rnn_cur_, d_prev_tok_,
                          hp_.markov_rank, RING_RNN);
        for (int s = 0; s < K; ++s) run_markov_step(s, true);
        rec.end_recording(q_);
        g_graphs_[bi] = std::make_unique<ExecGraph>(rec.finalize());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "dspark: greedy command graph unavailable for K=%d (%s)\n", K, e.what());
        g_graphs_[bi].reset();
    }
}

void DsparkEngine::draft_greedy(int last_tok, int pos0, int k, int32_t * out_toks) {
    if (k < 1 || k > hp_.max_draft()) throw std::runtime_error("dspark draft: k out of range");
    const int m = 1 + k;
    q_.memcpy(d_last_tok_, &last_tok, sizeof(int32_t));
    q_.memcpy(d_pos0_, &pos0, sizeof(int32_t));
    q_.memcpy(d_m_, &m, sizeof(int32_t));
    q_.memcpy(d_k_, &k, sizeof(int32_t));

    if (k == 3 || k == 7) {
        const int bi = (k == 3) ? 0 : 1;
        if (!g_graphs_[bi]) {
            if (bi == 0) build_greedy_graph<3>(0);
            else build_greedy_graph<7>(1);
        }
        if (g_graphs_[bi]) {
            q_.ext_oneapi_graph(*g_graphs_[bi]);
            q_.memcpy(out_toks, draft_out_, (size_t) k * sizeof(int32_t)).wait();
            return;
        }
    }
    if (k <= 3) record_backbone<4>();
    else record_backbone<8>();
    k_dspark_rnn_init(q_, d_last_tok_, d_pos0_, f_rnn_ring_, rnn_state_, f_rnn_cur_, d_prev_tok_,
                      hp_.markov_rank, RING_RNN);
    for (int s = 0; s < k; ++s) run_markov_step(s, true);
    q_.memcpy(out_toks, draft_out_, (size_t) k * sizeof(int32_t)).wait();
}

void DsparkEngine::draft_backbone(int last_tok, int pos0, int k) {
    if (k < 1 || k > hp_.max_draft()) throw std::runtime_error("dspark backbone: k out of range");
    const int m = 1 + k;
    q_.memcpy(d_last_tok_, &last_tok, sizeof(int32_t));
    q_.memcpy(d_pos0_, &pos0, sizeof(int32_t));
    q_.memcpy(d_m_, &m, sizeof(int32_t));
    q_.memcpy(d_k_, &k, sizeof(int32_t));

    if (k == 3 || k == 7) {
        const int bi = (k == 3) ? 0 : 1;
        if (!b_graphs_[bi]) {
            try {
                ModGraph rec(q_.get_context(), q_.get_device());
                rec.begin_recording(q_);
                if (bi == 0) record_backbone<4>();
                else record_backbone<8>();
                k_dspark_rnn_init(q_, d_last_tok_, d_pos0_, f_rnn_ring_, rnn_state_, f_rnn_cur_, d_prev_tok_,
                                  hp_.markov_rank, RING_RNN);
                rec.end_recording(q_);
                b_graphs_[bi] = std::make_unique<ExecGraph>(rec.finalize());
            } catch (const std::exception &) {
                b_graphs_[bi].reset();
            }
        }
        if (b_graphs_[bi]) {
            q_.ext_oneapi_graph(*b_graphs_[bi]);
            return;
        }
    }
    if (k <= 3) record_backbone<4>();
    else record_backbone<8>();
    k_dspark_rnn_init(q_, d_last_tok_, d_pos0_, f_rnn_ring_, rnn_state_, f_rnn_cur_, d_prev_tok_,
                      hp_.markov_rank, RING_RNN);
}

void DsparkEngine::draft_markov_step(int slot, int32_t prev_tok, float * out_logits) {
    if (slot < 0 || slot >= hp_.max_draft()) throw std::runtime_error("dspark markov step: slot out of range");
    q_.memcpy(d_prev_tok_, &prev_tok, sizeof(int32_t));
    run_markov_step(slot, false);
    q_.memcpy(out_logits, slot_logits_, (size_t) hp_.n_vocab * sizeof(float)).wait();
}

// ---------------------------------------------------------------------------------------------------
// Snapshot / Restore / Stash

size_t DsparkEngine::kv_bytes_(size_t ntok, size_t kvw) const {
    return 2 * (ntok * kv_val_bytes(kvw) + ntok * kvw / 32 * sizeof(sycl::half));
}

void DsparkEngine::kv_copy(int il, size_t tok0, size_t ntok, uint8_t * host, bool to_host) {
    const size_t len = kv_len_[il], nh = hp_.n_head_kv;
    const size_t vb = kv_val_bytes(ntok * HD), sb = ntok * HD / 32 * sizeof(sycl::half);
    size_t o = 0;
    for (int kind = 0; kind < 4; ++kind)
        for (size_t hh = 0; hh < nh; ++hh) {
            const size_t first = (hh * len + tok0) * HD;
            void * devp = kind == 0 ? (void *) (kc_[il] + kv_val_bytes(first)) : kind == 1 ? (void *) (ks_[il] + first / 32)
                        : kind == 2 ? (void *) (vc_[il] + kv_val_bytes(first)) : (void *) (vs_[il] + first / 32);
            const size_t n = kind % 2 ? sb : vb;
            if (to_host) q_.memcpy(host + o, devp, n);
            else q_.memcpy(devp, host + o, n);
            o += n;
        }
}

void DsparkEngine::save(Snapshot & s, int pos) {
    const size_t kvw = (size_t) hp_.n_head_kv * HD;
    s.f_rnn_ring.resize((size_t) RING_RNN * D);
    q_.memcpy(s.f_rnn_ring.data(), f_rnn_ring_, (size_t) RING_RNN * D * sizeof(float));
    size_t swa = 0;
    for (int il = 0; il < hp_.n_layer; ++il)
        if (hp_.is_swa[il]) swa += kv_bytes_(kv_len_[il], kvw);
    s.swa.resize(swa);
    size_t o = 0;
    for (int il = 0; il < hp_.n_layer; ++il)
        if (hp_.is_swa[il]) {
            kv_copy(il, 0, kv_len_[il], s.swa.data() + o, true);
            o += kv_bytes_(kv_len_[il], kvw);
        }
    q_.wait();
    s.pos = pos;
}

void DsparkEngine::restore(const Snapshot & s) {
    const size_t kvw = (size_t) hp_.n_head_kv * HD;
    if (s.f_rnn_ring.size() != (size_t) RING_RNN * D)
        throw std::runtime_error("dspark restore: invalid snapshot");
    q_.memcpy(f_rnn_ring_, s.f_rnn_ring.data(), (size_t) RING_RNN * D * sizeof(float));
    size_t o = 0, f = 0;
    for (int il = 0; il < hp_.n_layer; ++il) {
        if (hp_.is_swa[il]) {
            kv_copy(il, 0, kv_len_[il], const_cast<uint8_t *>(s.swa.data()) + o, false);
            o += kv_bytes_(kv_len_[il], kvw);
        } else if (!s.full.empty() && s.pos > 0) {
            kv_copy(il, 0, s.pos, const_cast<uint8_t *>(s.full.data()) + f, false);
            f += kv_bytes_(s.pos, kvw);
        }
    }
    q_.wait();
}

void DsparkEngine::stash(Snapshot & dst, const Snapshot & src) {
    const size_t kvw = (size_t) hp_.n_head_kv * HD;
    dst.pos = src.pos;
    dst.f_rnn_ring = src.f_rnn_ring;
    dst.swa = src.swa;
    size_t full = 0;
    if (src.pos > 0)
        for (int il = 0; il < hp_.n_layer; ++il)
            if (!hp_.is_swa[il]) full += kv_bytes_(src.pos, kvw);
    dst.full.resize(full);
    size_t o = 0;
    if (src.pos > 0)
        for (int il = 0; il < hp_.n_layer; ++il) {
            if (hp_.is_swa[il]) continue;
            kv_copy(il, 0, src.pos, dst.full.data() + o, true);
            o += kv_bytes_(src.pos, kvw);
        }
    q_.wait();
}

}  // namespace ink
