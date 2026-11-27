// Batched-decode verify path (VB in {2, 4, 8}) for speculative decoding.
// Runs up to VFAST=8 tokens through a recorded SYCL command graph per bucket so verify(T) scales with
// distinct-expert weight reads (~1.35x at T=2, ~1.9x at T=4, ~2.7x at T=8) rather than prefill's XMX tiles.
#include "fast.h"

#include "kernels.h"
#include "quant.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace sycl_exp = sycl::ext::oneapi::experimental;

namespace ink {

namespace {

#ifndef INK_VGUR
#define INK_VGUR 2
#endif
#ifndef INK_VDR
#define INK_VDR 2
#endif

template <uint32_t T>
constexpr bool has_vq8() {
    return has_q8<T>() || T == q::Q8_0R;
}

template <uint32_t T>
constexpr bool has_vdown_q8() {
    return has_down_q8<T>() || T == q::Q8_0R;
}

// Unpacked 32-value block in int8x4 DP4A form + up to 2 sub-block float scales + optional min/delta term.
struct DecodedU32 {
    int32_t w[8];
    float s0, s1;
    float m0;
};

template <uint32_t T>
inline DecodedU32 decode_unit_q8(const RowRef rr, int ib, const q::Tabs & tb) {
    DecodedU32 u{};
    if constexpr (T == q::NVFP4R) {
        const uint32_t * wp = (const uint32_t *) (rr.p + 16 * ib);
        const uint32_t w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3];
        u.w[0] = fp4x4(w0);
        u.w[1] = fp4x4(w1);
        u.w[2] = fp4x4(w0 >> 4);
        u.w[3] = fp4x4(w1 >> 4);
        u.w[4] = fp4x4(w2);
        u.w[5] = fp4x4(w3);
        u.w[6] = fp4x4(w2 >> 4);
        u.w[7] = fp4x4(w3 >> 4);
        const uint16_t sc = *(const uint16_t *) (rr.a + 2 * ib);
        u.s0 = q::ue4m3((uint8_t) (sc & 0xFFu));
        u.s1 = q::ue4m3((uint8_t) (sc >> 8));
    } else if constexpr (T == q::Q5_KR) {
        const uint32_t * wp = (const uint32_t *) (rr.p + 16 * ib);
        const uint32_t qh = *(const uint32_t *) (rr.a + 8 * ib);
        const uint16_t sm = *(const uint16_t *) (rr.a + 8 * ib + 4);
        const uint32_t dd = *(const uint32_t *) (rr.a2 + (ib / 8) * 4);
        const float d = h2f_bits((uint16_t) (dd & 0xFFFFu));
        const float dmin = h2f_bits((uint16_t) (dd >> 16));
        u.w[0] = (int32_t) ((wp[0] & 0x0F0F0F0Fu) | hi16x4(qh & 0xFu));
        u.w[1] = (int32_t) ((wp[1] & 0x0F0F0F0Fu) | hi16x4((qh >> 4) & 0xFu));
        u.w[2] = (int32_t) ((wp[2] & 0x0F0F0F0Fu) | hi16x4((qh >> 8) & 0xFu));
        u.w[3] = (int32_t) ((wp[3] & 0x0F0F0F0Fu) | hi16x4((qh >> 12) & 0xFu));
        u.w[4] = (int32_t) (((wp[0] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 16) & 0xFu));
        u.w[5] = (int32_t) (((wp[1] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 20) & 0xFu));
        u.w[6] = (int32_t) (((wp[2] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 24) & 0xFu));
        u.w[7] = (int32_t) (((wp[3] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 28) & 0xFu));
        u.s0 = d * (float) (sm & 0xFFu);
        u.m0 = -dmin * (float) (sm >> 8);
    } else if constexpr (T == q::Q4_KR) {
        const int b = ib / 8, j = ib % 8, g = j / 2, sh = (j & 1) ? 4 : 0;
        const uint32_t * wp = (const uint32_t *) (rr.p + b * 128 + g * 32);
        const uint16_t sm = *(const uint16_t *) (rr.a + ib * 2);
        const uint32_t dd = *(const uint32_t *) (rr.a2 + b * 4);
        const float d = h2f_bits((uint16_t) (dd & 0xFFFFu));
        const float dmin = h2f_bits((uint16_t) (dd >> 16));
        for (int k = 0; k < 8; ++k) u.w[k] = (int32_t) ((wp[k] >> sh) & 0x0F0F0F0Fu);
        u.s0 = d * (float) (sm & 0xFFu);
        u.m0 = -dmin * (float) (sm >> 8);
    } else if constexpr (T == q::Q5_0R) {
        const uint32_t * wp = (const uint32_t *) (rr.p + 16 * ib);
        const uint32_t qh = *(const uint32_t *) (rr.a + 4 * ib);
        u.w[0] = (int32_t) ((wp[0] & 0x0F0F0F0Fu) | hi16x4(qh & 0xFu));
        u.w[1] = (int32_t) ((wp[1] & 0x0F0F0F0Fu) | hi16x4((qh >> 4) & 0xFu));
        u.w[2] = (int32_t) ((wp[2] & 0x0F0F0F0Fu) | hi16x4((qh >> 8) & 0xFu));
        u.w[3] = (int32_t) ((wp[3] & 0x0F0F0F0Fu) | hi16x4((qh >> 12) & 0xFu));
        u.w[4] = (int32_t) (((wp[0] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 16) & 0xFu));
        u.w[5] = (int32_t) (((wp[1] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 20) & 0xFu));
        u.w[6] = (int32_t) (((wp[2] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 24) & 0xFu));
        u.w[7] = (int32_t) (((wp[3] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 28) & 0xFu));
        u.s0 = q::f16(rr.a2 + 2 * ib);
    } else if constexpr (T == q::Q8_0R) {
        const sycl::int4 * wp = (const sycl::int4 *) (rr.p + 32 * ib);
        const sycl::int4 w0 = wp[0], w1 = wp[1];
        for (int i = 0; i < 4; ++i) {
            u.w[i] = w0[i];
            u.w[4 + i] = w1[i];
        }
        u.s0 = q::f16(rr.a + 2 * ib);
    } else if constexpr (T == q::Q2_0R) {
        const sycl::uint2 wv = *(const sycl::uint2 *) (rr.p + 8 * ib);
        for (int w = 0; w < 2; ++w)
            for (int r = 0; r < 4; ++r) u.w[4 * w + r] = (int32_t) ((wv[w] >> (2 * r)) & 0x03030303u);
        u.s0 = q::f16(rr.a + 2 * (ib / 2));
    } else if constexpr (T == q::IQ1_SR) {
        const uint32_t w = *(const uint32_t *) (rr.p + 4 * ib);
        const uint32_t qh = *(const uint16_t *) (rr.a + 2 * ib);
        const uint16_t d16 = *(const uint16_t *) (rr.a2 + (ib / 8) * 2);
        const float dl = h2f_bits(d16) * (float) (2 * ((qh >> 12) & 7) + 1);
        const float delta = qh & 0x8000 ? -0.125f : 0.125f;
        for (int l = 0; l < 4; ++l) {
            const uint32_t * g32 = (const uint32_t *) tb.iq1s + 2 * (((w >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8));
            u.w[2 * l + 0] = (int32_t) g32[0];
            u.w[2 * l + 1] = (int32_t) g32[1];
        }
        u.s0 = dl;
        u.m0 = delta;
    } else {
        const uint32_t w0 = *(const uint32_t *) (rr.p + 8 * ib);
        const uint32_t w1 = *(const uint32_t *) (rr.p + 8 * ib + 4);
        const uint32_t wv[2] = {w0, w1};
        uint32_t aux = 0;
        uint16_t d16;
        if constexpr (T == q::IQ2_SR) {
            aux = *(const uint16_t *) (rr.a + 2 * ib);
            d16 = *(const uint16_t *) (rr.a2 + (ib / 8) * 2);
        } else if constexpr (T == q::IQ2_XXSR) {
            d16 = *(const uint16_t *) (rr.a + (ib / 8) * 2);
        } else {
            aux = rr.a[ib];
            d16 = *(const uint16_t *) (rr.a2 + (ib / 8) * 2);
        }
        for (int l = 0; l < 4; ++l) {
            const uint32_t * g32;
            uint32_t signs;
            if constexpr (T == q::IQ2_SR) {
                const uint32_t qh = aux & 0xFF;
                g32 = (const uint32_t *) tb.iq2s + 2 * (((wv[0] >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300));
                signs = (wv[1] >> (8 * l)) & 0xFF;
            } else if constexpr (T == q::IQ2_XXSR) {
                g32 = (const uint32_t *) tb.iq2xxs + 2 * ((wv[0] >> (8 * l)) & 0xFF);
                signs = tb.ksigns[(wv[1] >> (7 * l)) & 127];
            } else {
                const uint32_t v = (wv[l / 2] >> (16 * (l % 2))) & 0xFFFF;
                g32 = (const uint32_t *) tb.iq2xs + 2 * (v & 511);
                signs = tb.ksigns[v >> 9];
            }
            u.w[2 * l + 0] = signed4(g32[0], byte_mask4(signs & 0xF));
            u.w[2 * l + 1] = signed4(g32[1], byte_mask4(signs >> 4));
        }
        if constexpr (T == q::IQ2_XXSR) {
            u.s0 = u.s1 = h2f_bits(d16) * (0.5f + (float) (w1 >> 28)) * 0.25f;
        } else {
            const uint32_t sc = T == q::IQ2_SR ? (aux >> 8) & 0xFF : aux;
            const float d = h2f_bits(d16);
            u.s0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f;
            u.s1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
        }
    }
    return u;
}

template <uint32_t T>
inline float dot_decoded_q8(const DecodedU32 & u, const sycl::int4 x0, const sycl::int4 x1, float xdb, float xsb) {
    if constexpr (T == q::Q5_KR || T == q::Q4_KR) {
        int32_t isum = 0;
        for (int i = 0; i < 4; ++i) isum = dp4a(u.w[i], x0[i], isum);
        for (int i = 0; i < 4; ++i) isum = dp4a(u.w[4 + i], x1[i], isum);
        return u.s0 * xdb * (float) isum + u.m0 * xsb;
    } else if constexpr (T == q::Q5_0R) {
        int32_t isum = 0;
        for (int i = 0; i < 4; ++i) isum = dp4a(u.w[i], x0[i], isum);
        for (int i = 0; i < 4; ++i) isum = dp4a(u.w[4 + i], x1[i], isum);
        return u.s0 * xdb * (float) (isum - 16 * (int32_t) xsb);
    } else if constexpr (T == q::Q8_0R) {
        int32_t isum = 0;
        for (int i = 0; i < 4; ++i) isum = dp4a(u.w[i], x0[i], isum);
        for (int i = 0; i < 4; ++i) isum = dp4a(u.w[4 + i], x1[i], isum);
        return u.s0 * xdb * (float) isum;
    } else if constexpr (T == q::Q2_0R) {
        int32_t isum = 0;
        for (int i = 0; i < 4; ++i) isum = dp4a(u.w[i], x0[i], isum);
        for (int i = 0; i < 4; ++i) isum = dp4a(u.w[4 + i], x1[i], isum);
        return u.s0 * xdb * (float) (isum - (int32_t) xsb);
    } else if constexpr (T == q::IQ1_SR) {
        int32_t isum = 0;
        for (int i = 0; i < 4; ++i) isum = dp4a(u.w[i], x0[i], isum);
        for (int i = 0; i < 4; ++i) isum = dp4a(u.w[4 + i], x1[i], isum);
        return u.s0 * (xdb * (float) isum + u.m0 * xsb);
    } else {
        int32_t is0 = 0, is1 = 0;
        for (int i = 0; i < 4; ++i) is0 = dp4a(u.w[i], x0[i], is0);
        for (int i = 0; i < 4; ++i) is1 = dp4a(u.w[4 + i], x1[i], is1);
        return xdb * (u.s0 * (float) is0 + u.s1 * (float) is1);
    }
}

template <uint32_t T, int VB>
inline void lane_dot_batch(const RowRef rr, int cols, const float * x, const float * x2, int split, size_t xs, int lane,
                           const q::Tabs & tb, float acc[VB]) {
    for (int t = 0; t < VB; ++t) acc[t] = 0.0f;
    if constexpr (T == q::Q8_0R || T == q::Q6_KR) {
        for (int c = lane * 16; c < cols; c += SG * 16) {
            const sycl::int4 wv = *(const sycl::int4 *) (rr.p + c);
            const float d = (T == q::Q8_0R) ? q::f16(rr.a + (c / 32) * 2)
                                            : (q::f16(rr.a2 + (c / 256) * 2) * (float) (int8_t) rr.a[c / 16]);
            const float * xb = c < split ? x + c : x2 + (c - split);
            float s[VB]{};
            for (int k = 0; k < 4; ++k) {
                const int w = wv[k];
                const float w0 = (float) (int8_t) w, w1 = (float) (int8_t) (w >> 8);
                const float w2 = (float) (int8_t) (w >> 16), w3 = (float) (int8_t) (w >> 24);
                for (int t = 0; t < VB; ++t) {
                    const sycl::float4 xv = *(const sycl::float4 *) (xb + (size_t) t * xs + 4 * k);
                    s[t] += w0 * xv[0] + w1 * xv[1] + w2 * xv[2] + w3 * xv[3];
                }
            }
            for (int t = 0; t < VB; ++t) acc[t] += s[t] * d;
        }
    } else if constexpr (T == q::Q4_KR) {
        for (int s = lane; s < cols / 32; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (rr.p + 16 * s);
            const int b = s / 8, o = (s % 8) * 16, g = o / 32, l0 = o % 32;
            const float d = q::f16(rr.a2 + b * 4), dmin = q::f16(rr.a2 + b * 4 + 2);
            const int clo = 256 * b + 64 * g + l0, chi = clo + 32;
            const float * xlo = clo < split ? x + clo : x2 + (clo - split);
            const float * xhi = chi < split ? x + chi : x2 + (chi - split);
            const uint8_t * sm = rr.a + b * 16 + 4 * g;
            const float d0 = d * (float) sm[0], m0 = dmin * (float) sm[1], d1 = d * (float) sm[2], m1 = dmin * (float) sm[3];
            float slo[VB]{}, shi[VB]{}, xl[VB]{}, xh[VB]{};
            for (int w = 0; w < 4; ++w)
                for (int k = 0; k < 4; ++k) {
                    const uint32_t byte = (wv[w] >> (8 * k)) & 0xFFu;
                    const float qlo = (float) (byte & 0xFu), qhi = (float) (byte >> 4);
                    const int j = 4 * w + k;
                    for (int t = 0; t < VB; ++t) {
                        const float vlo = xlo[(size_t) t * xs + j];
                        const float vhi = xhi[(size_t) t * xs + j];
                        slo[t] += qlo * vlo;
                        shi[t] += qhi * vhi;
                        xl[t] += vlo;
                        xh[t] += vhi;
                    }
                }
            for (int t = 0; t < VB; ++t) acc[t] += d0 * slo[t] - m0 * xl[t] + d1 * shi[t] - m1 * xh[t];
        }
    } else if constexpr (T == q::Q5_KR || T == q::Q5_0R) {
        for (int s = lane; s < cols / 32; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (rr.p + 16 * s);
            const int c0 = 32 * s;
            const float * xsb = c0 < split ? x + c0 : x2 + (c0 - split);
            float d1, m1;
            uint32_t qh;
            if constexpr (T == q::Q5_KR) {
                qh = *(const uint32_t *) (rr.a + 8 * s);
                const uint16_t sm = *(const uint16_t *) (rr.a + 8 * s + 4);
                const int b = s / 8;
                d1 = q::f16(rr.a2 + 4 * b) * (float) (sm & 0xFFu);
                m1 = q::f16(rr.a2 + 4 * b + 2) * (float) (sm >> 8);
            } else {
                qh = *(const uint32_t *) (rr.a + 4 * s);
                d1 = q::f16(rr.a2 + 2 * s);
                m1 = 0.0f;
            }
            float sum[VB]{}, xsum[VB]{};
            for (int w = 0; w < 4; ++w)
                for (int k = 0; k < 4; ++k) {
                    const uint32_t byte = (wv[w] >> (8 * k)) & 0xFFu;
                    const int j = 4 * w + k;
                    if constexpr (T == q::Q5_KR) {
                        const float qlo = (float) ((byte & 0xFu) + ((qh >> j) & 1u ? 16u : 0u));
                        const float qhi = (float) ((byte >> 4) + ((qh >> (16 + j)) & 1u ? 16u : 0u));
                        for (int t = 0; t < VB; ++t) {
                            const float * xt = xsb + (size_t) t * xs;
                            sum[t] += qlo * xt[j] + qhi * xt[16 + j];
                            xsum[t] += xt[j] + xt[16 + j];
                        }
                    } else {
                        const float qlo = (float) ((int) ((byte & 0xFu) + ((qh >> j) & 1u ? 16u : 0u)) - 16);
                        const float qhi = (float) ((int) ((byte >> 4) + ((qh >> (16 + j)) & 1u ? 16u : 0u)) - 16);
                        for (int t = 0; t < VB; ++t) {
                            const float * xt = xsb + (size_t) t * xs;
                            sum[t] += qlo * xt[j] + qhi * xt[16 + j];
                        }
                    }
                }
            for (int t = 0; t < VB; ++t) acc[t] += (T == q::Q5_KR) ? (d1 * sum[t] - m1 * xsum[t]) : (d1 * sum[t]);
        }
    } else if constexpr (q::shape(T).elems == 1) {
        for (int c = lane; c < cols; c += SG) {
            const uint8_t * p = rr.p + (size_t) c * q::shape(T).bytes;
            const float v = T == q::F32 ? sycl::bit_cast<float>(q::rd32(p)) : T == q::F16 ? q::f16(p) : q::bf16(p);
            const float * xb = c < split ? x + c : x2 + (c - split);
            for (int t = 0; t < VB; ++t) acc[t] += v * xb[(size_t) t * xs];
        }
    } else {
        constexpr int V = step_vals<T>();
        for (int s = lane; s < cols / V; s += SG) {
            step_decode<T>(rr, s, [&](int c, float v) {
                const float * xb = c < split ? x + c : x2 + (c - split);
                for (int t = 0; t < VB; ++t) acc[t] += v * xb[(size_t) t * xs];
            }, tb);
        }
    }
}

template <int VB>
struct VMvQuantArgs {
    const float * x[3] = {};
    size_t xs[3] = {};
    int cols[3] = {};
    int8_t * xq[3] = {};
    float * xd[3] = {}, * xsum[3] = {};
    int off[4] = {};
    int nx = 0;
};

template <int VB>
sycl::event k_vmv_quant(sycl::queue & q, const VMvQuantArgs<VB> a) {
    const int nblk = a.off[a.nx];
    const size_t total = (size_t) VB * nblk;
    return q.parallel_for(sycl::nd_range<1>((total + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int gb = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
        if (gb >= (int) total) return;
        const int t = gb / nblk, lb = gb % nblk;
        int k = 0;
        while (k + 1 < a.nx && lb >= a.off[k + 1]) ++k;
        const int ib = lb - a.off[k], cols = a.cols[k], nbk = cols / 32;
        const int lane = (int) sg.get_local_id()[0];
        const float * row = a.x[k] + (size_t) t * a.xs[k] + 32 * ib;
        const float v0 = row[2 * lane], v1 = row[2 * lane + 1];
        const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
        const float sum = sycl::reduce_over_group(sg, v0 + v1, sycl::plus<float>());
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        int8_t * dst = a.xq[k] + (size_t) t * cols + 32 * ib;
        dst[2 * lane] = (int8_t) sycl::round(v0 * id);
        dst[2 * lane + 1] = (int8_t) sycl::round(v1 * id);
        if (lane == 0) {
            const int bidx = t * nbk + ib;
            a.xd[k][bidx] = d;
            a.xsum[k][bidx] = sum;
        }
    });
}

template <uint32_t T, int VB, bool STAGE = false>
sycl::event k_mv_batch_q8(sycl::queue & q, const VMvJob * jobs, int njobs, int total_rows) {
    const size_t groups = (size_t) (total_rows + RPG - 1) / RPG;
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint64_t, 1> gl(STAGE ? grid64<T>() : 1, cgh);
        sycl::local_accessor<uint8_t, 1> ksl(STAGE ? 128 : 1, cgh);
        cgh.parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            q::Tabs tb;
            if constexpr (STAGE) tb = stage<T>(it, &gl[0], &ksl[0]);
            const auto sg = it.get_sub_group();
            const int row = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
            if (row >= total_rows) return;
            int j = 0;
            while (j + 1 < njobs && row >= jobs[j + 1].row0) ++j;
            const VMvJob & b = jobs[j];
            const int r = row - b.row0, lane = (int) sg.get_local_id()[0];
            const RowRef rr = row_of(b.w, (size_t) r);
            const int nblk = b.cols / 32;
            float acc[VB]{};
            if constexpr (T == q::Q8_0R) {
                for (int ib = lane; ib < nblk; ib += SG) {
                    const sycl::int4 * wp = (const sycl::int4 *) (rr.p + 32 * ib);
                    const sycl::int4 w0 = wp[0], w1 = wp[1];
                    const float d = q::f16(rr.a + 2 * ib);
                    #pragma unroll
                    for (int t = 0; t < VB; ++t) {
                        const sycl::int4 * xv = (const sycl::int4 *) (b.xq + (size_t) t * b.cols + 32 * ib);
                        const sycl::int4 x0 = xv[0], x1 = xv[1];
                        int32_t is = 0;
                        is = dp4a(w0[0], x0[0], is);
                        is = dp4a(w0[1], x0[1], is);
                        is = dp4a(w0[2], x0[2], is);
                        is = dp4a(w0[3], x0[3], is);
                        is = dp4a(w1[0], x1[0], is);
                        is = dp4a(w1[1], x1[1], is);
                        is = dp4a(w1[2], x1[2], is);
                        is = dp4a(w1[3], x1[3], is);
                        acc[t] += d * b.xd[(size_t) t * nblk + ib] * (float) is;
                    }
                }
            } else if constexpr (T == q::Q6_KR) {
                for (int ib = lane; ib < nblk; ib += SG) {
                    const sycl::int4 * wp = (const sycl::int4 *) (rr.p + 32 * ib);
                    const sycl::int4 w0 = wp[0], w1 = wp[1];
                    const float d = q::f16(rr.a2 + (ib / 8) * 2);
                    const float sc0 = (float) (int8_t) rr.a[2 * ib];
                    const float sc1 = (float) (int8_t) rr.a[2 * ib + 1];
                    #pragma unroll
                    for (int t = 0; t < VB; ++t) {
                        const sycl::int4 * xv = (const sycl::int4 *) (b.xq + (size_t) t * b.cols + 32 * ib);
                        const sycl::int4 x0 = xv[0], x1 = xv[1];
                        int32_t is0 = 0, is1 = 0;
                        is0 = dp4a(w0[0], x0[0], is0);
                        is0 = dp4a(w0[1], x0[1], is0);
                        is0 = dp4a(w0[2], x0[2], is0);
                        is0 = dp4a(w0[3], x0[3], is0);
                        is1 = dp4a(w1[0], x1[0], is1);
                        is1 = dp4a(w1[1], x1[1], is1);
                        is1 = dp4a(w1[2], x1[2], is1);
                        is1 = dp4a(w1[3], x1[3], is1);
                        acc[t] += d * b.xd[(size_t) t * nblk + ib] * (sc0 * (float) is0 + sc1 * (float) is1);
                    }
                }
            } else if constexpr (T == q::Q4_KR) {
                for (int ib = lane; ib < nblk; ib += SG) {
                    const DecodedU32 u = decode_unit_q8<q::Q4_KR>(rr, ib, tb);
                    #pragma unroll
                    for (int t = 0; t < VB; ++t) {
                        const int blk = t * nblk + ib;
                        const sycl::int4 * xv = (const sycl::int4 *) (b.xq + (size_t) t * b.cols + 32 * ib);
                        acc[t] += dot_decoded_q8<q::Q4_KR>(u, xv[0], xv[1], b.xd[blk], b.xsum[blk]);
                    }
                }
            }
            for (int t = 0; t < VB; ++t) {
                const float v = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
                if (lane == t) b.y[(size_t) t * b.ys + r] = b.cap > 0.0f ? b.cap * sycl::tanh(v / b.cap) : v;
            }
        });
    });
}

template <uint32_t T, int VB, bool STAGE = false>
sycl::event k_mv_batch(sycl::queue & q, const VMvJob * jobs, int njobs, int total_rows) {
    const size_t groups = (size_t) (total_rows + RPG - 1) / RPG;
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint64_t, 1> gl(STAGE ? grid64<T>() : 1, cgh);
        sycl::local_accessor<uint8_t, 1> ksl(STAGE ? 128 : 1, cgh);
        cgh.parallel_for(sycl::nd_range<1>(groups * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            q::Tabs tb;
            if constexpr (STAGE) tb = stage<T>(it, &gl[0], &ksl[0]);
            const auto sg = it.get_sub_group();
            const int row = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
            if (row >= total_rows) return;
            int j = 0;
            while (j + 1 < njobs && row >= jobs[j + 1].row0) ++j;
            const VMvJob & b = jobs[j];
            const int r = row - b.row0, lane = (int) sg.get_local_id()[0];
            float acc[VB];
            lane_dot_batch<T, VB>(row_of(b.w, (size_t) r), b.cols, b.x, b.x2, b.split, b.xs, lane, tb, acc);
            for (int t = 0; t < VB; ++t) {
                const float v = sycl::reduce_over_group(sg, acc[t], sycl::plus<float>());
                if (lane == t) b.y[(size_t) t * b.ys + r] = b.cap > 0.0f ? b.cap * sycl::tanh(v / b.cap) : v;
            }
        });
    });
}

template <uint32_t T, int VB>
sycl::event k_mv_batch_wide(sycl::queue & q, const VMvJob * jobs, int njobs, int total_rows) {
    return q.parallel_for(sycl::nd_range<1>((size_t) total_rows * WG, WG), [=](sycl::nd_item<1> it) {
        const auto grp = it.get_group();
        const int row = (int) it.get_group(0), l = (int) it.get_local_id(0);
        int j = 0;
        while (j + 1 < njobs && row >= jobs[j + 1].row0) ++j;
        const VMvJob & b = jobs[j];
        const int r = row - b.row0;
        const uint8_t * w = b.w.p + (size_t) r * b.w.rb;
        float acc[VB]{};
        for (int c = l; c < b.cols; c += WG) {
            const uint8_t * p = w + (size_t) c * q::shape(T).bytes;
            const float v = T == q::F32 ? sycl::bit_cast<float>(q::rd32(p)) : T == q::F16 ? q::f16(p) : q::bf16(p);
            const float * xb = c < b.split ? b.x + c : b.x2 + (c - b.split);
            for (int t = 0; t < VB; ++t) acc[t] += v * xb[(size_t) t * b.xs];
        }
        for (int t = 0; t < VB; ++t) {
            const float sum = sycl::reduce_over_group(grp, acc[t], sycl::plus<float>());
            if (l == 0) b.y[(size_t) t * b.ys + r] = b.cap > 0.0f ? b.cap * sycl::tanh(sum / b.cap) : sum;
        }
    });
}

struct VMixArgs {
    const float * st[12] = {};  // each slot: [VFAST][D], token stride D
    int ns = 0;
    const float * cur = nullptr;
    const float * qw = nullptr;
    int nq = 1;
    const float * normw[4] = {};
    float * out[4] = {};        // each target slot: [VFAST][D], token stride D
    float * outn[4] = {};
    float * push = nullptr;
    float eps = 1e-6f;
};

template <int VB>
sycl::event k_vmix(sycl::queue & q, const VMixArgs a) {
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> red(RPG * 13, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) VB * a.nq * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto grp = it.get_group();
            const auto sg = it.get_sub_group();
            const int sgid = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];
            const int g = (int) it.get_group(0), tok = g / a.nq, j = g % a.nq, l = (int) it.get_local_id(0);
            const size_t toff = (size_t) tok * D;
            float c[PER];
            if (a.cur) {
                float ss = 0.0f;
                for (int k = 0; k < PER; ++k) {
                    c[k] = a.cur[toff + l + k * WG];
                    ss += c[k] * c[k];
                }
                ss = sycl::reduce_over_group(grp, ss, sycl::plus<float>());
                const float cs = 1.0f / sycl::sqrt(ss / D + 1e-6f);
                for (int k = 0; k < PER; ++k) c[k] *= cs;
                if (a.push && j == 0)
                    for (int k = 0; k < PER; ++k) a.push[toff + l + k * WG] = c[k];
            } else {
                for (int k = 0; k < PER; ++k) c[k] = 0.0f;
            }
            const int n = a.ns + (a.cur ? 1 : 0);
            float sc[13], mx = 0.0f;
            if (n == 1) {
                sc[0] = 0.0f;
            } else {
                mx = -INFINITY;
                const float * qr = a.qw + (size_t) j * D;
                const float inv = 1.0f / sycl::sqrt((float) D);
                float qv[PER];
                for (int k = 0; k < PER; ++k) qv[k] = qr[l + k * WG];
                for (int s = 0; s < n; ++s) {
                    float acc = 0.0f;
                    for (int k = 0; k < PER; ++k) acc += (s < a.ns ? a.st[s][toff + l + k * WG] : c[k]) * qv[k];
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
            float o[PER], ss = 0.0f;
            for (int k = 0; k < PER; ++k) {
                const int i = l + k * WG;
                float acc = 0.0f;
                for (int s = 0; s < n; ++s) acc += sc[s] * (s < a.ns ? a.st[s][toff + i] : c[k]);
                o[k] = acc / sum;
                ss += o[k] * o[k];
            }
            ss = sycl::reduce_over_group(grp, ss, sycl::plus<float>());
            if (a.outn[j]) {
                const float r = 1.0f / sycl::sqrt(ss / D + 1e-6f);
                for (int k = 0; k < PER; ++k) a.outn[j][toff + l + k * WG] = o[k] * r;
            }
            if (a.out[j]) {
                const float r = a.normw[j] ? 1.0f / sycl::sqrt(ss / D + a.eps) : 1.0f;
                for (int k = 0; k < PER; ++k) {
                    const int i = l + k * WG;
                    a.out[j][toff + i] = o[k] * r * (a.normw[j] ? a.normw[j][i] : 1.0f);
                }
            }
        });
    });
}

struct VCtlArgs {
    const float * gain[4] = {}, * forget[4] = {}, * g[4] = {}, * cur[4] = {}, * msg[4] = {};
    float * out[4] = {};
    int n = 0, rank = 64;
};

template <int VB>
sycl::event k_vctl_up(sycl::queue & q, const VCtlArgs a) {
    return q.parallel_for(sycl::nd_range<1>((size_t) a.n * D / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int row = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
        const int c = row / D, i = row % D, lane = (int) sg.get_local_id()[0];
        const float * gw = a.gain[c] + (size_t) i * a.rank;
        const float * fw = a.forget[c] + (size_t) i * a.rank;
        float ga[VB]{}, fo[VB]{};
        for (int k = lane; k < a.rank; k += SG) {
            const float wg = gw[k], wf = fw[k];
            #pragma unroll
            for (int t = 0; t < VB; ++t) {
                const float gv = a.g[c][(size_t) t * a.rank + k];
                ga[t] += wg * gv;
                fo[t] += wf * gv;
            }
        }
        #pragma unroll
        for (int t = 0; t < VB; ++t) {
            const float gsum = sycl::reduce_over_group(sg, ga[t], sycl::plus<float>());
            const float fsum = sycl::reduce_over_group(sg, fo[t], sycl::plus<float>());
            if (lane == t) {
                const size_t off = (size_t) t * D + i;
                const float x = a.cur[c][off];
                a.out[c][off] = x - sycl::tanh(fsum) * x + sycl::tanh(gsum) * a.msg[c][off];
            }
        }
    });
}

struct VNormArgs {
    const float * x[3] = {}, * w[3] = {};
    float * y[3] = {};
    int n = 0;
    float eps = 1e-6f;
};

template <int VB>
sycl::event k_vnorm(sycl::queue & q, const VNormArgs a) {
    return q.parallel_for(sycl::nd_range<1>((size_t) VB * a.n * WG, WG), [=](sycl::nd_item<1> it) {
        const int g = (int) it.get_group(0), tok = g / a.n, j = g % a.n, l = (int) it.get_local_id(0);
        const size_t toff = (size_t) tok * D;
        float v[PER], ss = 0.0f;
        for (int k = 0; k < PER; ++k) {
            v[k] = a.x[j][toff + l + k * WG];
            ss += v[k] * v[k];
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.0f / sycl::sqrt(ss / D + a.eps);
        for (int k = 0; k < PER; ++k) a.y[j][toff + l + k * WG] = v[k] * r * a.w[j][l + k * WG];
    });
}

template <int VB>
sycl::event k_vembed(sycl::queue & q, const W & e, const int32_t * vtok, const float * nw, float * vxa, float eps) {
    const uint8_t * ep = e.p;
    const size_t rb = e.row_bytes;
    const uint32_t type = e.type;
    const q::BlockShape bs = q::shape(type);
    return q.parallel_for(sycl::nd_range<1>((size_t) VB * WG, WG), [=](sycl::nd_item<1> it) {
        const int t = (int) it.get_local_id(0), tk = (int) it.get_group(0);
        const uint8_t * row = ep + (size_t) vtok[tk] * rb;
        float v[32], ss = 0.0f;
        const bool mine = t < D / 32;
        if (mine) {
            const int c0 = t * 32;
            q::decode32(type, row + (size_t) (c0 / bs.elems) * bs.bytes, (c0 % bs.elems) / 32, v);
            for (int j = 0; j < 32; ++j) ss += v[j] * v[j];
        }
        ss = sycl::reduce_over_group(it.get_group(), ss, sycl::plus<float>());
        const float r = 1.0f / sycl::sqrt(ss / D + eps);
        if (mine)
            for (int j = 0; j < 32; ++j) vxa[(size_t) tk * D + t * 32 + j] = v[j] * r * nw[t * 32 + j];
    });
}

template <int VB>
sycl::event k_vtopk(sycl::queue & q, const float * logits, const float * pb, int ne, int nu, float ws, int32_t * ids, float * wts) {
    return q.parallel_for(sycl::nd_range<1>((size_t) VB * 256, 256), [=](sycl::nd_item<1> it) {
        const auto grp = it.get_group();
        const int tok = (int) it.get_group(0), e = (int) it.get_local_id(0);
        const float p = e < ne ? sigmoidf(logits[(size_t) tok * 256 + e]) : 0.0f;
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
            ids[(size_t) tok * nu + rank] = e;
            wts[(size_t) tok * nu + rank] = p / (sum + 1e-20f) * ws;
        }
    });
}

template <int VB>
sycl::event k_vgroup(sycl::queue & q, const int32_t * ids, const int32_t * ntokp, int ne, int nu, int32_t * pairs, int32_t * tiles) {
    const int wg = ne <= 128 ? 128 : 256;
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int32_t, 1> s_ids(VB * 16, cgh);
        cgh.parallel_for(sycl::nd_range<1>(wg, wg), [=](sycl::nd_item<1> it) {
            const auto grp = it.get_group();
            const int e = (int) it.get_local_id(0);
            const int np = (*ntokp) * nu;
            for (int i = e; i < np; i += wg) s_ids[i] = ids[i];
            sycl::group_barrier(grp);
            int32_t my_p[VB]{};
            int cnt = 0;
            if (e < ne) {
                for (int i = 0; i < np; ++i)
                    if (s_ids[i] == e) my_p[cnt++] = i;
            }
            const int n_tiles_e = (cnt + V_MMAX - 1) / V_MMAX;
            const int pair_off = sycl::exclusive_scan_over_group(grp, cnt, sycl::plus<int>());
            const int tile_off = sycl::exclusive_scan_over_group(grp, n_tiles_e, sycl::plus<int>());
            if (e < ne) {
                for (int j = 0; j < cnt; ++j) pairs[pair_off + j] = my_p[j];
                for (int k = 0; k < n_tiles_e; ++k) {
                    const int s = k * V_MMAX, tid = tile_off + k;
                    tiles[1 + 3 * tid + 0] = e;
                    tiles[1 + 3 * tid + 1] = pair_off + s;
                    tiles[1 + 3 * tid + 2] = sycl::min(V_MMAX, cnt - s);
                }
            }
            if (e == ne - 1) tiles[0] = tile_off + n_tiles_e;
        });
    });
}

template <int VB>
sycl::event k_vquant_x(sycl::queue & q, const float * x, int8_t * xq, float * xd, float * xs) {
    constexpr int nblk = VB * (D / 32);
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

template <int VB>
sycl::event k_vquant_act(sycl::queue & q, const float * act, const int32_t * ntokp, int nu, int cols, bool perm,
                         int8_t * xq, float * xd, float * xs) {
    const int max_nblk = VB * nu * (cols / 32);
    return q.parallel_for(sycl::nd_range<1>((size_t) (max_nblk + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const int b = (int) (it.get_group(0) * RPG + sg.get_group_id()[0]);
        const int nblk = (*ntokp) * nu * (cols / 32);
        if (b >= nblk) return;
        const int lane = (int) sg.get_local_id()[0];
        const float v0 = act[(size_t) 32 * b + 2 * lane], v1 = act[(size_t) 32 * b + 2 * lane + 1];
        const float amax = sycl::reduce_over_group(sg, sycl::fmax(sycl::fabs(v0), sycl::fabs(v1)), sycl::maximum<float>());
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        const int q0 = (int) sycl::round(v0 * id), q1 = (int) sycl::round(v1 * id);
        const int qs = sycl::reduce_over_group(sg, q0 + q1, sycl::plus<int>());
        auto pos = [&](int v) { return perm ? (v / 16) * 16 + (v % 4) * 4 + (v % 16) / 4 : v; };
        xq[(size_t) 32 * b + pos(2 * lane)] = (int8_t) q0;
        xq[(size_t) 32 * b + pos(2 * lane + 1)] = (int8_t) q1;
        if (lane == 0) {
            xd[b] = d;
            xs[b] = (float) qs;
        }
    });
}

struct VGateUpArgs {
    WRef w;
    const int8_t * xq;
    const float * xd, * xs;
    const float * f;
    int cols, rows_e, nff, nu;
    const float * scale;
    const int32_t * pairs, * tiles;
    float * act;
    float beta, lb;
};

template <uint32_t T, int GR, int M>
inline void vgate_up_run(const VGateUpArgs & a, const sycl::sub_group & sg, const q::Tabs & tb,
                         int e, int i0, int first, int half, int l8, int lane) {
    int pr[M], tok[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        pr[m] = a.pairs[first + m];
        tok[m] = pr[m] / a.nu;
    }
    const size_t r0 = (size_t) e * a.rows_e + (half ? a.nff : 0) + i0;
    RowRef rows[GR];
    #pragma unroll
    for (int r = 0; r < GR; ++r) rows[r] = row_of(a.w, r0 + r);
    float acc[GR][M]{};
    const int nblk = a.cols / 32;
    for (int ib = l8; ib < nblk; ib += 8) {
        if constexpr (M == 1) {
            const int blk = tok[0] * nblk + ib;
            const sycl::int4 * xv = (const sycl::int4 *) (a.xq + (size_t) tok[0] * a.cols + 32 * ib);
            const sycl::int4 x0 = xv[0], x1 = xv[1];
            const float xdb = a.xd[blk], xsb = a.xs[blk];
            #pragma unroll
            for (int r = 0; r < GR; ++r) {
                if constexpr (T == q::NVFP4R) {
                    acc[r][0] += xdb * nvfp4_q8_dot(rows[r], ib, x0, x1);
                } else {
                    const DecodedU32 u = decode_unit_q8<T>(rows[r], ib, tb);
                    acc[r][0] += dot_decoded_q8<T>(u, x0, x1, xdb, xsb);
                }
            }
        } else {
            #pragma unroll
            for (int r = 0; r < GR; ++r) {
                const DecodedU32 u = decode_unit_q8<T>(rows[r], ib, tb);
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const int blk = tok[m] * nblk + ib;
                    const sycl::int4 * xv = (const sycl::int4 *) (a.xq + (size_t) tok[m] * a.cols + 32 * ib);
                    acc[r][m] += dot_decoded_q8<T>(u, xv[0], xv[1], a.xd[blk], a.xs[blk]);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < GR; ++r)
        #pragma unroll
        for (int m = 0; m < M; ++m)
            for (int off = 4; off > 0; off /= 2) acc[r][m] += sycl::permute_group_by_xor(sg, acc[r][m], off);
    const float s = a.scale ? a.scale[e] : 1.0f;
    #pragma unroll
    for (int r = 0; r < GR; ++r) {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const float gg = sycl::select_from_group(sg, acc[r][m], 0) * s;
            const float uu = sycl::select_from_group(sg, acc[r][m], 8) * s;
            if (lane == r * M + m) {
                const float g = a.beta * sycl::tanh(gg / a.beta) * sigmoidf(gg);
                a.act[(size_t) pr[m] * a.nff + i0 + r] = g * (a.lb > 0.0f ? a.lb * sycl::tanh(uu / a.lb) : uu);
            }
        }
    }
}

template <uint32_t T, int VB, bool STAGE = false>
sycl::event k_vgate_up_r(sycl::queue & q, const VGateUpArgs a) {
    constexpr int GR = INK_VGUR;
    const size_t nsg = (size_t) (VB * a.nu) * (a.nff / GR);
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint64_t, 1> gl(STAGE ? grid64<T>() : 1, cgh);
        sycl::local_accessor<uint8_t, 1> ksl(STAGE ? 128 : 1, cgh);
        cgh.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            q::Tabs tb;
            if constexpr (STAGE) tb = stage<T>(it, &gl[0], &ksl[0]);
            const auto sg = it.get_sub_group();
            const size_t gid = it.get_group(0) * RPG + sg.get_group_id()[0];
            if (gid >= nsg) return;
            const int nsg_row = a.nff / GR;
            const int tile = (int) (gid / nsg_row), i0 = (int) ((gid % nsg_row) * GR);
            if (tile >= a.tiles[0]) return;
            const int lane = (int) sg.get_local_id()[0], half = lane / 8, l8 = lane % 8;
            const int e = a.tiles[1 + 3 * tile], first = a.tiles[2 + 3 * tile], nb = a.tiles[3 + 3 * tile];
            switch (nb) {
                case 1: vgate_up_run<T, GR, 1>(a, sg, tb, e, i0, first, half, l8, lane); break;
                case 2: vgate_up_run<T, GR, 2>(a, sg, tb, e, i0, first, half, l8, lane); break;
                case 3: vgate_up_run<T, GR, 3>(a, sg, tb, e, i0, first, half, l8, lane); break;
                default: vgate_up_run<T, GR, 4>(a, sg, tb, e, i0, first, half, l8, lane); break;
            }
        });
    });
}

template <uint32_t T, int M>
inline void vgate_up_fp_run(const VGateUpArgs & a, const sycl::sub_group & sg, const q::Tabs & tb,
                            int e, int i, int first, int lane) {
    int pr[M];
    const float * xr[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        pr[m] = a.pairs[first + m];
        xr[m] = a.f + (size_t) (pr[m] / a.nu) * D;
    }
    const size_t r0 = (size_t) e * a.rows_e;
    const RowRef rg = row_of(a.w, r0 + i), ru = row_of(a.w, r0 + a.nff + i);
    float ag[M]{}, au[M]{};
    if constexpr (T == q::Q8_0R) {
        for (int c = lane * 16; c < a.cols; c += SG * 16) {
            const sycl::int4 wg = *(const sycl::int4 *) (rg.p + c);
            const sycl::int4 wu = *(const sycl::int4 *) (ru.p + c);
            const float dg = q::f16(rg.a + (c / 32) * 2), du = q::f16(ru.a + (c / 32) * 2);
            float sg_acc[M]{}, su_acc[M]{};
            for (int k = 0; k < 4; ++k) {
                const int vg = wg[k], vu = wu[k];
                const float g0 = (float) (int8_t) vg, g1 = (float) (int8_t) (vg >> 8), g2 = (float) (int8_t) (vg >> 16), g3 = (float) (int8_t) (vg >> 24);
                const float u0 = (float) (int8_t) vu, u1 = (float) (int8_t) (vu >> 8), u2 = (float) (int8_t) (vu >> 16), u3 = (float) (int8_t) (vu >> 24);
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const sycl::float4 xv = *(const sycl::float4 *) (xr[m] + c + 4 * k);
                    sg_acc[m] += g0 * xv[0] + g1 * xv[1] + g2 * xv[2] + g3 * xv[3];
                    su_acc[m] += u0 * xv[0] + u1 * xv[1] + u2 * xv[2] + u3 * xv[3];
                }
            }
            #pragma unroll
            for (int m = 0; m < M; ++m) {
                ag[m] += sg_acc[m] * dg;
                au[m] += su_acc[m] * du;
            }
        }
    } else {
        constexpr int V = step_vals<T>();
        for (int s = lane; s < a.cols / V; s += SG) {
            step_decode<T>(rg, s, [&](int c, float v) {
                #pragma unroll
                for (int m = 0; m < M; ++m) ag[m] += v * xr[m][c];
            }, tb);
            step_decode<T>(ru, s, [&](int c, float v) {
                #pragma unroll
                for (int m = 0; m < M; ++m) au[m] += v * xr[m][c];
            }, tb);
        }
    }
    const float sc = a.scale ? a.scale[e] : 1.0f;
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        const float gg = sycl::reduce_over_group(sg, ag[m], sycl::plus<float>()) * sc;
        const float uu = sycl::reduce_over_group(sg, au[m], sycl::plus<float>()) * sc;
        if (lane == m) {
            const float g = a.beta * sycl::tanh(gg / a.beta) * sigmoidf(gg);
            a.act[(size_t) pr[m] * a.nff + i] = g * (a.lb > 0.0f ? a.lb * sycl::tanh(uu / a.lb) : uu);
        }
    }
}

template <uint32_t T, int VB, bool STAGE = false>
sycl::event k_vgate_up_fp(sycl::queue & q, const VGateUpArgs a) {
    const size_t nsg = (size_t) (VB * a.nu) * a.nff;
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint64_t, 1> gl(STAGE ? grid64<T>() : 1, cgh);
        sycl::local_accessor<uint8_t, 1> ksl(STAGE ? 128 : 1, cgh);
        cgh.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            q::Tabs tb;
            if constexpr (STAGE) tb = stage<T>(it, &gl[0], &ksl[0]);
            const auto sg = it.get_sub_group();
            const size_t gid = it.get_group(0) * RPG + sg.get_group_id()[0];
            if (gid >= nsg) return;
            const int tile = (int) (gid / a.nff), i = (int) (gid % a.nff), lane = (int) sg.get_local_id()[0];
            if (tile >= a.tiles[0]) return;
            const int e = a.tiles[1 + 3 * tile], first = a.tiles[2 + 3 * tile], nb = a.tiles[3 + 3 * tile];
            switch (nb) {
                case 1: vgate_up_fp_run<T, 1>(a, sg, tb, e, i, first, lane); break;
                case 2: vgate_up_fp_run<T, 2>(a, sg, tb, e, i, first, lane); break;
                case 3: vgate_up_fp_run<T, 3>(a, sg, tb, e, i, first, lane); break;
                default: vgate_up_fp_run<T, 4>(a, sg, tb, e, i, first, lane); break;
            }
        });
    });
}

struct VDownQ8Args {
    WRef w;
    const int8_t * xq;
    const float * xd, * xs;
    const float * act;
    int cols, nu;
    const float * wts, * scale;
    const int32_t * pairs, * tiles;
    float * out;
};

template <uint32_t T, int RD, int M>
inline void vdown_q8_run(const VDownQ8Args & a, const sycl::sub_group & sg, const q::Tabs & tb,
                         int e, int i0, int first, int lane) {
    int pr[M], tok[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        pr[m] = a.pairs[first + m];
        tok[m] = pr[m] / a.nu;
    }
    RowRef rows[RD];
    #pragma unroll
    for (int r = 0; r < RD; ++r) rows[r] = row_of(a.w, (size_t) e * D + i0 + r);
    float acc[RD][M]{};
    const int nblk = a.cols / 32;
    for (int ib = lane; ib < nblk; ib += SG) {
        if constexpr (M == 1) {
            const int blk = pr[0] * nblk + ib;
            const sycl::int4 * xv = (const sycl::int4 *) (a.xq + (size_t) pr[0] * a.cols + 32 * ib);
            const sycl::int4 x0 = xv[0], x1 = xv[1];
            const float xdb = a.xd[blk], xsb = a.xs[blk];
            #pragma unroll
            for (int r = 0; r < RD; ++r) {
                if constexpr (T == q::NVFP4R) {
                    acc[r][0] += xdb * nvfp4_q8_dot(rows[r], ib, x0, x1);
                } else {
                    const DecodedU32 u = decode_unit_q8<T>(rows[r], ib, tb);
                    acc[r][0] += dot_decoded_q8<T>(u, x0, x1, xdb, xsb);
                }
            }
        } else {
            #pragma unroll
            for (int r = 0; r < RD; ++r) {
                const DecodedU32 u = decode_unit_q8<T>(rows[r], ib, tb);
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const int blk = pr[m] * nblk + ib;
                    const sycl::int4 * xv = (const sycl::int4 *) (a.xq + (size_t) pr[m] * a.cols + 32 * ib);
                    acc[r][m] += dot_decoded_q8<T>(u, xv[0], xv[1], a.xd[blk], a.xs[blk]);
                }
            }
        }
    }
    const float es = a.scale ? a.scale[e] : 1.0f;
    #pragma unroll
    for (int r = 0; r < RD; ++r) {
        #pragma unroll
        for (int m = 0; m < M; ++m) {
            const float y = sycl::reduce_over_group(sg, acc[r][m], sycl::plus<float>());
            if (lane == r * M + m)
                sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(
                    a.out[(size_t) tok[m] * D + i0 + r]).fetch_add(y * a.wts[pr[m]] * es);
        }
    }
}

template <uint32_t T, int VB>
sycl::event k_vdown_q8(sycl::queue & q, const VDownQ8Args a) {
    constexpr int RD = INK_VDR;
    const size_t nsg = (size_t) (VB * a.nu) * (D / RD);
    const q::Tabs tb;
    return q.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
        const auto sg = it.get_sub_group();
        const size_t gid = it.get_group(0) * RPG + sg.get_group_id()[0];
        if (gid >= nsg) return;
        const int nsg_row = D / RD;
        const int tile = (int) (gid / nsg_row), i0 = (int) ((gid % nsg_row) * RD);
        if (tile >= a.tiles[0]) return;
        const int lane = (int) sg.get_local_id()[0];
        const int e = a.tiles[1 + 3 * tile], first = a.tiles[2 + 3 * tile], nb = a.tiles[3 + 3 * tile];
        switch (nb) {
            case 1: vdown_q8_run<T, RD, 1>(a, sg, tb, e, i0, first, lane); break;
            case 2: vdown_q8_run<T, RD, 2>(a, sg, tb, e, i0, first, lane); break;
            case 3: vdown_q8_run<T, RD, 3>(a, sg, tb, e, i0, first, lane); break;
            default: vdown_q8_run<T, RD, 4>(a, sg, tb, e, i0, first, lane); break;
        }
    });
}

template <uint32_t T, int M>
inline void vdown_fp_run(const VDownQ8Args & a, const sycl::sub_group & sg, const q::Tabs & tb,
                         int e, int i, int first, int lane) {
    int pr[M];
    const float * xr[M];
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        pr[m] = a.pairs[first + m];
        xr[m] = a.act + (size_t) pr[m] * a.cols;
    }
    const RowRef rr = row_of(a.w, (size_t) e * D + i);
    float acc[M]{};
    if constexpr (T == q::Q8_0R) {
        for (int c = lane * 16; c < a.cols; c += SG * 16) {
            const sycl::int4 wv = *(const sycl::int4 *) (rr.p + c);
            const float d = q::f16(rr.a + (c / 32) * 2);
            float s[M]{};
            for (int k = 0; k < 4; ++k) {
                const int w = wv[k];
                const float w0 = (float) (int8_t) w, w1 = (float) (int8_t) (w >> 8), w2 = (float) (int8_t) (w >> 16), w3 = (float) (int8_t) (w >> 24);
                #pragma unroll
                for (int m = 0; m < M; ++m) {
                    const sycl::float4 xv = *(const sycl::float4 *) (xr[m] + c + 4 * k);
                    s[m] += w0 * xv[0] + w1 * xv[1] + w2 * xv[2] + w3 * xv[3];
                }
            }
            #pragma unroll
            for (int m = 0; m < M; ++m) acc[m] += s[m] * d;
        }
    } else {
        constexpr int V = step_vals<T>();
        for (int s = lane; s < a.cols / V; s += SG)
            step_decode<T>(rr, s, [&](int c, float v) {
                #pragma unroll
                for (int m = 0; m < M; ++m) acc[m] += v * xr[m][c];
            }, tb);
    }
    const float sc = a.scale ? a.scale[e] : 1.0f;
    #pragma unroll
    for (int m = 0; m < M; ++m) {
        const float y = sycl::reduce_over_group(sg, acc[m], sycl::plus<float>());
        if (lane == m)
            sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device, sycl::access::address_space::global_space>(
                a.out[(size_t) (pr[m] / a.nu) * D + i]).fetch_add(y * a.wts[pr[m]] * sc);
    }
}

template <uint32_t T, int VB, bool STAGE = false>
sycl::event k_vdown_fp(sycl::queue & q, const VDownQ8Args a) {
    const size_t nsg = (size_t) (VB * a.nu) * D;
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint64_t, 1> gl(STAGE ? grid64<T>() : 1, cgh);
        sycl::local_accessor<uint8_t, 1> ksl(STAGE ? 128 : 1, cgh);
        cgh.parallel_for(sycl::nd_range<1>((nsg + RPG - 1) / RPG * WG, WG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            q::Tabs tb;
            if constexpr (STAGE) tb = stage<T>(it, &gl[0], &ksl[0]);
            const auto sg = it.get_sub_group();
            const size_t gid = it.get_group(0) * RPG + sg.get_group_id()[0];
            if (gid >= nsg) return;
            const int tile = (int) (gid / D), i = (int) (gid % D), lane = (int) sg.get_local_id()[0];
            if (tile >= a.tiles[0]) return;
            const int e = a.tiles[1 + 3 * tile], first = a.tiles[2 + 3 * tile], nb = a.tiles[3 + 3 * tile];
            switch (nb) {
                case 1: vdown_fp_run<T, 1>(a, sg, tb, e, i, first, lane); break;
                case 2: vdown_fp_run<T, 2>(a, sg, tb, e, i, first, lane); break;
                case 3: vdown_fp_run<T, 3>(a, sg, tb, e, i, first, lane); break;
                default: vdown_fp_run<T, 4>(a, sg, tb, e, i, first, lane); break;
            }
        });
    });
}

template <int VB>
sycl::event k_vqkv_post(sycl::queue & q, const float * proj, const float * qclip, const float * rclip,
                        const float * kconv, const float * vconv, const float * kdyn_off, const float * vdyn_off,
                        const float * hist, int8_t * kc, int8_t * vc, sycl::half * ks, sycl::half * vs,
                        float * qf, float * rf, float * vraw_slot, const int32_t * pos0p, const int32_t * ntokp,
                        int nh, int nkv, int n_rel, int ring, int kvlen, int hc,
                        float qk_clip, float rel_clip, float alpha, float floor_, bool full, bool kv4) {
    (void) kdyn_off;
    (void) vdyn_off;
    constexpr size_t PJ = 8704;
    const int kvw = nkv * HD;
    const int groups = nh + 2 * nkv + nh;
    const float cnorm = 1.0f / std::sqrt((float) (4 * (full ? 2 : 1) + 1));
    return q.parallel_for(sycl::nd_range<1>((size_t) VB * groups * HD, HD), [=](sycl::nd_item<1> it) {
        const auto grp = it.get_group();
        const int gg = (int) it.get_group(0), t = (int) it.get_local_id(0);
        const int tok = gg / groups, g = gg % groups;
        if (tok >= *ntokp) return;
        const int pos = *pos0p + tok;
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
            const size_t off = is_v ? 4096 : 3072;
            float xx[4];
            for (int j = 0; j < 4; ++j) {
                const int tt = tok - 3 + j;
                xx[j] = tt >= 0 ? proj[(size_t) tt * PJ + off + c] : hist[ch * 3 + 3 + tt];
            }
            vraw_slot[(size_t) tok * hc + ch] = xx[3];
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
    });
}

struct VAttnArgs {
    const float * qf, * rf;
    const int8_t * kc, * vc;
    const sycl::half * ksc, * vsc;
    const float * rb;
    const int32_t * pos0p, * ntokp;
    float * part;
    int nh, nkv, ratio, n_rel, extent, n_swa, ring, kvlen, ch, nc, pstride;
    bool swa, kv4;
};

template <int AWG, int VB, int RATIO>
sycl::event k_vattn_r(sycl::queue & q, const VAttnArgs a) {
    constexpr int R = VB * RATIO;
    constexpr int NSG = AWG / SG;
    constexpr int SPU = NSG / VB;
    constexpr int DPL = HD / (SG * SPU);
    static_assert(SPU >= 1 && (SG * SPU * DPL == HD), "k_vattn_r geometry mismatch");
    static_assert(NSG <= SG, "k_vattn_r softmax reduction expects NSG <= SG");
    const float scale = 1.0f / std::sqrt((float) HD);
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<int32_t, 1> sq8((size_t) R * (HD / 4), cgh);
        sycl::local_accessor<float, 1> sqd((size_t) R * (HD / 32), cgh), sqs((size_t) R * (HD / 32), cgh);
        sycl::local_accessor<float, 1> sr((size_t) R * 16, cgh);
        sycl::local_accessor<sycl::half, 1> sp((size_t) R * a.ch, cgh);
        sycl::local_accessor<float, 1> sm(R, cgh), sred((size_t) 2 * NSG * R, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) a.nkv * a.nc * AWG, AWG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG)]] {
            const auto grp = it.get_group();
            const int g = (int) it.get_group(0), t = (int) it.get_local_id(0);
            const int nh = a.nh, ch = a.ch, nc = a.nc, pstride = a.pstride;
            const int n_rel = a.n_rel, extent = a.extent, n_swa = a.n_swa, ring = a.ring, kvlen = a.kvlen;
            const bool swa = a.swa, kv4 = a.kv4;
            const int kvh = g / nc, c = g % nc, h0 = kvh * RATIO;
            const int pos0 = *a.pos0p, ntok = *a.ntokp, p_last = pos0 + ntok - 1;
            const int lo = swa ? sycl::max(0, pos0 - n_swa + 1) : 0;
            const int k0 = lo + c * ch;
            if (k0 > p_last) {
                for (int u = 0; u < ntok; ++u)
                    #pragma unroll
                    for (int j = 0; j < RATIO; ++j) {
                        float * out = a.part + ((size_t) (u * nh + h0 + j) * pstride + c) * (HD + 2);
                        if (t == 0) {
                            out[0] = -INFINITY;
                            out[1] = 0.0f;
                        }
                        if (t < HD) out[2 + t] = 0.0f;
                    }
                return;
            }
            const int R_act = ntok * RATIO;
            for (int i = t; i < R_act * n_rel; i += AWG) {
                const int r = i / n_rel, rr = i % n_rel, u = r / RATIO, j = r % RATIO;
                sr[r * 16 + rr] = a.rf[((size_t) u * nh + h0 + j) * n_rel + rr];
            }
            {
                const auto sgq = it.get_sub_group();
                const int sgid = (int) sgq.get_group_id()[0], lane = (int) sgq.get_local_id()[0];
                for (int qb = sgid; qb < R_act * (HD / 32); qb += NSG) {
                    const int r = qb / (HD / 32), b = qb % (HD / 32), u = r / RATIO, j = r % RATIO;
                    const float * qrow = a.qf + ((size_t) u * nh + h0 + j) * HD + b * 32;
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
                    const int idx = ring ? k % ring : k;
                    const size_t kbase = ((size_t) kvh * kvlen + idx) * HD;
                    float dot[R]{};
                    for (int b = 0; b < HD / 32; ++b) {
                        const float ds = (float) a.ksc[kbase / 32 + b];
                        int32_t kw[8];
                        if (kv4) {
                            const sycl::uint4 w4 = *(const sycl::uint4 *) ((const uint8_t *) a.kc + (kbase + 32 * b) / 2);
                            for (int w = 0; w < 4; ++w) {
                                kw[2 * w] = (int32_t) (w4[w] & 0x0F0F0F0Fu);
                                kw[2 * w + 1] = (int32_t) ((w4[w] >> 4) & 0x0F0F0F0Fu);
                            }
                        } else {
                            const sycl::int4 * p4 = (const sycl::int4 *) (a.kc + kbase + 32 * b);
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
                    for (int u = 0; u < VB; ++u) {
                        if (u < ntok) {
                            const int p = pos0 + u, dist = p - k;
                            if (k <= p && (!swa || dist < n_swa)) {
                                #pragma unroll
                                for (int j = 0; j < RATIO; ++j) sc[u * RATIO + j] = dot[u * RATIO + j] * scale;
                                if (dist < extent) {
                                    for (int rr = 0; rr < n_rel; ++rr) {
                                        const float bb = a.rb[rr + n_rel * dist];
                                        #pragma unroll
                                        for (int j = 0; j < RATIO; ++j) {
                                            const int r = u * RATIO + j;
                                            sc[r] += bb * sr[r * 16 + rr];
                                        }
                                    }
                                }
                            }
                        }
                    }
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
                    const int p = pos0 + u;
                    const int nk = sycl::clamp(sycl::min(ch, p - k0 + 1), 0, ch);
                    float acc[RATIO][DPL]{};
                    for (int jj = 0; jj < nk; ++jj) {
                        const int kk = k0 + jj;
                        const int idx = ring ? kk % ring : kk;
                        const size_t vi = ((size_t) kvh * kvlen + idx) * HD + d_base;
                        float vvals[DPL];
                        if constexpr (DPL == 1) {
                            if (kv4) {
                                const uint8_t by = ((const uint8_t *) a.vc)[vi / 2];
                                vvals[0] = (float) ((int) ((d_base & 1) ? (by >> 4) : (by & 15)) - 8);
                            } else {
                                vvals[0] = (float) a.vc[vi];
                            }
                        } else if constexpr (DPL == 2) {
                            if (kv4) {
                                const uint8_t by = ((const uint8_t *) a.vc)[vi / 2];
                                vvals[0] = (float) ((int) (by & 15) - 8);
                                vvals[1] = (float) ((int) (by >> 4) - 8);
                            } else {
                                const sycl::vec<int8_t, 2> v2 = *(const sycl::vec<int8_t, 2> *) (a.vc + vi);
                                vvals[0] = (float) v2[0];
                                vvals[1] = (float) v2[1];
                            }
                        } else {
                            kv_get<DPL>(a.vc, kv4, vi, vvals);
                        }
                        const float vd = (float) a.vsc[vi / 32];
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
                        float * out = a.part + ((size_t) (u * nh + h0 + j) * pstride + c) * (HD + 2);
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

template <int AWG, int VB>
sycl::event k_vattn(sycl::queue & q, const VAttnArgs a) {
    if (a.ratio == 3) return k_vattn_r<AWG, VB, 3>(q, a);
    if (a.ratio == 2) return k_vattn_r<AWG, VB, 2>(q, a);
    if (a.ratio == 1) return k_vattn_r<AWG, VB, 1>(q, a);
    return k_vattn_r<AWG, VB, 4>(q, a);
}

template <int VB>
sycl::event k_vcombine(sycl::queue & q, const float * part, const float * proj, float * att, const int32_t * ntokp,
                       int nh, int nc, int pstride) {
    constexpr int CG = 8;
    constexpr size_t PJ = 8704;
    return q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sacc(CG * HD, cgh), sls(CG, cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) VB * nh * CG * HD, CG * HD), [=](sycl::nd_item<1> it) {
            const auto grp = it.get_group();
            const int g = (int) it.get_group(0), u = g / nh, hh = g % nh;
            if (u >= *ntokp) return;
            const int t = (int) it.get_local_id(0), d = t % HD, gq = t / HD;
            const float * base = part + ((size_t) u * nh + hh) * pstride * (HD + 2);
            const float * pr = proj + (size_t) u * PJ;
            const float sk = pr[5504 + hh];
            float ml = -INFINITY;
            for (int c = t; c < nc; c += CG * HD) ml = sycl::fmax(ml, base[c * (HD + 2)]);
            const float M = sycl::fmax(sk, sycl::reduce_over_group(grp, ml, sycl::maximum<float>()));
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
                float o = 0.0f, L = sycl::exp(sk - M);
                for (int g2 = 0; g2 < CG; ++g2) {
                    o += sacc[g2 * HD + t];
                    L += sls[g2];
                }
                att[(size_t) u * nh * HD + hh * HD + t] = o / L * sigmoidf(pr[5528 + hh * HD + t]);
            }
        });
    });
}

template <int VB>
sycl::event k_vconv_res(sycl::queue & q, const float * raw, const float * w, const float * hist, const float * res,
                        float * out, float * vraw_slot_off, const int32_t * ntokp, int hc, bool drop) {
    return q.parallel_for(sycl::range<2>((size_t) VB, (size_t) D), [=](sycl::item<2> it) {
        const int tok = (int) it[0], i = (int) it[1];
        if (tok >= *ntokp) return;
        float xx[4];
        for (int j = 0; j < 4; ++j) {
            const int tt = tok - 3 + j;
            xx[j] = tt >= 0 ? raw[(size_t) tt * D + i] : hist[i * 3 + 3 + tt];
        }
        vraw_slot_off[(size_t) tok * hc + i] = xx[3];
        float cv = 0.0f;
        for (int j = 0; j < 4; ++j) cv += w[j + 4 * i] * xx[j];
        const size_t o = (size_t) tok * D + i;
        out[o] = (drop ? 0.0f : res[o]) + (xx[3] + cv) * 0.44721359549995793f;
    });
}

template <int VB>
void k_vargmax(sycl::queue & q, const float * logits, int n_vocab, float * tmp_v, int32_t * tmp_i, int32_t * out_idx) {
    constexpr int NG = WG;
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sv(RPG, cgh);
        sycl::local_accessor<int32_t, 1> si(RPG, cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(VB, (size_t) NG * WG), sycl::range<2>(1, WG)),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int tok = (int) it.get_global_id(0);
            const float * x = logits + (size_t) tok * n_vocab;
            float bv = -INFINITY;
            int32_t bi = 1 << 30;
            for (int i = (int) it.get_global_id(1); i < n_vocab; i += NG * WG) {
                const float v = x[i];
                if (v > bv || (v == bv && i < bi)) {
                    bv = v;
                    bi = i;
                }
            }
            wg_argmax(it, &sv[0], &si[0], bv, bi);
            if (it.get_local_id(1) == 0) {
                const size_t g = (size_t) tok * NG + it.get_group(1);
                tmp_v[g] = bv;
                tmp_i[g] = bi;
            }
        });
    });
    q.submit([&](sycl::handler & cgh) {
        sycl::local_accessor<float, 1> sv(RPG, cgh);
        sycl::local_accessor<int32_t, 1> si(RPG, cgh);
        cgh.parallel_for(sycl::nd_range<2>(sycl::range<2>(VB, WG), sycl::range<2>(1, WG)),
                         [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(SG)]] {
            const int tok = (int) it.get_global_id(0), l = (int) it.get_local_id(1);
            const size_t off = (size_t) tok * NG + l;
            float bv = tmp_v[off];
            int32_t bi = tmp_i[off];
            wg_argmax(it, &sv[0], &si[0], bv, bi);
            if (l == 0) out_idx[tok] = bi;
        });
    });
}

}  // namespace

void FastEngine::ensure_vraw() {
    if (vraw_) return;
    const Hparams & h = hp_;
    vraw_ = dev<float>((size_t) h.n_slot() * VMAX * h.conv_channels());
    vlogits_ = dev<float>((size_t) VMAX * h.n_vocab);
    vhist_ = sycl::malloc_device<float *>((size_t) h.n_slot(), q_);
    allocs_.push_back(vhist_);
    std::vector<float *> hp(hist_.begin(), hist_.end());
    q_.memcpy(vhist_, hp.data(), hp.size() * sizeof(float *)).wait();
}

void FastEngine::init_verify() {
    if (vready_) return;
    ensure_vraw();
    const Hparams & h = hp_;
    const int swa_ch = std::min(swa_chunk, INK_SWA_AWG);
    max_vchunks_ = std::max((n_ctx_ + FCH - 1) / FCH, (std::min(h.n_swa + VFAST - 1, n_ctx_) + swa_ch - 1) / swa_ch);

    vtok_ = dev<int32_t>(VFAST);
    vpos0_ = dev<int32_t>(1);
    vntok_ = dev<int32_t>(1);

    const size_t VD = (size_t) VFAST * D;
    vxa_ = dev<float>(VD);
    vxb_ = dev<float>(VD);
    vprelude_ = dev<float>(VD);
    vstore_ = dev<float>((size_t) std::max(1, m_.schedule.n_vectors) * VD);
    vin_ = dev<float>(3 * VD);
    vinn_ = dev<float>(3 * VD);
    vmsg_ = dev<float>(4 * VD);
    vmsgn_ = dev<float>(4 * VD);
    vctl_ = dev<float>(3 * VD);
    vnin_ = dev<float>(3 * VD);
    vg_ = dev<float>((size_t) 4 * VFAST * 64);
    vproj_ = dev<float>((size_t) VFAST * 8704);
    vqf_ = dev<float>((size_t) VFAST * h.n_head * HD);
    vrf_ = dev<float>((size_t) VFAST * h.n_head * h.n_rel);
    vatt_ = dev<float>((size_t) VFAST * h.n_head * HD);
    vf_ = dev<float>(VD);
    vfraw_ = dev<float>(VD);
    vfn_ = dev<float>(VD);
    vo_ = dev<float>(VD);
    vfq_ = dev<int8_t>(VD);
    vfqd_ = dev<float>(VD / 32);
    vfqs_ = dev<float>(VD / 32);
    for (int k = 0; k < 3; ++k) {
        vmxq_[k] = dev<int8_t>((size_t) VFAST * 3072);
        vmxqd_[k] = dev<float>((size_t) VFAST * 96);
        vmxqs_[k] = dev<float>((size_t) VFAST * 96);
    }
    vact_ = dev<float>((size_t) VFAST * h.n_used * h.n_ff);
    vaq_ = dev<int8_t>((size_t) VFAST * h.n_used * h.n_ff);
    vaqd_ = dev<float>((size_t) VFAST * h.n_used * h.n_ff / 32);
    vaqs_ = dev<float>((size_t) VFAST * h.n_used * h.n_ff / 32);
    vrouter_ = dev<float>((size_t) VFAST * 256);
    vwts_ = dev<float>((size_t) VFAST * h.n_used);
    vids_ = dev<int32_t>((size_t) VFAST * h.n_used);
    vpairs_ = dev<int32_t>((size_t) VFAST * h.n_used);
    vtiles_ = dev<int32_t>(1 + 3 * (size_t) VFAST * h.n_used);

    const size_t need_vpart = (size_t) VFAST * h.n_head * max_vchunks_ * (HD + 2);
    const size_t pool_floats = (size_t) 23 * TB_ * D;
    vpart_ = (bin_ && need_vpart <= pool_floats) ? bin_ : dev<float>(need_vpart);

    vargmax_val_ = dev<float>((size_t) VFAST * WG);
    vargmax_idx_ = dev<int32_t>((size_t) VFAST * WG);
    vargmax_out_ = dev<int32_t>(VFAST);
    q_.wait();
    vready_ = true;
}

VMvJob * FastEngine::vjobs(const std::vector<VMvJob> & js) {
    if (vjob_next_ < vjob_cache_.size()) return vjob_cache_[vjob_next_++];
    sycl::queue setup(q_.get_context(), q_.get_device());
    VMvJob * d = sycl::malloc_device<VMvJob>(js.size(), setup);
    setup.memcpy(d, js.data(), js.size() * sizeof(VMvJob)).wait();
    allocs_.push_back(d);
    vjob_cache_.push_back(d);
    ++vjob_next_;
    return d;
}

std::vector<VMvGroup> FastEngine::plan_vmv(const std::vector<VMvJob> & js) {
    std::vector<VMvGroup> plan;
    std::vector<uint32_t> order;
    for (const VMvJob & j : js)
        if (std::find(order.begin(), order.end(), j.type) == order.end()) order.push_back(j.type);
    for (uint32_t t : order) {
        std::vector<VMvJob> g;
        int rows = 0;
        VMvGroup gr;
        gr.type = t;
        bool q8_ok = (t == q::Q8_0R || t == q::Q6_KR || t == q::Q4_KR);
        for (VMvJob j : js) {
            if (j.type != t) continue;
            j.row0 = rows;
            rows += j.rows;
            if (q8_ok) {
                if (j.split < j.cols || j.cols % 32 != 0 || j.cols > 3072) {
                    q8_ok = false;
                } else {
                    int idx = -1;
                    for (int k = 0; k < gr.nx; ++k)
                        if (gr.ux[k] == j.x && gr.uxs[k] == j.xs && gr.ucols[k] == j.cols) {
                            idx = k;
                            break;
                        }
                    if (idx < 0) {
                        if (gr.nx >= 3) q8_ok = false;
                        else {
                            idx = gr.nx++;
                            gr.ux[idx] = j.x;
                            gr.uxs[idx] = j.xs;
                            gr.ucols[idx] = j.cols;
                        }
                    }
                    if (q8_ok) {
                        j.xq = vmxq_[idx];
                        j.xd = vmxqd_[idx];
                        j.xsum = vmxqs_[idx];
                    }
                }
            }
            g.push_back(j);
        }
        if (!q8_ok) gr.nx = 0;
        gr.jobs = vjobs(g);
        gr.n = (int) g.size();
        gr.rows = rows;
        plan.push_back(gr);
    }
    return plan;
}

template <int VB>
void FastEngine::run_vmv(const char * label, const std::vector<VMvGroup> & plan) {
    static const bool q8_on = !getenv("INK_VMV_Q8") || std::string(getenv("INK_VMV_Q8")) != "0";
    for (const VMvGroup & g : plan) {
        if (q8_on && g.nx > 0) {
            VMvQuantArgs<VB> qa;
            qa.nx = g.nx;
            for (int k = 0; k < g.nx; ++k) {
                qa.x[k] = g.ux[k];
                qa.xs[k] = g.uxs[k];
                qa.cols[k] = g.ucols[k];
                qa.xq[k] = vmxq_[k];
                qa.xd[k] = vmxqd_[k];
                qa.xsum[k] = vmxqs_[k];
                qa.off[k + 1] = qa.off[k] + g.ucols[k] / 32;
            }
            P("v_mvq", (k_vmv_quant<VB>(q_, qa)));
        }
        dispatch_type(g.type, [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            if constexpr (T == q::Q8_0R || T == q::Q6_KR || T == q::Q4_KR) {
                if (q8_on && g.nx > 0) {
                    P(label, stage_tabs ? k_mv_batch_q8<T, VB, true>(q_, g.jobs, g.n, g.rows)
                                        : k_mv_batch_q8<T, VB, false>(q_, g.jobs, g.n, g.rows));
                    return;
                }
            }
            if constexpr (q::shape(T).elems == 1) {
                if (g.rows <= 512) {
                    P(label, (k_mv_batch_wide<T, VB>(q_, g.jobs, g.n, g.rows)));
                    return;
                }
            }
            P(label, stage_tabs ? k_mv_batch<T, VB, true>(q_, g.jobs, g.n, g.rows) : k_mv_batch<T, VB, false>(q_, g.jobs, g.n, g.rows));
        });
    }
}

template <int VB>
void FastEngine::vvisit(const Visit & v) {
    const Hparams & h = hp_;
    const LayerW & L = m_.layers[v.ip];
    const bool swa = h.is_swa[v.ip], loop = !v.bank.empty();
    const int nh = h.n_head, nkv = h.n_head_kv, kvw = nkv * HD, hc = h.conv_channels();
    const LoopW * lw = loop ? &m_.loops[v.ip - h.loop_start] : nullptr;
    auto vstore = [&](int id) { return vstore_ + (size_t) id * VFAST * D; };
    const W * norms[3] = {swa ? &L.attn_norm : &L.q_inp_norm, swa ? &L.attn_norm : &L.k_inp_norm, swa ? &L.attn_norm : &L.v_inp_norm};

    const float * nin[3], * inr[3], * inz[3];
    for (int j = 0; j < 3; ++j) {
        nin[j] = vnin_ + (size_t) (swa && !loop ? 0 : j) * VFAST * D;
        inr[j] = vin_ + (size_t) (swa ? 0 : j) * VFAST * D;
        inz[j] = vinn_ + (size_t) (swa ? 0 : j) * VFAST * D;
    }
    constexpr size_t PJ = 8704;
    float * Q = vproj_, * K = vproj_ + 3072, * V = vproj_ + 4096, * R = vproj_ + 5120, * sink = vproj_ + 5504,
          * gate = vproj_ + 5528, * kdyn = vproj_ + 8600, * vdyn = vproj_ + 8632;
    auto job = [](const W & w, const float * x, const float * x2, int split, size_t xs, float * y, size_t ys) {
        return VMvJob{wref(w), w.ktype, (int) w.ne[0], (int) w.ne[1], 0, split, x, x2, xs, y, ys, 0.0f, nullptr, nullptr, nullptr};
    };

    std::vector<VMvGroup> ctl_plan, ffn_ctl_plan;
    if (loop) {
        std::vector<VMvJob> js;
        for (int c = 0; c < 3; ++c) js.push_back(job(lw->down[c], inz[c], vmsgn_ + (size_t) c * VFAST * D, D, D, vg_ + (size_t) c * VFAST * 64, 64));
        ctl_plan = plan_vmv(js);
        ffn_ctl_plan = plan_vmv({job(lw->down[3], vfn_, vmsgn_ + (size_t) 3 * VFAST * D, D, D, vg_ + (size_t) 3 * VFAST * 64, 64)});
    }
    std::vector<VMvJob> pj = {job(L.wq, nin[0], nin[0], 1 << 30, D, Q, PJ),        job(L.wk, nin[1], nin[1], 1 << 30, D, K, PJ),
                              job(L.wv, nin[2], nin[2], 1 << 30, D, V, PJ),        job(L.rel, nin[0], nin[0], 1 << 30, D, R, PJ),
                              job(L.sink_proj, nin[0], nin[0], 1 << 30, D, sink, PJ), job(L.gate, nin[0], nin[0], 1 << 30, D, gate, PJ)};
    if (!swa) {
        pj.push_back(job(L.k_conv_proj, nin[1], nin[1], 1 << 30, D, kdyn, PJ));
        pj.push_back(job(L.v_conv_proj, nin[2], nin[2], 1 << 30, D, vdyn, PJ));
    }
    const std::vector<VMvGroup> proj_plan = plan_vmv(pj);
    const std::vector<VMvGroup> wo_plan = plan_vmv({job(L.wo, vatt_, vatt_, 1 << 30, (size_t) nh * HD, vo_, D)});
    const std::vector<VMvGroup> router_plan = plan_vmv({job(L.gate_inp, vf_, vf_, 1 << 30, D, vrouter_, 256)});
    if (vjobs_only_) return;

    if (v.restart >= 0) {
        VMixArgs a;
        a.cur = vxa_;
        a.push = vstore(v.restart);
        P("v_restart", (k_vmix<VB>(q_, a)));
        P("v_restart", q_.memcpy(vxa_, vprelude_, (size_t) VB * D * sizeof(float)));
    }

    // 1-3: attention inputs
    {
        VMixArgs a;
        a.qw = L.attn_res_query.ok() ? L.attn_res_query.f32() : nullptr;
        if (a.qw) {
            for (size_t s = 0; s < v.blocks.size(); ++s) a.st[s] = vstore(v.blocks[s]);
            a.ns = (int) v.blocks.size();
        }
        a.nq = a.qw ? (int) L.attn_res_query.ne[1] : 1;
        a.cur = vxa_;
        a.eps = h.eps;
        if (v.push >= 0) a.push = vstore(v.push);
        for (int j = 0; j < a.nq; ++j) {
            if (loop) {
                a.out[j] = vin_ + (size_t) j * VFAST * D;
                a.outn[j] = vinn_ + (size_t) j * VFAST * D;
            } else {
                a.out[j] = vnin_ + (size_t) j * VFAST * D;
                a.normw[j] = norms[j]->f32();
            }
        }
        P("v_amix", (k_vmix<VB>(q_, a)));
    }
    if (loop) {
        VMixArgs b;
        for (size_t s = 0; s < v.bank.size(); ++s) b.st[s] = vstore(v.bank[s]);
        b.ns = (int) v.bank.size();
        b.qw = lw->msg_query.f32();
        b.nq = 4;
        for (int j = 0; j < 4; ++j) {
            b.out[j] = vmsg_ + (size_t) j * VFAST * D;
            b.outn[j] = vmsgn_ + (size_t) j * VFAST * D;
        }
        P("v_mmix", (k_vmix<VB>(q_, b)));
        run_vmv<VB>("v_ctl_dn", ctl_plan);
        VCtlArgs c;
        c.n = 3;
        c.rank = (int) lw->gain[0].ne[0];
        for (int k = 0; k < 3; ++k) {
            c.gain[k] = lw->gain[k].f32();
            c.forget[k] = lw->forget[k].f32();
            c.g[k] = vg_ + (size_t) k * VFAST * 64;
            c.cur[k] = inr[k];
            c.msg[k] = vmsg_ + (size_t) k * VFAST * D;
            c.out[k] = vctl_ + (size_t) k * VFAST * D;
        }
        P("v_ctl_up", (k_vctl_up<VB>(q_, c)));
        VNormArgs nm;
        nm.n = 3;
        nm.eps = h.eps;
        for (int k = 0; k < 3; ++k) {
            nm.x[k] = vctl_ + (size_t) k * VFAST * D;
            nm.w[k] = norms[k]->f32();
            nm.y[k] = vnin_ + (size_t) k * VFAST * D;
        }
        P("v_norm", (k_vnorm<VB>(q_, nm)));
    }

    // 4-5: projections + Q/K/V/R post-process + KV cache + fused vraw_ K/V save
    run_vmv<VB>("v_proj", proj_plan);
    float * vraw_slot = vraw_ + (size_t) v.slot * VMAX * hc;
    P("v_qkv", (k_vqkv_post<VB>(q_, vproj_, L.q_clip.f32(), L.rel_clip.f32(), L.k_conv.f32(), L.v_conv.f32(), kdyn, vdyn, hist_[v.slot],
                                 kc_[v.slot], vc_[v.slot], ks_[v.slot], vs_[v.slot], vqf_, vrf_, vraw_slot, vpos0_, vntok_, nh, nkv,
                                 h.n_rel, swa ? kv_len_[v.slot] : 0, kv_len_[v.slot], hc, h.qk_clip, h.rel_clip, h.log_alpha,
                                 h.log_floor, !swa, kv4_)));

    // 6: batched split-K attention + combine
    const int swa_ch = std::min(swa_chunk, INK_SWA_AWG);
    const int ch = swa ? swa_ch : FCH;
    const int nc = swa ? (std::min(h.n_swa + VB - 1, n_ctx_) + ch - 1) / ch : max_vchunks_;
    {
        VAttnArgs aa{vqf_, vrf_, kc_[v.slot], vc_[v.slot], ks_[v.slot], vs_[v.slot], L.rel_bias.f32(), vpos0_, vntok_, vpart_,
                     nh, nkv, nh / nkv, h.n_rel, (int) L.rel_bias.ne[1], h.n_swa, swa ? kv_len_[v.slot] : 0, kv_len_[v.slot], ch, nc,
                     max_vchunks_, swa, kv4_};
        if (swa) P("v_attn_swa", (k_vattn<INK_SWA_AWG, VB>(q_, aa)));
        else P("v_attn_full", (k_vattn<256, VB>(q_, aa)));
        P(swa ? "v_comb_swa" : "v_comb_full", (k_vcombine<VB>(q_, vpart_, vproj_, vatt_, vntok_, nh, nc, max_vchunks_)));
    }

    // 7-9: wo + deferred 4-tap conv + fused vraw_ save + residual
    run_vmv<VB>("v_wo", wo_plan);
    P("v_conv_att", (k_vconv_res<VB>(q_, vo_, L.post_conv.f32(), hist_[v.slot] + (size_t) 2 * kvw * 3, vxa_, vxb_,
                                     vraw_slot + 2 * kvw, vntok_, hc, v.push >= 0)));

    // 10: FFN mix + loop ctl
    {
        VMixArgs a;
        for (size_t s = 0; s < v.blocks.size(); ++s) a.st[s] = vstore(v.blocks[s]);
        a.ns = (int) v.blocks.size();
        if (v.push >= 0) a.st[a.ns++] = vstore(v.push);
        a.cur = vxb_;
        a.qw = L.ffn_res_query.f32();
        a.nq = 1;
        a.eps = h.eps;
        if (loop) {
            a.out[0] = vfraw_;
            a.outn[0] = vfn_;
        } else {
            a.out[0] = vf_;
            a.normw[0] = L.ffn_norm.f32();
        }
        P("v_fmix", (k_vmix<VB>(q_, a)));
        if (loop) {
            run_vmv<VB>("v_fctl_dn", ffn_ctl_plan);
            VCtlArgs c;
            c.n = 1;
            c.rank = (int) lw->gain[3].ne[0];
            c.gain[0] = lw->gain[3].f32();
            c.forget[0] = lw->forget[3].f32();
            c.g[0] = vg_ + (size_t) 3 * VFAST * 64;
            c.cur[0] = vfraw_;
            c.msg[0] = vmsg_ + (size_t) 3 * VFAST * D;
            c.out[0] = vctl_;
            P("v_fctl_up", (k_vctl_up<VB>(q_, c)));
            VNormArgs nm;
            nm.n = 1;
            nm.eps = h.eps;
            nm.x[0] = vctl_;
            nm.w[0] = L.ffn_norm.f32();
            nm.y[0] = vf_;
            P("v_fnorm", (k_vnorm<VB>(q_, nm)));
        }
    }

    // 11: router + parallel single-WG bucketer
    run_vmv<VB>("v_router", router_plan);
    P("v_topk", (k_vtopk<VB>(q_, vrouter_, L.probs_b.ok() ? L.probs_b.f32() : nullptr, h.n_expert, h.n_used, h.w_scale, vids_, vwts_)));
    P("v_group", (k_vgroup<VB>(q_, vids_, vntok_, h.n_expert, h.n_used, vpairs_, vtiles_)));

    // 12a: grouped gate-up
    {
        bool q8 = false;
        dispatch_type(L.gate_up.ktype, [&](auto tag) { q8 = has_vq8<decltype(tag)::value>(); });
        if (q8) P("v_quant_x", (k_vquant_x<VB>(q_, vf_, vfq_, vfqd_, vfqs_)));
        VGateUpArgs ga{wref(L.gate_up), vfq_, vfqd_, vfqs_, vf_, (int) L.gate_up.ne[0], (int) L.gate_up.ne[1], h.n_ff, h.n_used,
                       L.gate_up_s.ok() ? L.gate_up_s.f32() : nullptr, vpairs_, vtiles_, vact_, h.situ_beta, h.situ_lb};
        dispatch_type(L.gate_up.ktype, [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            if constexpr (has_vq8<T>()) {
                if (gate_up_r && ga.nff % INK_VGUR == 0) {
                    P("v_gate_up", stage_tabs ? k_vgate_up_r<T, VB, true>(q_, ga) : k_vgate_up_r<T, VB, false>(q_, ga));
                    return;
                }
            }
            P("v_gate_up", stage_tabs ? k_vgate_up_fp<T, VB, true>(q_, ga) : k_vgate_up_fp<T, VB, false>(q_, ga));
        });
    }

    // 12b-13: grouped down + deferred 4-tap conv + fused vraw_ save + residual
    P("v_zero", q_.memset(vo_, 0, (size_t) VB * D * sizeof(float)));
    {
        VDownQ8Args da{wref(L.down), vaq_, vaqd_, vaqs_, vact_, (int) L.down.ne[0], h.n_used, vwts_,
                       L.down_s.ok() ? L.down_s.f32() : nullptr, vpairs_, vtiles_, vo_};
        dispatch_type(L.down.ktype, [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            if constexpr (has_vdown_q8<T>()) {
                if (down_q8 && da.cols % 32 == 0) {
                    P("v_quant_act", (k_vquant_act<VB>(q_, vact_, vntok_, h.n_used, da.cols, T == q::Q2_0R, vaq_, vaqd_, vaqs_)));
                    P("v_down", (k_vdown_q8<T, VB>(q_, da)));
                    return;
                }
            }
            P("v_down", stage_tabs ? k_vdown_fp<T, VB, true>(q_, da) : k_vdown_fp<T, VB, false>(q_, da));
        });
    }
    P("v_conv_ffn", (k_vconv_res<VB>(q_, vo_, L.ffn_post_conv.f32(), hist_[v.slot] + (size_t) (2 * kvw + D) * 3, vxb_, vxa_,
                                     vraw_slot + 2 * kvw + D, vntok_, hc, false)));
}

template <int VB>
void FastEngine::vrecord() {
    vjob_next_ = 0;
    const Hparams & h = hp_;
    const bool run = !vjobs_only_;
    if (run) P("v_embed", (k_vembed<VB>(q_, m_.tok_embd, vtok_, m_.tok_norm.f32(), vxa_, h.eps)));
    const Schedule & sch = m_.schedule;
    const auto & vs = sch.visits;
    for (size_t i = 0; i < vs.size(); ++i) {
        if (run && tgt_feats_ && (i == 5 || i == 23 || i == 41)) {
            const int tap = i == 5 ? 0 : (i == 23 ? 1 : 2);
            P("v_tap", q_.memcpy(tgt_feats_ + (size_t) tap * tap_cap_ * D, vxa_, (size_t) VB * D * sizeof(float)));
        }
        vvisit<VB>(vs[i]);
        if (run && (int) i + 1 == sch.prelude_end) P("v_prelude", q_.memcpy(vprelude_, vxa_, (size_t) VB * D * sizeof(float)));
    }
    const W & o = m_.output;
    const std::vector<VMvGroup> head = plan_vmv({VMvJob{wref(o), o.ktype, (int) o.ne[0], h.n_vocab, 0, 1 << 30, vf_, vf_, D,
                                                        vlogits_, (size_t) h.n_vocab, h.softcap}});
    if (!run) return;
    VMixArgs a;
    for (size_t s = 0; s < sch.out_blocks.size(); ++s) a.st[s] = vstore_ + (size_t) sch.out_blocks[s] * VFAST * D;
    a.ns = (int) sch.out_blocks.size();
    a.cur = vxa_;
    a.qw = m_.output_res_query.f32();
    a.nq = 1;
    a.eps = h.eps;
    a.out[0] = vf_;
    a.normw[0] = m_.output_norm.f32();
    P("v_out", (k_vmix<VB>(q_, a)));
    if (tgt_feats_) P("v_tap", q_.memcpy(tgt_feats_ + (size_t) 3 * tap_cap_ * D, vf_, (size_t) VB * D * sizeof(float)));
    run_vmv<VB>("v_head", head);
}

template <int VB>
void FastEngine::build_vgraph(int bi) {
    if (vjob_cache_.empty()) {
        vjobs_only_ = true;
        vrecord<VB>();
        vjobs_only_ = false;
    }
    try {
        ModGraph rec(q_.get_context(), q_.get_device());
        rec.begin_recording(q_);
        vrecord<VB>();
        rec.end_recording(q_);
        vgraphs_[bi] = std::make_unique<ExecGraph>(rec.finalize());
    } catch (const std::exception & e) {
        std::fprintf(stderr, "fast verify: command graph unavailable for VB=%d (%s), submitting directly\n", VB, e.what());
        vgraphs_[bi].reset();
    }
}

bool FastEngine::can_fast_verify(int n) const {
    static const bool on = !getenv("INK_FAST_VERIFY") || std::string(getenv("INK_FAST_VERIFY")) != "0";
    return on && n >= 1 && n <= VFAST && !cpu_;
}

int FastEngine::run_fast_verify(const std::vector<int> & tokens, int pos0) {
    const int n = (int) tokens.size();
    if (n < 1 || n > VFAST) throw std::runtime_error("fast verify: n out of range");
    if (pos0 + n > n_ctx_) throw std::runtime_error("verify: context is full");
    init_verify();
    const int bi = n <= 2 ? 0 : (n <= 4 ? 1 : 2);
    const int vb = bi == 0 ? 2 : (bi == 1 ? 4 : 8);
    int32_t pad[VFAST];
    for (int i = 0; i < VFAST; ++i) pad[i] = tokens[std::min(i, n - 1)];
    q_.memcpy(vtok_, pad, (size_t) vb * sizeof(int32_t));
    q_.memcpy(vpos0_, &pos0, sizeof(int32_t));
    q_.memcpy(vntok_, &n, sizeof(int32_t));
    if (vuse_graph_ && !profile) {
        if (!vgraphs_[bi]) {
            if (bi == 0) build_vgraph<2>(0);
            else if (bi == 1) build_vgraph<4>(1);
            else build_vgraph<8>(2);
        }
        if (vgraphs_[bi]) {
            q_.ext_oneapi_graph(*vgraphs_[bi]);
            vn_ = n;
            return vb;
        }
    }
    if (bi == 0) vrecord<2>();
    else if (bi == 1) vrecord<4>();
    else vrecord<8>();
    vn_ = n;
    return vb;
}

const std::vector<float> & FastEngine::verify(const std::vector<int> & tokens, int pos0) {
    const int n = (int) tokens.size();
    if (!can_fast_verify(n)) return verify_prefill(tokens, pos0);
    run_fast_verify(tokens, pos0);
    vlogits_host_.resize((size_t) n * hp_.n_vocab);
    q_.memcpy(vlogits_host_.data(), vlogits_, vlogits_host_.size() * sizeof(float)).wait();
    collect_profile();
    return vlogits_host_;
}

std::vector<int32_t> FastEngine::verify_argmax(const std::vector<int> & tokens, int pos0) {
    const int n = (int) tokens.size();
    const size_t V = hp_.n_vocab;
    if (!can_fast_verify(n)) {
        const auto & vl = verify_prefill(tokens, pos0);
        std::vector<int32_t> out(n);
        for (int i = 0; i < n; ++i) {
            const float * p = vl.data() + (size_t) i * V;
            out[i] = (int32_t) (std::max_element(p, p + V) - p);
        }
        return out;
    }
    const int vb = run_fast_verify(tokens, pos0);
    if (vb == 2) k_vargmax<2>(q_, vlogits_, (int) V, vargmax_val_, vargmax_idx_, vargmax_out_);
    else if (vb == 4) k_vargmax<4>(q_, vlogits_, (int) V, vargmax_val_, vargmax_idx_, vargmax_out_);
    else k_vargmax<8>(q_, vlogits_, (int) V, vargmax_val_, vargmax_idx_, vargmax_out_);
    std::vector<int32_t> out(n);
    q_.memcpy(out.data(), vargmax_out_, (size_t) n * sizeof(int32_t)).wait();
    return out;
}

}  // namespace ink
