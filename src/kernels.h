// Device helpers shared by the decode path (fast.cpp) and the prompt path (prefill.cpp): constants, weight
// references, per-format dispatch, and the per-lane dot products.
#pragma once

#include <sycl/sycl.hpp>
#ifdef __SYCL_DEVICE_ONLY__
#include <sycl/ext/oneapi/dot_product.hpp>
#endif

#include <cmath>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "model.h"
#include "quant.h"

namespace ink {

// ---------------------------------------------------------------------------------------------------
// constants

constexpr int SG = 16;        // sub-group size (Xe2 SIMD16)
constexpr int WG = 256;       // work-group size
constexpr int RPG = WG / SG;  // rows per work-group in sub-group-per-row kernels
constexpr int D = 2304;       // n_embd (checked in the constructor)
constexpr int PER = D / WG;   // values per thread in whole-vector kernels (9)
constexpr int CHUNK = 256;    // keys per attention work-group (sliding-window layers use swa_chunk)
#ifndef INK_AWG
#define INK_AWG 256
#endif
#ifndef INK_SWA_AWG
#define INK_SWA_AWG 128
#endif
constexpr int AWG = INK_AWG;  // decode attention work-group size
constexpr int FCH = AWG;      // keys per work-group in full-attention decode (one per thread)
constexpr int KPT = FCH / AWG;
constexpr int HD = 128;       // head dim

// A weight's planes (repacked formats use up to three) and one row of them.
struct WRef {
    const uint8_t * p = nullptr;
    size_t rb = 0;
    const uint8_t * a = nullptr;
    size_t arb = 0;
    const uint8_t * a2 = nullptr;
    size_t arb2 = 0;
};
struct RowRef {
    const uint8_t * p;
    const uint8_t * a;
    const uint8_t * a2;
};
inline RowRef row_of(const WRef & w, size_t r) { return {w.p + r * w.rb, w.a + r * w.arb, w.a2 + r * w.arb2}; }
inline WRef wref(const W & w) { return {w.p, w.row_bytes, w.aux, w.aux_row_bytes, w.aux2, w.aux2_row_bytes}; }

struct MvJob {
    WRef w;
    uint32_t type = 0;
    int cols = 0, rows = 0, row0 = 0;  // row0: first row of this job within its group
    int split = 0;                     // columns >= split read x2[c - split]
    const float * x = nullptr;
    const float * x2 = nullptr;
    float * y = nullptr;
    float cap = 0.0f;  // > 0: y = cap * tanh(y / cap)
};

// ---------------------------------------------------------------------------------------------------
// format specialisation

template <uint32_t T> struct TypeTag {
    static constexpr uint32_t value = T;
};

template <class F> void dispatch_type(uint32_t t, F && f) {
#define INK_T(x) \
    case q::x: f(TypeTag<q::x>{}); return;
    switch (t) {
        INK_T(F32) INK_T(F16) INK_T(BF16) INK_T(Q5_0) INK_T(Q8_0) INK_T(Q4_K) INK_T(Q5_K) INK_T(Q6_K) INK_T(IQ2_XXS)
        INK_T(IQ2_XS) INK_T(IQ3_XXS) INK_T(IQ1_S) INK_T(IQ2_S) INK_T(IQ4_XS) INK_T(MXFP4) INK_T(NVFP4) INK_T(Q2_0)
        INK_T(Q5_0R) INK_T(Q8_0R) INK_T(Q6_KR) INK_T(Q4_KR) INK_T(Q5_KR) INK_T(IQ2_XXSR) INK_T(IQ2_XSR) INK_T(IQ1_SR) INK_T(IQ2_SR)
        INK_T(NVFP4R) INK_T(Q2_0R)
    }
#undef INK_T
    throw std::runtime_error("unsupported weight type " + std::to_string(t));
}

// 64-bit words of codebook a format needs in local memory (1 = none)
template <uint32_t T> constexpr int grid64() {
    if constexpr (T == q::IQ2_XXS) return 256;
    else if constexpr (T == q::IQ2_XS) return 512;
    else if constexpr (T == q::IQ2_S) return 1024;
    else if constexpr (T == q::IQ1_S) return 2048;
    else if constexpr (T == q::IQ3_XXS) return 128;  // 256 x uint32
    else return 1;
}
template <uint32_t T> constexpr bool uses_ksigns() { return T == q::IQ2_XXS || T == q::IQ2_XS || T == q::IQ3_XXS; }

// Copy the format's codebook into local memory and point the tables at it (all work-items must call).
template <uint32_t T> inline q::Tabs stage(const sycl::nd_item<1> & it, uint64_t * g, uint8_t * ks) {
    q::Tabs tb;
    if constexpr (grid64<T>() > 1) {
        const int l = (int) it.get_local_id(0), n = (int) it.get_local_range(0);
        if constexpr (T == q::IQ3_XXS) {
            uint32_t * g32 = (uint32_t *) g;
            for (int i = l; i < 256; i += n) g32[i] = tables::iq3xxs_grid[i];
            tb.iq3xxs = g32;
        } else {
            const uint64_t * src = T == q::IQ2_XXS ? tables::iq2xxs_grid : T == q::IQ2_XS ? tables::iq2xs_grid
                                 : T == q::IQ2_S   ? tables::iq2s_grid
                                                   : tables::iq1s_grid;
            for (int i = l; i < grid64<T>(); i += n) g[i] = src[i];
            if constexpr (T == q::IQ2_XXS) tb.iq2xxs = g;
            else if constexpr (T == q::IQ2_XS) tb.iq2xs = g;
            else if constexpr (T == q::IQ2_S) tb.iq2s = g;
            else tb.iq1s = g;
        }
        if constexpr (uses_ksigns<T>()) {
            for (int i = l; i < 128; i += n) ks[i] = tables::ksigns_iq2xs[i];
            tb.ksigns = ks;
        }
        sycl::group_barrier(it.get_group());
    }
    return tb;
}

// this lane's share of one weight row . x (sum over the sub-group afterwards)
template <uint32_t T>
inline float lane_dot(const RowRef rr, int cols, const float * x, const float * x2, int split, int lane, const q::Tabs & tb) {
    constexpr q::BlockShape bs = q::shape(T);
    const uint8_t * row = rr.p;
    const uint8_t * arow = rr.a;
    float acc = 0.0f;
    if constexpr (T == q::NVFP4R) {
        // 32 values per step: two 16-value sub-blocks, each 8 bytes (low nibbles j, high nibbles j + 8)
        for (int s = lane; s < cols / 32; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
            const int c = 32 * s;
            const float * xs = c < split ? x + c : x2 + (c - split);
            for (int t = 0; t < 2; ++t) {
                const float d = q::ue4m3(arow[2 * s + t]);
                float sum = 0.0f;
                for (int w = 0; w < 2; ++w) {
                    const uint32_t word = wv[2 * t + w];
                    for (int b = 0; b < 4; ++b) {
                        const uint32_t byte = (word >> (8 * b)) & 0xFF;
                        const int j = 4 * w + b;
                        sum += (float) tables::kvalues_fp4[byte & 0xF] * xs[16 * t + j] + (float) tables::kvalues_fp4[byte >> 4] * xs[16 * t + 8 + j];
                    }
                }
                acc += sum * d;
            }
        }
        return acc;
    } else if constexpr (T == q::Q2_0R) {
        // 64 values per step (one block): 2 bits each, 00=-1 01=0 10=+1 11=+2
        for (int s = lane; s < cols / 64; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
            const int c = 64 * s;
            const float * xs = c < split ? x + c : x2 + (c - split);
            float sum = 0.0f;
            for (int w = 0; w < 4; ++w)
                for (int k = 0; k < 16; ++k) sum += (float) ((int) ((wv[w] >> (2 * k)) & 3) - 1) * xs[16 * w + k];
            acc += sum * q::f16(arow + 2 * s);
        }
        return acc;
    } else if constexpr (T == q::IQ2_SR) {
        // 64 values per step: two 32-value chunks of [4 grid bytes, 4 sign bytes]
        for (int s = lane; s < cols / 64; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
            for (int h = 0; h < 2; ++h) {
                const int ib = 2 * s + h, c = 32 * ib;
                const float * xs = c < split ? x + c : x2 + (c - split);
                const uint32_t idx = wv[2 * h], sg = wv[2 * h + 1];
                const uint32_t qh = arow[2 * ib], sc = arow[2 * ib + 1];
                const float d = q::f16(rr.a2 + (ib / 8) * 2);
                const float db0 = d * (0.5f + (float) (sc & 0xf)) * 0.25f, db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t grid = tb.iq2s[((idx >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300)];
                    const uint32_t signs = (sg >> (8 * l)) & 0xFF;
                    float sum = 0.0f;
                    for (int j = 0; j < 8; ++j) sum += (float) q::byte_of(grid, j) * (signs & (1u << j) ? -xs[8 * l + j] : xs[8 * l + j]);
                    acc += sum * (l < 2 ? db0 : db1);
                }
            }
        }
        return acc;
    } else if constexpr (T == q::IQ2_XXSR) {
        // 64 values per step: two chunks of (4 grid index bytes, u32 of 4x7 sign bits + 4-bit scale)
        for (int s = lane; s < cols / 64; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
            for (int h = 0; h < 2; ++h) {
                const int ib = 2 * s + h, c = 32 * ib;
                const float * xs = c < split ? x + c : x2 + (c - split);
                const uint32_t a0 = wv[2 * h], a1 = wv[2 * h + 1];
                const float db = q::f16(arow + (ib / 8) * 2) * (0.5f + (float) (a1 >> 28)) * 0.25f;
                float sum = 0.0f;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t grid = tb.iq2xxs[(a0 >> (8 * l)) & 0xFF];
                    const uint32_t signs = tb.ksigns[(a1 >> (7 * l)) & 127];
                    for (int j = 0; j < 8; ++j) sum += (float) q::byte_of(grid, j) * (signs & (1u << j) ? -xs[8 * l + j] : xs[8 * l + j]);
                }
                acc += sum * db;
            }
        }
        return acc;
    } else if constexpr (T == q::IQ2_XSR) {
        // 64 values per step: two chunks of 4 u16 (9-bit grid index, 7-bit sign index); scale byte per chunk
        for (int s = lane; s < cols / 64; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
            for (int h = 0; h < 2; ++h) {
                const int ib = 2 * s + h, c = 32 * ib;
                const float * xs = c < split ? x + c : x2 + (c - split);
                const uint32_t sc = arow[ib];
                const float d = q::f16(rr.a2 + (ib / 8) * 2);
                const float db0 = d * (0.5f + (float) (sc & 0xf)) * 0.25f, db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
                for (int l = 0; l < 4; ++l) {
                    const uint32_t v = (wv[2 * h + l / 2] >> (16 * (l % 2))) & 0xFFFF;
                    const uint64_t grid = tb.iq2xs[v & 511];
                    const uint32_t signs = tb.ksigns[v >> 9];
                    float sum = 0.0f;
                    for (int j = 0; j < 8; ++j) sum += (float) q::byte_of(grid, j) * (signs & (1u << j) ? -xs[8 * l + j] : xs[8 * l + j]);
                    acc += sum * (l < 2 ? db0 : db1);
                }
            }
        }
        return acc;
    } else if constexpr (T == q::IQ1_SR) {
        // 128 values per step: four chunks of 4 grid index bytes; qh (u16) per chunk
        for (int s = lane; s < cols / 128; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
            for (int h = 0; h < 4; ++h) {
                const int ib = 4 * s + h, c = 32 * ib;
                const float * xs = c < split ? x + c : x2 + (c - split);
                const uint32_t qh = q::rd16(arow + 2 * ib);
                const float dl = q::f16(rr.a2 + (ib / 8) * 2) * (float) (2 * ((qh >> 12) & 7) + 1);
                const float delta = qh & 0x8000 ? -0.125f : 0.125f;
                float sum = 0.0f, xsum = 0.0f;
                for (int l = 0; l < 4; ++l) {
                    const uint64_t grid = tb.iq1s[((wv[h] >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)];
                    for (int j = 0; j < 8; ++j) {
                        sum += (float) (int8_t) q::byte_of(grid, j) * xs[8 * l + j];
                        xsum += xs[8 * l + j];
                    }
                }
                acc += dl * (sum + delta * xsum);
            }
        }
        return acc;
    } else if constexpr (T == q::Q4_KR) {
        // 32 values per step: 16 bytes of one 64-value group, low nibbles -> chunk 2g, high -> chunk 2g + 1
        for (int s = lane; s < cols / 32; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
            const int b = s / 8, o = (s % 8) * 16, g = o / 32, l0 = o % 32;
            const float d = q::f16(rr.a2 + b * 4), dmin = q::f16(rr.a2 + b * 4 + 2);
            const int clo = 256 * b + 64 * g + l0, chi = clo + 32;
            const float * xlo = clo < split ? x + clo : x2 + (clo - split);
            const float * xhi = chi < split ? x + chi : x2 + (chi - split);
            float slo = 0.0f, shi = 0.0f, xl = 0.0f, xh = 0.0f;
            for (int w = 0; w < 4; ++w)
                for (int k = 0; k < 4; ++k) {
                    const uint32_t byte = (wv[w] >> (8 * k)) & 0xFF;
                    const int j = 4 * w + k;
                    slo += (float) (byte & 0xF) * xlo[j];
                    shi += (float) (byte >> 4) * xhi[j];
                    xl += xlo[j];
                    xh += xhi[j];
                }
            const uint8_t * sm = arow + b * 16 + 4 * g;  // (sc, m) of chunks 2g and 2g + 1
            acc += d * (float) sm[0] * slo - dmin * (float) sm[1] * xl + d * (float) sm[2] * shi - dmin * (float) sm[3] * xh;
        }
        return acc;
    } else if constexpr (T == q::Q5_KR) {
        // 32 values per step: 16 bytes of low nibbles (0..15 low, 16..31 high), u32 qh + (sc, m), (d, dmin) per 256
        for (int s = lane; s < cols / 32; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
            const uint32_t qh = *(const uint32_t *) (arow + 8 * s);
            const uint16_t sm = *(const uint16_t *) (arow + 8 * s + 4);
            const int b = s / 8, c0 = 32 * s;
            const float d = q::f16(rr.a2 + 4 * b), dmin = q::f16(rr.a2 + 4 * b + 2);
            const float d1 = d * (float) (sm & 0xFFu), m1 = dmin * (float) (sm >> 8);
            const float * xs = c0 < split ? x + c0 : x2 + (c0 - split);
            float sum = 0.0f, xsum = 0.0f;
            for (int w = 0; w < 4; ++w)
                for (int k = 0; k < 4; ++k) {
                    const uint32_t byte = (wv[w] >> (8 * k)) & 0xFFu;
                    const int j = 4 * w + k;
                    const float qlo = (float) ((byte & 0xFu) + ((qh >> j) & 1u ? 16u : 0u));
                    const float qhi = (float) ((byte >> 4) + ((qh >> (16 + j)) & 1u ? 16u : 0u));
                    sum += qlo * xs[j] + qhi * xs[16 + j];
                    xsum += xs[j] + xs[16 + j];
                }
            acc += d1 * sum - m1 * xsum;
        }
        return acc;
    } else if constexpr (T == q::Q5_0R) {
        // 32 values per step: 16 bytes low nibbles (0..15 low, 16..31 high), u32 qh, f16 d
        for (int s = lane; s < cols / 32; s += SG) {
            const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
            const uint32_t qh = *(const uint32_t *) (arow + 4 * s);
            const float d = q::f16(rr.a2 + 2 * s);
            const int c0 = 32 * s;
            const float * xs = c0 < split ? x + c0 : x2 + (c0 - split);
            float sum = 0.0f;
            for (int w = 0; w < 4; ++w)
                for (int k = 0; k < 4; ++k) {
                    const uint32_t byte = (wv[w] >> (8 * k)) & 0xFFu;
                    const int j = 4 * w + k;
                    const int qlo = (int) ((byte & 0xFu) + ((qh >> j) & 1u ? 16u : 0u)) - 16;
                    const int qhi = (int) ((byte >> 4) + ((qh >> (16 + j)) & 1u ? 16u : 0u)) - 16;
                    sum += (float) qlo * xs[j] + (float) qhi * xs[16 + j];
                }
            acc += d * sum;
        }
        return acc;
    }
    if constexpr (T == q::Q6_KR) {
        // int8 values, 16 per lane per step, scale d * sc per 16 values
        for (int c = lane * 16; c < cols; c += SG * 16) {
            const sycl::int4 wv = *(const sycl::int4 *) (row + c);
            const float * xs = c < split ? x + c : x2 + (c - split);
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) {
                const sycl::float4 xv = *(const sycl::float4 *) (xs + 4 * k);
                const int w = wv[k];
                s += (float) (int8_t) (w) * xv[0] + (float) (int8_t) (w >> 8) * xv[1] + (float) (int8_t) (w >> 16) * xv[2] +
                     (float) (int8_t) (w >> 24) * xv[3];
            }
            acc += s * (q::f16(rr.a2 + (c / 256) * 2) * (float) (int8_t) arow[c / 16]);
        }
    } else if constexpr (T == q::Q8_0R) {
        // 16 int8 values per lane per step, the sub-group reading 256 contiguous bytes
        for (int c = lane * 16; c < cols; c += SG * 16) {
            const sycl::int4 wv = *(const sycl::int4 *) (row + c);
            const float * xs = c < split ? x + c : x2 + (c - split);
            float s = 0.0f;
            for (int k = 0; k < 4; ++k) {
                const sycl::float4 xv = *(const sycl::float4 *) (xs + 4 * k);
                const int w = wv[k];
                s += (float) (int8_t) (w) * xv[0] + (float) (int8_t) (w >> 8) * xv[1] + (float) (int8_t) (w >> 16) * xv[2] +
                     (float) (int8_t) (w >> 24) * xv[3];
            }
            acc += s * q::f16(arow + (c / 32) * 2);
        }
    } else if constexpr (bs.elems == 1) {
        for (int c = lane; c < cols; c += SG) {
            const uint8_t * p = row + (size_t) c * bs.bytes;
            float v;
            if constexpr (T == q::F32) v = sycl::bit_cast<float>(q::rd32(p));
            else if constexpr (T == q::F16) v = q::f16(p);
            else v = q::bf16(p);
            acc += v * (c < split ? x[c] : x2[c - split]);
        }
    } else {
        for (int c0 = lane * 32; c0 < cols; c0 += SG * 32) {
            const float * xs = c0 < split ? x + c0 : x2 + (c0 - split);
            q::decode32f(T, row + (size_t) (c0 / bs.elems) * bs.bytes, (c0 % bs.elems) / 32,
                         [&](int k, float v) { acc += v * xs[k]; }, tb);
        }
    }
    return acc;
}

inline float sigmoidf(float v) { return 1.0f / (1.0f + sycl::exp(-v)); }

// ---------------------------------------------------------------------------------------------------
// Step decoders for the batched (prompt) kernels: one step of a row hands every value to emit(column, value).
// Repacked layouts step over 16 bytes of the value plane; original block formats over 32 values; elementwise
// formats over 16 values.

// ---- KV cache store: int8 (kv4 = false) or 4-bit (q4_0-style, two values per byte, low nibble = even dim),
// one f16 scale per 32 values. A work-group of HD threads holds one head's values, thread t = dim t; o is the
// value's index in the cache (row * kvw + head * HD + t), so o / 32 is its scale and o / 2 its 4-bit byte.
template <class G>
inline void kv_put(const G & grp, const sycl::sub_group & sg, int t, float x, bool kv4, int8_t * q, sycl::half * sc, size_t o) {
    if (!kv4) {
        float amax = 0.0f;
        for (int b = 0; b < HD / 32; ++b) {
            const float mb = sycl::reduce_over_group(grp, t / 32 == b ? sycl::fabs(x) : 0.0f, sycl::maximum<float>());
            if (t / 32 == b) amax = mb;
        }
        const sycl::half dh = (sycl::half) (amax / 127.0f);
        const float dq = (float) dh;
        q[o] = (int8_t) (dq > 0.0f ? sycl::clamp((int) sycl::rint(x / dq), -127, 127) : 0);
        if (t % 32 == 0) sc[o / 32] = dh;
        return;
    }
    float mx = 0.0f, mn = 0.0f;
    for (int b = 0; b < HD / 32; ++b) {
        const bool mine = t / 32 == b;
        const float hi = sycl::reduce_over_group(grp, mine ? x : -INFINITY, sycl::maximum<float>());
        const float lo = sycl::reduce_over_group(grp, mine ? x : INFINITY, sycl::minimum<float>());
        if (mine) {
            mx = hi;
            mn = lo;
        }
    }
    const float vmax = mx >= -mn ? mx : mn;  // the signed extreme maps to -8, as in q4_0
    const sycl::half dh = (sycl::half) (vmax / -8.0f);
    const float d = (float) dh;
    const int qv = d != 0.0f ? sycl::clamp((int) sycl::rint(x / d) + 8, 0, 15) : 8;
    const int other = sycl::permute_group_by_xor(sg, qv, 1);  // the neighbouring dim (same sub-group)
    if ((t & 1) == 0) ((uint8_t *) q)[o / 2] = (uint8_t) (qv | (other << 4));
    if (t % 32 == 0) sc[o / 32] = dh;
}

// n consecutive cached values starting at o (a multiple of n, n in {4, 8, 16}), unscaled (int8 or nibble - 8)
template <int N> inline void kv_get(const int8_t * q, bool kv4, size_t o, float * out) {
    if (!kv4) {
        const sycl::vec<int8_t, N> v = *(const sycl::vec<int8_t, N> *) (q + o);
        for (int e = 0; e < N; ++e) out[e] = (float) v[e];
    } else {
        const sycl::vec<uint8_t, N / 2> v = *(const sycl::vec<uint8_t, N / 2> *) ((const uint8_t *) q + o / 2);
        for (int e = 0; e < N / 2; ++e) {
            out[2 * e] = (float) ((int) (v[e] & 15) - 8);
            out[2 * e + 1] = (float) ((int) (v[e] >> 4) - 8);
        }
    }
}

template <uint32_t T> constexpr int step_vals() {
    if constexpr (T == q::Q8_0R || T == q::Q6_KR) return 16;
    else if constexpr (T == q::NVFP4R || T == q::Q4_KR || T == q::Q5_KR || T == q::Q5_0R) return 32;
    else if constexpr (T == q::Q2_0R || T == q::IQ2_SR || T == q::IQ2_XXSR || T == q::IQ2_XSR) return 64;
    else if constexpr (T == q::IQ1_SR) return 128;
    else if constexpr (q::shape(T).elems == 1) return 16;
    else return 32;
}

// One 32-value sub-block (ib) of an IQ1_SR row: a quarter of its step, so more threads share a step's work.
template <class Emit> inline void iq1s_part(const RowRef rr, int ib, Emit && emit, const q::Tabs & tb) {
    const uint32_t w = *(const uint32_t *) (rr.p + 4 * ib);
    const uint32_t qh = q::rd16(rr.a + 2 * ib);
    const float dl = q::f16(rr.a2 + (ib / 8) * 2) * (float) (2 * ((qh >> 12) & 7) + 1);
    const float delta = qh & 0x8000 ? -0.125f : 0.125f;
    const int c = 32 * ib;
    for (int l = 0; l < 4; ++l) {
        const uint64_t grid = tb.iq1s[((w >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)];
        for (int j = 0; j < 8; ++j) emit(c + 8 * l + j, dl * ((float) (int8_t) q::byte_of(grid, j) + delta));
    }
}

template <uint32_t T, class Emit> inline void step_decode(const RowRef rr, int s, Emit && emit, const q::Tabs & tb) {
    const uint8_t * row = rr.p;
    const uint8_t * arow = rr.a;
    if constexpr (T == q::Q8_0R || T == q::Q6_KR) {
        const int c0 = 16 * s;
        const sycl::int4 wv = *(const sycl::int4 *) (row + c0);
        const float d = T == q::Q8_0R ? q::f16(arow + (c0 / 32) * 2) : q::f16(rr.a2 + (c0 / 256) * 2) * (float) (int8_t) arow[c0 / 16];
        for (int k = 0; k < 4; ++k)
            for (int b = 0; b < 4; ++b) emit(c0 + 4 * k + b, (float) (int8_t) (wv[k] >> (8 * b)) * d);
    } else if constexpr (T == q::NVFP4R) {
        const int c0 = 32 * s;
        const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
        for (int t = 0; t < 2; ++t) {
            const float d = q::ue4m3(arow[2 * s + t]);
            for (int w = 0; w < 2; ++w)
                for (int b = 0; b < 4; ++b) {
                    const uint32_t byte = (wv[2 * t + w] >> (8 * b)) & 0xFF;
                    const int j = 4 * w + b;
                    emit(c0 + 16 * t + j, (float) tables::kvalues_fp4[byte & 0xF] * d);
                    emit(c0 + 16 * t + 8 + j, (float) tables::kvalues_fp4[byte >> 4] * d);
                }
        }
    } else if constexpr (T == q::Q2_0R) {
        const int c0 = 64 * s;
        const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
        const float d = q::f16(arow + 2 * s);
        for (int w = 0; w < 4; ++w)
            for (int k = 0; k < 16; ++k) emit(c0 + 16 * w + k, (float) ((int) ((wv[w] >> (2 * k)) & 3) - 1) * d);
    } else if constexpr (T == q::IQ2_SR || T == q::IQ2_XXSR || T == q::IQ2_XSR) {
        const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
        for (int h = 0; h < 2; ++h) {
            const int ib = 2 * s + h, c = 32 * ib;
            float db0, db1;
            if constexpr (T == q::IQ2_XXSR) {
                db0 = db1 = q::f16(arow + (ib / 8) * 2) * (0.5f + (float) (wv[2 * h + 1] >> 28)) * 0.25f;
            } else {
                const uint32_t sc = T == q::IQ2_SR ? arow[2 * ib + 1] : arow[ib];
                const float d = q::f16(rr.a2 + (ib / 8) * 2);
                db0 = d * (0.5f + (float) (sc & 0xf)) * 0.25f;
                db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
            }
            for (int l = 0; l < 4; ++l) {
                uint64_t grid;
                uint32_t signs;
                if constexpr (T == q::IQ2_SR) {
                    grid = tb.iq2s[((wv[2 * h] >> (8 * l)) & 0xFF) | ((arow[2 * ib] << (8 - 2 * l)) & 0x300)];
                    signs = (wv[2 * h + 1] >> (8 * l)) & 0xFF;
                } else if constexpr (T == q::IQ2_XXSR) {
                    grid = tb.iq2xxs[(wv[2 * h] >> (8 * l)) & 0xFF];
                    signs = tb.ksigns[(wv[2 * h + 1] >> (7 * l)) & 127];
                } else {
                    const uint32_t v = (wv[2 * h + l / 2] >> (16 * (l % 2))) & 0xFFFF;
                    grid = tb.iq2xs[v & 511];
                    signs = tb.ksigns[v >> 9];
                }
                const float dl = l < 2 ? db0 : db1;
                for (int j = 0; j < 8; ++j) emit(c + 8 * l + j, dl * (float) q::byte_of(grid, j) * (signs & (1u << j) ? -1.0f : 1.0f));
            }
        }
    } else if constexpr (T == q::IQ1_SR) {
        const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
        for (int h = 0; h < 4; ++h) {
            const int ib = 4 * s + h, c = 32 * ib;
            const uint32_t qh = q::rd16(arow + 2 * ib);
            const float dl = q::f16(rr.a2 + (ib / 8) * 2) * (float) (2 * ((qh >> 12) & 7) + 1);
            const float delta = qh & 0x8000 ? -0.125f : 0.125f;
            for (int l = 0; l < 4; ++l) {
                const uint64_t grid = tb.iq1s[((wv[h] >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8)];
                for (int j = 0; j < 8; ++j) emit(c + 8 * l + j, dl * ((float) (int8_t) q::byte_of(grid, j) + delta));
            }
        }
    } else if constexpr (T == q::Q4_KR) {
        const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
        const int b = s / 8, o = (s % 8) * 16, g = o / 32, l0 = o % 32;
        const float d = q::f16(rr.a2 + b * 4), dmin = q::f16(rr.a2 + b * 4 + 2);
        const uint8_t * sm = arow + b * 16 + 4 * g;
        const float d1 = d * (float) sm[0], m1 = dmin * (float) sm[1], d2 = d * (float) sm[2], m2 = dmin * (float) sm[3];
        const int clo = 256 * b + 64 * g + l0, chi = clo + 32;
        for (int w = 0; w < 4; ++w)
            for (int k = 0; k < 4; ++k) {
                const uint32_t byte = (wv[w] >> (8 * k)) & 0xFF;
                const int j = 4 * w + k;
                emit(clo + j, d1 * (float) (byte & 0xF) - m1);
                emit(chi + j, d2 * (float) (byte >> 4) - m2);
            }
    } else if constexpr (T == q::Q5_KR) {
        const int c0 = 32 * s;
        const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
        const uint32_t qh = *(const uint32_t *) (arow + 8 * s);
        const uint16_t sm = *(const uint16_t *) (arow + 8 * s + 4);
        const int b = s / 8;
        const float d = q::f16(rr.a2 + 4 * b), dmin = q::f16(rr.a2 + 4 * b + 2);
        const float d1 = d * (float) (sm & 0xFFu), m1 = dmin * (float) (sm >> 8);
        for (int w = 0; w < 4; ++w)
            for (int k = 0; k < 4; ++k) {
                const uint32_t byte = (wv[w] >> (8 * k)) & 0xFFu;
                const int j = 4 * w + k;
                emit(c0 + j, d1 * (float) ((byte & 0xFu) + ((qh >> j) & 1u ? 16u : 0u)) - m1);
                emit(c0 + 16 + j, d1 * (float) ((byte >> 4) + ((qh >> (16 + j)) & 1u ? 16u : 0u)) - m1);
            }
    } else if constexpr (T == q::Q5_0R) {
        const int c0 = 32 * s;
        const sycl::uint4 wv = *(const sycl::uint4 *) (row + 16 * s);
        const uint32_t qh = *(const uint32_t *) (arow + 4 * s);
        const float d = q::f16(rr.a2 + 2 * s);
        for (int w = 0; w < 4; ++w)
            for (int k = 0; k < 4; ++k) {
                const uint32_t byte = (wv[w] >> (8 * k)) & 0xFFu;
                const int j = 4 * w + k;
                emit(c0 + j, d * (float) ((int) ((byte & 0xFu) + ((qh >> j) & 1u ? 16u : 0u)) - 16));
                emit(c0 + 16 + j, d * (float) ((int) ((byte >> 4) + ((qh >> (16 + j)) & 1u ? 16u : 0u)) - 16));
            }
    } else if constexpr (q::shape(T).elems == 1) {
        const int c0 = 16 * s;
        for (int i = 0; i < 16; ++i) {
            const uint8_t * p = row + (size_t) (c0 + i) * q::shape(T).bytes;
            float v;
            if constexpr (T == q::F32) v = sycl::bit_cast<float>(q::rd32(p));
            else if constexpr (T == q::F16) v = q::f16(p);
            else v = q::bf16(p);
            emit(c0 + i, v);
        }
    } else {
        constexpr q::BlockShape bs = q::shape(T);
        const int c0 = 32 * s;
        q::decode32f(T, row + (size_t) (c0 / bs.elems) * bs.bytes, (c0 % bs.elems) / 32, [&](int i, float v) { emit(c0 + i, v); }, tb);
    }
}

// One batched matrix-vector job: y[t][r] = W[r] . x[t] for every token t (token-major x and y).
struct BJob {
    WRef w;
    uint32_t type = 0;
    int cols = 0, rows = 0, row0 = 0, split = 1 << 30;
    const float * x = nullptr;
    const float * x2 = nullptr;  // columns >= split read x2 (same token stride)
    size_t xs = 0;               // token stride of x / x2
    float * y = nullptr;
    size_t ys = 0;               // token stride of y
};
struct BJobs {
    BJob j[8];
    int n = 0;
};

// ---------------------------------------------------------------------------------------------------
// shared DP4A & argmax helpers (used by fast.cpp and verify.cpp)

template <uint32_t T> constexpr bool has_q8() {
    return T == q::IQ2_SR || T == q::IQ2_XXSR || T == q::IQ2_XSR || T == q::IQ1_SR || T == q::NVFP4R || T == q::Q4_KR || T == q::Q5_KR;
}
template <uint32_t T> constexpr bool has_down_q8() { return T == q::NVFP4R || T == q::Q2_0R || T == q::Q5_0R; }

// 4 bits -> 4 bytes of 0xFF where the bit is set
inline uint32_t byte_mask4(uint32_t n) { return ((n * 0x00204081u) & 0x01010101u) * 0xFFu; }
// 4 bits -> 4 bytes of 0x10 (16) where the bit is set, 0x00 otherwise
inline uint32_t hi16x4(uint32_t bits4) { return ((bits4 * 0x00204081u) & 0x01010101u) << 4; }
// 4 NVFP4 nibbles (one per byte, low 4 bits) -> 4 int8 values 2 * E2M1 = sign x {0, 1, 2, 3, 4, 6, 8, 12}
inline int32_t fp4x4(uint32_t n) {
    const uint32_t m = n & 0x07070707u;
    const uint32_t gt4 = ((m + 0x03030303u) >> 3) & 0x01010101u;  // m > 4
    const uint32_t eq7 = ((m + 0x01010101u) >> 3) & 0x01010101u;  // m == 7
    const uint32_t mask = gt4 * 0xFFu;
    const uint32_t v = m + ((m & mask) - (0x04040404u & mask)) + 2 * eq7;  // m, but 5 6 7 -> 6 8 12
    // negate each byte safely without inter-byte borrow (v <= 12 < 128; -0 stays 0)
    const uint32_t neg = (0x80808080u - v) ^ 0x80808080u;
    const uint32_t smask = ((n >> 3) & 0x01010101u) * 0xFFu;
    return (int32_t) (v ^ ((v ^ neg) & smask));
}
// apply 8 sign bits to 8 grid bytes (each <= 0x2b, so the negation stays inside its byte)
inline int32_t signed4(uint32_t g, uint32_t m) { return (int32_t) ((g ^ m) + (m & 0x01010101u)); }
inline float h2f_bits(uint16_t b) { return (float) sycl::bit_cast<sycl::half>(b); }

inline int32_t dp4a(int32_t a, int32_t b, int32_t c) {
#ifdef __SYCL_DEVICE_ONLY__
    return sycl::ext::oneapi::dot_acc(a, b, c);
#else
    for (int i = 0; i < 4; ++i) c += (int32_t) (int8_t) ((uint32_t) a >> (8 * i)) * (int32_t) (int8_t) ((uint32_t) b >> (8 * i));
    return c;
#endif
}

inline float nvfp4_q8_dot(const RowRef rr, int ub, const sycl::int4 x0, const sycl::int4 x1) {
    const uint32_t * wp = (const uint32_t *) (rr.p + 16 * ub);
    const uint32_t w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3];
    int32_t is0 = dp4a(fp4x4(w0), x0[0], 0);
    is0 = dp4a(fp4x4(w1), x0[1], is0);
    is0 = dp4a(fp4x4(w0 >> 4), x0[2], is0);
    is0 = dp4a(fp4x4(w1 >> 4), x0[3], is0);
    int32_t is1 = dp4a(fp4x4(w2), x1[0], 0);
    is1 = dp4a(fp4x4(w3), x1[1], is1);
    is1 = dp4a(fp4x4(w2 >> 4), x1[2], is1);
    is1 = dp4a(fp4x4(w3 >> 4), x1[3], is1);
    const uint16_t sc = *(const uint16_t *) (rr.a + 2 * ub);
    return q::ue4m3((uint8_t) (sc & 0xFFu)) * (float) is0 + q::ue4m3((uint8_t) (sc >> 8)) * (float) is1;
}

inline int32_t q5_q8_isum(const uint32_t * wp, uint32_t qh, const sycl::int4 x0, const sycl::int4 x1) {
    const uint32_t w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3];
    int32_t isum = dp4a((int32_t) ((w0 & 0x0F0F0F0Fu) | hi16x4(qh & 0xFu)), x0[0], 0);
    isum = dp4a((int32_t) ((w1 & 0x0F0F0F0Fu) | hi16x4((qh >> 4) & 0xFu)), x0[1], isum);
    isum = dp4a((int32_t) ((w2 & 0x0F0F0F0Fu) | hi16x4((qh >> 8) & 0xFu)), x0[2], isum);
    isum = dp4a((int32_t) ((w3 & 0x0F0F0F0Fu) | hi16x4((qh >> 12) & 0xFu)), x0[3], isum);
    isum = dp4a((int32_t) (((w0 >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 16) & 0xFu)), x1[0], isum);
    isum = dp4a((int32_t) (((w1 >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 20) & 0xFu)), x1[1], isum);
    isum = dp4a((int32_t) (((w2 >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 24) & 0xFu)), x1[2], isum);
    isum = dp4a((int32_t) (((w3 >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 28) & 0xFu)), x1[3], isum);
    return isum;
}

// Argmax over a work-group's (val, idx) pairs using sub-group XOR shuffles + SLM across RPG sub-groups.
template <int Dim>
inline void wg_argmax(const sycl::nd_item<Dim> & it, float * sv, int32_t * si, float & bv, int32_t & bi) {
    const auto sg = it.get_sub_group();
    const int sgid = (int) sg.get_group_id()[0], lane = (int) sg.get_local_id()[0];
    for (int off = SG / 2; off > 0; off /= 2) {
        const float ov = sycl::permute_group_by_xor(sg, bv, off);
        const int32_t oi = sycl::permute_group_by_xor(sg, bi, off);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        sv[sgid] = bv;
        si[sgid] = bi;
    }
    sycl::group_barrier(it.get_group());
    if (sgid == 0) {
        bv = lane < RPG ? sv[lane] : -INFINITY;
        bi = lane < RPG ? si[lane] : (1 << 30);
        for (int off = SG / 2; off > 0; off /= 2) {
            const float ov = sycl::permute_group_by_xor(sg, bv, off);
            const int32_t oi = sycl::permute_group_by_xor(sg, bi, off);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------
// batched-decode verify descriptors (Phase 1 speculative decoding, verify.cpp)

constexpr int VFAST = 8;
constexpr int V_MMAX = 4;

struct VMvJob {
    WRef w;
    uint32_t type = 0;
    int cols = 0, rows = 0, row0 = 0, split = 1 << 30;
    const float * x = nullptr;
    const float * x2 = nullptr;
    size_t xs = 0;
    float * y = nullptr;
    size_t ys = 0;
    float cap = 0.0f;
    const int8_t * xq = nullptr;
    const float * xd = nullptr, * xsum = nullptr;
};

struct VMvGroup {
    uint32_t type = 0;
    VMvJob * jobs = nullptr;
    int n = 0;
    int rows = 0;
    int nx = 0;
    const float * ux[3] = {};
    size_t uxs[3] = {};
    int ucols[3] = {};
};

}  // namespace ink
