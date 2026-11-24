// Device-side decoders for the 15 weight formats in the Inkling GGUF files.
// Each decoder turns one 32-value chunk of a block into floats, ported line for line from ggml's
// reference dequantize_row_* functions (llama.cpp ggml/src/ggml-quants.c, MIT licence) so that the
// engine reads exactly the weights llama.cpp reads. Layouts are the packed ggml block structs.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

#include "quant_tables.h"

namespace ink::q {

enum Type : uint32_t {
    F32 = 0, F16 = 1, Q8_0 = 8, Q4_K = 12, Q5_K = 13, Q6_K = 14, IQ2_XXS = 16, IQ2_XS = 17,
    IQ3_XXS = 18, IQ1_S = 19, IQ2_S = 22, IQ4_XS = 23, BF16 = 30, MXFP4 = 39, NVFP4 = 40, Q2_0 = 42,
    // repacked layouts (kernel formats, never in a file): values in one aligned plane, scales in another
    Q8_0R = 1008,
    Q6_KR = 1014,  // int8 values (q - 32), int8 scale per 16 values, fp16 d per 256
    Q4_KR = 1012,    // nibbles per block; (scale, min) bytes per 32; (d, dmin) per 256
    IQ2_XXSR = 1016, // 8 bytes per 32 values; d per 256
    IQ2_XSR = 1017,  // 8 bytes per 32; scale byte per 32; d per 256
    IQ1_SR = 1019,   // 4 bytes per 32; qh u16 per 32; d per 256
    IQ2_SR = 1022,   // [4 grid bytes, 4 sign bytes] per 32; [qh, scale] per 32; d per 256
    NVFP4R = 1040,   // nibbles per block; UE4M3 scale byte per 16
    Q2_0R = 1042,    // 16 bytes per 64 values; d per 64
};

inline uint16_t rd16(const uint8_t * p) { return (uint16_t) (p[0] | (p[1] << 8)); }
inline uint32_t rd32(const uint8_t * p) {
    return (uint32_t) p[0] | ((uint32_t) p[1] << 8) | ((uint32_t) p[2] << 16) | ((uint32_t) p[3] << 24);
}
inline float f16(const uint8_t * p) { return (float) sycl::bit_cast<sycl::half>(rd16(p)); }
inline float bf16(const uint8_t * p) { return sycl::bit_cast<float>((uint32_t) rd16(p) << 16); }
inline uint8_t byte_of(uint64_t v, int j) { return (uint8_t) (v >> (8 * j)); }

// ggml_e8m0_to_fp32_half
inline float e8m0_half(uint8_t x) {
    const uint32_t bits = x < 2 ? (0x00200000u << x) : ((uint32_t) (x - 1) << 23);
    return sycl::bit_cast<float>(bits);
}

// ggml_ue4m3_to_fp32 (already halved to match kvalues_fp4 = 2 * E2M1)
inline float ue4m3(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int e = (x >> 3) & 0xF;
    const int m = x & 0x7;
    const float raw = e == 0 ? sycl::ldexp((float) m, -9) : sycl::ldexp(1.0f + (float) m / 8.0f, e - 7);
    return raw * 0.5f;
}

inline void scale_min_k4(int j, const uint8_t * q, uint8_t & d, uint8_t & m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

struct BlockShape {
    uint32_t elems;
    uint32_t bytes;
};

constexpr BlockShape shape(uint32_t t) {
    switch (t) {
        case F32: return {1, 4};
        case F16: return {1, 2};
        case BF16: return {1, 2};
        case Q8_0: return {32, 34};
        case Q4_K: return {256, 144};
        case Q5_K: return {256, 176};
        case Q6_K: return {256, 210};
        case IQ2_XXS: return {256, 66};
        case IQ2_XS: return {256, 74};
        case IQ3_XXS: return {256, 98};
        case IQ1_S: return {256, 50};
        case IQ2_S: return {256, 82};
        case IQ4_XS: return {256, 136};
        case MXFP4: return {32, 17};
        case NVFP4: return {64, 36};
        case Q2_0: return {64, 18};
        case Q8_0R: return {32, 34};
        case Q6_KR: return {256, 210};
        case Q4_KR: return {256, 144};
        case IQ2_XXSR: return {256, 66};
        case IQ2_XSR: return {256, 74};
        case IQ1_SR: return {256, 50};
        case IQ2_SR: return {256, 82};
        case NVFP4R: return {64, 36};
        case Q2_0R: return {64, 18};
        default: return {0, 0};
    }
}

// Where the codebook tables live: global memory by default, or a work-group's local copy (see kernels
// that stage them). Pointers are generic, so either works.
struct Tabs {
    const uint64_t * iq2xxs = tables::iq2xxs_grid;
    const uint64_t * iq2xs = tables::iq2xs_grid;
    const uint64_t * iq2s = tables::iq2s_grid;
    const uint32_t * iq3xxs = tables::iq3xxs_grid;
    const uint64_t * iq1s = tables::iq1s_grid;
    const uint8_t * ksigns = tables::ksigns_iq2xs;
};

// Decode the 32 values [32*chunk, 32*chunk + 32) of block `b`, handing each to out(index, value). Block
// types only. With a compile-time `type` the switch folds away; with an accumulating `out` no temporary
// array is needed (the dot-product kernels).
template <class Out>
inline void decode32f(uint32_t type, const uint8_t * b, int chunk, Out && out, const Tabs & T) {
    using tables::kmask_iq2xs;
    using tables::kvalues_iq4nl;
    using tables::kvalues_fp4;
    switch (type) {
        case Q8_0: {
            const float d = f16(b);
            for (int j = 0; j < 32; ++j) out(j, (float) (int8_t) b[2 + j] * d);
            break;
        }
        case Q4_K: {  // d, dmin, scales[12], qs[128]
            const float d = f16(b), dmin = f16(b + 2);
            const int g = chunk / 2, hi = chunk & 1;
            uint8_t sc, m;
            scale_min_k4(2 * g + hi, b + 4, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            const uint8_t * q = b + 16 + 32 * g;
            for (int l = 0; l < 32; ++l) out(l, d1 * (hi ? (q[l] >> 4) : (q[l] & 0xF)) - m1);
            break;
        }
        case Q5_K: {  // d, dmin, scales[12], qh[32], qs[128]
            const float d = f16(b), dmin = f16(b + 2);
            const int g = chunk / 2, hi = chunk & 1;
            uint8_t sc, m;
            scale_min_k4(2 * g + hi, b + 4, sc, m);
            const float d1 = d * sc, m1 = dmin * m;
            const uint8_t * qh = b + 16;
            const uint8_t * ql = b + 48 + 32 * g;
            const uint8_t u = (uint8_t) ((hi ? 2 : 1) << (2 * g));
            for (int l = 0; l < 32; ++l) {
                out(l, d1 * ((hi ? (ql[l] >> 4) : (ql[l] & 0xF)) + (qh[l] & u ? 16 : 0)) - m1);
            }
            break;
        }
        case Q6_K: {  // ql[128], qh[64], scales[16], d
            const int half = chunk / 4, k = chunk % 4;
            const uint8_t * ql = b + 64 * half;
            const uint8_t * qh = b + 128 + 32 * half;
            const int8_t * sc = (const int8_t *) (b + 192) + 8 * half;
            const float d = f16(b + 208);
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                int q, s;
                switch (k) {
                    case 0: q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4); s = sc[is + 0]; break;
                    case 1: q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); s = sc[is + 2]; break;
                    case 2: q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4); s = sc[is + 4]; break;
                    default: q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); s = sc[is + 6]; break;
                }
                out(l, d * (float) s * (float) (int8_t) (q - 32));
            }
            break;
        }
        case IQ2_XXS: {  // d, qs u16[32]
            const float d = f16(b);
            const uint8_t * p = b + 2 + 8 * chunk;
            const uint32_t a0 = rd32(p), a1 = rd32(p + 4);
            const float db = d * (0.5f + (float) (a1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint64_t grid = T.iq2xxs[(a0 >> (8 * l)) & 0xFF];
                const uint8_t signs = T.ksigns[(a1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) out(8 * l + j, db * byte_of(grid, j) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
            }
            break;
        }
        case IQ2_XS: {  // d, qs u16[32], scales[8]
            const float d = f16(b);
            const uint8_t s = b[66 + chunk];
            const float db0 = d * (0.5f + (float) (s & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (float) (s >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint16_t v = rd16(b + 2 + 2 * (4 * chunk + l));
                const uint64_t grid = T.iq2xs[v & 511];
                const uint8_t signs = T.ksigns[v >> 9];
                const float dl = l < 2 ? db0 : db1;
                for (int j = 0; j < 8; ++j) out(8 * l + j, dl * byte_of(grid, j) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
            }
            break;
        }
        case IQ2_S: {  // d, qs[64] (grid low bytes, then signs), qh[8], scales[8]
            const float d = f16(b);
            const uint8_t * qs = b + 2 + 4 * chunk;
            const uint8_t * signs = b + 2 + 32 + 4 * chunk;
            const uint8_t qh = b[66 + chunk];
            const uint8_t s = b[74 + chunk];
            const float db0 = d * (0.5f + (float) (s & 0xf)) * 0.25f;
            const float db1 = d * (0.5f + (float) (s >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint64_t grid = T.iq2s[qs[l] | ((qh << (8 - 2 * l)) & 0x300)];
                const float dl = l < 2 ? db0 : db1;
                for (int j = 0; j < 8; ++j) out(8 * l + j, dl * byte_of(grid, j) * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f));
            }
            break;
        }
        case IQ3_XXS: {  // d, qs[64] grid indices, then 32 bytes of scales and signs
            const float d = f16(b);
            const uint8_t * qs = b + 2 + 8 * chunk;
            const uint32_t a = rd32(b + 2 + 64 + 4 * chunk);
            const float db = d * (0.5f + (float) (a >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = T.ksigns[(a >> (7 * l)) & 127];
                const uint32_t g1 = T.iq3xxs[qs[2 * l + 0]];
                const uint32_t g2 = T.iq3xxs[qs[2 * l + 1]];
                for (int j = 0; j < 4; ++j) {
                    out(8 * l + j + 0, db * byte_of(g1, j) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
                    out(8 * l + j + 4, db * byte_of(g2, j) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
                }
            }
            break;
        }
        case IQ1_S: {  // d, qs[32], qh u16[8]
            const float d = f16(b);
            const uint16_t qh = rd16(b + 34 + 2 * chunk);
            const float dl = d * (float) (2 * ((qh >> 12) & 7) + 1);
            const float delta = qh & 0x8000 ? -0.125f : 0.125f;
            const uint8_t * qs = b + 2 + 4 * chunk;
            for (int l = 0; l < 4; ++l) {
                const uint64_t grid = T.iq1s[qs[l] | (((qh >> (3 * l)) & 7) << 8)];
                for (int j = 0; j < 8; ++j) out(8 * l + j, dl * ((float) (int8_t) byte_of(grid, j) + delta));
            }
            break;
        }
        case IQ4_XS: {  // d, scales_h u16, scales_l[4], qs[128]
            const float d = f16(b);
            const uint16_t sh = rd16(b + 2);
            const int ls = ((b[4 + chunk / 2] >> (4 * (chunk % 2))) & 0xf) | (((sh >> (2 * chunk)) & 3) << 4);
            const float dl = d * (float) (ls - 32);
            const uint8_t * qs = b + 8 + 16 * chunk;
            for (int j = 0; j < 16; ++j) {
                out(j + 0, dl * kvalues_iq4nl[qs[j] & 0xf]);
                out(j + 16, dl * kvalues_iq4nl[qs[j] >> 4]);
            }
            break;
        }
        case MXFP4: {  // e (E8M0), qs[16]
            const float d = e8m0_half(b[0]);
            const uint8_t * qs = b + 1;
            for (int j = 0; j < 16; ++j) {
                out(j + 0, (float) kvalues_fp4[qs[j] & 0x0F] * d);
                out(j + 16, (float) kvalues_fp4[qs[j] >> 4] * d);
            }
            break;
        }
        case NVFP4: {  // d[4] (UE4M3, one per 16 values), qs[32]
            for (int t = 0; t < 2; ++t) {
                const int s = 2 * chunk + t;
                const float d = ue4m3(b[s]);
                const uint8_t * qs = b + 4 + 8 * s;
                for (int j = 0; j < 8; ++j) {
                    out(16 * t + j + 0, (float) kvalues_fp4[qs[j] & 0x0F] * d);
                    out(16 * t + j + 8, (float) kvalues_fp4[qs[j] >> 4] * d);
                }
            }
            break;
        }
        case Q2_0: {  // d, qs[16]: 2 bits per value, 00=-1 01=0 10=+1 11=+2
            const float d = f16(b);
            const uint8_t * qs = b + 2;
            for (int j = 0; j < 32; ++j) {
                const int e = 32 * chunk + j;
                const int q = (qs[e / 4] >> ((e % 4) * 2)) & 0x03;
                out(j, (float) (q - 1) * d);
            }
            break;
        }
        default:
            for (int j = 0; j < 32; ++j) out(j, 0.0f);
            break;
    }
}

inline void decode32(uint32_t type, const uint8_t * b, int chunk, float * y, const Tabs & T) {
    decode32f(type, b, chunk, [&](int i, float v) { y[i] = v; }, T);
}
inline void decode32(uint32_t type, const uint8_t * b, int chunk, float * y) { decode32(type, b, chunk, y, Tabs{}); }

// Decode a whole tensor (or a prefix of whole blocks) to float on the device. Correctness reference for
// the fused kernels; not the fast path.
inline sycl::event dequantize(sycl::queue & q, uint32_t type, const uint8_t * src, float * dst, int64_t n) {
    const BlockShape bs = shape(type);
    if (bs.elems == 1) {
        return q.parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> i) {
            const uint8_t * p = src + (size_t) i[0] * bs.bytes;
            dst[i] = type == F32 ? sycl::bit_cast<float>(rd32(p)) : type == F16 ? f16(p) : bf16(p);
        });
    }
    return q.parallel_for(sycl::range<1>((size_t) (n / 32)), [=](sycl::id<1> i) {
        const int64_t e0 = (int64_t) i[0] * 32;
        const int64_t blk = e0 / bs.elems;
        const int chunk = (int) ((e0 % bs.elems) / 32);
        decode32(type, src + blk * bs.bytes, chunk, dst + e0);
    });
}

}  // namespace ink::q
