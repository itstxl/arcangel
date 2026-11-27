// Pure C++20 CPU unit tests for Phase 1 speculative-decoding kernel math & indexing (no SYCL/GPU needed).
// Verifies:
//   1. Unpack-once DecodedU32 + dot_decoded_q8 equivalence vs unit_q8x / down_unit across all 7 DP4A formats
//   2. Batched dense lane_dot_batch<Q8_0R, VB> and <Q6_KR, VB> vs single-token lane_dot
//   3. Parallel single-WG k_vgroup expert bucketing & V_MMAX=4 splitting
//   4. SWA 17-chunk (nc_swa) wraparound & zero-atomic batched split-K attention vs single-token reference
//   5. Deferred 4-tap convolution + commit(keep) rollback
#include "quant_tables.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace {

using namespace ink;

constexpr int SG = 16;
constexpr int HD = 128;
constexpr int D = 2304;
constexpr int V_MMAX = 4;

enum Type : uint32_t {
    Q5_0R = 1006,
    Q8_0R = 1008,
    Q6_KR = 1014,
    Q4_KR = 1012,
    IQ2_XXSR = 1016,
    IQ2_XSR = 1017,
    IQ1_SR = 1019,
    IQ2_SR = 1022,
    NVFP4R = 1040,
    Q2_0R = 1042,
    Q5_KR = 1013,
};

struct Tabs {
    const uint64_t * iq2xxs = tables::iq2xxs_grid;
    const uint64_t * iq2xs = tables::iq2xs_grid;
    const uint64_t * iq2s = tables::iq2s_grid;
    const uint64_t * iq1s = tables::iq1s_grid;
    const uint8_t * ksigns = tables::ksigns_iq2xs;
};

struct RowRef {
    const uint8_t * p;
    const uint8_t * a;
    const uint8_t * a2;
};

// IEEE-754 half <-> float (round-to-nearest-even)
inline float f16_bits_to_f32(uint16_t h) {
    const uint32_t s = (uint32_t) (h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x3FFu;
    if (e == 0) {
        if (m == 0) return std::bit_cast<float>(s);
        float v = (float) m * (1.0f / (1 << 24));
        return (h & 0x8000u) ? -v : v;
    }
    if (e == 31) return std::bit_cast<float>(s | 0x7F800000u | (m << 13));
    return std::bit_cast<float>(s | ((e + 112u) << 23) | (m << 13));
}

inline uint16_t f32_to_f16_bits(float f) {
    const uint32_t x = std::bit_cast<uint32_t>(f);
    const uint32_t s = (x >> 16) & 0x8000u;
    const uint32_t e = (x >> 23) & 0xFFu;
    const uint32_t m = x & 0x7FFFFFu;
    if (e == 255) return (uint16_t) (s | 0x7C00u | (m ? 0x200u : 0u));
    const int exp = (int) e - 127 + 15;
    if (exp >= 31) return (uint16_t) (s | 0x7C00u);
    if (exp <= 0) {
        if (exp < -10) return (uint16_t) s;
        const uint32_t sig = m | 0x800000u;
        const int shift = 14 - exp;
        uint32_t r = sig >> shift;
        const uint32_t rem = sig & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        if (rem > half || (rem == half && (r & 1u))) ++r;
        return (uint16_t) (s | r);
    }
    uint32_t r = ((uint32_t) exp << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (r & 1u))) ++r;
    return (uint16_t) (s | r);
}

inline float f16(const uint8_t * p) { return f16_bits_to_f32((uint16_t) (p[0] | (p[1] << 8))); }

inline float ue4m3(uint8_t x) {
    if (x == 0x7F) return 0.0f;
    const uint32_t e = (x >> 3) & 0xFu;
    const uint32_t m = x & 0x7u;
    return e == 0 ? (float) m * (1.0f / 1024.0f) : std::bit_cast<float>(((e + 119u) << 23) | (m << 20));
}

inline uint32_t byte_mask4(uint32_t n) { return ((n * 0x00204081u) & 0x01010101u) * 0xFFu; }
inline uint32_t hi16x4(uint32_t bits4) { return ((bits4 * 0x00204081u) & 0x01010101u) << 4; }

inline int32_t fp4x4(uint32_t n) {
    const uint32_t m = n & 0x07070707u;
    const uint32_t gt4 = ((m + 0x03030303u) >> 3) & 0x01010101u;
    const uint32_t eq7 = ((m + 0x01010101u) >> 3) & 0x01010101u;
    const uint32_t mask = gt4 * 0xFFu;
    const uint32_t v = m + ((m & mask) - (0x04040404u & mask)) + 2 * eq7;
    const uint32_t neg = (0x80808080u - v) ^ 0x80808080u;
    const uint32_t smask = ((n >> 3) & 0x01010101u) * 0xFFu;
    return (int32_t) (v ^ ((v ^ neg) & smask));
}

inline int32_t signed4(uint32_t g, uint32_t m) { return (int32_t) ((g ^ m) + (m & 0x01010101u)); }

inline int32_t dot_acc(int32_t a, int32_t b, int32_t c) {
    int32_t acc = c;
    for (int i = 0; i < 4; ++i) {
        const int8_t av = (int8_t) ((uint32_t) a >> (8 * i));
        const int8_t bv = (int8_t) ((uint32_t) b >> (8 * i));
        acc += (int32_t) av * (int32_t) bv;
    }
    return acc;
}

struct int4_sim {
    int32_t v[4];
    int32_t operator[](int i) const { return v[i]; }
};

// ---------------------------------------------------------------------------------------------------
// Reference decoders from src/fast.cpp

inline float ref_nvfp4_q8_dot(const RowRef rr, int ub, const int4_sim x0, const int4_sim x1) {
    const uint32_t * wp = (const uint32_t *) (rr.p + 16 * ub);
    const uint32_t w0 = wp[0], w1 = wp[1], w2 = wp[2], w3 = wp[3];
    int32_t is0 = dot_acc(fp4x4(w0), x0[0], 0);
    is0 = dot_acc(fp4x4(w1), x0[1], is0);
    is0 = dot_acc(fp4x4(w0 >> 4), x0[2], is0);
    is0 = dot_acc(fp4x4(w1 >> 4), x0[3], is0);
    int32_t is1 = dot_acc(fp4x4(w2), x1[0], 0);
    is1 = dot_acc(fp4x4(w3), x1[1], is1);
    is1 = dot_acc(fp4x4(w2 >> 4), x1[2], is1);
    is1 = dot_acc(fp4x4(w3 >> 4), x1[3], is1);
    uint16_t sc;
    std::memcpy(&sc, rr.a + 2 * ub, 2);
    return ue4m3((uint8_t) (sc & 0xFFu)) * (float) is0 + ue4m3((uint8_t) (sc >> 8)) * (float) is1;
}

template <uint32_t T>
inline float ref_unit_q8x(const RowRef rr, int ib, const int4_sim x0, const int4_sim x1, float xdb, float xsb, const Tabs & tb) {
    if constexpr (T == NVFP4R) {
        return xdb * ref_nvfp4_q8_dot(rr, ib, x0, x1);
    } else if constexpr (T == Q5_KR) {
        const uint32_t * wp = (const uint32_t *) (rr.p + 16 * ib);
        uint32_t qh;
        uint16_t sm;
        uint32_t dd;
        std::memcpy(&qh, rr.a + 8 * ib, 4);
        std::memcpy(&sm, rr.a + 8 * ib + 4, 2);
        std::memcpy(&dd, rr.a2 + (ib / 8) * 4, 4);
        const float d = f16_bits_to_f32((uint16_t) (dd & 0xFFFFu));
        const float dmin = f16_bits_to_f32((uint16_t) (dd >> 16));
        const float sc = (float) (sm & 0xFFu);
        const float mn = (float) (sm >> 8);
        int32_t isum = dot_acc((int32_t) ((wp[0] & 0x0F0F0F0Fu) | hi16x4(qh & 0xFu)), x0[0], 0);
        isum = dot_acc((int32_t) ((wp[1] & 0x0F0F0F0Fu) | hi16x4((qh >> 4) & 0xFu)), x0[1], isum);
        isum = dot_acc((int32_t) ((wp[2] & 0x0F0F0F0Fu) | hi16x4((qh >> 8) & 0xFu)), x0[2], isum);
        isum = dot_acc((int32_t) ((wp[3] & 0x0F0F0F0Fu) | hi16x4((qh >> 12) & 0xFu)), x0[3], isum);
        isum = dot_acc((int32_t) (((wp[0] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 16) & 0xFu)), x1[0], isum);
        isum = dot_acc((int32_t) (((wp[1] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 20) & 0xFu)), x1[1], isum);
        isum = dot_acc((int32_t) (((wp[2] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 24) & 0xFu)), x1[2], isum);
        isum = dot_acc((int32_t) (((wp[3] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 28) & 0xFu)), x1[3], isum);
        return d * sc * xdb * (float) isum - dmin * mn * xsb;
    } else if constexpr (T == Q4_KR) {
        const int b = ib / 8, j = ib % 8, g = j / 2, sh = (j & 1) ? 4 : 0;
        const uint32_t * wp = (const uint32_t *) (rr.p + b * 128 + g * 32);
        uint16_t sm;
        uint32_t dd;
        std::memcpy(&sm, rr.a + ib * 2, 2);
        std::memcpy(&dd, rr.a2 + b * 4, 4);
        const float d = f16_bits_to_f32((uint16_t) (dd & 0xFFFFu));
        const float dmin = f16_bits_to_f32((uint16_t) (dd >> 16));
        int32_t isum = 0;
        for (int k = 0; k < 4; ++k) isum = dot_acc((int32_t) ((wp[k] >> sh) & 0x0F0F0F0Fu), x0[k], isum);
        for (int k = 0; k < 4; ++k) isum = dot_acc((int32_t) ((wp[4 + k] >> sh) & 0x0F0F0F0Fu), x1[k], isum);
        return d * (float) (sm & 0xFFu) * xdb * (float) isum - dmin * (float) (sm >> 8) * xsb;
    } else if constexpr (T == Q5_0R) {
        const uint32_t * wp = (const uint32_t *) (rr.p + 16 * ib);
        uint32_t qh;
        std::memcpy(&qh, rr.a + 4 * ib, 4);
        int32_t isum = dot_acc((int32_t) ((wp[0] & 0x0F0F0F0Fu) | hi16x4(qh & 0xFu)), x0[0], 0);
        isum = dot_acc((int32_t) ((wp[1] & 0x0F0F0F0Fu) | hi16x4((qh >> 4) & 0xFu)), x0[1], isum);
        isum = dot_acc((int32_t) ((wp[2] & 0x0F0F0F0Fu) | hi16x4((qh >> 8) & 0xFu)), x0[2], isum);
        isum = dot_acc((int32_t) ((wp[3] & 0x0F0F0F0Fu) | hi16x4((qh >> 12) & 0xFu)), x0[3], isum);
        isum = dot_acc((int32_t) (((wp[0] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 16) & 0xFu)), x1[0], isum);
        isum = dot_acc((int32_t) (((wp[1] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 20) & 0xFu)), x1[1], isum);
        isum = dot_acc((int32_t) (((wp[2] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 24) & 0xFu)), x1[2], isum);
        isum = dot_acc((int32_t) (((wp[3] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 28) & 0xFu)), x1[3], isum);
        return f16(rr.a2 + 2 * ib) * xdb * (float) (isum - 16 * (int32_t) xsb);
    } else if constexpr (T == Q2_0R) {
        const int32_t X[8] = {x0[0], x0[1], x0[2], x0[3], x1[0], x1[1], x1[2], x1[3]};
        const uint32_t * wv = (const uint32_t *) (rr.p + 8 * ib);
        int32_t is = 0;
        for (int w = 0; w < 2; ++w)
            for (int r = 0; r < 4; ++r) is = dot_acc((int32_t) ((wv[w] >> (2 * r)) & 0x03030303u), X[4 * w + r], is);
        return f16(rr.a + 2 * (ib / 2)) * (float) (is - (int32_t) xsb);
    } else if constexpr (T == IQ1_SR) {
        uint32_t w, qh = 0;
        uint16_t d16;
        std::memcpy(&w, rr.p + 4 * ib, 4);
        std::memcpy(&qh, rr.a + 2 * ib, 2);
        std::memcpy(&d16, rr.a2 + (ib / 8) * 2, 2);
        const float dl = f16_bits_to_f32(d16) * (float) (2 * ((qh >> 12) & 7) + 1);
        const float delta = qh & 0x8000 ? -0.125f : 0.125f;
        int32_t isum = 0;
        for (int l = 0; l < 4; ++l) {
            const uint32_t * g32 = (const uint32_t *) tb.iq1s + 2 * (((w >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8));
            const int32_t xa = l < 2 ? x0[2 * l] : x1[2 * l - 4], xb = l < 2 ? x0[2 * l + 1] : x1[2 * l - 3];
            isum = dot_acc((int32_t) g32[0], xa, isum);
            isum = dot_acc((int32_t) g32[1], xb, isum);
        }
        return dl * (xdb * (float) isum + delta * xsb);
    } else {
        uint32_t wv[2];
        std::memcpy(wv, rr.p + 8 * ib, 8);
        uint32_t aux = 0;
        uint16_t d16;
        if constexpr (T == IQ2_SR) {
            uint16_t a16;
            std::memcpy(&a16, rr.a + 2 * ib, 2);
            aux = a16;
            std::memcpy(&d16, rr.a2 + (ib / 8) * 2, 2);
        } else if constexpr (T == IQ2_XXSR) {
            std::memcpy(&d16, rr.a + (ib / 8) * 2, 2);
        } else {
            aux = rr.a[ib];
            std::memcpy(&d16, rr.a2 + (ib / 8) * 2, 2);
        }
        int32_t part[2] = {0, 0};
        for (int l = 0; l < 4; ++l) {
            const uint32_t * g32;
            uint32_t signs;
            if constexpr (T == IQ2_SR) {
                const uint32_t qh = aux & 0xFF;
                g32 = (const uint32_t *) tb.iq2s + 2 * (((wv[0] >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300));
                signs = (wv[1] >> (8 * l)) & 0xFF;
            } else if constexpr (T == IQ2_XXSR) {
                g32 = (const uint32_t *) tb.iq2xxs + 2 * ((wv[0] >> (8 * l)) & 0xFF);
                signs = tb.ksigns[(wv[1] >> (7 * l)) & 127];
            } else {
                const uint32_t v = (wv[l / 2] >> (16 * (l % 2))) & 0xFFFF;
                g32 = (const uint32_t *) tb.iq2xs + 2 * (v & 511);
                signs = tb.ksigns[v >> 9];
            }
            const int32_t xa = l < 2 ? x0[2 * l] : x1[2 * l - 4], xb = l < 2 ? x0[2 * l + 1] : x1[2 * l - 3];
            int32_t t = dot_acc(signed4(g32[0], byte_mask4(signs & 0xF)), xa, 0);
            t = dot_acc(signed4(g32[1], byte_mask4(signs >> 4)), xb, t);
            part[l / 2] += t;
        }
        float db0, db1;
        if constexpr (T == IQ2_XXSR) {
            db0 = db1 = f16_bits_to_f32(d16) * (0.5f + (float) (wv[1] >> 28)) * 0.25f;
        } else {
            const uint32_t sc = T == IQ2_SR ? (aux >> 8) & 0xFF : aux;
            const float d = f16_bits_to_f32(d16);
            db0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f;
            db1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
        }
        return xdb * (db0 * (float) part[0] + db1 * (float) part[1]);
    }
}

// ---------------------------------------------------------------------------------------------------
// Phase 1 unpack-once DecodedU32 + dot_decoded_q8

struct DecodedU32 {
    int32_t w[8];
    float s0, s1;
    float m0;
};

template <uint32_t T>
inline DecodedU32 decode_unit_q8(const RowRef rr, int ib, const Tabs & tb) {
    DecodedU32 u{};
    if constexpr (T == NVFP4R) {
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
        uint16_t sc;
        std::memcpy(&sc, rr.a + 2 * ib, 2);
        u.s0 = ue4m3((uint8_t) (sc & 0xFFu));
        u.s1 = ue4m3((uint8_t) (sc >> 8));
        u.m0 = 0.0f;
    } else if constexpr (T == Q5_KR) {
        const uint32_t * wp = (const uint32_t *) (rr.p + 16 * ib);
        uint32_t qh, dd;
        uint16_t sm;
        std::memcpy(&qh, rr.a + 8 * ib, 4);
        std::memcpy(&sm, rr.a + 8 * ib + 4, 2);
        std::memcpy(&dd, rr.a2 + (ib / 8) * 4, 4);
        const float d = f16_bits_to_f32((uint16_t) (dd & 0xFFFFu));
        const float dmin = f16_bits_to_f32((uint16_t) (dd >> 16));
        u.w[0] = (int32_t) ((wp[0] & 0x0F0F0F0Fu) | hi16x4(qh & 0xFu));
        u.w[1] = (int32_t) ((wp[1] & 0x0F0F0F0Fu) | hi16x4((qh >> 4) & 0xFu));
        u.w[2] = (int32_t) ((wp[2] & 0x0F0F0F0Fu) | hi16x4((qh >> 8) & 0xFu));
        u.w[3] = (int32_t) ((wp[3] & 0x0F0F0F0Fu) | hi16x4((qh >> 12) & 0xFu));
        u.w[4] = (int32_t) (((wp[0] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 16) & 0xFu));
        u.w[5] = (int32_t) (((wp[1] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 20) & 0xFu));
        u.w[6] = (int32_t) (((wp[2] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 24) & 0xFu));
        u.w[7] = (int32_t) (((wp[3] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 28) & 0xFu));
        u.s0 = d * (float) (sm & 0xFFu);
        u.s1 = 0.0f;
        u.m0 = -dmin * (float) (sm >> 8);
    } else if constexpr (T == Q4_KR) {
        const int b = ib / 8, j = ib % 8, g = j / 2, sh = (j & 1) ? 4 : 0;
        const uint32_t * wp = (const uint32_t *) (rr.p + b * 128 + g * 32);
        uint16_t sm;
        uint32_t dd;
        std::memcpy(&sm, rr.a + ib * 2, 2);
        std::memcpy(&dd, rr.a2 + b * 4, 4);
        const float d = f16_bits_to_f32((uint16_t) (dd & 0xFFFFu));
        const float dmin = f16_bits_to_f32((uint16_t) (dd >> 16));
        for (int k = 0; k < 8; ++k) u.w[k] = (int32_t) ((wp[k] >> sh) & 0x0F0F0F0Fu);
        u.s0 = d * (float) (sm & 0xFFu);
        u.s1 = 0.0f;
        u.m0 = -dmin * (float) (sm >> 8);
    } else if constexpr (T == Q5_0R) {
        const uint32_t * wp = (const uint32_t *) (rr.p + 16 * ib);
        uint32_t qh;
        std::memcpy(&qh, rr.a + 4 * ib, 4);
        u.w[0] = (int32_t) ((wp[0] & 0x0F0F0F0Fu) | hi16x4(qh & 0xFu));
        u.w[1] = (int32_t) ((wp[1] & 0x0F0F0F0Fu) | hi16x4((qh >> 4) & 0xFu));
        u.w[2] = (int32_t) ((wp[2] & 0x0F0F0F0Fu) | hi16x4((qh >> 8) & 0xFu));
        u.w[3] = (int32_t) ((wp[3] & 0x0F0F0F0Fu) | hi16x4((qh >> 12) & 0xFu));
        u.w[4] = (int32_t) (((wp[0] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 16) & 0xFu));
        u.w[5] = (int32_t) (((wp[1] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 20) & 0xFu));
        u.w[6] = (int32_t) (((wp[2] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 24) & 0xFu));
        u.w[7] = (int32_t) (((wp[3] >> 4) & 0x0F0F0F0Fu) | hi16x4((qh >> 28) & 0xFu));
        u.s0 = f16(rr.a2 + 2 * ib);
        u.s1 = 0.0f;
        u.m0 = 0.0f;
    } else if constexpr (T == Q2_0R) {
        const uint32_t * wv = (const uint32_t *) (rr.p + 8 * ib);
        for (int w = 0; w < 2; ++w)
            for (int r = 0; r < 4; ++r) u.w[4 * w + r] = (int32_t) ((wv[w] >> (2 * r)) & 0x03030303u);
        u.s0 = f16(rr.a + 2 * (ib / 2));
        u.s1 = 0.0f;
        u.m0 = 0.0f;
    } else if constexpr (T == IQ1_SR) {
        uint32_t w, qh = 0;
        uint16_t d16;
        std::memcpy(&w, rr.p + 4 * ib, 4);
        std::memcpy(&qh, rr.a + 2 * ib, 2);
        std::memcpy(&d16, rr.a2 + (ib / 8) * 2, 2);
        const float dl = f16_bits_to_f32(d16) * (float) (2 * ((qh >> 12) & 7) + 1);
        const float delta = qh & 0x8000 ? -0.125f : 0.125f;
        for (int l = 0; l < 4; ++l) {
            const uint32_t * g32 = (const uint32_t *) tb.iq1s + 2 * (((w >> (8 * l)) & 0xFF) | (((qh >> (3 * l)) & 7) << 8));
            u.w[2 * l + 0] = (int32_t) g32[0];
            u.w[2 * l + 1] = (int32_t) g32[1];
        }
        u.s0 = dl;
        u.s1 = 0.0f;
        u.m0 = delta;
    } else {
        uint32_t wv[2];
        std::memcpy(wv, rr.p + 8 * ib, 8);
        uint32_t aux = 0;
        uint16_t d16;
        if constexpr (T == IQ2_SR) {
            uint16_t a16;
            std::memcpy(&a16, rr.a + 2 * ib, 2);
            aux = a16;
            std::memcpy(&d16, rr.a2 + (ib / 8) * 2, 2);
        } else if constexpr (T == IQ2_XXSR) {
            std::memcpy(&d16, rr.a + (ib / 8) * 2, 2);
        } else {
            aux = rr.a[ib];
            std::memcpy(&d16, rr.a2 + (ib / 8) * 2, 2);
        }
        for (int l = 0; l < 4; ++l) {
            const uint32_t * g32;
            uint32_t signs;
            if constexpr (T == IQ2_SR) {
                const uint32_t qh = aux & 0xFF;
                g32 = (const uint32_t *) tb.iq2s + 2 * (((wv[0] >> (8 * l)) & 0xFF) | ((qh << (8 - 2 * l)) & 0x300));
                signs = (wv[1] >> (8 * l)) & 0xFF;
            } else if constexpr (T == IQ2_XXSR) {
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
        if constexpr (T == IQ2_XXSR) {
            u.s0 = u.s1 = f16_bits_to_f32(d16) * (0.5f + (float) (wv[1] >> 28)) * 0.25f;
        } else {
            const uint32_t sc = T == IQ2_SR ? (aux >> 8) & 0xFF : aux;
            const float d = f16_bits_to_f32(d16);
            u.s0 = d * (0.5f + (float) (sc & 0xF)) * 0.25f;
            u.s1 = d * (0.5f + (float) (sc >> 4)) * 0.25f;
        }
        u.m0 = 0.0f;
    }
    return u;
}

template <uint32_t T>
inline float dot_decoded_q8(const DecodedU32 & u, const int4_sim x0, const int4_sim x1, float xdb, float xsb) {
    if constexpr (T == Q5_KR || T == Q4_KR) {
        int32_t isum = 0;
        for (int i = 0; i < 4; ++i) isum = dot_acc(u.w[i], x0[i], isum);
        for (int i = 0; i < 4; ++i) isum = dot_acc(u.w[4 + i], x1[i], isum);
        return u.s0 * xdb * (float) isum + u.m0 * xsb;
    } else if constexpr (T == Q5_0R) {
        int32_t isum = 0;
        for (int i = 0; i < 4; ++i) isum = dot_acc(u.w[i], x0[i], isum);
        for (int i = 0; i < 4; ++i) isum = dot_acc(u.w[4 + i], x1[i], isum);
        return u.s0 * xdb * (float) (isum - 16 * (int32_t) xsb);
    } else if constexpr (T == Q2_0R) {
        int32_t isum = 0;
        for (int i = 0; i < 4; ++i) isum = dot_acc(u.w[i], x0[i], isum);
        for (int i = 0; i < 4; ++i) isum = dot_acc(u.w[4 + i], x1[i], isum);
        return u.s0 * xdb * (float) (isum - (int32_t) xsb);
    } else if constexpr (T == IQ1_SR) {
        int32_t isum = 0;
        for (int i = 0; i < 4; ++i) isum = dot_acc(u.w[i], x0[i], isum);
        for (int i = 0; i < 4; ++i) isum = dot_acc(u.w[4 + i], x1[i], isum);
        return u.s0 * (xdb * (float) isum + u.m0 * xsb);
    } else {
        int32_t is0 = 0, is1 = 0;
        for (int i = 0; i < 4; ++i) is0 = dot_acc(u.w[i], x0[i], is0);
        for (int i = 0; i < 4; ++i) is1 = dot_acc(u.w[4 + i], x1[i], is1);
        return xdb * (u.s0 * (float) is0 + u.s1 * (float) is1);
    }
}

template <uint32_t T>
void check_unpack_format(const char * name, std::mt19937 & rng, const Tabs & tb) {
    std::uniform_int_distribution<uint32_t> u32(0, 0xFFFFFFFFu);
    std::uniform_int_distribution<int> i8d(-127, 127);
    std::uniform_real_distribution<float> fdist(0.01f, 0.5f);
    double max_diff = 0.0;
    for (int iter = 0; iter < 500; ++iter) {
        uint8_t p[32]{}, a[8]{}, a2[4]{};
        for (int i = 0; i < 32; i += 4) {
            uint32_t r = u32(rng);
            std::memcpy(p + i, &r, 4);
        }
        for (int i = 0; i < 8; i += 4) {
            uint32_t r = u32(rng);
            std::memcpy(a + i, &r, 4);
        }
        uint16_t d16 = f32_to_f16_bits(fdist(rng));
        uint16_t dm16 = f32_to_f16_bits(fdist(rng));
        if constexpr (T == IQ2_XXSR || T == Q2_0R) {
            std::memcpy(a, &d16, 2);
        } else {
            std::memcpy(a2, &d16, 2);
            std::memcpy(a2 + 2, &dm16, 2);
        }
        const RowRef rr{p, a, a2};
        const DecodedU32 u = decode_unit_q8<T>(rr, 0, tb);
        for (int m = 0; m < 4; ++m) {
            int8_t xq[32];
            int32_t xsum = 0;
            for (int j = 0; j < 32; ++j) {
                xq[j] = (int8_t) i8d(rng);
                xsum += xq[j];
            }
            int4_sim x0{}, x1{};
            std::memcpy(x0.v, xq, 16);
            std::memcpy(x1.v, xq + 16, 16);
            const float xdb = (T == Q2_0R || T == Q5_0R) ? 1.0f : fdist(rng);
            const float xsb = (float) xsum * ((T == Q2_0R || T == Q5_0R) ? 1.0f : xdb);
            const float ref = ref_unit_q8x<T>(rr, 0, x0, x1, xdb, xsb, tb);
            const float got = dot_decoded_q8<T>(u, x0, x1, xdb, xsb);
            const double diff = std::abs((double) ref - (double) got);
            if (diff > max_diff) max_diff = diff;
            if (diff > 1e-5) {
                std::fprintf(stderr, "FAIL %s iter=%d m=%d ref=%g got=%g diff=%g\n", name, iter, m, ref, got, diff);
                std::exit(1);
            }
        }
    }
    std::printf("PASS unpack-once %-10s max_abs_diff=%.3g\n", name, max_diff);
}

// ---------------------------------------------------------------------------------------------------
// Test 2: lane_dot_batch<Q8_0R, VB> and <Q6_KR, VB>

template <uint32_t T>
float ref_lane_dot_dense(const RowRef rr, int cols, const float * x, int lane) {
    float acc = 0.0f;
    for (int c = lane * 16; c < cols; c += SG * 16) {
        int32_t wv[4];
        std::memcpy(wv, rr.p + c, 16);
        float s = 0.0f;
        for (int k = 0; k < 4; ++k) {
            const float * xv = x + c + 4 * k;
            const int w = wv[k];
            s += (float) (int8_t) (w) * xv[0] + (float) (int8_t) (w >> 8) * xv[1] +
                 (float) (int8_t) (w >> 16) * xv[2] + (float) (int8_t) (w >> 24) * xv[3];
        }
        if constexpr (T == Q8_0R) acc += s * f16(rr.a + (c / 32) * 2);
        else acc += s * (f16(rr.a2 + (c / 256) * 2) * (float) (int8_t) rr.a[c / 16]);
    }
    return acc;
}

template <uint32_t T, int VB>
void lane_dot_batch_dense(const RowRef rr, int cols, const float * x, size_t xs, int lane, float out[VB]) {
    for (int t = 0; t < VB; ++t) out[t] = 0.0f;
    for (int c = lane * 16; c < cols; c += SG * 16) {
        int32_t wv[4];
        std::memcpy(wv, rr.p + c, 16);
        const float d = (T == Q8_0R) ? f16(rr.a + (c / 32) * 2) : (f16(rr.a2 + (c / 256) * 2) * (float) (int8_t) rr.a[c / 16]);
        float s[VB]{};
        for (int k = 0; k < 4; ++k) {
            const int w = wv[k];
            const float w0 = (float) (int8_t) w, w1 = (float) (int8_t) (w >> 8);
            const float w2 = (float) (int8_t) (w >> 16), w3 = (float) (int8_t) (w >> 24);
            for (int t = 0; t < VB; ++t) {
                const float * xv = x + (size_t) t * xs + c + 4 * k;
                s[t] += w0 * xv[0] + w1 * xv[1] + w2 * xv[2] + w3 * xv[3];
            }
        }
        for (int t = 0; t < VB; ++t) out[t] += s[t] * d;
    }
}

void test_dense_batch(std::mt19937 & rng) {
    std::uniform_int_distribution<int> i8d(-127, 127);
    std::uniform_real_distribution<float> fdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> sdist(0.01f, 0.2f);
    constexpr int cols = D;
    std::vector<uint8_t> vals(cols), aux(cols / 16), aux2((cols / 256) * 2);
    for (auto & v : vals) v = (uint8_t) (int8_t) i8d(rng);
    for (int b = 0; b < cols / 32; ++b) {
        uint16_t d16 = f32_to_f16_bits(sdist(rng));
        std::memcpy(aux.data() + 2 * b, &d16, 2);
    }
    RowRef r8{vals.data(), aux.data(), nullptr};

    std::vector<uint8_t> aux6(cols / 16);
    for (auto & s : aux6) s = (uint8_t) (int8_t) (i8d(rng) / 2);
    for (int b = 0; b < cols / 256; ++b) {
        uint16_t d16 = f32_to_f16_bits(sdist(rng));
        std::memcpy(aux2.data() + 2 * b, &d16, 2);
    }
    RowRef r6{vals.data(), aux6.data(), aux2.data()};

    double max_diff = 0.0;
    for (int vb : {2, 4, 8}) {
        std::vector<float> x((size_t) vb * cols);
        for (auto & v : x) v = fdist(rng);
        auto check = [&](auto tag) {
            constexpr uint32_t T = decltype(tag)::value;
            const RowRef & rr = (T == Q8_0R) ? r8 : r6;
            float got_sum[8]{}, ref_sum[8]{};
            for (int lane = 0; lane < SG; ++lane) {
                float out[8]{};
                if (vb == 2) lane_dot_batch_dense<T, 2>(rr, cols, x.data(), cols, lane, out);
                else if (vb == 4) lane_dot_batch_dense<T, 4>(rr, cols, x.data(), cols, lane, out);
                else lane_dot_batch_dense<T, 8>(rr, cols, x.data(), cols, lane, out);
                for (int t = 0; t < vb; ++t) {
                    got_sum[t] += out[t];
                    ref_sum[t] += ref_lane_dot_dense<T>(rr, cols, x.data() + (size_t) t * cols, lane);
                }
            }
            for (int t = 0; t < vb; ++t) {
                double d = std::abs((double) got_sum[t] - (double) ref_sum[t]);
                if (d > max_diff) max_diff = d;
                if (d > 1e-5) {
                    std::fprintf(stderr, "FAIL dense batch T=%u vb=%d diff=%g\n", T, vb, d);
                    std::exit(1);
                }
            }
        };
        check(std::integral_constant<uint32_t, Q8_0R>{});
        check(std::integral_constant<uint32_t, Q6_KR>{});
    }
    std::printf("PASS dense lane_dot_batch (Q8_0R, Q6_KR; VB=2,4,8) max_abs_diff=%.3g\n", max_diff);
}

// ---------------------------------------------------------------------------------------------------
// Test 3: k_vgroup parallel single-WG bucketer & V_MMAX=4 splitter

void sim_k_vgroup(const int32_t * ids, int ntok, int ne, int nu, std::vector<int32_t> & pairs, std::vector<int32_t> & tiles) {
    const int np = ntok * nu;
    std::vector<int> cnt(ne, 0), nt(ne, 0), pair_off(ne, 0), tile_off(ne, 0);
    std::vector<std::vector<int32_t>> my_p(ne);
    for (int e = 0; e < ne; ++e) {
        for (int l = 0; l < np; ++l)
            if (ids[l] == e) my_p[e].push_back(l);
        cnt[e] = (int) my_p[e].size();
        nt[e] = (cnt[e] + V_MMAX - 1) / V_MMAX;
    }
    for (int e = 1; e < ne; ++e) {
        pair_off[e] = pair_off[e - 1] + cnt[e - 1];
        tile_off[e] = tile_off[e - 1] + nt[e - 1];
    }
    const int total_tiles = tile_off[ne - 1] + nt[ne - 1];
    tiles[0] = total_tiles;
    for (int e = 0; e < ne; ++e) {
        for (int j = 0; j < cnt[e]; ++j) pairs[pair_off[e] + j] = my_p[e][j];
        int tk = 0;
        for (int s = 0; s < cnt[e]; s += V_MMAX, ++tk) {
            const int tid = tile_off[e] + tk;
            tiles[1 + 3 * tid + 0] = e;
            tiles[1 + 3 * tid + 1] = pair_off[e] + s;
            tiles[1 + 3 * tid + 2] = std::min(V_MMAX, cnt[e] - s);
        }
    }
}

void test_k_vgroup(std::mt19937 & rng) {
    constexpr int ne = 128, nu = 8;
    std::uniform_int_distribution<int> ed(0, ne - 1);
    auto verify_case = [&](const std::vector<int32_t> & ids, int ntok, int vb) {
        std::vector<int32_t> pairs((size_t) vb * nu, -1), tiles(1 + 3 * vb * nu, -1);
        sim_k_vgroup(ids.data(), ntok, ne, nu, pairs, tiles);
        const int nt = tiles[0];
        if (nt < 0 || nt > ntok * nu) {
            std::fprintf(stderr, "FAIL k_vgroup nt=%d\n", nt);
            std::exit(1);
        }
        std::vector<bool> seen((size_t) ntok * nu, false);
        for (int t = 0; t < nt; ++t) {
            const int e = tiles[1 + 3 * t + 0];
            const int first = tiles[1 + 3 * t + 1];
            const int nb = tiles[1 + 3 * t + 2];
            if (e < 0 || e >= ne || nb < 1 || nb > V_MMAX) {
                std::fprintf(stderr, "FAIL k_vgroup tile=%d e=%d nb=%d\n", t, e, nb);
                std::exit(1);
            }
            for (int m = 0; m < nb; ++m) {
                const int pr = pairs[first + m];
                if (pr < 0 || pr >= ntok * nu || seen[pr] || ids[pr] != e) {
                    std::fprintf(stderr, "FAIL k_vgroup bad pair=%d\n", pr);
                    std::exit(1);
                }
                seen[pr] = true;
            }
        }
        for (int pr = 0; pr < ntok * nu; ++pr) {
            if (!seen[pr]) {
                std::fprintf(stderr, "FAIL k_vgroup missing pair=%d\n", pr);
                std::exit(1);
            }
        }
    };

    // Degenerate all-same-expert (all 8 tokens route to expert 17 -> two tiles of 4) and all-disjoint
    for (int vb : {2, 4, 8}) {
        std::vector<int32_t> same((size_t) vb * nu, 999);
        for (int i = 0; i < vb * nu; ++i) same[i] = (i % nu == 0) ? 17 : (i % 7);
        verify_case(same, vb, vb);

        std::vector<int32_t> disj((size_t) vb * nu);
        for (int i = 0; i < vb * nu; ++i) disj[i] = i % ne;
        verify_case(disj, vb, vb);
    }
    // 5000 random trials with ntok <= vb (padded tail contains poison expert ids)
    for (int iter = 0; iter < 5000; ++iter) {
        const int vb = (iter % 3 == 0) ? 2 : (iter % 3 == 1 ? 4 : 8);
        const int ntok = 1 + (iter % vb);
        std::vector<int32_t> ids((size_t) vb * nu, 127);
        for (int t = 0; t < ntok; ++t) {
            std::vector<int> pool(ne);
            for (int i = 0; i < ne; ++i) pool[i] = i;
            std::shuffle(pool.begin(), pool.end(), rng);
            for (int k = 0; k < nu; ++k) ids[t * nu + k] = pool[k];
        }
        verify_case(ids, ntok, vb);
    }
    std::printf("PASS k_vgroup (5000 random + degenerate cases, V_MMAX=%d)\n", V_MMAX);
}

// ---------------------------------------------------------------------------------------------------
// Test 4: Batched split-K attention + 17-chunk SWA wraparound

inline void kv_put_cpu(const float * x_head, bool kv4, int8_t * q, uint16_t * sc, size_t base_o) {
    for (int b = 0; b < HD / 32; ++b) {
        if (!kv4) {
            float amax = 0.0f;
            for (int l = 0; l < 32; ++l) amax = std::max(amax, std::abs(x_head[32 * b + l]));
            const uint16_t dh = f32_to_f16_bits(amax / 127.0f);
            const float dq = f16_bits_to_f32(dh);
            for (int l = 0; l < 32; ++l) {
                int v = dq > 0.0f ? std::clamp((int) std::nearbyint(x_head[32 * b + l] / dq), -127, 127) : 0;
                q[base_o + 32 * b + l] = (int8_t) v;
            }
            sc[base_o / 32 + b] = dh;
        } else {
            float hi = -INFINITY, lo = INFINITY;
            for (int l = 0; l < 32; ++l) {
                hi = std::max(hi, x_head[32 * b + l]);
                lo = std::min(lo, x_head[32 * b + l]);
            }
            const float vmax = hi >= -lo ? hi : lo;
            const uint16_t dh = f32_to_f16_bits(vmax / -8.0f);
            const float d = f16_bits_to_f32(dh);
            for (int l = 0; l < 32; l += 2) {
                int q0 = d != 0.0f ? std::clamp((int) std::nearbyint(x_head[32 * b + l] / d) + 8, 0, 15) : 8;
                int q1 = d != 0.0f ? std::clamp((int) std::nearbyint(x_head[32 * b + l + 1] / d) + 8, 0, 15) : 8;
                ((uint8_t *) q)[(base_o + 32 * b + l) / 2] = (uint8_t) (q0 | (q1 << 4));
            }
            sc[base_o / 32 + b] = dh;
        }
    }
}

inline void quantize_q_cpu(const float * qh, bool kv4, int32_t * sq8, float * sqd, float * sqs) {
    int8_t * dst = (int8_t *) sq8;
    for (int b = 0; b < HD / 32; ++b) {
        float amax = 0.0f;
        for (int l = 0; l < 32; ++l) amax = std::max(amax, std::abs(qh[32 * b + l]));
        const float d = amax / 127.0f, id = d > 0.0f ? 1.0f / d : 0.0f;
        int qs = 0;
        for (int l = 0; l < 32; ++l) {
            const int qv = (int) std::nearbyint(qh[32 * b + l] * id);
            qs += qv;
            const int w = l / 8, e = l % 8;
            const int at = kv4 ? (2 * w + (e & 1)) * 4 + e / 2 : l;
            dst[32 * b + at] = (int8_t) qv;
        }
        sqd[b] = d;
        sqs[b] = (float) qs;
    }
}

inline float score_dp4a_key(const int8_t * kc, const uint16_t * ksc, bool kv4, size_t kbase,
                            const int32_t * sq8, const float * sqd, const float * sqs, float scale) {
    float dot = 0.0f;
    for (int b = 0; b < HD / 32; ++b) {
        const float ds = f16_bits_to_f32(ksc[kbase / 32 + b]);
        int32_t kw[8];
        if (kv4) {
            uint32_t w4[4];
            std::memcpy(w4, (const uint8_t *) kc + (kbase + 32 * b) / 2, 16);
            for (int w = 0; w < 4; ++w) {
                kw[2 * w] = (int32_t) (w4[w] & 0x0F0F0F0Fu);
                kw[2 * w + 1] = (int32_t) ((w4[w] >> 4) & 0x0F0F0F0Fu);
            }
        } else {
            std::memcpy(kw, kc + kbase + 32 * b, 32);
        }
        int32_t is = 0;
        for (int w = 0; w < 8; ++w) is = dot_acc(kw[w], sq8[8 * b + w], is);
        const float val = kv4 ? (float) is - 8.0f * sqs[b] : (float) is;
        dot += ds * sqd[b] * val;
    }
    return dot * scale;
}

std::vector<float> ref_single_attn(const float * qh, const float * rh, float sink, float gate_h,
                                   const int8_t * kc, const int8_t * vc, const uint16_t * ksc, const uint16_t * vsc,
                                   const float * rb, int pos, int n_swa, int ring, bool swa, bool kv4) {
    constexpr int n_rel = 16;
    const int lo = swa ? std::max(0, pos - n_swa + 1) : 0;
    const float scale = 1.0f / std::sqrt((float) HD);
    int32_t sq8[HD / 4]{};
    float sqd[4]{}, sqs[4]{};
    quantize_q_cpu(qh, kv4, sq8, sqd, sqs);
    std::vector<float> sc(pos - lo + 1);
    float mx = sink;
    for (int k = lo; k <= pos; ++k) {
        const int idx = ring ? k % ring : k;
        float s = score_dp4a_key(kc, ksc, kv4, (size_t) idx * HD, sq8, sqd, sqs, scale);
        const int dist = pos - k;
        if (dist < n_swa)
            for (int r = 0; r < n_rel; ++r) s += rb[r + n_rel * dist] * rh[r];
        sc[k - lo] = s;
        mx = std::max(mx, s);
    }
    double L = std::exp((double) sink - mx);
    std::vector<double> acc(HD, 0.0);
    for (int k = lo; k <= pos; ++k) {
        const float p = std::exp(sc[k - lo] - mx);
        L += p;
        const int idx = ring ? k % ring : k;
        for (int d = 0; d < HD; ++d) {
            float vv;
            if (kv4) {
                const uint8_t by = ((const uint8_t *) vc)[((size_t) idx * HD + d) / 2];
                vv = (float) ((int) ((d & 1) ? (by >> 4) : (by & 15)) - 8);
            } else {
                vv = (float) vc[(size_t) idx * HD + d];
            }
            acc[d] += (double) p * (double) vv * (double) f16_bits_to_f32(vsc[((size_t) idx * HD + d) / 32]);
        }
    }
    std::vector<float> out(HD);
    const float sig = 1.0f / (1.0f + std::exp(-gate_h));
    for (int d = 0; d < HD; ++d) out[d] = (float) (acc[d] / L) * sig;
    return out;
}

std::vector<float> sim_k_vattn_combine(const float * q_batch, const float * r_batch, const float * sink_b, const float * gate_b,
                                       const int8_t * kc, const int8_t * vc, const uint16_t * ksc, const uint16_t * vsc,
                                       const float * rb, int pos0, int ntok, int vb, int ch, int nc,
                                       int n_swa, int ring, bool swa, bool kv4) {
    constexpr int n_rel = 16;
    const float scale = 1.0f / std::sqrt((float) HD);
    const int lo = swa ? std::max(0, pos0 - n_swa + 1) : 0;
    const int p_last = pos0 + ntok - 1;
    std::vector<int32_t> sq8((size_t) vb * (HD / 4));
    std::vector<float> sqd((size_t) vb * 4), sqs((size_t) vb * 4);
    for (int u = 0; u < ntok; ++u)
        quantize_q_cpu(q_batch + u * HD, kv4, sq8.data() + u * (HD / 4), sqd.data() + u * 4, sqs.data() + u * 4);

    // part[u][c][HD + 2]
    std::vector<float> part((size_t) vb * nc * (HD + 2), 0.0f);
    for (int c = 0; c < nc; ++c) {
        const int k0 = lo + c * ch;
        if (k0 > p_last) {
            for (int u = 0; u < vb; ++u) part[((size_t) u * nc + c) * (HD + 2) + 0] = -INFINITY;
            continue;
        }
        const int nk = std::min(ch, p_last - k0 + 1);
        std::vector<uint16_t> sp((size_t) vb * ch, 0);
        std::vector<float> mc(vb, -INFINITY), lc(vb, 0.0f);
        std::vector<float> sc((size_t) vb * ch, -INFINITY);
        for (int j = 0; j < nk; ++j) {
            const int k = k0 + j;
            const int idx = ring ? k % ring : k;
            for (int u = 0; u < ntok; ++u) {
                const int p = pos0 + u;
                if (k <= p && (!swa || p - k < n_swa)) {
                    float s = score_dp4a_key(kc, ksc, kv4, (size_t) idx * HD,
                                             sq8.data() + u * (HD / 4), sqd.data() + u * 4, sqs.data() + u * 4, scale);
                    const int dist = p - k;
                    if (dist < n_swa)
                        for (int r = 0; r < n_rel; ++r) s += rb[r + n_rel * dist] * r_batch[u * n_rel + r];
                    sc[u * ch + j] = s;
                    mc[u] = std::max(mc[u], s);
                }
            }
        }
        for (int u = 0; u < ntok; ++u) {
            float ls = 0.0f;
            for (int j = 0; j < nk; ++j) {
                const float s = sc[u * ch + j];
                const float pr = (s == -INFINITY) ? 0.0f : std::exp(s - mc[u]);
                const uint16_t ph = f32_to_f16_bits(pr);
                sp[u * ch + j] = ph;
                ls += pr;
            }
            lc[u] = ls;
            float * dst = part.data() + ((size_t) u * nc + c) * (HD + 2);
            dst[0] = mc[u];
            dst[1] = lc[u];
            for (int d = 0; d < HD; ++d) {
                float acc = 0.0f;
                for (int j = 0; j < nk; ++j) {
                    const int idx = ring ? (k0 + j) % ring : (k0 + j);
                    float vv;
                    if (kv4) {
                        const uint8_t by = ((const uint8_t *) vc)[((size_t) idx * HD + d) / 2];
                        vv = (float) ((int) ((d & 1) ? (by >> 4) : (by & 15)) - 8);
                    } else {
                        vv = (float) vc[(size_t) idx * HD + d];
                    }
                    acc += f16_bits_to_f32(sp[u * ch + j]) * vv * f16_bits_to_f32(vsc[((size_t) idx * HD + d) / 32]);
                }
                dst[2 + d] = acc;
            }
        }
    }

    std::vector<float> out((size_t) ntok * HD);
    for (int u = 0; u < ntok; ++u) {
        float M = sink_b[u];
        for (int c = 0; c < nc; ++c) M = std::max(M, part[((size_t) u * nc + c) * (HD + 2) + 0]);
        float L = std::exp(sink_b[u] - M);
        std::vector<float> acc(HD, 0.0f);
        for (int c = 0; c < nc; ++c) {
            const float * p = part.data() + ((size_t) u * nc + c) * (HD + 2);
            if (p[0] == -INFINITY) continue;
            const float w = std::exp(p[0] - M);
            L += p[1] * w;
            for (int d = 0; d < HD; ++d) acc[d] += p[2 + d] * w;
        }
        const float sig = 1.0f / (1.0f + std::exp(-gate_b[u]));
        for (int d = 0; d < HD; ++d) out[u * HD + d] = (acc[d] / L) * sig;
    }
    return out;
}

void test_swa_attention_and_17_chunks(std::mt19937 & rng) {
    constexpr int n_swa = 1024, TB = 256, ring = n_swa + TB, ch = 64, n_rel = 16;
    std::uniform_real_distribution<float> fdist(-0.5f, 0.5f);
    std::vector<float> rb((size_t) n_swa * n_rel);
    for (auto & v : rb) v = fdist(rng);

    double max_diff = 0.0;
    for (bool kv4 : {true, false}) {
        std::vector<int8_t> kc((size_t) ring * HD), vc((size_t) ring * HD);
        std::vector<uint16_t> ksc((size_t) ring * (HD / 32)), vsc((size_t) ring * (HD / 32));
        for (int pos = 0; pos < 2050; ++pos) {
            float k_h[HD], v_h[HD];
            for (int d = 0; d < HD; ++d) {
                k_h[d] = fdist(rng);
                v_h[d] = fdist(rng);
            }
            const int idx = pos % ring;
            kv_put_cpu(k_h, kv4, kc.data(), ksc.data(), (size_t) idx * HD);
            kv_put_cpu(v_h, kv4, vc.data(), vsc.data(), (size_t) idx * HD);
        }

        // First verify the 16-chunk bug at pos0 = 1024, VB = 8 (tokens 1024..1031 miss chunk 16!)
        // Check across critical SWA wraparound positions and batch sizes
        for (int vb : {2, 4, 8}) {
            const int nc_swa = (n_swa + vb - 1 + ch - 1) / ch;  // 17 for vb=8, ch=64
            for (int pos0 : {0, 5, 1018, 1020, 1023, 1024, 1026, 2000}) {
                const int ntok = (pos0 == 1024) ? vb : std::max(1, vb - 1);
                // Re-populate future keys [pos0 .. pos0 + ntok) into ring
                for (int u = 0; u < ntok; ++u) {
                    float k_h[HD], v_h[HD];
                    for (int d = 0; d < HD; ++d) {
                        k_h[d] = fdist(rng);
                        v_h[d] = fdist(rng);
                    }
                    const int idx = (pos0 + u) % ring;
                    kv_put_cpu(k_h, kv4, kc.data(), ksc.data(), (size_t) idx * HD);
                    kv_put_cpu(v_h, kv4, vc.data(), vsc.data(), (size_t) idx * HD);
                }
                std::vector<float> qb((size_t) vb * HD), rqb((size_t) vb * n_rel), sink(vb), gate(vb);
                for (auto & v : qb) v = fdist(rng);
                for (auto & v : rqb) v = fdist(rng);
                for (auto & v : sink) v = fdist(rng);
                for (auto & v : gate) v = fdist(rng);

                auto got = sim_k_vattn_combine(qb.data(), rqb.data(), sink.data(), gate.data(),
                                               kc.data(), vc.data(), ksc.data(), vsc.data(),
                                               rb.data(), pos0, ntok, vb, ch, nc_swa, n_swa, ring, true, kv4);
                for (int u = 0; u < ntok; ++u) {
                    auto ref = ref_single_attn(qb.data() + u * HD, rqb.data() + u * n_rel, sink[u], gate[u],
                                               kc.data(), vc.data(), ksc.data(), vsc.data(),
                                               rb.data(), pos0 + u, n_swa, ring, true, kv4);
                    for (int d = 0; d < HD; ++d) {
                        double diff = std::abs((double) got[u * HD + d] - (double) ref[d]);
                        if (diff > max_diff) max_diff = diff;
                        if (diff > 5e-3) {
                            std::fprintf(stderr, "FAIL SWA attn kv4=%d vb=%d pos0=%d u=%d diff=%g\n", kv4, vb, pos0, u, diff);
                            std::exit(1);
                        }
                    }
                }
                if (vb == 8 && pos0 == 1024) {
                    // Explicitly confirm nc=16 drops self-attention for pos >= 1025
                    auto bad = sim_k_vattn_combine(qb.data(), rqb.data(), sink.data(), gate.data(),
                                                   kc.data(), vc.data(), ksc.data(), vsc.data(),
                                                   rb.data(), pos0, ntok, vb, ch, 16, n_swa, ring, true, kv4);
                    double err_last = 0.0;
                    for (int d = 0; d < HD; ++d) err_last = std::max(err_last, std::abs((double) bad[7 * HD + d] - (double) got[7 * HD + d]));
                    if (err_last <= 1e-4) {
                        std::fprintf(stderr, "FAIL expected nc=16 truncation did not show mismatch!\n");
                        std::exit(1);
                    }
                }
            }
        }
    }
    std::printf("PASS SWA 17-chunk batched attention (kv4=0/1, wraparound pos=0..2000) max_abs_diff=%.3g\n", max_diff);
}

// ---------------------------------------------------------------------------------------------------
// Test 5: Deferred 4-tap convolution + commit(keep) rollback

void test_conv_rollback(std::mt19937 & rng) {
    constexpr int HC = 64, VMAX = 16;
    std::uniform_real_distribution<float> fdist(-2.0f, 2.0f);
    std::vector<float> w(4 * HC);
    for (auto & v : w) v = fdist(rng);

    std::vector<float> ref_hist(HC * 3, 0.0f);
    std::vector<float> spec_hist(HC * 3, 0.0f);
    std::vector<float> vraw((size_t) VMAX * HC, 0.0f);

    for (int step = 0; step < 200; ++step) {
        const int n = 1 + (step % 8);
        const int keep = step % (n + 1);  // exercises keep=0, partial, and full keep=n
        std::vector<float> raw((size_t) n * HC);
        for (auto & v : raw) v = fdist(rng);

        // Batched verify: compute outputs WITHOUT mutating spec_hist, save raw into vraw
        std::vector<float> spec_out((size_t) n * HC);
        for (int t = 0; t < n; ++t) {
            for (int c = 0; c < HC; ++c) {
                float cv = 0.0f;
                for (int j = 0; j < 4; ++j) {
                    const int tt = t - 3 + j;
                    const float x = tt >= 0 ? raw[(size_t) tt * HC + c] : spec_hist[c * 3 + 3 + tt];
                    cv += w[j + 4 * c] * x;
                }
                spec_out[(size_t) t * HC + c] = (raw[(size_t) t * HC + c] + cv) * 0.44721359549995793f;
                vraw[(size_t) t * HC + c] = raw[(size_t) t * HC + c];
            }
        }

        // Reference: step `keep` tokens one at a time
        std::vector<float> ref_out((size_t) keep * HC);
        for (int t = 0; t < keep; ++t) {
            for (int c = 0; c < HC; ++c) {
                float * h = ref_hist.data() + c * 3;
                const float xx[4] = {h[0], h[1], h[2], raw[(size_t) t * HC + c]};
                float cv = 0.0f;
                for (int j = 0; j < 4; ++j) cv += w[j + 4 * c] * xx[j];
                h[0] = xx[1];
                h[1] = xx[2];
                h[2] = xx[3];
                ref_out[(size_t) t * HC + c] = (xx[3] + cv) * 0.44721359549995793f;
            }
        }
        for (int i = 0; i < keep * HC; ++i) {
            if (spec_out[i] != ref_out[i]) {
                std::fprintf(stderr, "FAIL conv output mismatch step=%d i=%d\n", step, i);
                std::exit(1);
            }
        }

        // commit(keep)
        if (keep > 0) {
            for (int c = 0; c < HC; ++c) {
                float nv[3];
                for (int j = 0; j < 3; ++j) {
                    const int t = keep - 3 + j;
                    nv[j] = t >= 0 ? vraw[(size_t) t * HC + c] : spec_hist[c * 3 + 3 + t];
                }
                for (int j = 0; j < 3; ++j) spec_hist[c * 3 + j] = nv[j];
            }
        }
        for (int i = 0; i < HC * 3; ++i) {
            if (spec_hist[i] != ref_hist[i]) {
                std::fprintf(stderr, "FAIL conv history mismatch after commit(%d) step=%d\n", keep, step);
                std::exit(1);
            }
        }
    }
    std::printf("PASS deferred 4-tap conv + commit(keep) rollback (200 random steps, exact)\n");
}

}  // namespace

int main() {
    std::mt19937 rng(42);
    const Tabs tb;

    check_unpack_format<NVFP4R>("NVFP4R", rng, tb);
    check_unpack_format<Q5_KR>("Q5_KR", rng, tb);
    check_unpack_format<Q4_KR>("Q4_KR", rng, tb);
    check_unpack_format<Q5_0R>("Q5_0R", rng, tb);
    check_unpack_format<Q2_0R>("Q2_0R", rng, tb);
    check_unpack_format<IQ1_SR>("IQ1_SR", rng, tb);
    check_unpack_format<IQ2_SR>("IQ2_SR", rng, tb);
    check_unpack_format<IQ2_XSR>("IQ2_XSR", rng, tb);
    check_unpack_format<IQ2_XXSR>("IQ2_XXSR", rng, tb);

    test_dense_batch(rng);
    test_k_vgroup(rng);
    test_swa_attention_and_17_chunks(rng);
    test_conv_rollback(rng);
    return 0;
}
