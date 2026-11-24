#include "cpu_experts.h"

#include "quant.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ink {

namespace {

float g_ue4m3[256];  // UE4M3 scale byte -> float (halved, as q::ue4m3)
struct Init {
    Init() {
        for (int i = 0; i < 256; ++i) g_ue4m3[i] = q::ue4m3((uint8_t) i);
    }
} g_init;

inline float h2f(const uint8_t * p) {
    uint16_t h;
    std::memcpy(&h, p, 2);
    return (float) sycl::bit_cast<sycl::half>(h);
}

// x (n values) -> int8 in blocks of 32 with one scale each
void quantize(const float * x, int n, int8_t * q, float * d) {
    for (int b = 0; b < n / 32; ++b) {
        float amax = 0.0f;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(x[32 * b + j]));
        const float s = amax / 127.0f, is = s > 0.0f ? 1.0f / s : 0.0f;
        d[b] = s;
        for (int j = 0; j < 32; ++j) q[32 * b + j] = (int8_t) std::lrint(x[32 * b + j] * is);
    }
}

// ---- scalar reference dots (row of the repacked layout . int8 x)

float dot_nvfp4_scalar(const W & w, size_t row, int cols, const int8_t * xq, const float * xd) {
    const uint8_t * v = w.p + row * w.row_bytes;
    const uint8_t * a = w.aux + row * w.aux_row_bytes;
    float acc = 0.0f;
    for (int b = 0; b < cols / 64; ++b) {
        const uint8_t * qs = v + 32 * b;
        for (int s = 0; s < 4; ++s) {
            int32_t isum = 0;
            const int8_t * x = xq + 64 * b + 16 * s;
            for (int j = 0; j < 8; ++j) {
                isum += tables::kvalues_fp4[qs[8 * s + j] & 15] * x[j];
                isum += tables::kvalues_fp4[qs[8 * s + j] >> 4] * x[8 + j];
            }
            acc += (float) isum * g_ue4m3[a[4 * b + s]] * xd[(64 * b + 16 * s) / 32];
        }
    }
    return acc;
}

float dot_q8_scalar(const W & w, size_t row, int cols, const int8_t * xq, const float * xd) {
    const int8_t * v = (const int8_t *) (w.p + row * w.row_bytes);
    const uint8_t * a = w.aux + row * w.aux_row_bytes;
    float acc = 0.0f;
    for (int b = 0; b < cols / 32; ++b) {
        int32_t isum = 0;
        for (int j = 0; j < 32; ++j) isum += v[32 * b + j] * xq[32 * b + j];
        acc += (float) isum * h2f(a + 2 * b) * xd[b];
    }
    return acc;
}

// ---- AVX2

__attribute__((target("avx2,fma"))) inline float hsum(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    s = _mm_add_ss(s, _mm_movehdup_ps(s));
    return _mm_cvtss_f32(s);
}

// signed int8 dot of 32 bytes -> 8 int32 (pairs summed: lanes 0-3 = bytes 0-15, lanes 4-7 = bytes 16-31)
__attribute__((target("avx2,fma"))) inline __m256i dot32(__m256i w, __m256i x) {
    const __m256i p16 = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(x, w));
    return _mm256_madd_epi16(p16, _mm256_set1_epi16(1));
}

__attribute__((target("avx2,fma"))) float dot_nvfp4_avx2(const W & w, size_t row, int cols, const int8_t * xq, const float * xd) {
    const uint8_t * v = w.p + row * w.row_bytes;
    const uint8_t * a = w.aux + row * w.aux_row_bytes;
    const __m128i lut128 = _mm_loadu_si128((const __m128i *) tables::kvalues_fp4);
    const __m256i lut = _mm256_set_m128i(lut128, lut128);
    const __m128i m4 = _mm_set1_epi8(0x0F);
    __m256 acc = _mm256_setzero_ps();
    for (int c = 0; c < cols / 32; ++c) {  // 32 values = 2 sub-blocks of 16 = 16 bytes of nibbles
        const int b = c / 2, h = c % 2;
        const __m128i raw = _mm_loadu_si128((const __m128i *) (v + 32 * b + 16 * h));
        const __m128i lo = _mm_and_si128(raw, m4), hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
        // sub-block 2h: [lo bytes 0-7, hi bytes 0-7]; sub-block 2h+1: [lo bytes 8-15, hi bytes 8-15]
        const __m256i idx = _mm256_set_m128i(_mm_unpackhi_epi64(lo, hi), _mm_unpacklo_epi64(lo, hi));
        const __m256i wv = _mm256_shuffle_epi8(lut, idx);
        const __m256i xv = _mm256_loadu_si256((const __m256i *) (xq + 32 * c));
        const __m256 f = _mm256_cvtepi32_ps(dot32(wv, xv));
        const float sa = g_ue4m3[a[4 * b + 2 * h]] * xd[c], sb = g_ue4m3[a[4 * b + 2 * h + 1]] * xd[c];
        acc = _mm256_fmadd_ps(f, _mm256_set_m128(_mm_set1_ps(sb), _mm_set1_ps(sa)), acc);
    }
    return hsum(acc);
}

__attribute__((target("avx2,fma"))) float dot_q8_avx2(const W & w, size_t row, int cols, const int8_t * xq, const float * xd) {
    const uint8_t * v = w.p + row * w.row_bytes;
    const uint8_t * a = w.aux + row * w.aux_row_bytes;
    __m256 acc = _mm256_setzero_ps();
    for (int b = 0; b < cols / 32; ++b) {
        const __m256i wv = _mm256_loadu_si256((const __m256i *) (v + 32 * b));
        const __m256i xv = _mm256_loadu_si256((const __m256i *) (xq + 32 * b));
        const __m256 f = _mm256_cvtepi32_ps(dot32(wv, xv));
        acc = _mm256_fmadd_ps(f, _mm256_set1_ps(h2f(a + 2 * b) * xd[b]), acc);
    }
    return hsum(acc);
}

using DotFn = float (*)(const W &, size_t, int, const int8_t *, const float *);
DotFn pick(uint32_t ktype, bool avx2) {
    if (ktype == q::NVFP4R) return avx2 ? dot_nvfp4_avx2 : dot_nvfp4_scalar;
    if (ktype == q::Q8_0R) return avx2 ? dot_q8_avx2 : dot_q8_scalar;
    throw std::runtime_error("cpu experts: unsupported format");
}

inline float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

}  // namespace

bool CpuExperts::supports(uint32_t ktype) { return ktype == q::NVFP4R || ktype == q::Q8_0R; }

CpuExperts::CpuExperts(int threads) {
    avx2_ = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    if (threads <= 0) {
        if (const char * e = getenv("ARCANGEL_CPU_THREADS")) threads = atoi(e);
        if (threads <= 0) threads = std::max(1u, std::thread::hardware_concurrency() / 2);  // physical cores
    }
    for (int i = 1; i < threads; ++i) workers_.emplace_back([this] { worker(); });
}

CpuExperts::~CpuExperts() {
    stop_ = true;
    gen_.fetch_add(1);
    gen_.notify_all();
    for (auto & t : workers_) t.join();
}

void CpuExperts::worker() {
    uint32_t seen = 0;  // the generation at construction: no job can have been posted before the workers exist
    while (true) {
        // spin briefly (the next job usually comes within a few ms during decode), then sleep
        int spins = 0;
        while (gen_.load(std::memory_order_acquire) == seen) {
            if (++spins < 20000) _mm_pause();
            else gen_.wait(seen);
        }
        seen = gen_.load();
        if (stop_) return;
        for (int i; (i = next_.fetch_add(job_chunk_)) < job_n_;) job_(i, std::min(job_n_, i + job_chunk_));
        done_.fetch_add(1, std::memory_order_release);
    }
}

void CpuExperts::parallel(int n, const std::function<void(int, int)> & fn) {
    job_ = fn;
    job_n_ = n;
    job_chunk_ = std::max(1, n / (threads() * 8));
    next_ = 0;
    done_ = 0;
    gen_.fetch_add(1, std::memory_order_release);
    gen_.notify_all();
    for (int i; (i = next_.fetch_add(job_chunk_)) < job_n_;) job_(i, std::min(job_n_, i + job_chunk_));
    while (done_.load(std::memory_order_acquire) < (int) workers_.size()) _mm_pause();
}

void CpuExperts::run(const W & gu, const float * gscale, const W & dn, const float * dscale, const float * x,
                     const int32_t * ids, const float * wts, int nu, int nff, int D, float beta, float lb, float * y) {
    const DotFn gdot = pick(gu.ktype, avx2_), ddot = pick(dn.ktype, avx2_);
    const int gcols = (int) gu.ne[0], rows_e = (int) gu.ne[1], dcols = (int) dn.ne[0];
    xq_.resize(gcols);
    xd_.resize(gcols / 32);
    quantize(x, gcols, xq_.data(), xd_.data());
    act_.resize((size_t) nu * nff);
    // gate-up with SITU: (expert k, row i) pairs
    parallel(nu * nff, [&](int b, int e) {
        for (int r = b; r < e; ++r) {
            const int k = r / nff, i = r % nff, ex = ids[k];
            const float s = gscale ? gscale[ex] : 1.0f;
            const size_t r0 = (size_t) ex * rows_e;
            const float gg = gdot(gu, r0 + i, gcols, xq_.data(), xd_.data()) * s;
            const float uu = gdot(gu, r0 + nff + i, gcols, xq_.data(), xd_.data()) * s;
            const float g = beta * std::tanh(gg / beta) * sigmoid(gg);
            act_[r] = g * (lb > 0.0f ? lb * std::tanh(uu / lb) : uu);
        }
    });
    aq_.resize((size_t) nu * dcols);
    ad_.resize((size_t) nu * dcols / 32);
    for (int k = 0; k < nu; ++k) quantize(act_.data() + (size_t) k * nff, dcols, aq_.data() + (size_t) k * dcols, ad_.data() + (size_t) k * dcols / 32);
    // down: each output row sums the chosen experts
    parallel(D, [&](int b, int e) {
        for (int i = b; i < e; ++i) {
            float acc = 0.0f;
            for (int k = 0; k < nu; ++k) {
                const int ex = ids[k];
                acc += ddot(dn, (size_t) ex * D + i, dcols, aq_.data() + (size_t) k * dcols, ad_.data() + (size_t) k * dcols / 32) *
                       wts[k] * (dscale ? dscale[ex] : 1.0f);
            }
            y[i] = acc;
        }
    });
}

}  // namespace ink
