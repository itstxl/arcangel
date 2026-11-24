// Routed experts on the CPU, for expert offloading: the experts of some layers stay in pinned system RAM
// (INK_OFFLOAD) and, during decode, the CPU computes their gate-up, SITU and down for the chosen experts, since
// it reads system RAM ~3x faster than the GPU can over PCIe. Formats: the repacked NVFP4R and Q8_0R layouts
// (what UD-Q4_K_XL's experts use). AVX2 int8 dot products, a scalar fallback.
#pragma once

#include "model.h"

#include <atomic>
#include <cstdint>
#include <functional>
#include <thread>
#include <vector>

namespace ink {

class CpuExperts {
public:
    explicit CpuExperts(int threads = 0);  // 0: ARCANGEL_CPU_THREADS, else the physical cores
    ~CpuExperts();
    CpuExperts(const CpuExperts &) = delete;
    CpuExperts & operator=(const CpuExperts &) = delete;

    static bool supports(uint32_t ktype);

    // y[D] = sum_k wts[k] * dscale[e_k] * down_e(SITU(gate_e(x) * gscale, up_e(x) * gscale)), e_k = ids[k]
    void run(const W & gate_up, const float * gscale, const W & down, const float * dscale, const float * x,
             const int32_t * ids, const float * wts, int n_used, int n_ff, int D, float beta, float lb, float * y);
    int threads() const { return (int) workers_.size() + 1; }

private:
    void parallel(int n, const std::function<void(int, int)> & fn);  // fn(begin, end) over [0, n) in chunks
    void worker();

    std::vector<std::thread> workers_;
    std::function<void(int, int)> job_;
    int job_n_ = 0, job_chunk_ = 1;
    std::atomic<int> next_{0}, done_{0};
    std::atomic<uint32_t> gen_{0};
    std::atomic<bool> stop_{false};
    bool avx2_ = false;
    std::vector<int8_t> xq_, aq_;
    std::vector<float> xd_, ad_, act_;
};

}  // namespace ink
