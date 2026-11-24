// The one GPU the engine runs on, and what it can hold.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>

namespace ink {

struct DeviceInfo {
    std::string name;
    std::string backend;
    uint64_t total_bytes = 0;
    uint64_t free_bytes = 0;   // 0 when the driver does not report it
    uint64_t max_alloc_bytes = 0;
    uint32_t compute_units = 0;
    std::string sub_group_sizes;
};

// Picks the first Level Zero GPU (Intel Arc). Enables the Level Zero sysman layer first so that the
// card's real free memory can be read; that number is what the memory planner budgets against.
sycl::queue make_queue(bool profiling = false);  // profiling: kernel timestamps for FastEngine::profile
DeviceInfo describe(const sycl::device & dev);

}  // namespace ink
