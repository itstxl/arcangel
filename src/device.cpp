#include "device.h"

#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

namespace ink {

sycl::queue make_queue(bool profiling) {
    // must be set before the runtime initialises Level Zero
    setenv("ZES_ENABLE_SYSMAN", "1", 0);

    // ARCANGEL_DEVICE=n picks the n-th Level Zero GPU; otherwise the one with the most memory (skips an
    // integrated GPU, and picks the Arc card on machines with several).
    std::vector<sycl::device> gpus;
    for (const auto & dev : sycl::device::get_devices(sycl::info::device_type::gpu))
        if (dev.get_backend() == sycl::backend::ext_oneapi_level_zero) gpus.push_back(dev);
    if (gpus.empty()) throw std::runtime_error("no Level Zero GPU found (is intel-compute-runtime installed?)");
    size_t pick = 0;
    if (const char * e = getenv("ARCANGEL_DEVICE")) {
        pick = (size_t) atoi(e);
        if (pick >= gpus.size()) throw std::runtime_error("ARCANGEL_DEVICE=" + std::string(e) + ": only " +
                                                          std::to_string(gpus.size()) + " Level Zero GPU(s)");
    } else {
        for (size_t i = 1; i < gpus.size(); ++i)
            if (gpus[i].get_info<sycl::info::device::global_mem_size>() >
                gpus[pick].get_info<sycl::info::device::global_mem_size>())
                pick = i;
    }
    const sycl::device & dev = gpus[pick];
    if (profiling) return sycl::queue(dev, {sycl::property::queue::in_order{}, sycl::property::queue::enable_profiling{}});
    return sycl::queue(dev, sycl::property::queue::in_order{});
}

DeviceInfo describe(const sycl::device & dev) {
    DeviceInfo d;
    d.name = dev.get_info<sycl::info::device::name>();
    d.backend = dev.get_backend() == sycl::backend::ext_oneapi_level_zero ? "level_zero" : "other";
    d.total_bytes = dev.get_info<sycl::info::device::global_mem_size>();
    d.max_alloc_bytes = dev.get_info<sycl::info::device::max_mem_alloc_size>();
    d.compute_units = dev.get_info<sycl::info::device::max_compute_units>();
    if (!dev.get_info<sycl::info::device::host_unified_memory>() && dev.has(sycl::aspect::ext_intel_free_memory)) {
        d.free_bytes = dev.get_info<sycl::ext::intel::info::device::free_memory>();
    }
    for (size_t s : dev.get_info<sycl::info::device::sub_group_sizes>()) {
        d.sub_group_sizes += (d.sub_group_sizes.empty() ? "" : ",") + std::to_string(s);
    }
    return d;
}

}  // namespace ink
