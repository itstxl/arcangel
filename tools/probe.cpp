// ink-probe MODEL.gguf : read the model's header (read-only), describe the GPU, and check that the
// model's bytes would fit in the card's free memory. Moves at most 64 MiB through the GPU.
#include "device.h"
#include "gguf_file.h"

#include <cstdio>
#include <map>
#include <string>

static double gib(uint64_t b) { return (double) b / (1ull << 30); }

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    try {
        ink::GgufFile f(argv[1]);
        const std::string arch = f.get("general.architecture").s;
        std::printf("model: %s (%s), %zu tensors, file %.2f GiB\n", f.get("general.name").s.c_str(), arch.c_str(),
                    f.tensors().size(), gib(f.file_size()));
        for (const char * k : {"block_count", "embedding_length", "expert_count", "expert_used_count", "loop.start",
                               "loop.end", "loop.count", "attention.sliding_window", "shortconv.l_cache"}) {
            const ink::GgufValue * v = f.find(arch + "." + k);
            if (v) std::printf("  %-28s %lld\n", k, (long long) v->as_int());
        }

        std::map<std::string, uint64_t> by_type;
        uint64_t total = 0;
        for (const auto & t : f.tensors()) {
            by_type[ink::type_info(t.type)->name] += t.n_bytes;
            total += t.n_bytes;
        }
        std::printf("weights: %.2f GiB\n", gib(total));
        for (const auto & [name, bytes] : by_type) std::printf("  %-8s %7.3f GiB\n", name.c_str(), gib(bytes));

        sycl::queue q = ink::make_queue();
        const ink::DeviceInfo d = ink::describe(q.get_device());
        std::printf("device: %s [%s], %u compute units, sub-groups {%s}\n", d.name.c_str(), d.backend.c_str(),
                    d.compute_units, d.sub_group_sizes.c_str());
        std::printf("  memory %.2f GiB total, %s free, max single allocation %.2f GiB\n", gib(d.total_bytes),
                    d.free_bytes ? (std::to_string(gib(d.free_bytes)).substr(0, 5) + " GiB").c_str() : "unknown",
                    gib(d.max_alloc_bytes));
        if (d.free_bytes) {
            std::printf("  weights fit with %.2f GiB left for KV, drafter and scratch\n", gib(d.free_bytes) - gib(total));
        }

        // smoke test: 64 MiB round trip through the GPU
        const size_t n = 16u << 20;
        float * dev = sycl::malloc_device<float>(n, q);
        q.fill(dev, 1.5f, n).wait();
        float check = 0;
        q.memcpy(&check, dev + n - 1, sizeof(float)).wait();
        sycl::free(dev, q);
        std::printf("smoke test: %s\n", check == 1.5f ? "ok" : "FAILED");
        return check == 1.5f ? 0 : 1;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
