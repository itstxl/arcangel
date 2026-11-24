// bench_prefill MODEL.gguf N_TOKENS [N_CTX] [profile] : time the prompt path on N_TOKENS tokens (a fixed
// cycle of real token ids), report tokens per second, and with `profile` the kernel time per label.
#include "device.h"
#include "fast.h"
#include "gguf_file.h"
#include "model.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int main_impl(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf N_TOKENS [N_CTX] [profile]\n", argv[0]);
        return 2;
    }
    const int n = std::atoi(argv[2]);
    const int n_ctx = argc > 3 ? std::atoi(argv[3]) : n + 64;
    const bool prof = argc > 4 && std::string(argv[4]) == "profile";
    ink::GgufFile f(argv[1]);
    sycl::queue q = ink::make_queue(prof);
    ink::Model model(q, f, true);
    const int chunk = std::getenv("INK_CHUNK") ? std::atoi(std::getenv("INK_CHUNK")) : 256;
    ink::FastEngine eng(q, model, n_ctx, false, chunk);
    const int cyc[8] = {791, 6864, 315, 9822, 374, 12366, 13, 578};
    std::vector<int> toks(n);
    for (int i = 0; i < n; ++i) toks[i] = cyc[i % 8];
    // warm-up on a short prompt (kernel first-use costs), then the timed run from position 0 again
    eng.prefill(std::vector<int>(toks.begin(), toks.begin() + std::min(n, 64)), 0);
    eng.profile = prof;
    if (std::getenv("INK_XMX")) eng.use_xmx = std::string(std::getenv("INK_XMX")) != "0";
    const auto t0 = std::chrono::steady_clock::now();
    eng.prefill(toks, 0);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("prefill: %d tokens in %.2f s, %.1f tok/s (chunks of %d)\n", n, s, n / s, eng.prefill_chunk());
    if (prof) eng.print_profile();
    return 0;
}

int main(int argc, char ** argv) {
    try {
        return main_impl(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
