// bench_decode MODEL.gguf [N_TOKENS] [N_CTX] : time the single-token decode path.
// Feeds a fixed token sequence one token at a time and reports milliseconds per token after warm-up.
#include "device.h"
#include "engine.h"
#include "fast.h"
#include "gguf_file.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <string>

static int main_impl(int argc, char ** argv);

int main(int argc, char ** argv) {
    try {
        return main_impl(argc, argv);
    } catch (const std::exception & e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}

static int main_impl(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf [N_TOKENS] [N_CTX] [ref|fast|graph|profile] [START_POS]\n", argv[0]);
        return 2;
    }
    const int n = argc > 2 ? std::atoi(argv[2]) : 32;
    const int n_ctx = argc > 3 ? std::atoi(argv[3]) : 256;
    ink::GgufFile f(argv[1]);
    const bool prof = argc > 4 && std::string(argv[4]) == "profile";
    sycl::queue q = ink::make_queue(prof);
    const std::string mode = argc > 4 ? (prof ? std::string("fast") : std::string(argv[4])) : "graph";
    ink::Model model(q, f, mode != "ref");
    std::unique_ptr<ink::Engine> ref;
    std::unique_ptr<ink::FastEngine> fast;
    if (mode == "ref") ref = std::make_unique<ink::Engine>(q, model, n_ctx);
    else fast = std::make_unique<ink::FastEngine>(q, model, n_ctx, mode == "graph");
    if (fast && prof) fast->profile = true;
    if (fast && std::getenv("INK_SWA_CHUNK")) fast->swa_chunk = std::atoi(std::getenv("INK_SWA_CHUNK"));
    if (fast && std::getenv("INK_STAGE")) fast->stage_tabs = std::string(std::getenv("INK_STAGE")) != "0";
    auto step = [&](int t, int p) { if (ref) ref->step(t, p); else fast->step(t, p); };
    std::printf("engine: %s%s\n", mode.c_str(), fast && mode == "graph" && !fast->graph_active() ? " (graph unavailable)" : "");

    // START_POS > 0 times decode deep in the context; earlier positions hold whatever the cache has (timing only)
    const int start = argc > 5 ? std::atoi(argv[5]) : 0;
    const int warm = 3;
    const int toks[8] = {791, 6864, 315, 9822, 374, 12366, 13, 578};
    double total = 0;
    for (int i = 0; i < warm + n; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        step(toks[i % 8], start + i);
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (i >= warm) total += ms;
    }
    std::printf("decode at %d: %d tokens, %.2f ms/token, %.2f tok/s\n", start, n, total / n, 1000.0 * n / total);
    if (fast && prof) fast->print_profile();
    return 0;
}
