// test_dequant MODEL.gguf : for every weight format in the model, decode real tensors on the GPU and
// compare value for value with llama.cpp's own CPU decoder (ggml type_traits->to_float).
// Pass = every value bit-identical, or within 1e-6 relative where float rounding may differ.
#include "device.h"
#include "gguf_file.h"
#include "quant.h"

#include "ggml.h"  // from llama.cpp-inkling, linked against its libggml-base

#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <vector>

int main(int argc, char ** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]);
        return 2;
    }
    ink::GgufFile f(argv[1]);
    sycl::queue q = ink::make_queue();

    // up to three tensors per format, at most 4M values each, so every format is exercised on real data
    std::map<uint32_t, std::vector<const ink::GgufTensor *>> picks;
    for (const auto & t : f.tensors()) {
        auto & v = picks[t.type];
        if (v.size() < 3) v.push_back(&t);
    }

    int failures = 0;
    for (const auto & [type, list] : picks) {
        const auto * tt = ggml_get_type_traits((ggml_type) type);
        const ink::q::BlockShape bs = ink::q::shape(type);
        for (const ink::GgufTensor * t : list) {
            int64_t n = std::min<int64_t>(t->n_elements(), 4 << 20);
            n -= n % (bs.elems == 1 ? 32 : bs.elems);
            if (n <= 0) n = t->n_elements();
            const size_t nbytes = (size_t) (n / bs.elems) * bs.bytes;

            std::vector<float> ref(n), got(n);
            if (type == 0) {
                std::memcpy(ref.data(), t->data, n * sizeof(float));
            } else {
                tt->to_float(t->data, ref.data(), n);
            }

            uint8_t * dsrc = sycl::malloc_device<uint8_t>(nbytes, q);
            float * ddst = sycl::malloc_device<float>(n, q);
            q.memcpy(dsrc, t->data, nbytes).wait();
            ink::q::dequantize(q, type, dsrc, ddst, n).wait();
            q.memcpy(got.data(), ddst, n * sizeof(float)).wait();
            sycl::free(dsrc, q);
            sycl::free(ddst, q);

            int64_t exact = 0, close = 0, bad = 0, first_bad = -1;
            double max_abs = 0;
            for (int64_t i = 0; i < n; ++i) {
                if (std::memcmp(&ref[i], &got[i], sizeof(float)) == 0) { ++exact; continue; }
                const double a = std::fabs((double) ref[i] - got[i]);
                max_abs = std::max(max_abs, a);
                if (a <= 1e-6 * std::max(1.0, std::fabs((double) ref[i]))) { ++close; continue; }
                if (first_bad < 0) first_bad = i;
                ++bad;
            }
            const bool ok = bad == 0;
            failures += !ok;
            std::printf("%-8s %-36s %9lld values: %s (exact %lld, close %lld, wrong %lld, max diff %.3g)\n",
                        ggml_type_name((ggml_type) type), t->name.c_str(), (long long) n, ok ? "PASS" : "FAIL",
                        (long long) exact, (long long) close, (long long) bad, max_abs);
            if (!ok) {
                std::printf("         first wrong at %lld: ref %.9g got %.9g\n", (long long) first_bad,
                            ref[first_bad], got[first_bad]);
            }
        }
    }
    std::printf("%s\n", failures ? "SOME FORMATS FAILED" : "ALL FORMATS MATCH llama.cpp");
    return failures ? 1 : 0;
}
