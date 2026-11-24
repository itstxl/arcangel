// gguf-retype IN.gguf OUT.gguf NAME=TYPE [NAME=TYPE ...]
//
// Copies a GGUF and re-quantises the named tensors (dequantise with ggml, quantise to TYPE: q8_0, q5_k, bf16,
// f16, ...). Used to test on a 12 GB card the weight formats that only the bigger quants use, e.g. the Q8_0
// expert tensors of UD-Q4_K_XL: retype a few expert tensors of UD-Q2_K_XL, then compare the engine with
// llama.cpp on the result (ref-dump + test-forward). Plain CPU code, linked against ggml.
#include "ggml.h"
#include "gguf.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static ggml_type parse_type(const std::string & s) {
    for (int t = 0; t < GGML_TYPE_COUNT; ++t) {
        const char * n = ggml_type_name((ggml_type) t);
        if (n && strcasecmp(n, s.c_str()) == 0) return (ggml_type) t;
    }
    fprintf(stderr, "unknown type %s\n", s.c_str());
    exit(2);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s IN.gguf OUT.gguf NAME=TYPE [...]\n", argv[0]);
        return 2;
    }
    std::map<std::string, ggml_type> want;
    for (int i = 3; i < argc; ++i) {
        std::string a = argv[i];
        auto eq = a.find('=');
        if (eq == std::string::npos) return 2;
        want[a.substr(0, eq)] = parse_type(a.substr(eq + 1));
    }

    ggml_context * meta = nullptr;
    gguf_init_params ip = {/*no_alloc*/ true, &meta};
    gguf_context * src = gguf_init_from_file(argv[1], ip);
    if (!src) return 1;
    int fd = open(argv[1], O_RDONLY);
    struct stat st;
    fstat(fd, &st);
    const uint8_t * base = (const uint8_t *) mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    const size_t data0 = gguf_get_data_offset(src);

    gguf_context * dst = gguf_init_empty();
    gguf_set_kv(dst, src);
    const int64_t n = gguf_get_n_tensors(src);
    ggml_init_params cp = {(size_t) n * ggml_tensor_overhead() + (1 << 20), nullptr, true};
    ggml_context * nctx = ggml_init(cp);
    std::vector<std::vector<uint8_t>> owned(n);
    std::vector<const void *> datas(n);
    std::vector<size_t> sizes(n);
    for (int64_t i = 0; i < n; ++i) {
        const char * name = gguf_get_tensor_name(src, i);
        ggml_tensor * t = ggml_get_tensor(meta, name);
        const uint8_t * p = base + data0 + gguf_get_tensor_offset(src, i);
        ggml_tensor * nt = ggml_dup_tensor(nctx, t);
        ggml_set_name(nt, name);
        auto it = want.find(name);
        if (it == want.end()) {
            datas[i] = p;
            sizes[i] = ggml_nbytes(t);
        } else {
            const int64_t cols = t->ne[0], rows = ggml_nelements(t) / cols;
            nt->type = it->second;
            nt->nb[0] = ggml_type_size(nt->type);
            nt->nb[1] = nt->nb[0] * (cols / ggml_blck_size(nt->type));
            for (int d = 2; d < GGML_MAX_DIMS; ++d) nt->nb[d] = nt->nb[d - 1] * nt->ne[d - 1];
            const auto * tr = ggml_get_type_traits(t->type);
            std::vector<float> f((size_t) cols);
            owned[i].resize(ggml_nbytes(nt));
            const size_t rb_in = ggml_row_size(t->type, cols), rb_out = ggml_row_size(nt->type, cols);
            for (int64_t r = 0; r < rows; ++r) {
                if (t->type == GGML_TYPE_F32) std::memcpy(f.data(), p + r * rb_in, cols * 4);
                else tr->to_float(p + r * rb_in, f.data(), cols);
                ggml_quantize_chunk(nt->type, f.data(), owned[i].data() + r * rb_out, 0, 1, cols, nullptr);
            }
            datas[i] = owned[i].data();
            sizes[i] = owned[i].size();
            printf("%s: %s -> %s (%.0f -> %.0f MiB)\n", name, ggml_type_name(t->type), ggml_type_name(nt->type),
                   ggml_nbytes(t) / 1048576.0, sizes[i] / 1048576.0);
            want.erase(it);
        }
        gguf_add_tensor(dst, nt);
    }
    for (auto & [k, v] : want) fprintf(stderr, "warning: no tensor named %s\n", k.c_str());

    FILE * out = fopen(argv[2], "wb");
    if (!out || !gguf_write_to_file_ptr(dst, out, /*only_meta*/ true)) return 1;
    const size_t align = gguf_get_alignment(dst);
    std::vector<uint8_t> pad(align, 0);
    for (int64_t i = 0; i < n; ++i) {
        fwrite(datas[i], 1, sizes[i], out);
        const size_t rem = sizes[i] % align;
        if (rem) fwrite(pad.data(), 1, align - rem, out);
    }
    fclose(out);
    printf("wrote %s\n", argv[2]);
    return 0;
}
