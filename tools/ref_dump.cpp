// ref_dump : run llama.cpp (CPU, the reference) on a prompt and save its named intermediate tensors, so
// every engine layer can be compared with what llama.cpp computes.
//
//   ref_dump MODEL.gguf OUT_DIR "prompt text" [N_DECODE] [--list] [--only a,b,c]
//
// Step 0 is the prompt in one batch; steps 1..N_DECODE feed the greedy next token one at a time (the
// single-token decode path). Files: OUT_DIR/s<step>_<tensor name>.bin = int64 ne[4] then float32 values
// (ggml order, ne[0] fastest); OUT_DIR/tokens.txt; OUT_DIR/s<step>_logits.bin for the last position.
// --list prints every tensor name in the graph instead of saving. Reads the model; writes only OUT_DIR.
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <map>
#include <set>
#include <string>
#include <vector>

struct State {
    std::string dir;
    int step = 0;
    bool list = false;
    std::set<std::string> only;  // tensor base names (before "-<layer>") to save; empty = the defaults
    std::map<std::string, int> seen;
    std::set<std::string> listed;
    int saved = 0;
};

static const std::set<std::string> kDefault = {
    "inp_embd", "Qcur", "Kcur", "Vcur", "Rcur", "kqv_out", "attn_out", "ffn_inp", "ffn_moe_logits",
    "ffn_moe_probs", "ffn_moe_topk", "ffn_moe_weights_scaled", "ffn_out", "l_out", "result_norm", "result_output",
};

static std::string base_name(const char * name) {
    std::string s(name);
    const size_t dash = s.rfind('-');
    return dash == std::string::npos ? s : s.substr(0, dash);
}

static bool on_tensor(ggml_tensor * t, bool ask, void * ud) {
    auto & st = *(State *) ud;
    if (st.list) {
        if (ask && t->name[0] && st.listed.insert(t->name).second) {
            std::printf("%-40s %-6s [%lld, %lld, %lld, %lld]\n", t->name, ggml_type_name(t->type), (long long) t->ne[0],
                        (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3]);
        }
        return false;
    }
    const auto & want = st.only.empty() ? kDefault : st.only;
    const bool wanted = want.count(base_name(t->name)) > 0;
    if (ask) return wanted;
    if (!wanted) return true;
    if (!ggml_is_contiguous(t)) {
        std::fprintf(stderr, "skip %s: not contiguous\n", t->name);
        return true;
    }
    const int64_t n = ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_I32) {
        std::vector<int32_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(int32_t));
        for (int64_t i = 0; i < n; ++i) out[i] = (float) tmp[i];
    } else if (const auto * tt = ggml_get_type_traits(t->type); tt->to_float) {
        std::vector<uint8_t> raw(ggml_nbytes(t));
        ggml_backend_tensor_get(t, raw.data(), 0, raw.size());
        tt->to_float(raw.data(), out.data(), n);
    } else {
        std::fprintf(stderr, "skip %s: type %s\n", t->name, ggml_type_name(t->type));
        return true;
    }
    std::string name = "s" + std::to_string(st.step) + "_" + t->name;
    const int k = st.seen[name]++;
    if (k > 0) name += "#" + std::to_string(k);  // a name used twice in one graph
    FILE * f = std::fopen((st.dir + "/" + name + ".bin").c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "cannot write %s\n", name.c_str());
        return true;
    }
    const int64_t ne[4] = {t->ne[0], t->ne[1], t->ne[2], t->ne[3]};
    std::fwrite(ne, sizeof(ne), 1, f);
    std::fwrite(out.data(), sizeof(float), n, f);
    std::fclose(f);
    ++st.saved;
    return true;
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s MODEL.gguf OUT_DIR \"prompt\" [N_DECODE] [--list] [--only a,b]\n", argv[0]);
        return 2;
    }
    State st;
    st.dir = argv[2];
    const std::string prompt = argv[3];
    int n_decode = 0;
    for (int i = 4; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--list")) {
            st.list = true;
        } else if (!std::strcmp(argv[i], "--only") && i + 1 < argc) {
            std::string s = argv[++i];
            for (size_t p = 0; p <= s.size();) {
                const size_t q = s.find(',', p);
                st.only.insert(s.substr(p, q == std::string::npos ? std::string::npos : q - p));
                if (q == std::string::npos) break;
                p = q + 1;
            }
        } else {
            n_decode = std::atoi(argv[i]);
        }
    }

    llama_backend_init();
    // CPU only: the reference, and it must never touch the GPU (the user's server may be using it).
    // An empty device list stops weight offload; op_offload = false below stops compute offload.
    static ggml_backend_dev_t no_devices[1] = {nullptr};
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.devices = no_devices;
    llama_model * model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const llama_vocab * vocab = llama_model_get_vocab(model);

    const int n_tok = -llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), nullptr, 0, false, true);
    std::vector<llama_token> tokens(n_tok);
    llama_tokenize(vocab, prompt.c_str(), (int) prompt.size(), tokens.data(), n_tok, false, true);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 2048;
    cp.n_batch = cp.n_ubatch = 1024;
    cp.n_threads = cp.n_threads_batch = 6;
    cp.op_offload = false;
    cp.cb_eval = on_tensor;
    cp.cb_eval_user_data = &st;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;

    if (!st.list) {
        FILE * f = std::fopen((st.dir + "/tokens.txt").c_str(), "w");
        for (llama_token t : tokens) std::fprintf(f, "%d\n", t);
        std::fclose(f);
    }

    const int n_vocab = llama_vocab_n_tokens(vocab);
    auto save_logits = [&](int step) {
        const float * logits = llama_get_logits_ith(ctx, -1);
        FILE * f = std::fopen((st.dir + "/s" + std::to_string(step) + "_logits.bin").c_str(), "wb");
        const int64_t ne[4] = {n_vocab, 1, 1, 1};
        std::fwrite(ne, sizeof(ne), 1, f);
        std::fwrite(logits, sizeof(float), n_vocab, f);
        std::fclose(f);
        int best = 0;
        for (int i = 1; i < n_vocab; ++i) if (logits[i] > logits[best]) best = i;
        return best;
    };

    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), n_tok)) != 0) return 1;
    if (st.list) {
        llama_free(ctx);
        llama_model_free(model);
        return 0;
    }
    llama_token next = save_logits(0);
    std::printf("step 0: %d prompt tokens, next %d\n", n_tok, next);
    for (int s = 1; s <= n_decode; ++s) {
        st.step = s;
        if (llama_decode(ctx, llama_batch_get_one(&next, 1)) != 0) return 1;
        next = save_logits(s);
        std::printf("step %d: next %d\n", s, next);
    }
    std::printf("saved %d tensors to %s\n", st.saved, st.dir.c_str());
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
