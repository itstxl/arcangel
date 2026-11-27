// C API for the serving layer (server/inkling_server.py via ctypes): one engine per handle, prompt and
// decode calls returning logits, and prompt-cache snapshots. Errors return a negative value; ink_error()
// says why.
#include "device.h"
#include "fast.h"
#include "gguf_file.h"
#include "model.h"

#include <cstring>
#include <map>
#include <memory>
#include <string>

namespace {

struct Handle {
    std::unique_ptr<ink::GgufFile> file;
    sycl::queue q;
    std::unique_ptr<ink::Model> model;
    std::unique_ptr<ink::FastEngine> eng;
    std::map<int, ink::FastEngine::Snapshot> snaps;
};

thread_local std::string g_error;

template <typename F> int guarded(F && f) {
    try {
        return f();
    } catch (const std::exception & e) {
        g_error = e.what();
        return -1;
    }
}

}  // namespace

extern "C" {

const char * ink_error() { return g_error.c_str(); }

void * ink_open(const char * path, int n_ctx, int prompt_chunk, int kv_bits) {
    try {
        auto h = std::make_unique<Handle>();
        h->file = std::make_unique<ink::GgufFile>(path);
        h->q = ink::make_queue();
        h->model = std::make_unique<ink::Model>(h->q, *h->file, true);
        if (prompt_chunk > 0) {
            h->eng = std::make_unique<ink::FastEngine>(h->q, *h->model, n_ctx, true, prompt_chunk, kv_bits);
        } else {
            // auto: the largest prompt chunk whose buffers fit the free VRAM (bigger chunks read the prompt faster)
            for (int c : {1024, 512, 256, 128}) {
                try {
                    h->eng = std::make_unique<ink::FastEngine>(h->q, *h->model, n_ctx, true, c, kv_bits);
                    break;
                } catch (const std::exception & e) {
                    if (c == 128) throw;
                }
            }
        }
        return h.release();
    } catch (const std::exception & e) {
        g_error = e.what();
        return nullptr;
    }
}

void ink_close(void * hp) { delete (Handle *) hp; }

int ink_n_vocab(void * hp) { return ((Handle *) hp)->eng->hp().n_vocab; }
int ink_chunk(void * hp) { return ((Handle *) hp)->eng->prefill_chunk(); }

// Speculative decoding: run n tokens (<= 16) from pos0, logits after each of them into logits[n][vocab]; then
// ink_commit(keep) to keep the first `keep` (decoding continues at pos0 + keep).
int ink_verify(void * hp, const int32_t * toks, int n, int pos0, float * logits) {
    return guarded([&] {
        auto * h = (Handle *) hp;
        const auto & l = h->eng->verify(std::vector<int>(toks, toks + n), pos0);
        if (logits) std::memcpy(logits, l.data(), l.size() * sizeof(float));
        return 0;
    });
}

int ink_verify_argmax(void * hp, const int32_t * toks, int n, int pos0, int32_t * out_toks) {
    return guarded([&] {
        auto * h = (Handle *) hp;
        const auto out = h->eng->verify_argmax(std::vector<int>(toks, toks + n), pos0);
        if (out_toks) std::memcpy(out_toks, out.data(), out.size() * sizeof(int32_t));
        return 0;
    });
}

// Early-exit logits (n_pass loop passes) for the token at pos, state left unchanged; then call ink_step.
int ink_draft(void * hp, int32_t token, int pos, int n_pass, float * logits) {
    return guarded([&] {
        const auto & l = ((Handle *) hp)->eng->draft(token, pos, n_pass);
        if (logits) std::memcpy(logits, l.data(), l.size() * sizeof(float));
        return 0;
    });
}

int ink_commit(void * hp, int keep) {
    return guarded([&] {
        ((Handle *) hp)->eng->commit(keep);
        return 0;
    });
}

int ink_prefill(void * hp, const int32_t * toks, int n, int pos0, float * logits) {
    return guarded([&] {
        auto * h = (Handle *) hp;
        const auto & l = h->eng->prefill(std::vector<int>(toks, toks + n), pos0);
        if (logits) std::memcpy(logits, l.data(), l.size() * sizeof(float));
        return 0;
    });
}

int ink_step(void * hp, int32_t tok, int pos, float * logits) {
    return guarded([&] {
        auto * h = (Handle *) hp;
        const auto & l = h->eng->step(tok, pos);
        if (logits) std::memcpy(logits, l.data(), l.size() * sizeof(float));
        return 0;
    });
}

int ink_step_argmax(void * hp, int32_t tok, int pos) {
    return guarded([&] {
        return ((Handle *) hp)->eng->step_argmax(tok, pos);
    });
}

int ink_save(void * hp, int id, int pos) {
    return guarded([&] {
        auto * h = (Handle *) hp;
        h->eng->save(h->snaps[id], pos);
        return 0;
    });
}

// returns the snapshot's position, or -1 when there is none with this id
int ink_restore(void * hp, int id) {
    return guarded([&] {
        auto * h = (Handle *) hp;
        auto it = h->snaps.find(id);
        if (it == h->snaps.end() || it->second.pos < 0) return -1;
        h->eng->restore(it->second);
        return it->second.pos;
    });
}

// dst = snapshot src plus the full-attention KV up to src's position (the whole conversation at that point)
int ink_stash(void * hp, int src, int dst) {
    return guarded([&] {
        auto * h = (Handle *) hp;
        auto it = h->snaps.find(src);
        if (it == h->snaps.end() || it->second.pos < 0) return -1;
        ink::FastEngine::Snapshot tmp;
        h->eng->stash(tmp, it->second);
        h->snaps[dst] = std::move(tmp);
        return 0;
    });
}

int ink_drop(void * hp, int id) {
    return guarded([&] {
        ((Handle *) hp)->snaps.erase(id);
        return 0;
    });
}

}  // extern "C"
