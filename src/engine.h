// Inkling 2 Mini forward pass, one token at a time (the decode path). Phase 1: correctness first —
// simple kernels, float32 activations, a host sync per layer for expert routing. Every step follows
// SPEC.md; names passed to the debug hook match llama.cpp's tensor names so tests can compare.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "model.h"

namespace ink {

using DebugHook = std::function<void(const std::string & name, const float * dev, int64_t n)>;

class Engine {
public:
    Engine(sycl::queue & q, const Model & m, int n_ctx);
    ~Engine();
    Engine(const Engine &) = delete;
    Engine & operator=(const Engine &) = delete;

    // Feed one token at position pos (positions start at 0 and must be consecutive). Returns the logits
    // over the real vocabulary.
    const std::vector<float> & step(int token, int pos);

    const Hparams & hp() const { return hp_; }
    uint64_t weight_bytes() const { return m_.weight_bytes; }
    DebugHook hook;

private:
    struct Scratch;  // device work buffers

    float * vec();                  // a scratch vector of n_embd floats, valid for the current token
    float * buf(size_t n);          // a scratch buffer of n floats, valid for the current token
    void emit(const std::string & name, int slot, const float * p, int64_t n);

    void matvec(const W & w, const float * x, float * y, int64_t expert = 0);
    void rmsnorm(const float * x, const W * w, float * y, int len, int rows);
    void mix(const std::vector<const float *> & blocks, const float * cur, const W & query, float * out);
    void control(const float * cur, const float * msg, const LoopW & lw, int c, float * out);
    void conv(float * x, const W & w, const float * dyn, int heads, int slot, int offset, int channels);
    void clip(float * v, const W * scale, int rows, int len, float limit);
    void layer(int slot, int ip, float * x, std::vector<const float *> & blocks, const float * messages,
               const LoopW * lw, int pos, float tau);

    sycl::queue & q_;
    const Model & m_;
    const Hparams & hp_;
    int n_ctx_;
    std::vector<void *> allocs_;

    std::vector<float *> k_cache_, v_cache_, conv_hist_;  // per cache slot
    float * pool_ = nullptr;
    size_t pool_size_ = 0, pool_used_ = 0;
    int32_t * ids_dev_ = nullptr;
    float * wts_dev_ = nullptr;
    std::vector<float> logits_;
};

}  // namespace ink
