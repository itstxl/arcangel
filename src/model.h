// The Inkling 2 Mini weights on the GPU, still in their GGUF formats, plus the hyperparameters and the
// static schedule of the 48 layer visits. Shared by the reference engine and the fast engine.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <string>
#include <vector>

#include "gguf_file.h"

namespace ink {

// A weight on the device, still in its GGUF format.
struct W {
    uint32_t type = 0;   // format in the file
    uint32_t ktype = 0;  // format the kernels read: type, or a repacked layout (quant.h, e.g. Q8_0R)
    int64_t ne[4] = {1, 1, 1, 1};
    const uint8_t * p = nullptr;
    size_t row_bytes = 0;  // bytes of one ne[0]-row (repacked: bytes of one row in the value plane)
    const uint8_t * aux = nullptr;  // repacked: second plane (scales)
    size_t aux_row_bytes = 0;       // repacked: bytes of one row in the second plane
    const uint8_t * aux2 = nullptr; // repacked: third plane (per-block scales), if the layout has one
    size_t aux2_row_bytes = 0;
    bool ok() const { return p != nullptr; }
    const float * f32() const { return (const float *) p; }
};

struct LayerW {
    W attn_norm, q_inp_norm, k_inp_norm, v_inp_norm;  // SWA: attn_norm; full: three input norms
    W wq, wk, wv, wo, gate, q_clip, sink_proj, rel, rel_bias, rel_clip;
    W k_conv, v_conv, post_conv, ffn_post_conv, k_conv_proj, v_conv_proj;
    W attn_res_query, ffn_res_query, ffn_norm, gate_inp, probs_b, gate_up, gate_up_s, down, down_s;
};

struct LoopW {
    W msg_query;  // [n_embd, 4]: Q, K, V, FFN
    W down[4], gain[4], forget[4];
};

struct Hparams {
    int n_embd = 0, n_head = 0, n_head_kv = 0, head_dim = 0, n_layer = 0;
    int loop_start = 0, loop_end = 0, n_pass = 0;
    int n_expert = 0, n_used = 0, n_ff = 0, n_conv = 0, n_rel = 0, n_swa = 0, block = 0, n_vocab = 0;
    float eps = 1e-6f, qk_clip = 0, rel_clip = 0, log_floor = 0, log_alpha = 0, situ_beta = 0, situ_lb = 0;
    float w_scale = 1, softcap = 0;
    std::vector<bool> is_swa;
    int n_core() const { return loop_end - loop_start; }
    int n_slot() const { return n_layer + (n_pass - 1) * n_core(); }
    int conv_channels() const { return 2 * n_head_kv * head_dim + 2 * n_embd; }
};

// One layer visit in execution order. Vector ids index a per-token store of n_embd vectors: blocks are
// stored RMS-normalised (they are only ever read through the depth mix), `prelude` raw.
struct Visit {
    int slot = 0;                  // cache slot (KV, conv history)
    int ip = 0;                    // physical layer
    int pass = 0;                  // loop pass (0 outside the core)
    std::vector<int> blocks;       // normalised blocks the depth mix reads, before this visit
    int push = -1;                 // >= 0: store this visit's normalised input as that block id
    // loop passes > 0: the message bank (normalised block ids + the previous pass output's id)
    std::vector<int> bank;
    int restart = -1;              // first visit of a pass > 0: id to store the previous pass output in
                                   // (normalised), then x restarts from the prelude
};

struct Schedule {
    std::vector<Visit> visits;
    std::vector<int> out_blocks;  // blocks the output mix reads
    int n_vectors = 0;            // ids used
    int prelude_end = 0;          // visits before this index are the prelude; save x after them
};

class Model {
public:
    // repack: store some formats in kernel-friendly layouts (the fast engine); the reference engine needs false
    Model(sycl::queue & q, const GgufFile & f, bool repack = false);
    ~Model();
    Model(const Model &) = delete;
    Model & operator=(const Model &) = delete;

    Hparams hp;
    W tok_embd, tok_norm, output, output_norm, output_res_query;
    std::vector<LayerW> layers;
    std::vector<LoopW> loops;
    Schedule schedule;
    uint64_t weight_bytes = 0;  // on the GPU
    uint64_t host_bytes = 0;    // offloaded expert weights in pinned system RAM (INK_OFFLOAD)
    std::vector<bool> offloaded;  // per physical layer: routed experts in system RAM

    // Throws when `extra` bytes beyond the weights would not fit in the card's free memory with a margin.
    static void check_fits(sycl::queue & q, uint64_t need, const std::string & what);

private:
    sycl::queue & q_;
    std::vector<void *> allocs_;
};

}  // namespace ink
