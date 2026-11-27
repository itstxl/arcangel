// Native DSpark (dflash + markov RNN head) speculative drafter for Inkling 2 Mini.
// Follows dspark_phase2_design.md: 3 interleaved SWA/Full backbone layers over target feature taps
// [5, 23, 41, 48] + block-uniform bidirectional mask diffusion + serial rank-256 Markov RNN head.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "gguf_file.h"
#include "model.h"

namespace ink {

struct DsparkHparams {
    int n_embd = 0;      // 2304
    int n_ff = 0;        // 9216
    int n_layer = 0;     // 3
    int n_head = 0;      // 24
    int n_head_kv = 0;   // 8
    int head_dim = 0;    // 128
    int rope_dim = 0;    // 128
    int n_swa = 0;       // 1025
    int block_size = 0;  // 8 (1 anchor + up to 7 draft slots)
    int markov_rank = 0; // 256
    int n_vocab = 0;     // 200058
    float eps = 1e-6f;
    float rope_freq_base = 1000000.0f;
    float softcap = 30.0f;
    bool sample_from_anchor = false;
    bool has_markov_context = true;
    std::vector<bool> is_swa;
    std::vector<int> target_layers;  // [5, 23, 41, 48]

    int max_draft() const { return block_size - 1; }
};

struct DsparkLayerW {
    W ctx_mix;          // [4] F32
    W ctx_norm;         // [2304] F32
    W attn_k_ctx;       // [2304, 1024] Q8_0R
    W attn_v_ctx;       // [2304, 1024] Q8_0R
    W attn_k_ctx_norm;  // [128] F32
    W attn_norm;        // [2304] F32
    W attn_q;           // [2304, 3072] Q8_0R
    W attn_k;           // [2304, 1024] Q8_0R
    W attn_v;           // [2304, 1024] Q8_0R
    W attn_q_norm;      // [128] F32
    W attn_k_norm;      // [128] F32
    W attn_output;      // [3072, 2304] Q8_0R
    W ffn_norm;         // [2304] F32
    W ffn_gate;         // [2304, 9216] Q8_0R
    W ffn_up;           // [2304, 9216] Q8_0R
    W ffn_down;         // [9216, 2304] Q8_0R
};

class DsparkModel {
public:
    // w2_q8 = true quantizes/repacks BF16 markov_w2 [256, n_vocab] to Q8_0R at load time
    // (~97.7 MiB -> ~51.8 MiB read per Markov slot). INK_DSPARK_W2_Q8=0 overrides to keep BF16.
    DsparkModel(sycl::queue & q, const GgufFile & f, const Model & target, bool w2_q8 = true);
    ~DsparkModel();
    DsparkModel(const DsparkModel &) = delete;
    DsparkModel & operator=(const DsparkModel &) = delete;

    DsparkHparams hp;
    W tok_norm;         // [2304] F32 (applied to anchor token embedding only)
    W mask_embd;        // [2304, 7] F32
    W output_norm;      // [2304] F32
    W markov_ctx_mix;   // [4] F32
    W markov_ctx_norm;  // [2304] F32
    W joint_proj_w;     // [5120, 768] BF16
    W joint_proj_b;     // [768] F32
    W markov_w1;        // [256, 200058] BF16
    W markov_w2_w;      // [256, 200058] Q8_0R or BF16
    W markov_w2_b;      // [200058] F32
    std::vector<DsparkLayerW> layers;
    uint64_t weight_bytes = 0;

private:
    sycl::queue & q_;
    std::vector<void *> allocs_;
};

class DsparkEngine {
public:
    DsparkEngine(sycl::queue & q, const Model & target, const DsparkModel & dm,
                 int n_ctx, int kv_bits = 4, int inject_chunk = 8);
    ~DsparkEngine();
    DsparkEngine(const DsparkEngine &) = delete;
    DsparkEngine & operator=(const DsparkEngine &) = delete;

    const DsparkHparams & hp() const { return hp_; }
    int kv_bits() const { return kv4_ ? 4 : 8; }

    // Mode A: inject `n` committed tokens starting at `pos0` from target feature taps
    // stored on device in tap-major layout [4][tap_cap][D].
    void inject(const float * tgt_feats_dev, int tap_cap, int n, int pos0);

    // Mode B + C (greedy): run block-diffusion backbone on [last_tok, mask[0..k-1]] at `pos0..`
    // followed by k serial Markov RNN steps with greedy argmax; writes `k` drafted token ids to out_toks.
    void draft_greedy(int last_tok, int pos0, int k, int32_t * out_toks);

    // Mode B only (for temperature > 0): runs backbone and computes uncapped base_logits[0..k-1][V],
    // resets RNN state and f_rnn pointer for `pos0`.
    void draft_backbone(int last_tok, int pos0, int k);

    // Single serial Markov RNN step `slot` (0 <= slot < k) given `prev_tok`; updates internal RNN state,
    // adds markov_w2 bias to base_logits[slot], applies softcap=30, and copies [n_vocab] floats to out_logits.
    void draft_markov_step(int slot, int32_t prev_tok, float * out_logits);

    // Snapshot / restore / stash mirroring FastEngine's KV & SWA state.
    struct Snapshot {
        int pos = -1;
        std::vector<float> f_rnn_ring;
        std::vector<uint8_t> swa;   // SWA layers (is_swa[il]) ring contents
        std::vector<uint8_t> full;  // full-attention layers [0, pos)
    };
    void save(Snapshot & s, int pos);
    void restore(const Snapshot & s);
    void stash(Snapshot & dst, const Snapshot & src);

private:
    static constexpr int RING_RNN = 9;  // block_size + 1
    static constexpr int M_MAX = 8;     // 1 anchor + max 7 draft slots
    static constexpr int FCH = 256;     // full-attention KV chunk
    static constexpr int SCH = 128;     // SWA KV chunk

    template <typename T> T * dev(size_t n);
    size_t kv_val_bytes(size_t nvals) const { return kv4_ ? nvals / 2 : nvals; }
    size_t kv_bytes_(size_t ntok, size_t kvw) const;
    void kv_copy(int il, size_t tok0, size_t ntok, uint8_t * host, bool to_host);

    template <int NB> void inject_tile(const float * tgt_feats_dev, int tap_cap, int tok_off, int n, int pos0);
    template <int M> void record_backbone();
    void run_backbone(int last_tok, int pos0, int k);
    void run_markov_step(int slot, bool greedy_update);
    template <int K> void build_greedy_graph(int bi);

    sycl::queue & q_;
    const Model & target_;
    const DsparkModel & dm_;
    const DsparkHparams & hp_;
    int n_ctx_ = 0;
    bool kv4_ = false;
    int inject_chunk_ = 8;
    int max_chunks_ = 0;
    std::vector<void *> allocs_;

    // 3-layer KV cache (same head-major quantized layout as FastEngine)
    std::vector<int8_t *> kc_, vc_;
    std::vector<sycl::half *> ks_, vs_;
    std::vector<int> kv_len_;

    // Persistent states & RoPE frequencies
    float * inv_freq_ = nullptr;    // [HD / 2]
    float * f_rnn_ring_ = nullptr;  // [9][D]

    // Dynamic scalar parameters on device (for ExecGraph reuse)
    int32_t * d_last_tok_ = nullptr;
    int32_t * d_pos0_ = nullptr;
    int32_t * d_m_ = nullptr;       // 1 + k
    int32_t * d_k_ = nullptr;       // k
    int32_t * d_prev_tok_ = nullptr;

    // Mode B / C scratch buffers
    float * x_ = nullptr;           // [M_MAX][D]
    float * h_ = nullptr;           // [M_MAX][D]
    int8_t * hq_ = nullptr;         // [M_MAX][D]
    float * hqd_ = nullptr;         // [M_MAX][D / 32]
    float * qkv_ = nullptr;         // [M_MAX][3072 + 1024 + 1024]
    float * qf_ = nullptr;          // [M_MAX][n_head][HD]
    float * kblk_ = nullptr;        // [M_MAX][n_head_kv][HD]
    float * vblk_ = nullptr;        // [M_MAX][n_head_kv][HD]
    float * part_ = nullptr;        // [M_MAX][n_head][max_chunks_][HD + 2]
    float * att_ = nullptr;         // [M_MAX][n_head * HD]
    int8_t * attq_ = nullptr;       // [M_MAX][n_head * HD]
    float * attqd_ = nullptr;       // [M_MAX][n_head * HD / 32]
    float * act_ = nullptr;         // [M_MAX][n_ff]
    int8_t * actq_ = nullptr;       // [M_MAX][n_ff]
    float * actqd_ = nullptr;       // [M_MAX][n_ff / 32]
    float * hidden_ = nullptr;      // [M_MAX - 1][D] (slots 1..k)
    float * base_logits_ = nullptr; // [M_MAX - 1][n_vocab]

    // Mode C Markov RNN scratch
    float * rnn_state_ = nullptr;   // [markov_rank]
    float * joint_in_ = nullptr;    // [256 + 256 + D + D = 5120]
    float * joint_out_ = nullptr;   // [768]
    float * markov_ = nullptr;      // [markov_rank]
    int8_t * markov_q_ = nullptr;   // [markov_rank]
    float * markov_qd_ = nullptr;   // [markov_rank / 32]
    float * slot_logits_ = nullptr; // [n_vocab] (probabilistic step output)
    float * argmax_val_ = nullptr;  // [WG]
    int32_t * argmax_idx_ = nullptr;// [WG]
    int32_t * draft_out_ = nullptr; // [M_MAX - 1]
    float * f_rnn_cur_ = nullptr;   // points into f_rnn_ring_ via device offset or copied [D] (copied for ExecGraph!)

    // Mode A inject scratch (sized for up to 8 tokens per tile)
    float * inj_c_ = nullptr;       // [8][D]
    int8_t * inj_q_ = nullptr;      // [8][D]
    float * inj_qd_ = nullptr;      // [8][D / 32]
    float * inj_kv_ = nullptr;      // [8][2 * 1024]

    using ExecGraph = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
    using ModGraph = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable>;
    std::unique_ptr<ExecGraph> g_graphs_[2];  // 0: K=3 (M=4), 1: K=7 (M=8)
    std::unique_ptr<ExecGraph> b_graphs_[2];  // backbone-only graphs (K=3, K=7)
};

}  // namespace ink
