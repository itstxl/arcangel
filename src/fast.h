// The fast decode path: ~10 fused kernels per layer visit (~17 on loop passes > 0), expert routing on
// the GPU, and the whole token recorded once as a SYCL command graph and replayed every token.
// Numerics follow the reference engine (engine.h); tests compare the two and both against llama.cpp.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "cpu_experts.h"
#include "model.h"

namespace ink {

struct MvJob;  // one matrix-vector job (kernels.h)
struct BJob;   // one batched matrix-vector job (kernels.h)

// Jobs of one weight format, launched as one format-specialised kernel.
struct MvGroup {
    uint32_t type = 0;
    MvJob * jobs = nullptr;
    int n = 0;     // jobs
    int rows = 0;  // total rows
};

class FastEngine {
public:
    // use_graph = false submits kernels directly (debugging, and the per-visit hook)
    // prompt_chunk: tokens per prompt chunk (bigger = fewer expert-weight decodes per token, more memory)
    // kv_bits: 8 (int8) or 4 (q4_0-style) K/V cache; env INK_KV overrides
    FastEngine(sycl::queue & q, const Model & m, int n_ctx, bool use_graph = true, int prompt_chunk = 256, int kv_bits = 8);
    ~FastEngine();
    FastEngine(const FastEngine &) = delete;
    FastEngine & operator=(const FastEngine &) = delete;

    const std::vector<float> & step(int token, int pos);

    // The prompt: tokens at positions pos0, pos0 + 1, ... processed in chunks of up to prefill_chunk() tokens.
    // Returns the logits after the last token; decoding continues with step(next, pos0 + tokens.size()).
    const std::vector<float> & prefill(const std::vector<int> & tokens, int pos0);
    int prefill_chunk() const { return TB_; }

    // Speculative decoding: run tokens[0..n) (n <= VMAX) at positions pos0.. in one batch and return the logits
    // after every one of them ([n][vocab]). The K/V cache is written for all n; the convolution history is not
    // advanced until commit(keep), which keeps the first `keep` tokens (decoding then continues at pos0 + keep;
    // the cache entries past it are overwritten later).
    static constexpr int VMAX = 16;
    const std::vector<float> & verify(const std::vector<int> & tokens, int pos0);
    void commit(int keep);
    bool use_xmx = true;
    bool gate_up_r = true;  // multi-row codebook gate-up (A/B switch, env INK_GUR=0)
    bool down_q8 = true;  // expert down with int8 activations (DP4A) for NVFP4 / Q2_0 (A/B switch, env INK_DOWNQ8=0)  // prompt matrix multiplies on the XMX units (A/B switch)

    // Prompt-cache snapshot: the state that is not positional (convolution histories) plus the sliding-window
    // rings, copied to host memory. Full-attention KV stays valid below the snapshot's position, so restoring a
    // snapshot taken at position p lets a prompt that shares the first p tokens continue from p.
    struct Snapshot {
        int pos = -1;
        std::vector<float> hist;
        std::vector<uint8_t> swa;   // every sliding-window slot's ring (see kv_copy), in slot order
        std::vector<uint8_t> full;  // optional: the full-attention slots' cache for positions [0, pos)
    };
    void save(Snapshot & s, int pos);
    void restore(const Snapshot & s);
    // dst = src plus the full-attention KV of positions [0, src.pos) as the engine holds it now (a whole
    // conversation, so another request can use the engine and this one can come back later)
    void stash(Snapshot & dst, const Snapshot & src);
    bool is_swa_slot(int slot) const;
    int kv_bits() const { return kv4_ ? 4 : 8; }

    // Called after each visit of a prompt chunk with "l_out-<slot>", the device array [n][n_embd] and n.
    std::function<void(const std::string & name, const float * dev, int n)> bhook;

    const Hparams & hp() const { return hp_; }
    uint64_t weight_bytes() const { return m_.weight_bytes; }
    bool graph_active() const { return !graphs_.empty(); }
    int cpu_expert_layers() const;  // offloaded layers whose experts the CPU computes during decode

    // Kernel timing by label (needs a queue from make_queue(true) and no graph); print_profile() reports
    // milliseconds per token for each label.
    bool profile = false;
    void print_profile() const;

    // Stage codebook tables in work-group local memory (A/B switch; set before the first step).
    // keys per attention work-group on sliding-window layers (64, 128 or 256; A/B switch, set before the first step)
    int swa_chunk = 64;  // measured best at 30k context (SWA attention 3.8 -> 1.7 ms vs 256)
    bool stage_tabs = false;  // measured: no gain on the B580 (the cache already serves the tables)

    // Called after each visit with "l_out-<slot>" (only without the graph).
    std::function<void(const std::string & name, const float * dev, int64_t n)> hook;

private:
    void record();            // submit one whole token's kernels to q_
    void prepare_jobs();      // build every job table once, submitting nothing
    void build_graph();       // record one token as a command graph
    void visit(const Visit & v);
    void bvisit(const Visit & v, int n, int pos0);  // a prompt chunk of n tokens
    void run_bmm(const char * label, const std::vector<BJob> & js, int ntok);
    MvJob * jobs(const std::vector<MvJob> & js);  // the next job table (built once, then reused in order)
    std::vector<MvGroup> plan_mv(const std::vector<MvJob> & js);  // group by format, build tables
    void run_mv(const char * label, const std::vector<MvGroup> & plan);

    sycl::queue & q_;
    const Model & m_;
    const Hparams & hp_;
    int n_ctx_;
    std::vector<void *> allocs_;
    template <typename T> T * dev(size_t n);

    // dynamic inputs of a token
    int32_t * token_ = nullptr;
    int32_t * pos_ = nullptr;

    // caches per slot: full-attention slots hold n_ctx positions, sliding-window slots a ring of n_swa
    // K and V cache: int8 values with one f16 scale per 32 values (q8_0-style), per slot
    bool kv4_ = false;  // 4-bit cache values
    size_t kv_val_bytes(size_t nvals) const { return kv4_ ? nvals / 2 : nvals; }
    size_t kv_bytes_(size_t ntok, size_t kvw) const;  // one slot's snapshot bytes for ntok tokens
    std::vector<int8_t *> kc_, vc_;
    std::vector<sycl::half *> ks_, vs_;
    void kv_copy(int slot, size_t tok0, size_t ntok, uint8_t * host, bool to_host);  // K, K scales, V, V scales
    std::vector<float *> hist_;
    std::vector<int> kv_len_;

    // activations
    float * xa_ = nullptr, * xb_ = nullptr, * prelude_ = nullptr, * store_ = nullptr;  // store_: normalised blocks
    float * in_ = nullptr, * inn_ = nullptr, * msg_ = nullptr, * msgn_ = nullptr, * ctl_ = nullptr, * nin_ = nullptr;
    float * g_ = nullptr, * proj_ = nullptr, * qf_ = nullptr, * rf_ = nullptr, * part_ = nullptr, * att_ = nullptr;
    float * f_ = nullptr, * fraw_ = nullptr, * fn_ = nullptr, * act_ = nullptr, * logits_dev_ = nullptr, * wts_ = nullptr;
    int32_t * ids_ = nullptr;
    int8_t * fq_ = nullptr;           // the FFN input quantised to int8 (codebook experts)
    int8_t * aq_ = nullptr;           // the experts' activations quantised to int8 (DP4A down)
    float * aqd_ = nullptr, * aqs_ = nullptr;
    float * fqd_ = nullptr, * fqs_ = nullptr;
    int max_chunks_ = 0;
    std::vector<MvJob *> job_cache_;
    size_t job_next_ = 0;
    bool jobs_only_ = false;
    float * router_logits_ = nullptr;

    using ExecGraph = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::executable>;
    using ModGraph = sycl::ext::oneapi::experimental::command_graph<sycl::ext::oneapi::experimental::graph_state::modifiable>;
    // decode graph, split into segments at the layers whose experts the CPU computes (expert offloading): replay
    // segment i, then the CPU experts of cpu_seq_[i], then segment i + 1
    std::vector<std::unique_ptr<ExecGraph>> graphs_;
    std::unique_ptr<ModGraph> rec_;  // the segment being recorded
    std::vector<int> cpu_seq_;       // physical layer of each split, in decode order
    void split(int ip);              // end the current segment (graph) or run the CPU experts now (direct)
    void cpu_experts(int ip);
    std::unique_ptr<class CpuExperts> cpu_;
    std::vector<bool> cpu_layer_;    // per physical layer: experts on the CPU during decode
    std::vector<std::vector<float>> cpu_gs_, cpu_ds_;  // per layer: the experts' gate-up / down scales (host)
    float * hx_ = nullptr;           // pinned: FFN input (D), then ids (as int32) and weights of the chosen experts
    float * hy_ = nullptr;           // pinned: the CPU's expert sum (D)
    // hot-expert cache (offload): per CPU-expert layer, the most-used experts copied to VRAM; per token the GPU
    // computes the chosen experts that are cached while the CPU computes the rest (ARCANGEL_EXPERT_CACHE_MB)
    struct ExpertCache {
        W gu, dn;                    // the cached experts' rows, same layouts as the layer's tensors
        std::vector<int> slot_e;     // slot -> expert (-1: empty)
        std::vector<double> count;   // routing counts (decayed) per expert
    };
    std::vector<ExpertCache> ecache_;  // per physical layer (empty for others)
    int cache_slots_ = 0;
    int32_t * cmap_ = nullptr;       // device [n_layer][n_expert]: expert -> slot (-1)
    int32_t * gslot_ = nullptr;      // device [8]: slot of each chosen expert (-1: on the CPU)
    float * gcoef_ = nullptr, * gsc_ = nullptr;  // device [8]: routing weight x down scale, gate-up scale
    float * ygpu_ = nullptr;         // device [D]: the cached experts' sum
    std::vector<char> seg_cache_;    // per graph segment: 1 = the cached-experts segment (runs during the CPU)
    void split_cache();
    W stage(const W & w, int ip);   // prompt path: offloaded expert tensor streamed into stage_pool_
    void reload_cache();            // after staging overwrote the cache: copy its experts back
    uint8_t * stage_pool_ = nullptr;  // = the hot-expert cache's memory (one allocation)
    size_t stage_bytes_ = 0;
    bool staged_ = false;
    void refresh_cache();
    int tokens_since_refresh_ = 0;
    bool cache_warm_ = false;
public:
    int expert_cache_slots() const { return cache_slots_; }
private:
    std::vector<float> logits_;

    // prompt chunks: token-major [TB_][...] arrays (prefill.cpp)
    int TB_ = 256;
    int32_t * btok_ = nullptr, * bids_ = nullptr, * bpairs_ = nullptr, * btiles_ = nullptr;
    float * bxa_ = nullptr, * bxb_ = nullptr, * bprelude_ = nullptr;
    sycl::half * bstore_ = nullptr;  // prompt path: the depth-mix states, f16
    float * bin_ = nullptr, * binn_ = nullptr, * bnin_ = nullptr, * bctl_ = nullptr, * bmsg_ = nullptr, * bmsgn_ = nullptr;
    float * bgain_ = nullptr, * bforget_ = nullptr, * bg_ = nullptr, * bproj_ = nullptr, * bq_ = nullptr, * br_ = nullptr;
    float * batt_ = nullptr, * bo_ = nullptr, * bf_ = nullptr, * bfraw_ = nullptr, * bfn_ = nullptr;
    float * blogits_ = nullptr, * bwts_ = nullptr;
    sycl::half * bact_ = nullptr;  // expert activations [pair][n_ff], f16 (the down GEMM reads f16 anyway)
    void P(const char * label, sycl::event e);
    // convolution-history update after a prompt chunk (verify mode: saves the chunk's inputs for commit instead)
    void bhist(const char * label, const float * raw, size_t rs, int slot, int off, int channels, int n);
    bool verify_ = false;
    int vn_ = 0;
    float * vraw_ = nullptr;     // [slot][VMAX][conv channels]: verify chunk inputs per slot
    float * vlogits_ = nullptr;  // [VMAX][vocab]
    float ** vhist_ = nullptr;   // device copy of hist_ (the per-slot pointers), for commit
    void embed(int n);           // embedding rows of btok_[0..n) into bxa_
    std::vector<float> vlogits_host_;
    std::vector<std::pair<const char *, sycl::event>> prof_events_;
    std::vector<std::pair<std::string, double>> prof_ms_;  // label -> total ms
    int prof_tokens_ = 0;
};

}  // namespace ink
