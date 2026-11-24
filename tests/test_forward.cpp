// test_forward MODEL.gguf REF_DIR : run the engine token by token through the prompt saved by ref_dump
// (plus its decode steps) and compare every layer visit with llama.cpp's tensors.
//
// llama.cpp's CPU path quantises activations to 8 bits before each quantised matmul, the engine keeps
// float32, so values differ slightly; what matters is that the error stays small through 48 visits and
// that the final token distribution matches (top-1 and KL divergence).
#include "device.h"
#include "engine.h"
#include "fast.h"
#include "gguf_file.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <string>
#include <vector>

struct Ref {
    int64_t ne[4] = {0, 0, 0, 0};
    std::vector<float> v;
    bool ok = false;
};

static Ref load_ref(const std::string & path) {
    Ref r;
    std::ifstream f(path, std::ios::binary);
    if (!f) return r;
    f.read((char *) r.ne, sizeof(r.ne));
    const int64_t n = r.ne[0] * r.ne[1] * r.ne[2] * r.ne[3];
    r.v.resize(n);
    f.read((char *) r.v.data(), n * sizeof(float));
    r.ok = (bool) f;
    return r;
}

struct Score {
    double cos_min = 1, cos_sum = 0, rel_max = 0;
    int n = 0;
};

template <typename E> int run(E & eng, sycl::queue & q, const std::string & dir);
static int run_prefill(ink::FastEngine & eng, sycl::queue & q, const std::string & dir);

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
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s MODEL.gguf REF_DIR [ref|fast|graph]\n", argv[0]);
        return 2;
    }
    const std::string dir = argv[2];
    const std::string mode = argc > 3 ? argv[3] : "ref";
    ink::GgufFile f(argv[1]);
    sycl::queue q = ink::make_queue();
    ink::Model model(q, f, mode != "ref");
    std::printf("loaded %.2f GiB of weights, engine: %s\n", model.weight_bytes / double(1 << 30), mode.c_str());
    if (mode == "ref") {
        ink::Engine eng(q, model, 64);
        return run(eng, q, dir);
    }
    ink::FastEngine eng(q, model, 64, mode == "graph");
    std::printf("command graph: %s\n", eng.graph_active() ? "on" : "off");
    if (mode == "prefill") return run_prefill(eng, q, dir);
    return run(eng, q, dir);
}

template <typename E> int run(E & eng, sycl::queue & q, const std::string & dir) {

    std::vector<int> tokens;
    {
        std::ifstream tf(dir + "/tokens.txt");
        for (int t; tf >> t;) tokens.push_back(t);
    }

    std::map<std::string, Ref> cache;
    std::map<std::string, Score> scores;  // by tensor name (without step)
    int step = 0, col = 0;
    int64_t step_tokens = 1;  // tokens llama.cpp processed in this step's batch
    std::vector<float> host;
    eng.hook = [&](const std::string & name, const float * dev, int64_t n) {
        const std::string base = name.substr(0, name.find('-'));
        if (base != "l_out" && base != "kqv_out" && base != "ffn_out" && base != "attn_out" && base != "result_norm") return;
        const std::string file = dir + "/s" + std::to_string(step) + "_" + name + ".bin";
        auto it = cache.find(file);
        if (it == cache.end()) it = cache.emplace(file, load_ref(file)).first;
        const Ref & r = it->second;
        if (!r.ok || r.ne[0] != n) return;
        // llama.cpp keeps some tensors (result_norm) only for the last tokens of a batch: align from the end
        const int64_t ci = col - (step_tokens - r.ne[1]);
        if (ci < 0 || ci >= r.ne[1]) return;
        host.resize(n);
        q.memcpy(host.data(), dev, n * sizeof(float)).wait();
        const float * ref = r.v.data() + (size_t) ci * n;
        double dot = 0, na = 0, nb = 0, nd = 0;
        for (int64_t i = 0; i < n; ++i) {
            dot += (double) host[i] * ref[i];
            na += (double) host[i] * host[i];
            nb += (double) ref[i] * ref[i];
            nd += ((double) host[i] - ref[i]) * ((double) host[i] - ref[i]);
        }
        const double c = dot / (std::sqrt(na * nb) + 1e-30);
        Score & s = scores[name];
        s.cos_min = std::min(s.cos_min, c);
        s.cos_sum += c;
        s.rel_max = std::max(s.rel_max, std::sqrt(nd / (nb + 1e-30)));
        s.n++;
    };

    auto compare_logits = [&](const std::vector<float> & ours, int st) {
        Ref r = load_ref(dir + "/s" + std::to_string(st) + "_logits.bin");
        if (!r.ok) return;
        const int n = (int) std::min<size_t>(ours.size(), r.v.size());
        auto softmax = [&](const float * l, std::vector<double> & p) {
            double m = -1e30;
            for (int i = 0; i < n; ++i) m = std::max(m, (double) l[i]);
            double s = 0;
            p.resize(n);
            for (int i = 0; i < n; ++i) s += (p[i] = std::exp(l[i] - m));
            for (double & v : p) v /= s;
        };
        std::vector<double> pr, po;
        softmax(r.v.data(), pr);
        softmax(ours.data(), po);
        double kl = 0;
        int top_r = 0, top_o = 0;
        for (int i = 0; i < n; ++i) {
            if (pr[i] > 0) kl += pr[i] * std::log(pr[i] / std::max(po[i], 1e-30));
            if (pr[i] > pr[top_r]) top_r = i;
            if (po[i] > po[top_o]) top_o = i;
        }
        std::printf("step %d logits: llama.cpp top %d (p=%.3f), engine top %d (p=%.3f), KL %.5f -> %s\n", st, top_r,
                    pr[top_r], top_o, po[top_o], kl, top_r == top_o ? "MATCH" : "DIFFERENT");
    };

    std::vector<float> logits;
    step_tokens = (int64_t) tokens.size();
    for (size_t t = 0; t < tokens.size(); ++t) {
        step = 0;
        col = (int) t;
        logits = eng.step(tokens[t], (int) t);
    }
    compare_logits(logits, 0);
    // decode steps: feed the tokens llama.cpp chose
    int pos = (int) tokens.size();
    for (int st = 1; st <= 2; ++st) {
        Ref r = load_ref(dir + "/s" + std::to_string(st - 1) + "_logits.bin");
        if (!r.ok) break;
        int best = 0;
        for (size_t i = 1; i < r.v.size(); ++i) if (r.v[i] > r.v[best]) best = (int) i;
        step = st;
        col = 0;
        step_tokens = 1;
        logits = eng.step(best, pos++);
        compare_logits(logits, st);
    }

    // per visit, in execution order
    const auto & hp = eng.hp();
    std::vector<int> order;
    for (int i = 0; i < hp.loop_start; ++i) order.push_back(i);
    for (int i = hp.loop_start; i < hp.loop_end; ++i) order.push_back(i);
    for (int p = 1; p < hp.n_pass; ++p)
        for (int i = 0; i < hp.loop_end - hp.loop_start; ++i) order.push_back(hp.n_layer + (p - 1) * (hp.loop_end - hp.loop_start) + i);
    for (int i = hp.loop_end; i < hp.n_layer; ++i) order.push_back(i);
    std::printf("\nvisit slot   l_out cos(min/mean) rel.err | kqv_out cos min | ffn_out cos min\n");
    int v = 0;
    for (int slot : order) {
        const auto & lo = scores["l_out-" + std::to_string(slot)];
        const auto & ko = scores["kqv_out-" + std::to_string(slot)];
        const auto & fo = scores["ffn_out-" + std::to_string(slot)];
        std::printf("%5d %4d   %.5f / %.5f  %7.4f | %.5f        | %.5f   (%d tokens)\n", v++, slot, lo.cos_min,
                    lo.n ? lo.cos_sum / lo.n : 0.0, lo.rel_max, ko.cos_min, fo.cos_min, lo.n);
    }
    const auto & rn = scores["result_norm"];
    std::printf("result_norm cos min %.5f\n", rn.cos_min);
    return 0;
}

// The prompt as one batched chunk: every visit compared over all prompt tokens, then two decode steps.
static int run_prefill(ink::FastEngine & eng, sycl::queue & q, const std::string & dir) {
    std::vector<int> tokens;
    {
        std::ifstream tf(dir + "/tokens.txt");
        for (int t; tf >> t;) tokens.push_back(t);
    }
    std::map<std::string, std::pair<double, double>> res;  // name -> (min cos, max rel err)
    std::vector<float> host;
    eng.bhook = [&](const std::string & name, const float * dev, int n) {
        Ref r = load_ref(dir + "/s0_" + name + ".bin");
        const int64_t d = 2304;
        if (!r.ok || r.ne[0] != d || r.ne[1] != n) return;
        host.resize((size_t) n * d);
        q.memcpy(host.data(), dev, host.size() * sizeof(float)).wait();
        double cmin = 1, rmax = 0;
        for (int t = 0; t < n; ++t) {
            double dot = 0, na = 0, nb = 0, nd = 0;
            for (int64_t i = 0; i < d; ++i) {
                const double a = host[(size_t) t * d + i], b = r.v[(size_t) t * d + i];
                dot += a * b; na += a * a; nb += b * b; nd += (a - b) * (a - b);
            }
            cmin = std::min(cmin, dot / (std::sqrt(na * nb) + 1e-30));
            rmax = std::max(rmax, std::sqrt(nd / (nb + 1e-30)));
        }
        res[name] = {cmin, rmax};
    };
    auto compare = [&](const std::vector<float> & ours, int st) {
        Ref r = load_ref(dir + "/s" + std::to_string(st) + "_logits.bin");
        const int n = (int) std::min<size_t>(ours.size(), r.v.size());
        auto softmax = [&](const float * l, std::vector<double> & p) {
            double m = -1e30;
            for (int i = 0; i < n; ++i) m = std::max(m, (double) l[i]);
            double s = 0;
            p.resize(n);
            for (int i = 0; i < n; ++i) s += (p[i] = std::exp(l[i] - m));
            for (double & v : p) v /= s;
        };
        std::vector<double> pr, po;
        softmax(r.v.data(), pr);
        softmax(ours.data(), po);
        double kl = 0;
        int tr = 0, to = 0;
        for (int i = 0; i < n; ++i) {
            if (pr[i] > 0) kl += pr[i] * std::log(pr[i] / std::max(po[i], 1e-30));
            if (pr[i] > pr[tr]) tr = i;
            if (po[i] > po[to]) to = i;
        }
        std::printf("step %d logits: llama.cpp top %d (p=%.3f), engine top %d (p=%.3f), KL %.5f -> %s\n", st, tr, pr[tr], to, po[to], kl,
                    tr == to ? "MATCH" : "DIFFERENT");
        return tr;
    };
    std::vector<float> logits = eng.prefill(tokens, 0);
    int next = compare(logits, 0);
    int pos = (int) tokens.size();
    for (int st = 1; st <= 2; ++st) {
        logits = eng.step(next, pos++);
        next = compare(logits, st);
    }
    std::printf("\nprompt chunk, per visit (min cosine over the %zu tokens, max relative error):\n", tokens.size());
    for (const auto & [name, v] : res) std::printf("  %-10s %.5f  %.4f\n", name.c_str(), v.first, v.second);
    return 0;
}
