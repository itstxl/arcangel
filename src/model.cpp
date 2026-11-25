#include "model.h"

#include "device.h"
#include "quant.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace ink {

namespace {

// host = true: the tensor lives in pinned host memory (USM host allocation) that the GPU reads over PCIe and
// the CPU can read directly: expert offloading. `total` counts device bytes only, `host_total` the rest.
W load_impl(sycl::queue & q, const GgufFile & f, const std::string & name, bool required, uint64_t & total,
       std::vector<void *> & allocs, bool repack, bool host, uint64_t * host_total) {
    W w;
    const GgufTensor * t = f.tensor(name);
    if (!t) {
        if (required) throw std::runtime_error("model is missing tensor " + name);
        return w;
    }
    auto alloc = [&](size_t n) { return host ? sycl::malloc_host<uint8_t>(n, q) : sycl::malloc_device<uint8_t>(n, q); };
    uint64_t dummy = 0;
    uint64_t & tot = host ? (host_total ? *host_total : dummy) : total;
    // a pinned host destination is filled by the CPU: q.memcpy between two host buffers goes through the
    // driver's staging copies (measured: minutes for UD-Q4_K_XL's 8 GiB of offloaded experts)
    auto upload = [&](void * dst, const void * src, size_t n) {
        if (host) std::memcpy(dst, src, n);
        else q.memcpy(dst, src, n).wait();
    };
    auto * dev = alloc(t->n_bytes);
    if (!dev) throw std::runtime_error("device allocation failed for " + name);
    allocs.push_back(dev);
    tot += t->n_bytes;
    const q::BlockShape bs = q::shape(t->type);
    w.type = w.ktype = t->type;
    for (int i = 0; i < 4; ++i) w.ne[i] = t->ne[i];
    w.p = dev;
    w.row_bytes = (size_t) (t->ne[0] / bs.elems) * bs.bytes;
    const int64_t cols = t->ne[0], rows = t->n_elements() / cols;
    if (repack && t->type == q::Q8_0 && cols % 32 == 0) {
        // Q8_0R: int8 values of every row contiguous (row stride = cols bytes), then the fp16 scales
        // (row stride = cols / 32 halves). Same values as the file, lossless.
        std::vector<uint8_t> host(t->n_bytes);
        const int64_t nb = cols / 32;
        uint8_t * vals = host.data();
        uint8_t * scl = host.data() + (size_t) rows * cols;
        for (int64_t r = 0; r < rows; ++r) {
            const uint8_t * src = t->data + (size_t) r * w.row_bytes;
            for (int64_t b = 0; b < nb; ++b) {
                std::memcpy(scl + ((size_t) r * nb + b) * 2, src + b * 34, 2);
                std::memcpy(vals + (size_t) r * cols + b * 32, src + b * 34 + 2, 32);
            }
        }
        upload(dev, host.data(), t->n_bytes);
        w.ktype = q::Q8_0R;
        w.row_bytes = (size_t) cols;
        w.aux = dev + (size_t) rows * cols;
        w.aux_row_bytes = (size_t) nb * 2;
        return w;
    }
    // Repacked layouts: allocate the planes, fill them on the host, upload, point the weight at them.
    auto planes = [&](uint32_t ktype, size_t vrow, size_t arow, size_t a2row, auto && fill) {
        const size_t vb = (size_t) rows * vrow, ab = (size_t) rows * arow, a2b = (size_t) rows * a2row;
        auto * d2 = alloc(vb + ab + a2b);
        if (!d2) throw std::runtime_error("device allocation failed for " + name);
        allocs.back() = d2;
        sycl::free(dev, q);
        tot += vb + ab + a2b - t->n_bytes;
        std::vector<uint8_t> host(vb + ab + a2b);
        for (int64_t r = 0; r < rows; ++r)
            fill(t->data + (size_t) r * w.row_bytes, host.data() + (size_t) r * vrow, host.data() + vb + (size_t) r * arow,
                 host.data() + vb + ab + (size_t) r * a2row);
        upload(d2, host.data(), host.size());
        w.ktype = ktype;
        w.p = d2;
        w.row_bytes = vrow;
        w.aux = arow ? d2 + vb : nullptr;
        w.aux_row_bytes = arow;
        w.aux2 = a2row ? d2 + vb + ab : nullptr;
        w.aux2_row_bytes = a2row;
        return w;
    };
    const int64_t n256 = cols / 256, n64 = cols / 64, n32 = cols / 32;
    if (repack && t->type == q::NVFP4 && cols % 64 == 0) {
        return planes(q::NVFP4R, cols / 2, cols / 16, 0, [&](const uint8_t * src, uint8_t * v, uint8_t * a, uint8_t *) {
            for (int64_t b = 0; b < n64; ++b) {
                std::memcpy(v + b * 32, src + b * 36 + 4, 32);
                std::memcpy(a + b * 4, src + b * 36, 4);
            }
        });
    }
    if (repack && t->type == q::Q2_0 && cols % 64 == 0) {
        return planes(q::Q2_0R, cols / 4, n64 * 2, 0, [&](const uint8_t * src, uint8_t * v, uint8_t * a, uint8_t *) {
            for (int64_t b = 0; b < n64; ++b) {
                std::memcpy(v + b * 16, src + b * 18 + 2, 16);
                std::memcpy(a + b * 2, src + b * 18, 2);
            }
        });
    }
    if (repack && t->type == q::IQ2_S && cols % 256 == 0) {
        return planes(q::IQ2_SR, n32 * 8, n32 * 2, n256 * 2, [&](const uint8_t * src, uint8_t * v, uint8_t * a, uint8_t * a2) {
            for (int64_t b = 0; b < n256; ++b) {
                const uint8_t * blk = src + b * 82;
                std::memcpy(a2 + b * 2, blk, 2);
                for (int ib = 0; ib < 8; ++ib) {
                    const int64_t c = b * 8 + ib;
                    std::memcpy(v + c * 8, blk + 2 + 4 * ib, 4);       // grid index low bytes
                    std::memcpy(v + c * 8 + 4, blk + 34 + 4 * ib, 4);  // sign bytes
                    a[c * 2 + 0] = blk[66 + ib];                       // qh
                    a[c * 2 + 1] = blk[74 + ib];                       // two 4-bit scales
                }
            }
        });
    }
    if (repack && t->type == q::IQ2_XXS && cols % 256 == 0) {
        return planes(q::IQ2_XXSR, n256 * 64, n256 * 2, 0, [&](const uint8_t * src, uint8_t * v, uint8_t * a, uint8_t *) {
            for (int64_t b = 0; b < n256; ++b) {
                std::memcpy(v + b * 64, src + b * 66 + 2, 64);
                std::memcpy(a + b * 2, src + b * 66, 2);
            }
        });
    }
    if (repack && t->type == q::IQ2_XS && cols % 256 == 0) {
        return planes(q::IQ2_XSR, n256 * 64, n32, n256 * 2, [&](const uint8_t * src, uint8_t * v, uint8_t * a, uint8_t * a2) {
            for (int64_t b = 0; b < n256; ++b) {
                std::memcpy(v + b * 64, src + b * 74 + 2, 64);
                std::memcpy(a + b * 8, src + b * 74 + 66, 8);
                std::memcpy(a2 + b * 2, src + b * 74, 2);
            }
        });
    }
    if (repack && t->type == q::IQ1_S && cols % 256 == 0) {
        return planes(q::IQ1_SR, n256 * 32, n32 * 2, n256 * 2, [&](const uint8_t * src, uint8_t * v, uint8_t * a, uint8_t * a2) {
            for (int64_t b = 0; b < n256; ++b) {
                std::memcpy(v + b * 32, src + b * 50 + 2, 32);
                std::memcpy(a + b * 16, src + b * 50 + 34, 16);
                std::memcpy(a2 + b * 2, src + b * 50, 2);
            }
        });
    }
    if (repack && t->type == q::Q4_K && cols % 256 == 0) {
        return planes(q::Q4_KR, n256 * 128, n256 * 16, n256 * 4, [&](const uint8_t * src, uint8_t * v, uint8_t * a, uint8_t * a2) {
            for (int64_t b = 0; b < n256; ++b) {
                const uint8_t * blk = src + b * 144;
                std::memcpy(a2 + b * 4, blk, 4);  // d, dmin
                std::memcpy(v + b * 128, blk + 16, 128);
                for (int j = 0; j < 8; ++j) {     // 6-bit (scale, min) of each 32 values, unpacked to bytes
                    uint8_t sc, m;
                    q::scale_min_k4(j, blk + 4, sc, m);
                    a[b * 16 + 2 * j] = sc;
                    a[b * 16 + 2 * j + 1] = m;
                }
            }
        });
    }
    if (repack && t->type == q::Q6_K && cols % 256 == 0) {
        // Q6_KR: the 6-bit values unpacked to int8 (q - 32), the int8 sub-block scales, the fp16 block scales.
        // Lossless: value = d * scale * q exactly as the file decodes. 8 + 0.5 + 0.06 bits per value.
        const int64_t nb = cols / 256;
        const size_t vbytes = (size_t) rows * cols, sbytes = (size_t) rows * cols / 16, dbytes = (size_t) rows * nb * 2;
        auto * dev2 = alloc(vbytes + sbytes + dbytes);
        if (!dev2) throw std::runtime_error("device allocation failed for " + name);
        allocs.back() = dev2;
        sycl::free(dev, q);
        tot += vbytes + sbytes + dbytes - t->n_bytes;
        std::vector<uint8_t> host(vbytes + sbytes + dbytes);
        int8_t * vals = (int8_t *) host.data();
        int8_t * scl = (int8_t *) (host.data() + vbytes);
        uint8_t * dd = host.data() + vbytes + sbytes;
        for (int64_t r = 0; r < rows; ++r) {
            for (int64_t b = 0; b < nb; ++b) {
                const uint8_t * blk = t->data + (size_t) r * w.row_bytes + b * 210;
                const uint8_t * ql0 = blk, * qh0 = blk + 128;
                const int8_t * sc0 = (const int8_t *) (blk + 192);
                std::memcpy(dd + ((size_t) r * nb + b) * 2, blk + 208, 2);
                int8_t * y = vals + (size_t) r * cols + b * 256;
                for (int n = 0; n < 2; ++n) {
                    const uint8_t * ql = ql0 + 64 * n, * qh = qh0 + 32 * n;
                    for (int l = 0; l < 32; ++l) {
                        y[128 * n + l + 0] = (int8_t) (((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32);
                        y[128 * n + l + 32] = (int8_t) (((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32);
                        y[128 * n + l + 64] = (int8_t) (((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32);
                        y[128 * n + l + 96] = (int8_t) (((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32);
                    }
                }
                // sub-block j (16 values) uses scales[j]: the reference indexes sc[is + 0/2/4/6] with is = l/16
                // over each half, i.e. scale index = (value index within the block) / 16
                for (int j = 0; j < 16; ++j) scl[(size_t) r * (cols / 16) + b * 16 + j] = sc0[j];
            }
        }
        upload(dev2, host.data(), host.size());
        w.ktype = q::Q6_KR;
        w.p = dev2;
        w.row_bytes = (size_t) cols;
        w.aux = dev2 + vbytes;
        w.aux_row_bytes = (size_t) cols / 16;
        w.aux2 = dev2 + vbytes + sbytes;
        w.aux2_row_bytes = (size_t) nb * 2;
        return w;
    }
    if (host) std::memcpy(dev, t->data, t->n_bytes);
    else q.memcpy(dev, t->data, t->n_bytes);
    return w;
}

// file pages already copied, dropped about every GiB (device uploads read the mapping asynchronously, so each
// batch waits for the queue once; per tensor that made start-up 2.5x slower)
std::vector<std::pair<const uint8_t *, size_t>> g_release;
size_t g_release_bytes = 0;
bool g_release_on = false;  // only with offloading: otherwise a cached file makes restarts ~2.5x faster
void release_now(sycl::queue & q, const GgufFile & f) {
    if (g_release.empty()) return;
    q.wait();
    for (auto & [p, n] : g_release) f.release(p, n);
    g_release.clear();
    g_release_bytes = 0;
}
void release_later(sycl::queue & q, const GgufFile & f, const uint8_t * p, size_t n) {
    if (!g_release_on) return;
    g_release.emplace_back(p, n);
    if ((g_release_bytes += n) >= (1ull << 30)) release_now(q, f);
}

W load(sycl::queue & q, const GgufFile & f, const std::string & name, bool required, uint64_t & total,
       std::vector<void *> & allocs, bool repack, bool host = false, uint64_t * host_total = nullptr) {
    W w = load_impl(q, f, name, required, total, allocs, repack, host, host_total);
    if (const GgufTensor * t = f.tensor(name)) release_later(q, f, t->data, t->n_bytes);
    return w;
}

Schedule make_schedule_impl(const Hparams & h) {
    Schedule s;
    int next = 0;
    std::vector<int> blocks;
    auto visit = [&](int slot, int ip, int pass) {
        Visit v;
        v.slot = slot;
        v.ip = ip;
        v.pass = pass;
        v.blocks = blocks;
        if (ip % h.block == 0) {
            v.push = next++;
            blocks.push_back(v.push);
        }
        return v;
    };
    for (int il = 0; il < h.loop_start; ++il) s.visits.push_back(visit(il, il, 0));
    s.prelude_end = (int) s.visits.size();
    const std::vector<int> prelude_blocks = blocks;
    for (int pass = 0; pass < h.n_pass; ++pass) {
        std::vector<int> bank;
        int restart = -1;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            restart = next++;
            bank.push_back(restart);
            blocks = prelude_blocks;
        }
        for (int ip = h.loop_start; ip < h.loop_end; ++ip) {
            const int slot = pass == 0 ? ip : h.n_layer + (pass - 1) * h.n_core() + (ip - h.loop_start);
            Visit v = visit(slot, ip, pass);
            v.bank = bank;
            if (ip == h.loop_start) v.restart = restart;
            s.visits.push_back(v);
        }
    }
    for (int il = h.loop_end; il < h.n_layer; ++il) s.visits.push_back(visit(il, il, 0));
    s.out_blocks = blocks;
    s.n_vectors = next;
    return s;
}

}  // namespace

Schedule make_schedule(const Hparams & h) { return make_schedule_impl(h); }

void Model::check_fits(sycl::queue & q, uint64_t need, const std::string & what) {
    const DeviceInfo d = describe(q.get_device());
    const uint64_t margin = 512ull << 20;
    if (d.free_bytes && need + margin > d.free_bytes) {
        throw std::runtime_error("not enough free VRAM for " + what + ": need " + std::to_string(need >> 20) +
                                 " MiB + " + std::to_string(margin >> 20) + " MiB margin, the card has " +
                                 std::to_string(d.free_bytes >> 20) + " MiB free (is another model loaded?)");
    }
}

Model::Model(sycl::queue & q, const GgufFile & f, bool repack) : q_(q) {
    const std::string a = f.get("general.architecture").s + ".";
    auto gi = [&](const std::string & k) { return (int) f.get(a + k).as_int(); };
    auto gf = [&](const std::string & k) { return (float) f.get(a + k).as_float(); };
    auto first = [&](const std::string & k) {
        const GgufValue & v = f.get(a + k);
        return (int) (v.kind == GgufValue::ARRAY ? v.arr.at(0).as_int() : v.as_int());
    };
    hp.n_embd = gi("embedding_length");
    hp.n_layer = gi("block_count");
    hp.n_head = first("attention.head_count");
    hp.n_head_kv = first("attention.head_count_kv");
    hp.head_dim = gi("attention.key_length");
    hp.loop_start = gi("loop.start");
    hp.loop_end = gi("loop.end");
    hp.n_pass = gi("loop.count");
    hp.n_expert = gi("expert_count");
    hp.n_used = gi("expert_used_count");
    hp.n_ff = gi("expert_feed_forward_length");
    hp.n_conv = gi("shortconv.l_cache");
    hp.n_rel = gi("attention.relative_length");
    hp.n_swa = gi("attention.sliding_window");
    hp.block = gi("attn_res.block_size");
    hp.n_vocab = gi("vocab_size_unpadded");
    hp.eps = gf("attention.layer_norm_rms_epsilon");
    hp.qk_clip = gf("attention.qk_clip");
    hp.rel_clip = gf("attention.relative_clip");
    hp.log_floor = gf("attention.log_scale_floor");
    hp.log_alpha = gf("attention.log_scale_alpha");
    hp.situ_beta = gf("activation.situ_beta");
    hp.situ_lb = gf("activation.situ_linear_beta");
    hp.w_scale = f.find(a + "expert_weights_scale") ? gf("expert_weights_scale") : 1.0f;
    hp.softcap = f.find(a + "final_logit_softcapping") ? gf("final_logit_softcapping") : 0.0f;
    for (const auto & v : f.get(a + "attention.sliding_window_pattern").arr) hp.is_swa.push_back(v.b);
    if (hp.n_conv != 4) throw std::runtime_error("engine assumes 4 convolution taps");
    if (hp.head_dim != 128) throw std::runtime_error("engine assumes head dim 128");

    // Expert offloading (INK_OFFLOAD): the routed experts of chosen layers stay in pinned system RAM. "once" =
    // the layers outside the loop core, which run once per token (the core's run n_pass times): half the expert
    // bytes for a quarter of the per-token expert reads. Or a list of layers, e.g. "0-5,18-23".
    std::vector<bool> off(hp.n_layer, false);
    if (const char * e = getenv("INK_OFFLOAD")) {
        const std::string spec = e;
        if (spec == "once") {
            for (int i = 0; i < hp.n_layer; ++i) off[i] = i < hp.loop_start || i >= hp.loop_end;
        } else if (!spec.empty() && spec != "none") {
            size_t i = 0;
            while (i < spec.size()) {
                size_t j = spec.find(',', i);
                if (j == std::string::npos) j = spec.size();
                const std::string part = spec.substr(i, j - i);
                const size_t dash = part.find('-');
                const int lo = std::stoi(part.substr(0, dash)), hi = dash == std::string::npos ? lo : std::stoi(part.substr(dash + 1));
                for (int l = lo; l <= hi && l < hp.n_layer; ++l) off[l] = true;
                i = j + 1;
            }
        }
    }
    offloaded = off;
    auto is_off = [&](const std::string & n) {
        if (n.rfind("blk.", 0) != 0 || n.find("_exps.weight") == std::string::npos) return false;
        return (bool) off[std::stoi(n.substr(4))];
    };
    uint64_t weights = 0, host_need = 0;
    for (const auto & t : f.tensors()) (is_off(t.name) ? host_need : weights) += t.n_bytes;
    check_fits(q_, weights, "the weights");
    if (host_need) {
        // pinned memory cannot be swapped: refuse rather than push the machine into swap (keep 4 GiB spare)
        uint64_t avail = 0;
        if (FILE * mi = fopen("/proc/meminfo", "r")) {
            char key[64];
            unsigned long long kb;
            while (fscanf(mi, "%63s %llu kB", key, &kb) == 2)
                if (std::string(key) == "MemAvailable:") avail = kb << 10;
            fclose(mi);
        }
        if (avail && host_need + (4ull << 30) > avail)
            throw std::runtime_error("not enough free system RAM for the offloaded experts: need " + std::to_string(host_need >> 20) +
                                     " MiB + 4096 MiB spare, " + std::to_string(avail >> 20) + " MiB available");
    }

    g_release_on = host_need > 0;
    auto L = [&](const std::string & n, bool req = true) {
        return load(q_, f, n, req, weight_bytes, allocs_, repack, is_off(n), &host_bytes);
    };
    // the embedding is read one row at a time by its file format (decode32), never by a matrix kernel: keep the
    // file layout (repacking a Q8_0 embedding, as in UD-Q4_K_XL, made every embedding row garbage)
    tok_embd = load(q_, f, "token_embd.weight", true, weight_bytes, allocs_, false);
    tok_norm = L("token_embd_norm.weight");
    output = L("output.weight");
    output_norm = L("output_norm.weight");
    output_res_query = L("output_res_query.weight");
    layers.resize(hp.n_layer);
    for (int i = 0; i < hp.n_layer; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".";
        LayerW & l = layers[i];
        const bool swa = hp.is_swa[i];
        l.attn_norm = L(p + "attn_norm.weight", swa);
        l.q_inp_norm = L(p + "attn_q_inp_norm.weight", !swa);
        l.k_inp_norm = L(p + "attn_k_inp_norm.weight", !swa);
        l.v_inp_norm = L(p + "attn_v_inp_norm.weight", !swa);
        l.k_conv_proj = L(p + "attn_k_conv_proj.weight", !swa);
        l.v_conv_proj = L(p + "attn_v_conv_proj.weight", !swa);
        l.wq = L(p + "attn_q.weight");
        l.wk = L(p + "attn_k.weight");
        l.wv = L(p + "attn_v.weight");
        l.wo = L(p + "attn_output.weight");
        l.gate = L(p + "attn_gate.weight");
        l.q_clip = L(p + "attn_q_clip.weight");
        l.sink_proj = L(p + "attn_sink_proj.weight");
        l.rel = L(p + "attn_rel.weight");
        l.rel_bias = L(p + "attn_rel_bias.weight");
        l.rel_clip = L(p + "attn_rel_clip.weight");
        l.k_conv = L(p + "attn_k_conv.weight");
        l.v_conv = L(p + "attn_v_conv.weight");
        l.post_conv = L(p + "attn_post_conv.weight");
        l.ffn_post_conv = L(p + "ffn_post_conv.weight");
        l.attn_res_query = L(p + "attn_res_query.weight", i > 0);
        l.ffn_res_query = L(p + "ffn_res_query.weight");
        l.ffn_norm = L(p + "ffn_norm.weight");
        l.gate_inp = L(p + "ffn_gate_inp.weight");
        l.probs_b = L(p + "exp_probs_b.bias", false);
        l.gate_up = L(p + "ffn_gate_up_exps.weight");
        l.gate_up_s = L(p + "ffn_gate_up_exps.scale", false);
        l.down = L(p + "ffn_down_exps.weight");
        l.down_s = L(p + "ffn_down_exps.scale", false);
    }
    const char * ctrl[4] = {"q", "k", "v", "ffn"};
    loops.resize(hp.n_core());
    for (int i = hp.loop_start; i < hp.loop_end; ++i) {
        const std::string p = "blk." + std::to_string(i) + ".loop_";
        LoopW & lw = loops[i - hp.loop_start];
        lw.msg_query = L(p + "msg_query.weight");
        for (int c = 0; c < 4; ++c) {
            lw.down[c] = L(p + ctrl[c] + "_ctrl_down.weight");
            lw.gain[c] = L(p + ctrl[c] + "_ctrl_gain.weight");
            lw.forget[c] = L(p + ctrl[c] + "_ctrl_forget.weight");
        }
    }
    schedule = make_schedule(hp);
    release_now(q_, f);  // the last batch of copied file pages
    q_.wait();
}

Model::~Model() {
    q_.wait();
    for (void * p : allocs_) sycl::free(p, q_);
}

}  // namespace ink
