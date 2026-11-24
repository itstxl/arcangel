#include "engine.h"

#include "device.h"
#include "quant.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace ink {

namespace {

constexpr int WG = 256;  // work-group size for row reductions

}  // namespace

Engine::Engine(sycl::queue & q, const Model & m, int n_ctx) : q_(q), m_(m), hp_(m.hp), n_ctx_(n_ctx) {
    const int n_slot = hp_.n_slot();
    const size_t kv_floats = (size_t) n_ctx * hp_.n_head_kv * hp_.head_dim;
    const size_t conv_floats = (size_t) hp_.conv_channels() * 3;
    pool_size_ = 64u << 20;  // floats of per-token scratch
    Model::check_fits(q_, (uint64_t) n_slot * (2 * kv_floats + conv_floats) * sizeof(float) + pool_size_ * sizeof(float),
                      "the reference engine's caches");
    for (int s = 0; s < n_slot; ++s) {
        k_cache_.push_back(sycl::malloc_device<float>(kv_floats, q_));
        v_cache_.push_back(sycl::malloc_device<float>(kv_floats, q_));
        conv_hist_.push_back(sycl::malloc_device<float>(conv_floats, q_));
        q_.fill(conv_hist_.back(), 0.0f, conv_floats);
        allocs_.insert(allocs_.end(), {k_cache_.back(), v_cache_.back(), conv_hist_.back()});
    }
    pool_ = sycl::malloc_device<float>(pool_size_, q_);
    ids_dev_ = sycl::malloc_device<int32_t>(64, q_);
    wts_dev_ = sycl::malloc_device<float>(64, q_);
    allocs_.insert(allocs_.end(), {pool_, ids_dev_, wts_dev_});
    q_.wait();
}

Engine::~Engine() {
    q_.wait();
    for (void * p : allocs_) sycl::free(p, q_);
}

float * Engine::buf(size_t n) {
    n = (n + 63) / 64 * 64;
    if (pool_used_ + n > pool_size_) throw std::runtime_error("scratch pool exhausted");
    float * p = pool_ + pool_used_;
    pool_used_ += n;
    return p;
}

float * Engine::vec() { return buf(hp_.n_embd); }

void Engine::emit(const std::string & name, int slot, const float * p, int64_t n) {
    if (hook) hook(slot < 0 ? name : name + "-" + std::to_string(slot), p, n);
}

void Engine::matvec(const W & w, const float * x, float * y, int64_t expert) {
    const int64_t rows = w.ne[1], cols = w.ne[0];
    const uint8_t * base = w.p + (size_t) expert * rows * w.row_bytes;
    const size_t rb = w.row_bytes;
    const uint32_t type = w.type;
    const q::BlockShape bs = q::shape(type);
    q_.parallel_for(sycl::range<1>((size_t) rows), [=](sycl::id<1> r) {
        const uint8_t * row = base + r[0] * rb;
        float acc = 0.0f;
        if (bs.elems == 1) {
            for (int64_t c = 0; c < cols; ++c) {
                const uint8_t * p = row + c * bs.bytes;
                const float v = type == q::F32 ? sycl::bit_cast<float>(q::rd32(p)) : type == q::F16 ? q::f16(p) : q::bf16(p);
                acc += v * x[c];
            }
        } else {
            float t[32];
            for (int64_t c0 = 0; c0 < cols; c0 += 32) {
                q::decode32(type, row + (c0 / bs.elems) * bs.bytes, (int) ((c0 % bs.elems) / 32), t);
                for (int j = 0; j < 32; ++j) acc += t[j] * x[c0 + j];
            }
        }
        y[r] = acc;
    });
}

void Engine::rmsnorm(const float * x, const W * w, float * y, int len, int rows) {
    const float eps = hp_.eps;
    const float * wp = w && w->ok() ? (const float *) w->p : nullptr;
    q_.parallel_for(sycl::nd_range<1>((size_t) rows * WG, WG), [=](sycl::nd_item<1> it) {
        const int r = (int) it.get_group(0), l = (int) it.get_local_id(0);
        const float * xr = x + (size_t) r * len;
        float s = 0.0f;
        for (int i = l; i < len; i += WG) s += xr[i] * xr[i];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float scale = 1.0f / sycl::sqrt(s / (float) len + eps);
        for (int i = l; i < len; i += WG) y[(size_t) r * len + i] = xr[i] * scale * (wp ? wp[i] : 1.0f);
    });
}

void Engine::mix(const std::vector<const float *> & blocks, const float * cur, const W & query, float * out) {
    const int d = hp_.n_embd;
    const int ns = (int) blocks.size() + 1;
    if (ns > 16) throw std::runtime_error("mix: too many blocks");
    float * S = buf((size_t) ns * d);
    for (int k = 0; k < ns - 1; ++k) q_.memcpy(S + (size_t) k * d, blocks[k], d * sizeof(float));
    q_.memcpy(S + (size_t) (ns - 1) * d, cur, d * sizeof(float));
    // the states are RMS-normalised without a weight, eps 1e-6 (inkling2.cpp stack/mix)
    const float eps = 1e-6f;
    q_.parallel_for(sycl::nd_range<1>((size_t) ns * WG, WG), [=](sycl::nd_item<1> it) {
        const int r = (int) it.get_group(0), l = (int) it.get_local_id(0);
        float * xr = S + (size_t) r * d;
        float s = 0.0f;
        for (int i = l; i < d; i += WG) s += xr[i] * xr[i];
        s = sycl::reduce_over_group(it.get_group(), s, sycl::plus<float>());
        const float scale = 1.0f / sycl::sqrt(s / (float) d + eps);
        for (int i = l; i < d; i += WG) xr[i] *= scale;
    });
    const float * qw = (const float *) query.p;
    const int nq = (int) query.ne[1];
    const float inv = 1.0f / std::sqrt((float) d);
    q_.parallel_for(sycl::nd_range<1>((size_t) nq * WG, WG), [=](sycl::nd_item<1> it) {
        const int j = (int) it.get_group(0), l = (int) it.get_local_id(0);
        float sc[16];
        float m = -INFINITY;
        for (int s = 0; s < ns; ++s) {
            float a = 0.0f;
            for (int i = l; i < d; i += WG) a += S[(size_t) s * d + i] * qw[(size_t) j * d + i];
            sc[s] = sycl::reduce_over_group(it.get_group(), a, sycl::plus<float>()) * inv;
            m = sycl::fmax(m, sc[s]);
        }
        float sum = 0.0f;
        for (int s = 0; s < ns; ++s) {
            sc[s] = sycl::exp(sc[s] - m);
            sum += sc[s];
        }
        for (int i = l; i < d; i += WG) {
            float acc = 0.0f;
            for (int s = 0; s < ns; ++s) acc += sc[s] * S[(size_t) s * d + i];
            out[(size_t) j * d + i] = acc / sum;
        }
    });
}

void Engine::control(const float * cur, const float * msg, const LoopW & lw, int c, float * out) {
    const int d = hp_.n_embd;
    float * cat = buf(2 * (size_t) d);
    q_.memcpy(cat, cur, d * sizeof(float));
    q_.memcpy(cat + d, msg, d * sizeof(float));
    rmsnorm(cat, nullptr, cat, d, 2);  // rms(cur) and rms(msg), eps 1e-6
    float * g = buf(lw.down[c].ne[1]);
    matvec(lw.down[c], cat, g);
    float * gain = vec();
    float * forget = vec();
    matvec(lw.gain[c], g, gain);
    matvec(lw.forget[c], g, forget);
    q_.parallel_for(sycl::range<1>(d), [=](sycl::id<1> i) {
        out[i] = cur[i] - sycl::tanh(forget[i]) * cur[i] + sycl::tanh(gain[i]) * msg[i];
    });
}

void Engine::conv(float * x, const W & w, const float * dyn, int heads, int slot, int offset, int channels) {
    float * hist = conv_hist_[slot] + (size_t) offset * 3;
    const float * wp = (const float *) w.p;
    const float norm = 1.0f / std::sqrt((float) (4 * (dyn ? 2 : 1) + 1));
    const int per_head = dyn ? channels / heads : 1;
    q_.parallel_for(sycl::range<1>(channels), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        const float xx[4] = {hist[c * 3 + 0], hist[c * 3 + 1], hist[c * 3 + 2], x[c]};
        float out = 0.0f;
        for (int j = 0; j < 4; ++j) out += wp[j + 4 * c] * xx[j];
        if (dyn) {
            const int h = c / per_head;
            for (int lag = 0; lag < 4; ++lag) out += dyn[lag + 4 * h] * xx[3 - lag];
        }
        hist[c * 3 + 0] = xx[1];
        hist[c * 3 + 1] = xx[2];
        hist[c * 3 + 2] = xx[3];
        x[c] = (xx[3] + out) * norm;
    });
}

void Engine::clip(float * v, const W * scale, int rows, int len, float limit) {
    const float * sp = scale && scale->ok() ? (const float *) scale->p : nullptr;
    q_.parallel_for(sycl::range<1>(rows), [=](sycl::id<1> id) {
        float * r = v + (size_t) id[0] * len;
        float s = 0.0f;
        for (int i = 0; i < len; ++i) {
            if (sp) r[i] *= sp[(size_t) id[0] * len + i];
            s += r[i] * r[i];
        }
        const float f = sycl::clamp(limit / (sycl::sqrt(s) + 1e-6f), 0.0f, 1.0f);
        for (int i = 0; i < len; ++i) r[i] *= f;
    });
}

void Engine::layer(int slot, int ip, float * x, std::vector<const float *> & blocks, const float * messages,
                   const LoopW * lw, int pos, float tau) {
    const LayerW & L = m_.layers[ip];
    const Hparams & h = hp_;
    const int d = h.n_embd, nh = h.n_head, nkv = h.n_head_kv, hd = h.head_dim;
    const bool swa = h.is_swa[ip];
    const bool full = !swa;

    float * inp_sa = vec();
    q_.memcpy(inp_sa, x, d * sizeof(float));

    // 1-3: attention inputs
    const float * in[3] = {x, x, x};
    if (L.attn_res_query.ok()) {
        float * mixed = buf((size_t) L.attn_res_query.ne[1] * d);
        mix(blocks, x, L.attn_res_query, mixed);
        if (swa) in[0] = in[1] = in[2] = mixed;
        else for (int j = 0; j < 3; ++j) in[j] = mixed + (size_t) j * d;
    }
    float * nin[3];
    if (messages) {
        for (int j = 0; j < 3; ++j) {
            float * o = vec();
            control(in[j], messages + (size_t) j * d, *lw, j, o);
            nin[j] = o;
        }
        for (int j = 0; j < 3; ++j) rmsnorm(nin[j], swa ? &L.attn_norm : j == 0 ? &L.q_inp_norm : j == 1 ? &L.k_inp_norm : &L.v_inp_norm, nin[j], d, 1);
    } else if (swa) {
        nin[0] = vec();
        rmsnorm(in[0], &L.attn_norm, nin[0], d, 1);
        nin[1] = nin[2] = nin[0];
    } else {
        const W * norms[3] = {&L.q_inp_norm, &L.k_inp_norm, &L.v_inp_norm};
        for (int j = 0; j < 3; ++j) {
            nin[j] = vec();
            rmsnorm(in[j], norms[j], nin[j], d, 1);
        }
    }

    // 4-6: projections, convolutions, clipping
    float * Q = buf((size_t) nh * hd);
    float * K = buf((size_t) nkv * hd);
    float * V = buf((size_t) nkv * hd);
    float * R = buf((size_t) nh * h.n_rel);
    float * sinks = buf(nh);
    matvec(L.wq, nin[0], Q);
    matvec(L.wk, nin[1], K);
    matvec(L.wv, nin[2], V);
    matvec(L.rel, nin[0], R);
    float * kdyn = nullptr, * vdyn = nullptr;
    if (full) {
        kdyn = buf(L.k_conv_proj.ne[1]);
        vdyn = buf(L.v_conv_proj.ne[1]);
        matvec(L.k_conv_proj, nin[1], kdyn);
        matvec(L.v_conv_proj, nin[2], vdyn);
    }
    conv(K, L.k_conv, kdyn, nkv, slot, 0, nkv * hd);
    conv(V, L.v_conv, vdyn, nkv, slot, nkv * hd, nkv * hd);
    clip(Q, &L.q_clip, nh, hd, h.qk_clip);
    clip(K, nullptr, nkv, hd, h.qk_clip);
    rmsnorm(V, nullptr, V, hd, nkv);
    clip(R, &L.rel_clip, nh, h.n_rel, h.rel_clip);
    if (full && tau != 1.0f) {
        q_.parallel_for(sycl::range<1>((size_t) nh * hd), [=](sycl::id<1> i) { Q[i] *= tau; });
        q_.parallel_for(sycl::range<1>((size_t) nh * h.n_rel), [=](sycl::id<1> i) { R[i] *= tau; });
    }
    matvec(L.sink_proj, nin[0], sinks);
    emit("Qcur", slot, Q, nh * hd);
    emit("Kcur", slot, K, nkv * hd);
    emit("Vcur", slot, V, nkv * hd);

    // 7: banded attention with sinks over this slot's cache
    float * kc = k_cache_[slot], * vc = v_cache_[slot];
    q_.memcpy(kc + (size_t) pos * nkv * hd, K, (size_t) nkv * hd * sizeof(float));
    q_.memcpy(vc + (size_t) pos * nkv * hd, V, (size_t) nkv * hd * sizeof(float));
    float * att = buf((size_t) nh * hd);
    {
        const float * rb = (const float *) L.rel_bias.p;
        const int extent = (int) L.rel_bias.ne[1], n_rel = h.n_rel, n_swa = h.n_swa, ratio = nh / nkv;
        const float scale = 1.0f / std::sqrt((float) hd);
        q_.parallel_for(sycl::range<1>(nh), [=](sycl::id<1> id) {
            const int hh = (int) id[0], kvh = hh / ratio;
            const float * qh = Q + (size_t) hh * hd;
            const float * rh = R + (size_t) hh * n_rel;
            const int k0 = swa ? sycl::max(0, pos - n_swa + 1) : 0;
            auto score = [&](int k) {
                const float * kk = kc + ((size_t) k * nkv + kvh) * hd;
                float s = 0.0f;
                for (int i = 0; i < hd; ++i) s += qh[i] * kk[i];
                s *= scale;
                const int dist = pos - k;
                if (dist < extent) {
                    float b = 0.0f;
                    for (int r = 0; r < n_rel; ++r) b += rb[r + n_rel * dist] * rh[r];
                    s += b;
                }
                return s;
            };
            const float sink = sinks[hh];
            float m = sink;
            for (int k = k0; k <= pos; ++k) m = sycl::fmax(m, score(k));
            float acc[128];
            for (int i = 0; i < hd; ++i) acc[i] = 0.0f;
            float den = sycl::exp(sink - m);
            for (int k = k0; k <= pos; ++k) {
                const float p = sycl::exp(score(k) - m);
                den += p;
                const float * vv = vc + ((size_t) k * nkv + kvh) * hd;
                for (int i = 0; i < hd; ++i) acc[i] += p * vv[i];
            }
            for (int i = 0; i < hd; ++i) att[(size_t) hh * hd + i] = acc[i] / den;
        });
    }
    emit("kqv_out", slot, att, nh * hd);

    // 8: output gate, projection, convolution
    float * g = buf((size_t) nh * hd);
    matvec(L.gate, nin[0], g);
    q_.parallel_for(sycl::range<1>((size_t) nh * hd), [=](sycl::id<1> i) { att[i] *= 1.0f / (1.0f + sycl::exp(-g[i])); });
    float * o = vec();
    matvec(L.wo, att, o);
    conv(o, L.post_conv, nullptr, 1, slot, 2 * nkv * hd, d);
    emit("attn_out", slot, o, d);

    // 9: residual, with a new block at the start of each group of `block` layers
    if (ip % h.block == 0) {
        blocks.push_back(inp_sa);
        q_.memcpy(x, o, d * sizeof(float));
    } else {
        q_.parallel_for(sycl::range<1>(d), [=](sycl::id<1> i) { x[i] = inp_sa[i] + o[i]; });
    }
    float * ffn_inp = vec();
    q_.memcpy(ffn_inp, x, d * sizeof(float));
    emit("ffn_inp", slot, ffn_inp, d);

    // 10-11: FFN input and routing
    float * f = vec();
    mix(blocks, x, L.ffn_res_query, f);
    if (messages) {
        float * c = vec();
        control(f, messages + 3 * (size_t) d, *lw, 3, c);
        f = c;
    }
    rmsnorm(f, &L.ffn_norm, f, d, 1);
    float * logits = buf(h.n_expert);
    matvec(L.gate_inp, f, logits);
    {
        const float * pb = L.probs_b.ok() ? (const float *) L.probs_b.p : nullptr;
        const int ne = h.n_expert, nu = h.n_used;
        const float ws = h.w_scale;
        int32_t * ids = ids_dev_;
        float * wts = wts_dev_;
        q_.single_task([=]() {
            float p[256], sel[256];
            for (int e = 0; e < ne; ++e) {
                p[e] = 1.0f / (1.0f + sycl::exp(-logits[e]));
                sel[e] = p[e] + (pb ? pb[e] : 0.0f);
            }
            float sum = 0.0f;
            for (int k = 0; k < nu; ++k) {
                int best = -1;
                for (int e = 0; e < ne; ++e) if (sel[e] > -INFINITY && (best < 0 || sel[e] > sel[best])) best = e;
                ids[k] = best;
                wts[k] = p[best];
                sum += p[best];
                sel[best] = -INFINITY;
            }
            for (int k = 0; k < nu; ++k) wts[k] = wts[k] / (sum + 1e-20f) * ws;
        });
    }
    int32_t ids[64];
    float wts[64];
    q_.memcpy(ids, ids_dev_, h.n_used * sizeof(int32_t));
    q_.memcpy(wts, wts_dev_, h.n_used * sizeof(float)).wait();

    // 12: the chosen experts, SITU activation, weighted sum
    float * y = vec();
    q_.fill(y, 0.0f, d);
    const int nff = h.n_ff;
    const float beta = h.situ_beta, lb = h.situ_lb;
    for (int k = 0; k < h.n_used; ++k) {
        const int e = ids[k];
        float * gu = buf(2 * (size_t) nff);
        matvec(L.gate_up, f, gu, e);
        const float * gus = L.gate_up_s.ok() ? (const float *) L.gate_up_s.p : nullptr;
        float * act = buf(nff);
        q_.parallel_for(sycl::range<1>(nff), [=](sycl::id<1> i) {
            const float s = gus ? gus[e] : 1.0f;
            const float gg = gu[i] * s, uu = gu[nff + i] * s;
            const float a = beta * sycl::tanh(gg / beta) * (1.0f / (1.0f + sycl::exp(-gg)));
            act[i] = a * (lb > 0.0f ? lb * sycl::tanh(uu / lb) : uu);
        });
        float * yd = vec();
        matvec(L.down, act, yd, e);
        const float * dws = L.down_s.ok() ? (const float *) L.down_s.p : nullptr;
        const float wk = wts[k];
        q_.parallel_for(sycl::range<1>(d), [=](sycl::id<1> i) { y[i] += wk * yd[i] * (dws ? dws[e] : 1.0f); });
    }

    // 13: convolution and residual
    conv(y, L.ffn_post_conv, nullptr, 1, slot, 2 * nkv * hd + d, d);
    emit("ffn_out", slot, y, d);
    q_.parallel_for(sycl::range<1>(d), [=](sycl::id<1> i) { x[i] = ffn_inp[i] + y[i]; });
    emit("l_out", slot, x, d);
}

const std::vector<float> & Engine::step(int token, int pos) {
    if (pos >= n_ctx_) throw std::runtime_error("context is full");
    pool_used_ = 0;
    const Hparams & h = hp_;
    const int d = h.n_embd;

    // embedding: decode the token's row, then RMS norm with token_embd_norm
    float * x = vec();
    {
        const W & e = m_.tok_embd;
        const uint8_t * row = e.p + (size_t) token * e.row_bytes;
        const uint32_t type = e.type;
        const q::BlockShape bs = q::shape(type);
        q_.parallel_for(sycl::range<1>(d / 32), [=](sycl::id<1> i) {
            const int64_t c0 = (int64_t) i[0] * 32;
            q::decode32(type, row + (c0 / bs.elems) * bs.bytes, (int) ((c0 % bs.elems) / 32), x + c0);
        });
        rmsnorm(x, &m_.tok_norm, x, d, 1);
    }
    const float tau = 1.0f + h.log_alpha * std::log(std::max(1.0f, (float) (pos + 1) / h.log_floor));

    std::vector<const float *> blocks;
    for (int il = 0; il < h.loop_start; ++il) layer(il, il, x, blocks, nullptr, nullptr, pos, tau);

    float * prelude = vec();
    q_.memcpy(prelude, x, d * sizeof(float));
    const std::vector<const float *> prelude_blocks = blocks;
    const int n_core = h.loop_end - h.loop_start;
    for (int pass = 0; pass < h.n_pass; ++pass) {
        std::vector<const float *> bank;
        float * bank_out = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = vec();
            q_.memcpy(bank_out, x, d * sizeof(float));
            q_.memcpy(x, prelude, d * sizeof(float));
            blocks = prelude_blocks;
        }
        for (int ip = h.loop_start; ip < h.loop_end; ++ip) {
            const int slot = pass == 0 ? ip : h.n_layer + (pass - 1) * n_core + (ip - h.loop_start);
            if (pass == 0) {
                layer(slot, ip, x, blocks, nullptr, nullptr, pos, tau);
                continue;
            }
            const LoopW & lw = m_.loops[ip - h.loop_start];
            float * messages = buf(4 * (size_t) d);
            mix(bank, bank_out, lw.msg_query, messages);
            layer(slot, ip, x, blocks, messages, &lw, pos, tau);
        }
    }
    for (int il = h.loop_end; il < h.n_layer; ++il) layer(il, il, x, blocks, nullptr, nullptr, pos, tau);

    // output: depth mix, norm, LM head, softcap, drop the padded vocabulary
    float * hcur = vec();
    mix(blocks, x, m_.output_res_query, hcur);
    rmsnorm(hcur, &m_.output_norm, hcur, d, 1);
    emit("result_norm", -1, hcur, d);
    float * lg = buf(m_.output.ne[1]);
    matvec(m_.output, hcur, lg);
    logits_.resize(h.n_vocab);
    q_.memcpy(logits_.data(), lg, h.n_vocab * sizeof(float)).wait();
    if (h.softcap > 0.0f) {
        for (float & v : logits_) v = h.softcap * std::tanh(v / h.softcap);
    }
    return logits_;
}

}  // namespace ink
