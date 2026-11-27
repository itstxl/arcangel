#!/usr/bin/env python3
"""Unit tests for copy-drafting speculative decoding in server/inkling_server.py (no GPU/.so required)."""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "server"))

import inkling_server as srv  # noqa: E402


class StubTokenizer:
    """Minimal ByteLevel-like tokenizer stub supporting decode(ids, skip_special_tokens=False)."""

    def __init__(self, id_to_bytes, specials):
        self.id_to_bytes = dict(id_to_bytes)
        self.specials = dict(specials)
        self.tok_to_id = {v: k for k, v in specials.items()}

    def token_to_id(self, s):
        return self.tok_to_id.get(s)

    def encode(self, text, add_special_tokens=False):
        ids = []
        for ch in text.encode("utf-8"):
            ids.append(ch + 100)
        return SimpleNamespace(ids=ids)

    def decode(self, ids, skip_special_tokens=False):
        buf = bytearray()
        out = []
        for tid in ids:
            if tid in self.specials:
                if buf:
                    out.append(buf.decode("utf-8", errors="replace"))
                    buf.clear()
                if not skip_special_tokens:
                    out.append(self.specials[tid])
            elif tid in self.id_to_bytes:
                buf.extend(self.id_to_bytes[tid])
            else:
                buf.extend(bytes([tid - 100]) if 100 <= tid < 356 else b"?")
        if buf:
            out.append(buf.decode("utf-8", errors="replace"))
        return "".join(out)


def make_specials():
    return {
        1: srv.T_MODEL,
        2: srv.T_THINK,
        3: srv.T_TEXT,
        4: srv.T_TOOL,
        5: srv.T_END,
        6: srv.T_STOP,
        7: "<|endoftext|>",
    }


class MockEngine:
    """Deterministic/logit-table mock of Engine verifying exact commit() & KV state invariants."""

    def __init__(self, n_vocab, n_ctx, next_fn, logits_fn=None):
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.next_fn = next_fn
        self.logits_fn = logits_fn
        self.logits = np.zeros(n_vocab, dtype=np.float32)
        self.tokens = []
        self.committed_hist = []  # exact token sequence committed to conv history
        self.kv_slots = {}        # pos -> tok (simulates speculative KV overwrite + rollback)
        self.uncommitted_verify = False
        self.calls = {"prefill": 0, "step": 0, "verify": 0, "commit": 0}

    def _fill_logits(self, seq):
        if self.logits_fn is not None:
            self.logits[:] = self.logits_fn(seq)
        else:
            self.logits.fill(-10.0)
            self.logits[self.next_fn(seq)] = 10.0

    def load_prompt(self, toks):
        self.calls["prefill"] += 1
        self.tokens = list(toks)
        self.committed_hist = list(toks)
        self.kv_slots = {i: t for i, t in enumerate(toks)}
        self._fill_logits(self.committed_hist)
        return len(toks)

    def step(self, tok, pos):
        assert not self.uncommitted_verify, "step() called with uncommitted verify()"
        assert pos == len(self.committed_hist), f"step pos={pos} != hist={len(self.committed_hist)}"
        self.calls["step"] += 1
        self.committed_hist.append(tok)
        self.kv_slots[pos] = tok
        self._fill_logits(self.committed_hist)

    def step_argmax(self, tok, pos):
        self.step(tok, pos)
        return int(np.argmax(self.logits))

    def verify(self, toks, pos0):
        assert not self.uncommitted_verify, "verify() called before commit()"
        assert pos0 == len(self.committed_hist), f"verify pos0={pos0} != hist={len(self.committed_hist)}"
        assert pos0 + len(toks) <= self.n_ctx, "verify exceeded n_ctx"
        self.calls["verify"] += 1
        self.uncommitted_verify = True
        self._pending_batch = list(toks)
        self._pending_pos0 = pos0
        out = np.zeros((len(toks), self.n_vocab), dtype=np.float32)
        seq = list(self.committed_hist)
        for i, tok in enumerate(toks):
            self.kv_slots[pos0 + i] = tok
            seq.append(tok)
            if self.logits_fn is not None:
                out[i] = self.logits_fn(seq)
            else:
                out[i].fill(-10.0)
                out[i, self.next_fn(seq)] = 10.0
        return out

    def verify_argmax(self, toks, pos0):
        vl = self.verify(toks, pos0)
        return [int(np.argmax(vl[i])) for i in range(len(toks))]

    def commit(self, keep):
        assert self.uncommitted_verify, "commit() without verify()"
        assert 0 <= keep <= len(self._pending_batch)
        self.calls["commit"] += 1
        self.committed_hist.extend(self._pending_batch[:keep])
        self.uncommitted_verify = False


def make_server(eng, tok, spec_copy=False, spec_draft_k=3, spec_min_match=8, max_tokens=256):
    s = object.__new__(srv.Server)
    s.args = SimpleNamespace(
        reasoning_effort="low",
        max_tokens=max_tokens,
        temperature=0.0,
        spec_copy=spec_copy,
        spec_draft_k=spec_draft_k,
        spec_min_match=spec_min_match,
    )
    s.tok = tok
    s.tmpl = SimpleNamespace(render=lambda **kw: "")
    s.ids = {name: tok.token_to_id(name) for name in (srv.T_MODEL, srv.T_THINK, srv.T_TEXT, srv.T_TOOL, srv.T_END, srv.T_STOP)}
    s.eos_id = tok.token_to_id("<|endoftext|>")
    s.eng = eng
    s.rng = np.random.default_rng(123)
    s.render = lambda body: ""
    return s


def test_prompt_lookup_drafter_invariants():
    prompt = [10, 11, 12, 13, 14, 15, 16, 17, 99, 98, 97]
    d = srv.PromptLookupDrafter(prompt, min_match=4)
    # Current suffix [17, 99, 98, 97] must never match itself
    assert d.propose(4) == []
    # Append [10, 11, 12, 13]: matches earlier occurrence of length 4 -> predicts [14, 15, 16, 17]
    for tok in (10, 11, 12, 13):
        d.append(tok)
    assert d.propose(4) == [14, 15, 16, 17]
    assert d.propose(2) == [14, 15]
    assert d.propose(0) == []

    # Longest match preferred (8 over 4) and latest occurrence wins
    d2 = srv.PromptLookupDrafter(
        [1, 2, 3, 4, 5, 6, 7, 8, 50, 51, 9, 9, 5, 6, 7, 8, 60, 61, 1, 2, 3, 4, 5, 6, 7, 8],
        min_match=4,
        n_grams=(8, 4),
    )
    assert d2.propose(2) == [50, 51]  # 8-gram match [1..8] beats 4-gram [5..8] -> [60, 61]

    d2.reset([1, 2, 3])
    assert d2.propose(3) == []


def test_greedy_equivalence_full_partial_and_zero_accept():
    # Prompt contains repeated 10-token segments; target follows part of them, diverges, then hits another match
    seg_a = list(range(200, 212))  # 12 bytes
    seg_b = list(range(220, 232))  # 12 bytes
    prompt = [1, 3] + seg_a + [40, 41, 42] + seg_b + [5]
    # Desired generated stream:
    # [1, 3] + seg_a (full accept on draft) + [99] (0-accept mismatch!) + seg_b[:9] + [245, 246] (partial accept) + [5, 6]
    desired = [1, 3] + seg_a + [99] + seg_b[:9] + [245, 246, 5, 6]

    def oracle(seq):
        gen_len = len(seq) - len(prompt)
        return desired[gen_len] if gen_len < len(desired) else 6

    tok = StubTokenizer({}, make_specials())
    body = {"messages": [], "temperature": 0.0, "max_tokens": 128}

    # Baseline (no spec decode)
    eng_base = MockEngine(400, 512, oracle)
    eng_base.tok_override = prompt
    s_base = make_server(eng_base, tok, spec_copy=False)
    s_base.tok.encode = lambda p, add_special_tokens=False: SimpleNamespace(ids=list(prompt))
    events_base = []
    res_base = s_base.generate(body, lambda k, v: events_base.append((k, v)))

    for k in (1, 3, 7):
        eng_spec = MockEngine(400, 512, oracle)
        s_spec = make_server(eng_spec, tok, spec_copy=True, spec_draft_k=k, spec_min_match=8)
        s_spec.tok.encode = lambda p, add_special_tokens=False: SimpleNamespace(ids=list(prompt))
        events_spec = []
        res_spec = s_spec.generate(body, lambda kind, v: events_spec.append((kind, v)))

        assert res_spec["finish"] == res_base["finish"] == "stop"
        assert res_spec["timing"]["completion_tokens"] == res_base["timing"]["completion_tokens"]
        assert res_spec["timing"]["spec_accepted"] > 0
        assert events_spec == events_base
        assert eng_spec.tokens == eng_base.tokens
        assert eng_spec.committed_hist == eng_base.committed_hist
        assert not eng_spec.uncommitted_verify
        assert eng_spec.calls["verify"] > 0


def test_mid_draft_stop_and_max_new_boundaries():
    # Prompt ends with 8 tokens that earlier preceded [101, 102, T_STOP, 103]
    prefix = [110, 111, 112, 113, 114, 115, 116, 117]
    prompt = [1, 3] + prefix + [101, 102, 6, 103] + prefix
    desired = [1, 3, 101, 102, 6, 999]  # stop token appears at draft index 2

    def oracle(seq):
        return desired[len(seq) - len(prompt)]

    tok = StubTokenizer({}, make_specials())
    tok.encode = lambda p, add_special_tokens=False: SimpleNamespace(ids=list(prompt))

    eng = MockEngine(1024, 512, oracle)
    s = make_server(eng, tok, spec_copy=True, spec_draft_k=3, spec_min_match=8)
    # Prefix before stop: first 2 tokens [1, 3] generated normally, then prefix matches!
    # Wait: prompt already ends with `prefix`, so after emitting t=1, suffix is `prefix + [1]` (no match).
    # Test explicit mid-draft stop by ending prompt with `prefix[:-1]` and generating `prefix[-1]` first.
    prompt2 = [1, 3] + prefix + [101, 102, 6, 103] + prefix[:-1]
    desired2 = [prefix[-1], 101, 102, 6, 999]
    tok.encode = lambda p, add_special_tokens=False: SimpleNamespace(ids=list(prompt2))
    eng = MockEngine(1024, 512, lambda seq: desired2[len(seq) - len(prompt2)])
    s = make_server(eng, tok, spec_copy=True, spec_draft_k=3, spec_min_match=8)
    ev = []
    res = s.generate({"messages": [], "temperature": 0.0}, lambda k, v: ev.append((k, v)))
    assert res["finish"] == "stop"
    # Stop token (6) is at draft[2]: only [prefix[-1], 101, 102] are committed; 6 is NOT committed
    assert eng.committed_hist == prompt2 + [prefix[-1], 101, 102]
    assert eng.tokens == eng.committed_hist
    assert res["timing"]["completion_tokens"] == 4  # prefix[-1], 101, 102, 6
    assert not eng.uncommitted_verify

    # max_new clipping: max_new=3 must stop cleanly at 3 tokens without exceeding max_new
    eng2 = MockEngine(1024, 512, lambda seq: desired2[len(seq) - len(prompt2)])
    s2 = make_server(eng2, tok, spec_copy=True, spec_draft_k=7, spec_min_match=8, max_tokens=3)
    res2 = s2.generate({"messages": [], "temperature": 0.0, "max_tokens": 3}, lambda k, v: None)
    assert res2["finish"] == "length"
    assert res2["timing"]["completion_tokens"] == 3
    assert not eng2.uncommitted_verify

    # n_ctx clipping: leave just enough room for 2 more tokens
    n_ctx = len(prompt2) + 3
    eng3 = MockEngine(1024, n_ctx, lambda seq: desired2[len(seq) - len(prompt2)])
    s3 = make_server(eng3, tok, spec_copy=True, spec_draft_k=7, spec_min_match=8)
    res3 = s3.generate({"messages": [], "temperature": 0.0}, lambda k, v: None)
    assert res3["timing"]["completion_tokens"] <= 2
    assert len(eng3.committed_hist) <= n_ctx - 1
    assert not eng3.uncommitted_verify


def test_multibyte_utf8_and_tool_call_burst_streaming():
    emoji_bytes = list("🚀✨".encode("utf-8"))  # 4-byte + 3-byte UTF-8 characters
    tool_json = list(b'{"name":"ping","args":{"x":1}}')
    prefix = [100 + b for b in b"prefix01"]

    id_to_bytes = {100 + b: bytes([b]) for b in range(256)}
    for i, b in enumerate(emoji_bytes):
        id_to_bytes[400 + i] = bytes([b])
    emoji_toks = list(range(400, 400 + len(emoji_bytes)))
    tool_toks = [100 + b for b in tool_json]
    # Prompt has [T_MODEL, T_TEXT] + prefix + emoji + ... and ends with [T_MODEL, T_TEXT] + prefix[:-1]
    prompt = [1, 3] + prefix + emoji_toks + [5, 1, 4] + tool_toks + [5, 6] + [1, 3] + prefix[:-1]
    desired = [1, 3] + prefix + emoji_toks + [5, 1, 4] + tool_toks + [5, 6]

    tok = StubTokenizer(id_to_bytes, make_specials())
    tok.encode = lambda p, add_special_tokens=False: SimpleNamespace(ids=list(prompt))

    eng = MockEngine(600, 512, lambda seq: desired[len(seq) - len(prompt)] if len(seq) - len(prompt) < len(desired) else 6)
    s = make_server(eng, tok, spec_copy=True, spec_draft_k=7, spec_min_match=8)
    ev = []
    res = s.generate({"messages": [], "temperature": 0.0}, lambda k, v: ev.append((k, v)))
    assert res["finish"] == "tool_calls"
    text_out = "".join(v for k, v in ev if k == "content")
    assert "\ufffd" not in text_out
    assert text_out == "prefix01🚀✨"
    assert len(res["tool_calls"]) == 1
    assert res["tool_calls"][0]["function"]["name"] == "ping"
    assert res["tool_calls"][0]["function"]["arguments"] == '{"x": 1}'


def test_lossless_probabilistic_rejection_sampling():
    rng = np.random.default_rng(2026)
    logits = np.array([2.1, 1.4, 0.3, -1.2, -20.0], dtype=np.float32)

    # Edge case 1: drafted token has 0 probability under top_k=1 -> always rejected, returns argmax
    vlogits_k1 = np.stack([logits, logits])
    a, nxt = srv.verify_accept_probabilistic([1], vlogits_k1, temperature=1.0, top_p=1.0, top_k=1, rng=rng)
    assert (a, nxt) == (0, 0)

    # Edge case 2: drafted token has probability 1.0 under top_k=1 -> always accepted
    a2, nxt2 = srv.verify_accept_probabilistic([0], vlogits_k1, temperature=1.0, top_p=1.0, top_k=1, rng=rng)
    assert (a2, nxt2) == (1, 0)

    # Statistical equivalence: arbitrary wrong/partially-matching draft distribution must yield exact target distribution
    idx, target_p = srv.filtered_probs(logits, temperature=0.8, top_p=0.92, top_k=4)
    expected = np.zeros(5, dtype=np.float64)
    for t_idx, p_val in zip(idx, target_p):
        expected[t_idx] = p_val

    vlogits = np.stack([logits, logits])
    counts = np.zeros(5, dtype=np.int64)
    trials = 150000
    for i in range(trials):
        draft_tok = i % 5  # intentionally cycle over every token, including out-of-nucleus token 4
        acc, nxt = srv.verify_accept_probabilistic([draft_tok], vlogits, temperature=0.8, top_p=0.92, top_k=4, rng=rng)
        emitted = draft_tok if acc == 1 else nxt
        counts[emitted] += 1

    empirical = counts / trials
    max_err = float(np.max(np.abs(empirical - expected)))
    assert max_err < 0.005, f"max distribution error {max_err:.5f}: empirical={empirical}, expected={expected}"
