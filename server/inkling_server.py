#!/usr/bin/env python3
"""OpenAI-compatible server for ArcAngel, the Inkling 2 Mini engine (libinkling.so).

    inkling_server.py --model MODEL.gguf [--n-ctx 32768] [--port 8082] [--reasoning-effort low]

Endpoints: GET /health, GET /v1/models, POST /v1/chat/completions (streaming and not, reasoning in
`reasoning_content`, OpenAI tool calls). One request at a time; the engine keeps the last conversation, so a
request that extends it only processes the new tokens. When a request diverges from what the engine holds
(for example the previous answer re-sent without its reasoning), the engine rolls back to the snapshot taken
at the end of the last prompt and continues from there.
"""
import argparse
import ctypes
import hashlib
import json
import os
import re
import select
import socket
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import jinja2
import numpy as np
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
TEMPLATE_SHA256 = (HERE / "chat_template.sha256").read_text(encoding="ascii").strip()

# special tokens of the Inkling 2 vocabulary
T_MODEL = "<|message_model|>"
T_THINK = "<|content_thinking|>"
T_TEXT = "<|content_text|>"
T_TOOL = "<|content_invoke_tool_json|>"
T_END = "<|end_message|>"
T_STOP = "<|content_model_end_sampling|>"


# --------------------------------------------------------------------------------------------------
# engine

class Engine:
    SNAP_PROMPT, SNAP_ZERO = 0, 9

    def __init__(self, lib_path, model, n_ctx, chunk, kv_bits=8, stash_ram_gb=None):
        self.lib = ctypes.CDLL(str(lib_path))
        L = self.lib
        L.ink_open.restype = ctypes.c_void_p
        L.ink_open.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        L.ink_error.restype = ctypes.c_char_p
        L.ink_n_vocab.argtypes = [ctypes.c_void_p]
        f32p = ctypes.POINTER(ctypes.c_float)
        L.ink_prefill.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int, f32p]
        L.ink_step.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int, f32p]
        L.ink_step_argmax.argtypes = [ctypes.c_void_p, ctypes.c_int32, ctypes.c_int]
        L.ink_step_argmax.restype = ctypes.c_int
        L.ink_save.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.ink_restore.argtypes = [ctypes.c_void_p, ctypes.c_int]
        L.ink_stash.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
        L.ink_drop.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.h = L.ink_open(model.encode(), n_ctx, chunk, kv_bits)
        if not self.h:
            raise RuntimeError("engine: " + L.ink_error().decode())
        self.n_ctx = n_ctx
        self.n_vocab = L.ink_n_vocab(self.h)
        L.ink_chunk.argtypes = [ctypes.c_void_p]
        self.chunk = L.ink_chunk(self.h)
        self.logits = np.zeros(self.n_vocab, dtype=np.float32)
        self.lp = self.logits.ctypes.data_as(f32p)
        self.tokens = []  # what the engine has processed, in order
        # host RAM per stashed prompt token: 8 full-attention slots x K and V x 8 heads x 128 dims, plus the
        # f16 scale per 32 values (the sliding-window slots add a fixed ~10-20 MB per stash)
        self.stash_bpt = 8 * 2 * 1024 * (kv_bits / 8 + 2 / 32)
        if stash_ram_gb is None:  # default: a quarter of the machine's RAM
            try:
                kb = int(next(l for l in open("/proc/meminfo") if l.startswith("MemTotal")).split()[1])
                stash_ram_gb = kb / 2**20 / 4
            except (OSError, StopIteration, ValueError):
                stash_ram_gb = 4.0
        self.stash_budget = stash_ram_gb * 2**30
        self.check(L.ink_save(self.h, self.SNAP_ZERO, 0))  # the empty state (zero convolution history)

    def check(self, rc):
        if rc < 0:
            raise RuntimeError("engine: " + self.lib.ink_error().decode())
        return rc

    def prefill(self, toks, pos0):
        arr = (ctypes.c_int32 * len(toks))(*toks)
        self.check(self.lib.ink_prefill(self.h, arr, len(toks), pos0, self.lp))

    def step(self, tok, pos):
        self.check(self.lib.ink_step(self.h, tok, pos, self.lp))

    def step_argmax(self, tok, pos):
        return self.check(self.lib.ink_step_argmax(self.h, tok, pos))

    # Whole conversations stashed in host RAM (full KV up to their prompt end), so a short unrelated request
    # (opencode's titles and summaries) does not cost the main conversation a full re-read.
    STASH_MIN = 1024  # only conversations with prompts at least this long are worth stashing
    STASH_MAX = 2
    stash = None
    prompt_len = 0

    @staticmethod
    def lcp(a, b):
        n = min(len(a), len(b))
        i = 0
        while i < n and a[i] == b[i]:
            i += 1
        return i

    def _stash_active(self):
        if self.stash is None:
            self.stash, self.next_id = [], 10
        if self.prompt_len < self.STASH_MIN:
            return
        ptoks = self.tokens[:self.prompt_len]
        if any(p == ptoks for _, p in self.stash):
            return
        sid = self.next_id
        self.next_id += 1
        self.check(self.lib.ink_stash(self.h, self.SNAP_PROMPT, sid))
        self.stash.append((sid, ptoks))
        self._trim_stash()

    def _trim_stash(self, keep=0):
        """Keep at most STASH_MAX stashes and within the host RAM budget (oldest go first); never drops the
        newest `keep` entries."""
        while len(self.stash) > keep and (len(self.stash) > self.STASH_MAX or
                              sum(len(p) for _, p in self.stash) * self.stash_bpt > self.stash_budget):
            old, _ = self.stash.pop(0)
            self.lib.ink_drop(self.h, old)

    def load_prompt(self, toks):
        """Bring the engine to the state after `toks`, reusing what it holds. Returns tokens processed."""
        if len(toks) >= self.n_ctx:
            raise ValueError(f"prompt of {len(toks)} tokens does not fit the {self.n_ctx}-token context")
        lcp = self.lcp(self.tokens, toks)
        best, best_l = None, 0
        for i, (sid, ptoks) in enumerate(self.stash or []):
            l = self.lcp(ptoks, toks)
            if l > best_l:
                best, best_l = i, l
        if best is not None and best_l > lcp and best_l >= self.STASH_MIN:
            # an earlier conversation fits better: stash the live one, bring that one back
            sid, ptoks = self.stash.pop(best)  # take it out first so stashing cannot evict it
            self._stash_active()
            self.stash.append((sid, ptoks))  # most recently used last
            self._trim_stash(keep=1)  # it is restored next
            pos = self.check(self.lib.ink_restore(self.h, sid))
            self.check(self.lib.ink_save(self.h, self.SNAP_PROMPT, pos))
            self.tokens, self.prompt_len = list(ptoks), pos
            lcp = self.lcp(self.tokens, toks)
            print(f"cache: switched to a stashed conversation ({pos} tokens)", flush=True)
        elif lcp < self.STASH_MIN and self.prompt_len >= self.STASH_MIN:
            self._stash_active()  # about to overwrite a long conversation
        have = self.tokens
        if lcp == len(have) and lcp < len(toks):
            start = lcp  # the prompt extends what the engine holds
        else:
            # diverged (or nothing new): roll back to the prompt snapshot if it lies inside the shared prefix
            # (strictly before the prompt end, so there is at least one token to process), else start over
            pos = self.lib.ink_restore(self.h, self.SNAP_PROMPT)  # -1: no snapshot yet
            if 0 < pos <= lcp and pos < len(toks):
                start = pos
            else:
                self.check(self.lib.ink_restore(self.h, self.SNAP_ZERO))
                start = 0
        self.prefill(toks[start:], start)
        self.tokens = list(toks)
        self.prompt_len = len(toks)
        self.check(self.lib.ink_save(self.h, self.SNAP_PROMPT, len(toks)))
        return len(toks) - start


# --------------------------------------------------------------------------------------------------
# chat template

def _tojson(value, indent=None, sort_keys=False, separators=None, ensure_ascii=False):
    return json.dumps(value, indent=indent, sort_keys=sort_keys, separators=separators, ensure_ascii=ensure_ascii)


def _raise(msg):
    raise jinja2.exceptions.TemplateError(msg)


def load_template(path):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != TEMPLATE_SHA256:
        raise ValueError(f"SHA-256 checksum mismatch for chat template: {path}")
    # the Hugging Face {% generation %} block only marks assistant text for training; render its contents
    src = re.sub(r"\{%-?\s*(end)?generation\s*-?%\}", "", raw.decode("utf-8"))
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    env.globals = {"namespace": env.globals["namespace"], "raise_exception": _raise}
    env.filters = {name: env.filters[name] for name in ("dictsort", "float", "trim")}
    env.filters["tojson"] = _tojson
    return env.from_string(src)


def normalise_messages(messages):
    """OpenAI messages -> what the Inkling template expects (tool call arguments as objects)."""
    out = []
    for m in messages:
        m = dict(m)
        if m.get("content") is None:
            m["content"] = ""
        if m.get("tool_calls"):
            tcs = []
            for tc in m["tool_calls"]:
                tc = json.loads(json.dumps(tc))
                fn = tc.get("function", {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        fn["arguments"] = {}
                tcs.append(tc)
            m["tool_calls"] = tcs
        if m.get("role") == "developer":
            m["role"] = "system"
        out.append(m)
    return out


# --------------------------------------------------------------------------------------------------
# sampling and output parsing

def sample(logits, temperature, top_p, top_k, rng):
    if temperature <= 0:
        return int(np.argmax(logits))
    k = top_k if top_k and top_k > 0 else 4096
    k = min(k, logits.size)
    idx = np.argpartition(logits, -k)[-k:]
    vals = logits[idx].astype(np.float64) / temperature
    order = np.argsort(-vals)
    idx, vals = idx[order], vals[order]
    p = np.exp(vals - vals[0])
    p /= p.sum()
    if 0 < top_p < 1:
        keep = int(np.searchsorted(np.cumsum(p), top_p) + 1)
        idx, p = idx[:keep], p[:keep] / p[:keep].sum()
    return int(rng.choice(idx, p=p))


class OutputParser:
    """Turns generated token ids into reasoning / text deltas and tool calls, as they arrive."""

    def __init__(self, tok, ids):
        self.tok = tok
        self.ids = ids
        self.mode = None  # None (between blocks), "header", "think", "text", "tool"
        self.block = []   # token ids of the current block's body
        self.sent = ""    # text of the current block already emitted
        self.anchor_ids = []
        self.anchor_text = ""
        self.pending = []
        self.header = []  # token ids of a tool header (the tool's name)
        self.tool_calls = []
        self.events = []  # (kind, value): ("reasoning", str) / ("content", str) / ("tool", dict)

    def _reset_block(self, mode, tok_list=None):
        self.mode = mode
        self.block = list(tok_list) if tok_list else []
        self.sent = ""
        self.anchor_ids = []
        self.anchor_text = ""
        self.pending = list(tok_list) if tok_list else []

    def _set_anchor(self):
        cand = self.pending[-1:]
        cand_txt = self.tok.decode(cand, skip_special_tokens=False)
        if not cand_txt.endswith("\ufffd") and not cand_txt.startswith("\ufffd"):
            self.anchor_ids, self.anchor_text = cand, cand_txt
            return
        cand = (self.anchor_ids + self.pending)[-4:]
        cand_txt = self.tok.decode(cand, skip_special_tokens=False)
        if not cand_txt.endswith("\ufffd") and not cand_txt.startswith("\ufffd"):
            self.anchor_ids, self.anchor_text = cand, cand_txt
        else:
            self.anchor_ids, self.anchor_text = [], ""

    def _flush(self, final=False):
        if not self.pending:
            return
        chunk = self.tok.decode(self.anchor_ids + self.pending, skip_special_tokens=False)
        if not final and chunk.endswith("\ufffd"):
            return  # a multi-byte character is not complete yet
        if chunk.startswith(self.anchor_text):
            new = chunk[len(self.anchor_text):]
            if new and self.mode in ("think", "text"):
                self.events.append(("reasoning" if self.mode == "think" else "content", new))
            self.sent += new
            self._set_anchor()
            self.pending = []
            return
        # Fallback if a token alters the decoded anchor boundary
        text = self.tok.decode(self.block, skip_special_tokens=False)
        if not final and text.endswith("\ufffd"):
            return
        new = text[len(self.sent):]
        if new and self.mode in ("think", "text"):
            self.events.append(("reasoning" if self.mode == "think" else "content", new))
        self.sent = text
        self._set_anchor()
        self.pending = []

    def feed(self, t):
        i = self.ids
        if t == i[T_MODEL]:
            self._reset_block("header")
            self.header = []
        elif self.mode == "header" and t == i[T_THINK]:
            self.mode = "think"
        elif self.mode == "header" and t == i[T_TEXT]:
            self.mode = "text"
        elif self.mode == "header" and t == i[T_TOOL]:
            self.mode = "tool"
        elif t == i[T_END]:
            if self.mode in ("think", "text"):
                self._flush(final=True)
            elif self.mode == "tool":
                self._close_tool()
            self._reset_block(None)
        elif self.mode == "header":
            self.header.append(t)
        elif self.mode in ("think", "text", "tool"):
            self.block.append(t)
            if self.mode != "tool":
                self.pending.append(t)
                self._flush()
        else:
            # text outside any block (should not happen): treat as content
            self._reset_block("text", [t])
            self._flush()

    def _close_tool(self):
        name = self.tok.decode(self.header, skip_special_tokens=False).strip()
        body = self.tok.decode(self.block, skip_special_tokens=False)
        try:
            obj = json.loads(body)
            args = obj.get("args", {})
            name = obj.get("name", name) or name
        except json.JSONDecodeError:
            args = {"_raw": body}
        call = {"id": "call_" + uuid.uuid4().hex[:24], "type": "function",
                "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}
        self.tool_calls.append(call)
        self.events.append(("tool", call))

    def take(self):
        ev, self.events = self.events, []
        return ev


# --------------------------------------------------------------------------------------------------
# HTTP

class Server:
    def __init__(self, args):
        self.args = args
        assets = Path(args.assets) if args.assets else HERE
        self.tmpl = load_template(assets / "chat_template.jinja")
        self.tok = Tokenizer.from_file(str(assets / "tokenizer.json"))
        self.ids = {s: self.tok.token_to_id(s) for s in (T_MODEL, T_THINK, T_TEXT, T_TOOL, T_END, T_STOP)}
        self.eos_id = self.tok.token_to_id("<|endoftext|>")
        lib = Path(args.lib) if args.lib else HERE.parent / "build" / "libinkling.so"
        t0 = time.time()
        self.eng = Engine(lib, args.model, args.n_ctx, args.chunk, args.kv_bits, args.stash_ram_gb)
        print(f"engine ready in {time.time() - t0:.1f} s: context {args.n_ctx}, {args.kv_bits}-bit KV cache, prompt chunk {self.eng.chunk}, vocab {self.eng.n_vocab}", flush=True)
        self.model_info = {
            "id": args.name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "arcangel",
            "aliases": [args.name],
            "tags": [],
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "context_length": self.eng.n_ctx,
            "meta": {"n_ctx": self.eng.n_ctx, "n_vocab": self.eng.n_vocab},
        }
        self.lock = threading.Lock()
        self.rng = np.random.default_rng()

    def render(self, body):
        kw = dict(body.get("chat_template_kwargs") or {})
        effort = body.get("reasoning_effort") or kw.get("reasoning_effort") or self.args.reasoning_effort
        tools = body.get("tools") or None
        if body.get("tool_choice") == "none":
            tools = None
        prompt = self.tmpl.render(messages=normalise_messages(body.get("messages", [])), tools=tools,
                                  reasoning_effort=effort, add_generation_prompt=True)
        return prompt

    def generate(self, body, emit, gone=None):
        """Run one chat completion; emit(kind, value) for streaming. Returns the final result dict.
        gone(): True when the client has disconnected (checked every 16 tokens; generation stops)."""
        prompt = self.render(body)
        toks = self.tok.encode(prompt, add_special_tokens=False).ids
        max_new = int(body.get("max_completion_tokens") or body.get("max_tokens") or self.args.max_tokens)
        max_new = max(1, min(max_new, self.eng.n_ctx - len(toks) - 1))
        temp = float(body.get("temperature", self.args.temperature))
        top_p = float(body.get("top_p", 1.0))
        top_k = int(body.get("top_k", 0) or 0)
        rng = np.random.default_rng(body["seed"]) if body.get("seed") is not None else self.rng
        parser = OutputParser(self.tok, self.ids)
        t0 = time.time()
        processed = self.eng.load_prompt(toks)
        t1 = time.time()
        pos = len(toks)
        stop_id = self.ids[T_STOP]
        eos_id = self.eos_id
        n_gen, finish = 0, "length"
        gen = []
        next_tok = sample(self.eng.logits, temp, top_p, top_k, rng)
        while n_gen < max_new:
            t = next_tok
            n_gen += 1
            gen.append(t)
            if t == stop_id or t == eos_id:
                finish = "stop"
                break
            parser.feed(t)
            for kind, val in parser.take():
                emit(kind, val)
            if pos + 1 >= self.eng.n_ctx or n_gen >= max_new:
                break
            if gone is not None and n_gen % 16 == 0 and gone():
                finish = "cancelled"
                break
            if temp <= 0:
                next_tok = self.eng.step_argmax(t, pos)
            else:
                self.eng.step(t, pos)
                next_tok = sample(self.eng.logits, temp, top_p, top_k, rng)
            self.eng.tokens.append(t)
            pos += 1
        t2 = time.time()
        if os.environ.get("ARCANGEL_TRACE"):  # developer aid: the token ids of every request, one JSON line each
            with open(os.environ["ARCANGEL_TRACE"], "a") as f:
                f.write(json.dumps({"prompt": toks, "gen": gen, "temp": temp}) + "\n")
        if parser.mode in ("think", "text"):
            parser._flush(final=True)
            for kind, val in parser.take():
                emit(kind, val)
        if parser.tool_calls and finish == "stop":
            finish = "tool_calls"
        timing = {"prompt_tokens": len(toks), "prompt_processed": processed, "prompt_s": round(t1 - t0, 3),
                  "completion_tokens": n_gen, "gen_s": round(t2 - t1, 3),
                  "gen_tok_s": round(n_gen / max(t2 - t1, 1e-9), 2)}
        print(f"request: prompt {len(toks)} ({processed} new, {t1 - t0:.2f} s), generated {n_gen} "
              f"at {timing['gen_tok_s']} tok/s, finish {finish}", flush=True)
        return {"finish": finish, "tool_calls": parser.tool_calls, "timing": timing}


def make_handler(server):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *a):
            pass

        def gone(self):
            """The client closed the connection (readable with no data = EOF)."""
            try:
                r, _, _ = select.select([self.connection], [], [], 0)
                return bool(r) and self.connection.recv(1, socket.MSG_PEEK) == b""
            except OSError:
                return True

        def _json(self, code, obj):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                return self._json(200, {"status": "ok"})
            if self.path.rstrip("/") == "/v1/models":
                return self._json(200, {"object": "list", "data": [server.model_info]})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if self.path.rstrip("/") != "/v1/chat/completions":
                return self._json(404, {"error": {"message": "not found"}})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError as e:
                return self._json(400, {"error": {"message": f"bad JSON: {e}"}})
            cid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            stream = bool(body.get("stream"))
            with server.lock:
                if stream:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Transfer-Encoding", "chunked")
                    self.end_headers()
                    n_tools = [0]

                    def send(obj):
                        data = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
                        self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
                        self.wfile.flush()

                    def chunk(delta, finish=None, usage=None):
                        c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": server.args.name,
                             "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                        if usage is not None:
                            c["usage"] = usage
                        send(c)

                    def emit(kind, val):
                        if kind == "content":
                            chunk({"content": val})
                        elif kind == "reasoning":
                            chunk({"reasoning_content": val})
                        elif kind == "tool":
                            chunk({"tool_calls": [{"index": n_tools[0], **val}]})
                            n_tools[0] += 1

                    try:
                        chunk({"role": "assistant", "content": ""})
                        res = server.generate(body, emit, self.gone)
                        if res["finish"] == "cancelled":
                            print("client went away: generation stopped", flush=True)
                            return
                        t = res["timing"]
                        usage = {"prompt_tokens": t["prompt_tokens"], "completion_tokens": t["completion_tokens"],
                                 "total_tokens": t["prompt_tokens"] + t["completion_tokens"]}
                        chunk({}, res["finish"], usage if (body.get("stream_options") or {}).get("include_usage") else None)
                        send("[DONE]")
                        self.wfile.write(b"0\r\n\r\n")
                    except (BrokenPipeError, ConnectionResetError):
                        print("client went away", flush=True)
                    except Exception as e:  # report inside the stream
                        print(f"error: {e}", flush=True)
                        try:
                            send({"error": {"message": str(e)}})
                            self.wfile.write(b"0\r\n\r\n")
                        except Exception:
                            pass
                    return
                parts = {"content": [], "reasoning": []}

                def emit(kind, val):
                    if kind in parts:
                        parts[kind].append(val)

                try:
                    res = server.generate(body, emit, self.gone)
                    if res["finish"] == "cancelled":
                        print("client went away: generation stopped", flush=True)
                        return
                except Exception as e:
                    return self._json(500, {"error": {"message": str(e)}})
                msg = {"role": "assistant", "content": "".join(parts["content"]) or None}
                if parts["reasoning"]:
                    msg["reasoning_content"] = "".join(parts["reasoning"])
                if res["tool_calls"]:
                    msg["tool_calls"] = res["tool_calls"]
                t = res["timing"]
                self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": server.args.name,
                                 "choices": [{"index": 0, "message": msg, "finish_reason": res["finish"]}],
                                 "usage": {"prompt_tokens": t["prompt_tokens"], "completion_tokens": t["completion_tokens"],
                                           "total_tokens": t["prompt_tokens"] + t["completion_tokens"]},
                                 "timings": t})

    return Handler


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--lib", default=None, help="libinkling.so (default: ../build/libinkling.so)")
    ap.add_argument("--assets", default=None, help="directory with tokenizer.json and chat_template.jinja (default: next to this file)")
    ap.add_argument("--n-ctx", type=int, default=32768)
    ap.add_argument("--chunk", type=int, default=0, help="prompt chunk size (0: the largest of 1024/512/256/128 that fits the free VRAM)")
    ap.add_argument("--kv-bits", type=int, default=8, choices=(8, 4), help="K/V cache precision")
    ap.add_argument("--stash-ram-gb", type=float, default=None,
                    help="host RAM for stashed conversations (default: a quarter of the machine's RAM)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8082)
    ap.add_argument("--name", default="inkling-2-mini")
    ap.add_argument("--reasoning-effort", default="low")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=8192)
    args = ap.parse_args()
    server = Server(args)
    class Quiet(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError)):
                return  # a client hung up mid-request: nothing to report
            super().handle_error(request, client_address)

    httpd = Quiet((args.host, args.port), make_handler(server))
    print(f"listening on http://{args.host}:{args.port}", flush=True)
    httpd.serve_forever()


if __name__ == "__main__":
    main()
