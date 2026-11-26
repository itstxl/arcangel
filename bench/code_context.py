#!/usr/bin/env python3
"""Measure prefill and decode on a fixed, tokenized source-code workload."""
import argparse
import ctypes as ct
import hashlib
import json
import os
import time
from pathlib import Path

from tokenizers import Tokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("corpus", type=Path, help="text file containing source code and coding context")
    parser.add_argument("--library", type=Path, default=Path("build/libinkling.so"))
    parser.add_argument("--tokenizer", type=Path, default=Path("server/tokenizer.json"))
    parser.add_argument("--context", type=int, default=131072)
    parser.add_argument("--prompt", type=int, default=130048)
    parser.add_argument("--append", type=int, default=1024, help="time this many prompt tokens separately")
    parser.add_argument("--decode", type=int, default=128)
    parser.add_argument("--chunk", type=int, choices=(0, 128, 256, 512, 1024, 2048, 4096), default=0,
                        help="prompt chunk size (0: auto)")
    parser.add_argument("--runs", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.append < args.prompt or min(args.decode, args.runs) < 1:
        parser.error("require 0 <= append < prompt, positive decode tokens and positive runs")
    if args.prompt + args.decode > args.context:
        parser.error("prompt and decode tokens must fit in context")
    if os.environ.get("INK_OFFLOAD") or os.environ.get("INK_KV", "8") != "8":
        parser.error("this benchmark requires full VRAM and an 8-bit KV cache")

    corpus = args.corpus.read_bytes()
    tokenizer = args.tokenizer.read_bytes()
    tokens = Tokenizer.from_buffer(tokenizer).encode(corpus.decode(), add_special_tokens=False).ids
    if len(tokens) < args.prompt + args.decode:
        parser.error(f"corpus has only {len(tokens)} tokens")
    prefix_len = args.prompt - args.append
    prefix = (ct.c_int32 * prefix_len)(*tokens[:prefix_len])
    append = (ct.c_int32 * args.append)(*tokens[prefix_len:args.prompt])

    lib = ct.CDLL(str(args.library.resolve()))
    handle_type = ct.c_void_p
    lib.ink_error.restype = ct.c_char_p
    lib.ink_open.argtypes = [ct.c_char_p, ct.c_int, ct.c_int, ct.c_int]
    lib.ink_open.restype = handle_type
    lib.ink_close.argtypes = [handle_type]
    lib.ink_close.restype = None
    lib.ink_chunk.argtypes = [handle_type]
    lib.ink_prefill.argtypes = [handle_type, ct.POINTER(ct.c_int32), ct.c_int, ct.c_int, ct.c_void_p]
    lib.ink_step.argtypes = [handle_type, ct.c_int32, ct.c_int, ct.c_void_p]
    lib.ink_save.argtypes = [handle_type, ct.c_int, ct.c_int]
    lib.ink_restore.argtypes = [handle_type, ct.c_int]

    def check(status):
        if status < 0:
            raise RuntimeError(lib.ink_error().decode())

    h = lib.ink_open(args.model.encode(), args.context, args.chunk, 8)
    if not h:
        raise RuntimeError(lib.ink_error().decode())
    try:
        # Position zero alone does not reset the convolution history.
        check(lib.ink_save(h, 0, 0))
        print(json.dumps(dict(context=args.context, kv_bits=8, prefix=prefix_len, append=args.append,
                              decode=args.decode, chunk=lib.ink_chunk(h),
                              corpus_sha256=hashlib.sha256(corpus).hexdigest(),
                              tokenizer_sha256=hashlib.sha256(tokenizer).hexdigest())), flush=True)
        warm_len = min(prefix_len, lib.ink_chunk(h))
        check(lib.ink_prefill(h, prefix, warm_len, 0, None))
        check(lib.ink_step(h, tokens[warm_len], warm_len, None))
        for run in range(args.runs):
            check(lib.ink_restore(h, 0))
            t0 = time.perf_counter()
            check(lib.ink_prefill(h, prefix, prefix_len, 0, None))
            t1 = time.perf_counter()
            if args.append:
                check(lib.ink_prefill(h, append, args.append, prefix_len, None))
            t2 = time.perf_counter()
            for pos in range(args.prompt, args.prompt + args.decode):
                check(lib.ink_step(h, tokens[pos], pos, None))
            t3 = time.perf_counter()
            print(json.dumps(dict(run=run + 1, prefill_tok_s=args.prompt / (t2 - t0),
                                  append_tok_s=args.append / (t2 - t1) if args.append else None,
                                  decode_tok_s=args.decode / (t3 - t2))), flush=True)
    finally:
        lib.ink_close(h)


if __name__ == "__main__":
    main()
