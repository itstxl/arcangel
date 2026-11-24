#!/usr/bin/env python3
"""Expected speed-up of copy-drafting speculative decoding, from a server trace and a cost model.

    ARCANGEL_TRACE=trace.jsonl server/run.sh ...      # record real requests
    bench/spec_sim.py trace.jsonl

Drafting: the longest earlier match (16/8/4/3/2 tokens) of the current suffix in prompt + output proposes the
next K tokens, only when the match is at least `minmatch` tokens long. A verify of T = 1 + K tokens costs
the dense layers once, the per-token parts T times and the experts T * f(T) times, f = the measured share of
distinct experts among T consecutive tokens (offloaded layers, opencode traffic, 2026-11-24). Costs in ms
per token for a B580 (offload: UD-Q4_K_XL with the CPU-expert layers and the hot-expert cache).
"""
import json
import sys

F = {1: 1.0, 2: 0.82, 3: 0.73, 4: 0.66, 5: 0.62, 6: 0.58, 7: 0.55, 8: 0.51, 9: 0.49}


def cost(T, setup):
    if setup == "offload":  # dense + head once, attention/small per token, experts (GPU + CPU), overhead
        return 5.5 + 4 * T + 16.4 * T * F[T] + 1.5
    return 5.0 + 4 * T + 6.0 * T * F[T] + 1.0


def run(rows, K, minmatch, setup):
    base = tot = 0.0
    for r in rows:
        seq, gen, idx = list(r["prompt"]), r["gen"], {}

        def add(end):
            for n in (2, 3, 4, 8, 16):
                if end >= n:
                    idx[(n, tuple(seq[end - n:end]))] = end

        for e in range(1, len(seq)):
            add(e)
        i = 0
        while i < len(gen):
            draft = []
            for n in (16, 8, 4, 3, 2):
                if n < minmatch or len(seq) < n:
                    continue
                j = idx.get((n, tuple(seq[-n:])))
                if j is not None:
                    draft = seq[j:j + K]
                    break
            a = 0
            while a < len(draft) and i + a < len(gen) and gen[i + a] == draft[a]:
                a += 1
            tot += cost(1 + len(draft), setup)
            for t in gen[i:i + a + 1]:
                seq.append(t)
                add(len(seq) - 1)
            i += a + 1
        base += cost(1, setup) * len(gen)
    return base / tot


if __name__ == "__main__":
    rows = [json.loads(l) for l in open(sys.argv[1])]
    for setup in ("offload", "vram"):
        print(setup)
        for K in (1, 2, 3, 4, 8):
            print(f"  K={K} " + "  ".join(f"match>={m:2d}: x{run(rows, K, m, setup):.2f}" for m in (2, 4, 8, 16)))
