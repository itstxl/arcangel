#!/usr/bin/env python3
"""Estimate what copy-based speculative decoding (prompt lookup) would gain, from a server trace.

    ARCANGEL_TRACE=trace.jsonl server/run.sh ...   # record real requests (one JSON line each)
    bench/draft_sim.py trace.jsonl [K ...]

For every generated token the drafter looks for the latest earlier occurrence of the last N tokens (N = 4, 3,
then 2) in prompt + output so far and proposes the K tokens that followed it. A verify step checks the drafts in
one batch and always yields one more token (the model's own next token), so a step yields accepted + 1 tokens.
The sampled tokens in the trace stand in for the model's choice: with sampling, a draft d is accepted with
probability p(d), which is also the chance that the trace sampled d, so the estimate is unbiased.
"""
import json
import sys


def simulate(prompt, gen, K, nmax=4, nmin=2):
    seq = list(prompt)
    # (n, ngram) -> position just after its latest occurrence, for occurrences ending before the last token
    # (the current suffix itself is indexed one step later, so a lookup never finds itself)
    idx = {}

    def index_upto(end):
        for n in range(nmin, nmax + 1):
            if end >= n:
                idx[(n, tuple(seq[end - n:end]))] = end

    for e in range(1, len(seq)):
        index_upto(e)

    steps = drafted_steps = drafted = accepted = 0
    i = 0
    while i < len(gen):
        draft = []
        for n in range(nmax, nmin - 1, -1):
            j = idx.get((n, tuple(seq[-n:]))) if len(seq) >= n else None
            if j is not None:
                draft = seq[j:j + K]
                break
        a = 0
        while a < len(draft) and i + a < len(gen) and gen[i + a] == draft[a]:
            a += 1
        steps += 1
        if draft:
            drafted_steps += 1
            drafted += len(draft)
            accepted += a
        for t in gen[i:i + a + 1]:
            seq.append(t)
            index_upto(len(seq) - 1)
        i += a + 1
    return steps, drafted_steps, drafted, accepted


def main():
    rows = [json.loads(l) for l in open(sys.argv[1])]
    Ks = [int(k) for k in sys.argv[2:]] or [4, 8, 16]
    n_gen = sum(len(r["gen"]) for r in rows)
    print(f"{len(rows)} requests, {n_gen} generated tokens, prompts {min(len(r['prompt']) for r in rows)}-"
          f"{max(len(r['prompt']) for r in rows)} tokens")
    for K in Ks:
        S = D = DR = A = 0
        for r in rows:
            s, d, dr, a = simulate(r["prompt"], r["gen"], K)
            S, D, DR, A = S + s, D + d, DR + dr, A + a
        print(f"K={K:2d}: {n_gen / S:.2f} tokens per step; drafts on {D / S:.0%} of steps, "
              f"{A / max(DR, 1):.0%} of drafted tokens accepted, {A / max(D, 1):.2f} accepted per drafted step")
    # per request, K=8
    print("per request (K=8): generated, tokens/step")
    for r in rows:
        s, d, dr, a = simulate(r["prompt"], r["gen"], 8)
        print(f"  {len(r['prompt']):6d} prompt, {len(r['gen']):5d} gen, {len(r['gen']) / s:.2f}")


if __name__ == "__main__":
    main()
