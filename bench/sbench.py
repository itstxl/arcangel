#!/usr/bin/env python3
"""Streaming benchmark client for an OpenAI-compatible server.

usage: sbench.py PORT NAME [NAME ...]     (NAME = a file in bench/requests/ without .json)

Per request it prints: prompt tokens, time to first token (reasoning or content), output tokens and
the client-side decode rate after the first token, llama-server's own timings and draft acceptance
when the server reports them, and the start of the answer.
"""
import json, sys, time, urllib.request
from pathlib import Path

port = sys.argv[1]
reqdir = Path(__file__).resolve().parent / "requests"

for name in sys.argv[2:]:
    body = json.loads((reqdir / f"{name}.json").read_text())
    body.update({"stream": True, "stream_options": {"include_usage": True}})
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; content = ""; reasoning = 0; usage = {}; timings = {}
    try:
        with urllib.request.urlopen(req, timeout=1200) as r:
            for raw in r:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                usage = ev.get("usage") or usage
                timings = ev.get("timings") or timings
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    piece = d.get("content") or ""
                    rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
                    if (piece or rpiece) and t_first is None:
                        t_first = time.time()
                    content += piece
                    reasoning += len(rpiece)
    except Exception as e:  # keep going so one failure does not hide the others
        print(f"  {name:7s} FAILED: {e}")
        continue
    t1 = time.time()
    pt = usage.get("prompt_tokens", timings.get("prompt_n", 0))
    ct = usage.get("completion_tokens", timings.get("predicted_n", 0))
    ttft = (t_first or t1) - t0
    rate = (ct - 1) / (t1 - t_first) if t_first and ct > 1 and t1 > t_first else 0.0
    extra = ""
    if timings.get("draft_n"):
        extra = f" | draft {timings['draft_n_accepted']}/{timings['draft_n']} ({100 * timings['draft_n_accepted'] / timings['draft_n']:.0f}%)"
    if timings.get("predicted_per_second"):
        extra += f" | server gen {timings['predicted_per_second']:.1f} t/s"
    print(f"  {name:7s} prompt {pt:6d} | TTFT {ttft:6.1f}s | out {ct:5d} @ {rate:5.1f} t/s{extra} | reasoning {reasoning} ch")
    print("          answer:", content.strip().replace("\n", " ")[:100])
