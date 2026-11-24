import json, sys, urllib.request, random
port, mode = sys.argv[1], sys.argv[2]
def ask(content):
    body = {"messages": [{"role": "user", "content": content}], "max_tokens": 1500, "temperature": 0, "reasoning_effort": "none"}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=3600).read())
    m = r["choices"][0]["message"]
    t = r["timings"]; t["finish"] = r["choices"][0]["finish_reason"]; t["reasoning"] = (m.get("reasoning_content") or "")[-200:]
    return m.get("content") or "", t
random.seed(7)
words = "river stone cloud lamp garden window paper copper violet engine harbor meadow".split()
def filler(n): return [" ".join(random.choice(words) for _ in range(12)) + "." for _ in range(n)]
if mode == "multi":   # ~60k tokens, 12 facts scattered, ask for 4
    doc = filler(3800)
    names = ["ALPHA", "BRAVO", "CEDAR", "DELTA", "EMBER", "FALCON", "GARNET", "HOLLY", "IVORY", "JASPER", "KELP", "LOTUS"]
    vals = {n: f"{random.randint(1000, 9999)}" for n in names}
    for i, n in enumerate(names):
        doc.insert(int(len(doc) * (i + 0.5) / len(names)) + i, f"Record: the code for {n} is {vals[n]}.")
    ask_for = ["BRAVO", "FALCON", "IVORY", "LOTUS"]
    a, t = ask("\n".join(doc) + "\n\nList the codes for " + ", ".join(ask_for) + ", one per line as NAME=CODE. Nothing else.")
    got = sum(f"{n}={vals[n]}" in a.replace(" ", "") for n in ask_for)
    print(f"multi-needle: {got}/4 correct; prompt {t['prompt_tokens']} in {t['prompt_s']} s; finish {t['finish']}; answer {a.strip()!r}; reasoning tail {t['reasoning']!r}; expected {[f'{n}={vals[n]}' for n in ask_for]}")
else:                  # ~90k tokens, one needle at three depths
    base = filler(5700)
    for frac, code in ((0.1, "4410-MARLIN"), (0.55, "2093-PLOVER"), (0.95, "7738-TERN")):
        doc = list(base); doc.insert(int(len(doc) * frac), f"The secret access code is {code}.")
        a, t = ask("\n".join(doc) + "\n\nWhat is the secret access code mentioned above? Answer with the code only.")
        print(f"needle {frac:.0%}: {'OK ' if code in a else 'MISS'} {a.strip()!r} (prompt {t['prompt_tokens']}, {t['prompt_s']} s, decode {t['gen_tok_s']} tok/s)")
