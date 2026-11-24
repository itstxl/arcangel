#!/usr/bin/env python3
"""Generate the benchmark request files in bench/requests/.

short  - one-paragraph answer, no context
code   - ~700 tokens of new code
long   - ~10k-token needle-in-a-haystack prompt (code PURPLE-HERON-4471)
follow - the long prompt + one more turn (measures prompt-cache reuse and decode at depth)
edit   - return a 150-line Python file with one function renamed (draft-friendly copying)

Reasoning effort is left to the server default unless EFFORT is given as argv[1].
"""
import json, random, re, sys
from pathlib import Path

out = Path(__file__).resolve().parent / "requests"
out.mkdir(exist_ok=True)
effort = sys.argv[1] if len(sys.argv) > 1 else None


def save(name, messages, max_tokens):
    body = {"messages": messages, "max_tokens": max_tokens, "temperature": 0}
    if effort:
        body["chat_template_kwargs"] = {"reasoning_effort": effort}
    (out / f"{name}.json").write_text(json.dumps(body))


save("short", [{"role": "user", "content": "In two sentences, explain why the sky is blue."}], 300)
save("code", [{"role": "user", "content": (
    "Write a complete Python module implementing an LRU cache class with get, put, delete, resize and a "
    "__repr__, using collections.OrderedDict, with type hints and docstrings, followed by a pytest test "
    "file covering every method. Output only code.")}], 1200)

random.seed(1)
words = "the river city market morning lantern quiet signal winter harbor garden engine ledger window orchard".split()
paras = [" ".join(random.choice(words) for _ in range(9)) + "." for _ in range(7000)]
paras.insert(4700, "The secret access code for the vault is PURPLE-HERON-4471.")
long_msgs = [{"role": "user", "content": " ".join(paras) +
              "\n\nWhat is the secret access code for the vault? Answer with the code only."}]
save("long", long_msgs, 400)
save("follow", long_msgs + [{"role": "assistant", "content": "PURPLE-HERON-4471"},
                            {"role": "user", "content": "Now spell that code backwards."}], 300)

src_path = Path.home() / ".local/src/llama.cpp/gguf-py/gguf/lazy.py"
src = "".join(src_path.read_text().splitlines(keepends=True)[:150])
old = [n for n in re.findall(r"def (\w+)\(", src) if not n.startswith("__") and len(n) > 5][0]
save("edit", [{"role": "user", "content": (
    f"Here is a Python file:\n\n```python\n{src}```\n\nRename the function `{old}` to `{old}_v2` everywhere it "
    "appears (definition and every call). Return the complete updated file in one code block and nothing else.")}],
     3000)
print("wrote", sorted(p.name for p in out.glob("*.json")), "| rename:", old, "| effort:", effort or "server default")
