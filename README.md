# ArcAngel

A fast local coding model for Intel Arc Battlemage cards. ArcAngel is an inference engine written for one
model, [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), and one GPU family, Intel Xe2
(Arc B570, B580, Pro B50, Pro B60, Pro B70). It serves an OpenAI-compatible API, so opencode and other
coding agents can use it.

- **Everything stays in VRAM.** No offload to system RAM or disk: the engine refuses to start rather than
  spill.
- **Long context.** 96k tokens on a 12 GB B580, up to 512k on a 32 GB B70.
- **Fast.** On a B580: 65 tok/s generation at short context, 43 tok/s at 90k, prompts read at 600-900 tok/s.
  That is about twice llama.cpp's generation speed and three to four times its prompt speed on the same card.

## Tested speeds

Arc B580 (12 GB), UD-Q2_K_XL, 96k context, 4-bit K/V cache, 1024-token prompt chunks. For comparison,
llama.cpp (Vulkan) was run on the same card with the same model, including its DSpark draft model for
speculative decoding.

| Generation (tokens/s) | ArcAngel | llama.cpp |
|---|---|---|
| short context | **65** | 33-38 |
| at 4k tokens | **62** | |
| at 12k (a real opencode session) | **57** | 17-21 |
| at 30k | **54** | ~18 |
| at 90k | **43** | (does not fit) |

| Prompt reading (tokens/s) | ArcAngel | llama.cpp |
|---|---|---|
| 4k-token prompt | **900** | ~225 |
| 16k | **825** | |
| 32k | **780** | |
| 90k | **627** (144 s) | (does not fit) |

Accuracy: against llama.cpp's CPU output the engine matches the top token on every step tested, with KL
divergence 0.001-0.002 using an 8-bit cache and 0.006-0.016 using the 4-bit cache. With the 4-bit cache, it
found hidden facts placed at 10%, 55% and 95% of a 90k-token prompt, and 4 of 4 facts in a 60k
multi-fact test. How the speed was reached, step by step: [PERF.md](PERF.md).

## Install

```
git clone https://github.com/itstxl/arcangel && cd arcangel
./install.sh
```

The installer detects the card, asks before it changes anything, and:

1. picks the profile for the card (below): which quantisation to download, context length, cache precision
2. checks the compiler and GPU runtime; on Arch-based distributions it can install them with pacman
3. downloads the model from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF),
   reusing a copy that is already on disk
4. builds the engine for the card's chip (kernels compiled ahead of time; takes a few minutes)
5. writes `~/.config/arcangel/arcangel.env` and a systemd user service, `arcangel`
6. adds the model to opencode, if you use it, and starts the server

Re-running it is safe; it skips what is already done. `./install.sh --help` lists the options (choose the card,
use your own GGUF, change context or port, run without questions). `./uninstall.sh` removes the service, the
opencode entry and the settings, and keeps the downloaded model unless you pass `--models`.

### Card profiles

| Card | VRAM | Model | Context | K/V cache | Status |
|---|---|---|---|---|---|
| Arc B570 | 10 GB | UD-IQ1_S (7.8 GB) | 32k | 4-bit | untested |
| Arc B580 | 12 GB | UD-Q2_K_XL (9.1 GB) | 96k | 4-bit | **tested** |
| Arc Pro B50 | 16 GB | UD-Q2_K_XL (9.1 GB) | 256k | 8-bit | untested |
| Arc Pro B60 | 24 GB | UD-Q4_K_XL (17.6 GB) | 256k | 8-bit | untested |
| Arc Pro B70 | 32 GB | UD-Q4_K_XL (17.6 GB) | 512k | 8-bit | untested |

The profiles live in [cards.tsv](cards.tsv). Only the B580 has run the whole setup. The other profiles were
sized from each card's memory: weights, plus the K/V cache, plus prompt buffers, plus a 0.5 GB margin, plus
about 1 GB for a desktop on the same card. What has been checked for them:

- the weight formats of Q4_K_XL and Q4_K_M (Q8_0 and Q5_K experts) match llama.cpp on a B580
  (`tests/format_test.sh`);
- the engine builds for the B70's chip (BMG-G31).

If you run one of the untested cards, please report how it went. If a profile does not fit (for example,
because the desktop uses more VRAM), re-run the installer with a smaller `--ctx` or with `--kv-bits 4`.

Why these models:
- The B60 and B70 get Q4_K_XL, not Q8_K_XL. Both use the same NVFP4 experts for 40 of the 48 expert tensors;
  Q8_K_XL only widens the dense layers to BF16, which costs bandwidth for little gain. The B70 spends its
  extra memory on context instead.
- Q3_K_XL, IQ3_XXS and IQ4_XS use weight formats the engine does not support yet (Q3_K, IQ3_S, IQ4_NL, Q5_0).

## Use

```
systemctl --user start arcangel        # or: server/run.sh (foreground)
systemctl --user stop arcangel         # frees the VRAM
journalctl --user -u arcangel -f       # logs
```

The API is at `http://127.0.0.1:8082/v1`, with model name `inkling-2-mini`. In opencode, pick
**Inkling 2 Mini (ArcAngel)**. The installer adds a provider like this to `~/.config/opencode/opencode.json`:

```json
"arcangel": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "ArcAngel",
  "options": { "baseURL": "http://127.0.0.1:8082/v1" },
  "models": {
    "inkling-2-mini": {
      "name": "Inkling 2 Mini (ArcAngel, 96k)",
      "tool_call": true, "reasoning": true,
      "limit": { "context": 98304, "output": 8192 }
    }
  }
}
```

**Settings** are in `~/.config/arcangel/arcangel.env`: model, context, cache bits, port and reasoning effort.
Restart the service after editing it.

**Conversation cache.** The server keeps the current conversation on the GPU and continues it without
re-reading the prompt. It also stashes up to two other long conversations in system RAM, so an agent's side
requests (titles, summaries) do not cost a full re-read. The stash is limited to a quarter of RAM by default
(`ARCANGEL_STASH_RAM_GB`).

**Reasoning effort** defaults to `low` (the model's own default is high and slow); requests can pass
`reasoning_effort`.

**VRAM.** The engine refuses to start rather than spill into system RAM. If it says there is not enough free
VRAM, close other GPU programs (another model server, games) or lower the context. On a card that also drives
your desktop, the desktop's use moves around: on the B580, 0.8-1.2 GB. The prompt chunk size (1024, 512, 256 or
128 tokens) is picked at start-up from what is free.

**Multiple GPUs.** The engine uses the Level Zero GPU with the most memory; set `ARCANGEL_DEVICE=n` to pick
another.

## Requirements

- An Intel Arc Battlemage (Xe2) card on Linux with the `xe` (or `i915`) driver
- Intel oneAPI DPC++/C++ compiler 2025.1 or newer (tested: 2026.0), Level Zero, and the Intel compute runtime
  with `ocloc`
- CMake 3.24+, Python 3.10+, about 10-20 GB of disk for the model

On Arch-based systems (including CachyOS and Omarchy), the installer can install these:
`intel-oneapi-dpcpp-cpp intel-compute-runtime level-zero-loader cmake ninja python curl`. On other
distributions, install them yourself first (see `install.sh` for package names); only Arch has been tested.

## How it works

Inkling 2 Mini is a mixture-of-experts model (128 experts, 8 active per token) with looped layers: 24 physical
layers run as 48 layer visits. The engine is SYCL C++ for Level Zero, with:

- **Kernels per weight format** for the model's 15 formats (IQ1_S, IQ2_*, Q2_0, NVFP4, MXFP4, Q4_K-Q8_0, ...),
  bit-exact with ggml. Weights are repacked losslessly at load into layouts that suit the GPU.
- **Prompt path** on the XMX matrix units: grouped expert GEMMs and flash attention.
- **Decode path** as a recorded SYCL graph, with int8 DP4A experts and attention.
- **K/V cache** in int8 or 4-bit, which is what makes 96k fit on 12 GB.

Every change is checked against llama.cpp's CPU output (KL ~0.001-0.002, same top token). [SPEC.md](SPEC.md)
walks through the forward pass. [PERF.md](PERF.md) records what made it fast, what did not, and why.

| Path | What |
|---|---|
| `install.sh`, `uninstall.sh`, `cards.tsv` | installer and card profiles |
| `server/` | OpenAI-compatible server (`inkling_server.py`, `run.sh`) |
| `src/gguf_file.*` | read-only mmap GGUF v3 reader |
| `src/device.*` | Level Zero device selection |
| `src/quant.h`, `src/quant_tables.h` | the weight formats (tables generated by `tools/gen_tables.py`) |
| `src/model.*` | weights, schedule of the 48 layer visits, repacking at load, VRAM check |
| `src/engine.*` | naive reference forward pass (f32, one token at a time) |
| `src/kernels.h` | shared kernel helpers: per-format row dot products and step decoders |
| `src/fast.*` | `FastEngine`: decode (graph replay), K/V cache, snapshots |
| `src/prefill.cpp` | the prompt path: XMX GEMMs, grouped XMX experts, XMX flash attention |
| `src/capi.cpp` | C API, built as `build/libinkling.so` |
| `tests/` | `test-forward` (logits vs llama.cpp), `test-dequant`, `gguf-retype` + `format_test.sh` |
| `tools/` | `bench-decode`, `bench-prefill`, `ref-dump` (llama.cpp CPU answer key), `ink-probe` |
| `bench/` | request-level benchmark against an OpenAI-compatible server |

## Development

```
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DINK_AOT_TARGETS=bmg_g21
cmake --build build
```

`INK_AOT_TARGETS` is `bmg_g21` (B570, B580, B50, B60), `bmg_g31` (B70), or both, comma separated. Stop the
server before benchmarks: the GPU is usually the desktop's too, and long runs can make the desktop lag.

```
M=path/to/Inkling-2-Mini-UD-Q2_K_XL.gguf
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/tbb/latest/lib
build/bench-prefill $M 4096 8192 profile     # env INK_CHUNK (prompt chunk), INK_XMX=0
build/bench-decode $M 48 32768 graph 16000   # tokens, context, mode, start position; env INK_KV=4
```

The correctness tests compare against llama.cpp on the CPU, so they need a llama.cpp build with Inkling 2
support (from source). Configure with `-DINK_LLAMA_TESTS=ON -DINK_LLAMA_DIR=path/to/llama.cpp`, then:

```
build/ref-dump $M ref/p1 "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is" 2
build/test-forward $M ref/p1 prefill        # also: ref | fast | graph; expect MATCH, KL ~0.001-0.002
tests/format_test.sh build $M ~/arcangel-format-test   # Q8_0 / Q5_K expert formats
```

`ref-dump` must stay CPU-only (`devices={nullptr}`, `op_offload=false`), so that it never loads a second copy
of the model onto the GPU.

## License

MIT, see [LICENSE](LICENSE). The model and tokenizer are Apache-2.0 and are downloaded, not included; see
[NOTICE](NOTICE).
