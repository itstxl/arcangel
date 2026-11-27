# ArcAngel

A fast local coding model for Intel Arc Battlemage cards. ArcAngel is an inference engine written for one
model, [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), and one GPU family, Intel Xe2
(Arc B570, B580, Pro B50, Pro B60, Pro B70). It serves an OpenAI-compatible API, so opencode and other
coding agents can use it.

- **Everything stays in VRAM.** No offload to system RAM or disk: the engine refuses to start rather than
  spill.
- **Long context.** 96k tokens on a 12 GB B580, up to 512k on a 32 GB B70.
- **Fast.** On a B580: 65 tok/s generation at short context, 47 tok/s at 90k, prompts read at 600-925 tok/s.
  That is about twice llama.cpp's generation speed at short context, three times at 30k, and four times its
  prompt speed on the same card.

## Tested speeds

Arc B580 (12 GB), UD-Q2_K_XL, 96k context, 4-bit K/V cache, 1024-token prompt chunks (measured 2026-11-25,
`bench-decode` / `bench-prefill`, two runs each within 1%). For comparison,
llama.cpp (Vulkan) was run on the same card with the same model, including its DSpark draft model for
speculative decoding.

| Generation (tokens/s) | ArcAngel | llama.cpp |
|---|---|---|
| short context | **65** | 33-38 |
| at 4k tokens | **64** | |
| at 12k (llama.cpp: a real opencode session) | **62** | 17-21 |
| at 30k | **57** | ~18 |
| at 90k | **47** | (does not fit) |

| Prompt reading (tokens/s) | ArcAngel | llama.cpp |
|---|---|---|
| 4k-token prompt | **925** | ~225 |
| 16k | **870** | |
| 32k | **815** | |
| 90k | **597** (151 s) | (does not fit) |

Accuracy: against llama.cpp's CPU output the engine matches the top token on every step tested, with KL
divergence 0.001-0.002 using an 8-bit cache and 0.006-0.016 using the 4-bit cache. With the 4-bit cache, it
found hidden facts placed at 10%, 55% and 95% of a 90k-token prompt, and 4 of 4 facts in a 60k
multi-fact test. How the speed was reached, step by step: [PERF.md](PERF.md).

## Install

```
git clone https://github.com/itstxl/arcangel && cd arcangel
./install.sh
```

That is all. The installer asks two questions, which card (it detects yours) and which setup (full VRAM or
offload, see below), shows a plan of what it will do, and after one "Go ahead?" does the rest:

1. installs the compiler and GPU runtime if they are missing (asks for your password)
2. downloads the model from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF),
   or reuses a copy already on disk
3. builds the engine for your card (a few minutes)
4. sets up the `arcangel` service and, if you use opencode, adds the model to it
5. starts the server and checks that it answers

Running it again is safe (it skips what is done) and is also how you change card or setup. To remove
everything: `./uninstall.sh` (the downloaded model is kept unless you add `--models`).

<details><summary>Advanced: options for scripted installs</summary>

`./install.sh --help` lists them: `--card ID --setup vram|offload -y` installs without questions; `--model FILE`
uses a GGUF you already have; `--ctx N`, `--kv-bits 4|8` and `--port N` override the recipe;
`--no-service`, `--no-opencode`, `--no-deps` skip steps; `--build-only` only builds; `--list-cards` prints every
card's two recipes.
</details>

### Two setups per card

Every card has two recipes, and the installer asks which one you want (`--setup vram` or `--setup offload`):

- **Full VRAM**: the whole model on the card. Fastest, but the card's memory decides the quant and the context.
- **Offload**: the routed experts of the 12 layers that run once per token (layers 0-5 and 18-23; the 12 in the
  loop run three times) stay in pinned system RAM, and the CPU computes them during generation. The card then
  has room for a better quant (B570, B580, B50) or far more context (B60, B70). Generation is about half as
  fast, and it needs ~32 GB of system RAM (8.3 GB of experts locked in memory, plus headroom).

| Card | Full VRAM: model, context, cache | Offload: model, context, cache | Status |
|---|---|---|---|
| Arc B570, 10 GB | UD-IQ1_S, 32k, 4-bit | UD-Q4_K_XL, 32k, 4-bit (one more loop layer in RAM) | both untested |
| Arc B580, 12 GB | UD-Q2_K_XL, 96k, 4-bit | UD-Q4_K_XL, 96k, 4-bit | **both tested** |
| Arc Pro B50, 16 GB | UD-Q2_K_XL, 256k, 8-bit | UD-Q4_K_XL, 256k, 8-bit | both untested |
| Arc Pro B60, 24 GB | UD-Q4_K_XL, 256k, 8-bit | UD-Q4_K_XL, 1M, 4-bit | **both tested** (by a contributor) |
| Arc Pro B70, 32 GB | UD-Q4_K_XL, 512k, 8-bit | UD-Q4_K_XL, 1M, 8-bit | both untested |

Measured on the B580 (Ryzen 5 5600X, 32 GB DDR4), 2026-11-25. Offload generation depends on how well the
hot-expert cache fits the text: 36-37 tok/s replaying real opencode requests, up to 40 in the benchmark.


| | Full VRAM (Q2_K_XL) | Offload (Q4_K_XL) |
|---|---|---|
| Generation, short context | 65 tok/s | 36-40 tok/s |
| Generation at 30k | 57 tok/s | 35-37 tok/s |
| Prompt reading, 4k / 16k | 925 / 870 tok/s | 585 / 570 tok/s |
| Weights in VRAM / in RAM | 8.5 / 0 GiB | 8.1 / 8.3 GiB |

How the offload setup works: per token the GPU does the dense layers and the looped layers' experts (~15 ms
on a B580), the CPU the offloaded experts, reading them from RAM at ~37 GB/s (what dual-channel DDR4
delivers; DDR5 speeds this half up directly). The spare VRAM holds a cache of each offloaded layer's most-used
experts (13 of 128 per layer on a B580 at 96k; `ARCANGEL_EXPERT_CACHE_MB` to set it): the GPU computes the
cached ones at the same time as the CPU does the rest, which took CPU time per token from ~14 to ~10 ms on
real opencode traffic. The cache follows what the model actually routes to, refreshed every 256 tokens. When reading a prompt,
the same VRAM is used to stream each offloaded layer's experts in with the copy engine (14 GB/s, against
~8.6 GB/s for the GPU reading RAM directly), and the cache is refilled afterwards. Switch setups any time with `./install.sh --card <card> --setup vram|offload`.

The profiles live in [cards.tsv](cards.tsv). The B580's were run here; the Arc Pro B60's were run by a
contributor (Debian 13, through the Docker image, Nov 2026):

| Arc Pro B60 | Result |
|---|---|
| full VRAM: Q4_K_XL, 256k, 8-bit cache | 51.1 tok/s generation (short), 43.7 at 30k, 864 tok/s prompt at 4k; loads in 7 s; needles at 10/55/95% of 90k 3/3, 60k multi-fact 4/4, tool calls (streaming and not) |
| offload: Q4_K_XL, 1M, 4-bit cache | 34.8 tok/s generation; loads in 12 s; a needle in a 300k-token prompt found |

The others (B570, B50, B70) were sized from each card's memory: weights, plus the K/V cache, plus prompt
buffers, plus a 0.5 GB margin, plus about 1 GB for a desktop on the same card. What has been checked for them:

- the weight formats of Q4_K_XL and Q4_K_M (Q8_0 and Q5_K experts) match llama.cpp on a B580
  (`tests/format_test.sh`), and Q4_K_XL runs end to end on a B580 and a B60;
- the engine builds for the B70's chip (BMG-G31), but has not run on one;
- contexts up to 300k have run (on the B60); the 512k and 1M profiles have not been filled to the end.

If you run one of the untested cards, please report how it went. If a profile does not fit (for example,
because the desktop uses more VRAM), re-run the installer with a smaller `--ctx` or with `--kv-bits 4`.

Why these models:
- Q8_K_XL is not used: its experts are the same NVFP4 as Q4_K_XL's in layers 0-19; it only widens the dense
  layers and layers 20-23 (already Q8_0 in Q4_K_XL) to BF16. So the B60 and B70 spend memory on context.
- Q3_K_XL, IQ3_XXS and IQ4_XS use weight formats the engine does not support yet (Q3_K, IQ3_S, IQ4_NL, Q5_0).

## Run with Docker

Use Docker with BuildKit and the Docker Compose plugin on an x86-64 Linux host with an Intel Arc Battlemage GPU, the `xe` driver, and GPU firmware (Linux 6.12 or later).
The image includes the userspace GPU runtime and checksum-verified tokenizer and chat template, so you don't need to run `install.sh` on the host.
Download the GGUF for your card from [the model repository](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) separately and use a copy you trust.

Create `.env` from the template and fill in your user ID, group ID, and render device's group ID automatically:

```fish
make .env
```

On a multi-GPU host, check `/dev/dri/by-path/` to identify your Arc card's render device and pass it to the setup command, for example `make .env RENDER_NODE=/dev/dri/renderD129`.
If you already have a `compose.env` from an older checkout, rename it to `.env` to keep your settings.
If `.env` already exists, edit it directly; the setup command won't overwrite it.

The default is the B580 full-VRAM profile, with `~/models/Inkling-2-Mini-UD-Q2_K_XL.gguf` and `/dev/dri/renderD128`.
Set `MODEL` to a filename relative to `MODELS_DIR`, and set `N_CTX` and `KV_BITS` to match your profile in [cards.tsv](cards.tsv).
The image defaults to `bmg_g21` (B570, B580, Pro B50 and Pro B60); for a Pro B70, set `INK_AOT_TARGETS=bmg_g31` in `.env`.
For offload, set `INK_OFFLOAD` to the layer ranges for your GGUF and profile; the template shows the B60 offload settings.

```fish
docker compose run --rm --build --service-ports arcangel
```

Compose loads `.env` automatically, builds the image, and starts the server in the foreground with your models directory mounted read-only at `/models`.
Once the model is loaded, check `curl -fsS http://127.0.0.1:8082/health`.
If startup reports insufficient VRAM, lower `N_CTX` or stop other GPU workloads.
Keep `HOST` unset or set to `127.0.0.1` for local use: the API does not authenticate requests.
An exported shell variable takes precedence over the value in `.env`.

### Probe and benchmark in Docker

Build the optional `diagnostics` image to run `ink-probe`, `bench-decode`, and `bench-prefill` with the same Intel runtime as the server.
The default image does not build or ship these tools.

```fish
docker build --target diagnostics -t arcangel:diagnostics .
```

When running this image, use `--entrypoint /opt/arcangel/build/ink-probe` to probe the GPU instead of starting the server.
Pass through your render node and mount the model as for the server, then pass the container path to the GGUF after the image name.
For a Pro B70, add `--build-arg INK_AOT_TARGETS=bmg_g31` to the build command.

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

The installer installs these itself (with sudo, after asking):

| Distribution | Status |
|---|---|
| Arch, CachyOS, Omarchy, EndeavourOS | tested end to end on a B580 |
| Ubuntu 24.04, 25.10 | packages and build tested (in a container, no GPU); adds Intel's oneAPI apt repository, and on 24.04 Intel's graphics PPA |
| Fedora 43 | packages and build tested (in a container, no GPU); adds Intel's oneAPI repository |
| Debian 13 | no Intel GPU compute runtime in Debian: install it from [intel/compute-runtime releases](https://github.com/intel/compute-runtime/releases) first |
| others | install the list above yourself, then run the installer |

Before downloading the model it checks the oneAPI version (2025.1+), the kernel (6.12+ for Battlemage), that
the GPU has the `xe` or `i915` driver and that Level Zero can see it, so a missing driver shows up in seconds,
not after a 10 GB download. `./install.sh --build-only` installs the tools and builds the engine without a GPU.

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
