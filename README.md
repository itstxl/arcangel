# ArcAngel

**A fast local coding model for Intel Arc GPUs.** ArcAngel is an inference engine written for one model,
[Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), and one GPU family, Intel Xe2: the
Arc B570 and B580, the Arc Pro B50, B60 and B70, and the Arc 140V integrated in Lunar Lake laptops. It serves an
OpenAI-compatible API, so [opencode](https://opencode.ai) and other coding agents can use it directly.

- **Fast.** On an Arc B580: 67 tok/s generation, still 48 tok/s with 90k tokens of context, prompts read at
  650-950 tok/s, and ~100 tok/s when the model is copying code (speculative decoding). That is about twice
  llama.cpp's generation speed on the same card, three times at 30k, and four times its prompt speed.
- **Long context.** 96k tokens on a 12 GB B580; the Pro cards' recipes go up to 1M (300k tested so far, on
  a B60).
- **Two setups per card.** Everything in VRAM for speed, or part of the model in system RAM for a better
  quantization or more context.
- **One-command install.** `./install.sh` detects the card, picks the right model, and sets up the server and
  opencode. Native or in Docker.

## Contents

- [Quick start](#quick-start)
- [Supported GPUs](#supported-gpus)
- [Performance](#performance)
- [Using ArcAngel](#using-arcangel)
- [Docker](#docker)
- [Requirements](#requirements)
- [How it works](#how-it-works)
- [Development](#development)
- [Contributors](#contributors)

## Quick start

```
git clone https://github.com/itstxl/arcangel && cd arcangel
./install.sh
```

The installer asks which card you have (it detects it), which setup you want (full VRAM or offload, see
[below](#supported-gpus)) and, if Docker is available, whether to run natively or in a container. It shows
a plan, and after one "Go ahead?" it:

1. installs the compiler and GPU runtime if they are missing (asks for your password), or builds a Docker image
2. downloads the right model from [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF),
   or reuses a copy already on disk
3. builds the engine for your card (a few minutes)
4. sets up the `arcangel` service and, if you use opencode, adds the model to it
5. starts the server and checks that it answers

Then, in opencode, pick **Inkling 2 Mini (ArcAngel)**. Running the installer again is safe and is also how you
change card or setup. `./uninstall.sh` removes the service, the opencode entry and the settings (the model is
kept unless you add `--models`).

<details><summary>Options for scripted installs</summary>

`./install.sh --help` lists them all. The main ones:

| Option | Effect |
|---|---|
| `--card ID`, `--setup vram\|offload`, `-y` | install without questions (`--list-cards` shows the IDs and recipes) |
| `--docker` / `--native` | choose how it runs |
| `--model FILE` | use a GGUF you already have |
| `--ctx N`, `--kv-bits 4\|8`, `--port N` | override the recipe |
| `--no-service`, `--no-opencode`, `--no-deps` | skip a step |
| `--build-only` | only install the tools and build (no GPU needed) |

</details>

## Supported GPUs

Every card has two recipes ([cards.tsv](cards.tsv)), and the installer asks which one you want:

- **Full VRAM**: the whole model on the GPU. The fastest, but the GPU's memory decides the quantization and
  the context.
- **Offload**: the experts of the 12 layers that run once per token stay in pinned system RAM, and the CPU
  computes them while the GPU does the rest. That leaves room for a better quantization (B570, B580, B50) or
  far more context (B60, B70). Generation is about 1/3 slower, and it needs ~32 GB of system RAM.

| GPU | Memory | Full VRAM: model, context, K/V cache | Offload: model, context, K/V cache | Tested |
|---|---|---|---|---|
| Arc B570 | 10 GB | UD-IQ1_S, 32k, 4-bit | UD-Q4_K_XL, 32k, 4-bit | not yet |
| **Arc B580** | 12 GB | UD-Q2_K_XL, 96k, 4-bit | UD-Q4_K_XL, 96k, 4-bit | **both** |
| Arc Pro B50 | 16 GB | UD-Q2_K_XL, 256k, 8-bit | UD-Q4_K_XL, 256k, 8-bit | not yet |
| **Arc Pro B60** | 24 GB | UD-Q4_K_XL, 256k, 8-bit | UD-Q4_K_XL, 1M, 4-bit | **both** (contributor) |
| Arc Pro B70 | 32 GB | UD-Q4_K_XL, 512k, 8-bit | UD-Q4_K_XL, 1M, 8-bit | not yet (builds) |
| **Arc 140V** (Lunar Lake iGPU) | 32 GB shared | UD-Q4_K_M, 32k, 8-bit | UD-Q2_K_XL, 96k, 4-bit (no offload: memory is shared) | **full VRAM** (contributor) |

The untested recipes were sized from each card's memory. If one does not fit, for example because your
desktop uses more VRAM, re-run the installer with a smaller `--ctx` or with `--kv-bits 4`. If you run one of
them, please report how it went.

<details><summary>Why these quantizations</summary>

- Q8_K_XL is not offered: its experts are the same NVFP4 as Q4_K_XL's in layers 0-19, and it only widens the
  dense layers and layers 20-23 (already Q8_0 in Q4_K_XL) to BF16. So the larger cards spend their memory on
  context instead.
- Q3_K_XL, IQ3_XXS and IQ4_XS use weight formats the engine does not support yet (Q3_K, IQ3_S, IQ4_NL).
- The B570's offload recipe keeps one more looped layer in RAM, to fit in 10 GB.

</details>

## Performance

### Arc B580 (12 GB), full VRAM

UD-Q2_K_XL, 96k context, 4-bit K/V cache. Measured 2026-11-27 with `bench-decode` / `bench-prefill`, two runs
each, within 1% of each other. llama.cpp (Vulkan, with its DSpark draft model) was run on the same card with
the same model.

| Generation | ArcAngel | llama.cpp |
|---|---|---|
| short context | **67 tok/s** | 33-38 tok/s |
| at 4k | **66** | |
| at 12k | **64** | 17-21 (a real opencode session) |
| at 30k | **60** | ~18 |
| at 90k | **48** | does not fit |

| Prompt reading | ArcAngel | llama.cpp |
|---|---|---|
| 4k tokens | **949 tok/s** | ~225 tok/s |
| 16k | **899** | |
| 32k | **850** | |
| 90k | **659** (137 s) | does not fit |

**Speculative decoding** (on by default in the full-VRAM setups) adds to that whenever the model repeats text
from the conversation, as coding agents do when they rewrite a file or quote code in an edit. Measured
through the server, temperature 1:

| | copy-heavy edit | prose |
|---|---|---|
| plain decoding | 64 tok/s | 65 tok/s |
| **copy drafting** (default) | **100 tok/s** | 65 tok/s |
| DSpark draft model (opt-in) | 83 tok/s | 42 tok/s |

### Arc B580, offload

UD-Q4_K_XL, 96k context, 4-bit cache, Ryzen 5 5600X with 32 GB of DDR4.

| | Offload (Q4_K_XL) | Full VRAM (Q2_K_XL), for comparison |
|---|---|---|
| Generation, short context | 44 tok/s | 67 tok/s |
| Generation at 30k | 41 tok/s | 60 tok/s |
| Prompt reading, 4k / 16k | 609 / 590 tok/s | 949 / 899 tok/s |
| Weights on the GPU / in RAM | 8.1 / 8.3 GiB | 8.5 / 0 GiB |

The CPU half reads the offloaded experts at what dual-channel DDR4 delivers (~37 GB/s); faster RAM (DDR5)
speeds it up directly. On replayed real opencode traffic generation is a few tok/s lower than in the
benchmark, because the cache of frequently used experts hits less often.

### Arc Pro B60 (24 GB)

Reported by [@marcellbaan](https://github.com/marcellbaan) (Debian 13, through the Docker image):

| Recipe | Result |
|---|---|
| full VRAM: Q4_K_XL, 256k, 8-bit | 51 tok/s generation, 44 at 30k, 864 tok/s prompt at 4k; long-context recall tests pass |
| offload: Q4_K_XL, 1M, 4-bit | 35 tok/s generation; a fact hidden in a 300k-token prompt found |

### Arc 140V (Lunar Lake)

Reported by [@alpha2beta](https://github.com/alpha2beta) (Core Ultra 7 258V, 32 GB LPDDR5X-8533), UD-Q4_K_M,
32k context:

| | ArcAngel | llama.cpp (Vulkan + DSpark) |
|---|---|---|
| Generation, short context / at 16k | **21 / 18.5 tok/s** | ~12 tok/s |
| Prompt reading, 2k | **226 tok/s** | 46-49 tok/s |
| Code edit with DSpark (K=3) | **23.4 tok/s** (1.21x) | |
| Refactor with copy drafting, 128k context | **29.6 tok/s** (98% of drafts accepted) | |
| HumanEval (164 tasks) | 91.5% base / 88.4% plus | |

On the slower integrated GPU the DSpark draft model pays off; on discrete cards copy drafting does.
Details: [docs/arcangel_spec_decode.md](docs/arcangel_spec_decode.md).

### Accuracy

Every change is checked against llama.cpp's CPU output for the same model: the engine predicts the same top
token on every step tested, with a KL divergence of 0.001-0.002 using the 8-bit K/V cache and 0.006-0.016
using the 4-bit one. With the 4-bit cache it finds facts hidden at 10%, 55% and 95% of a 90k-token prompt,
and 4 of 4 facts in a 60k multi-fact test. Speculative decoding does not change the output distribution.

How the speed was reached, including everything that was tried and did not work: [PERF.md](PERF.md).

## Using ArcAngel

```
systemctl --user start arcangel        # or, in the foreground: server/run.sh
systemctl --user stop arcangel         # frees the GPU memory
journalctl --user -u arcangel -f       # logs
```

The API is at `http://127.0.0.1:8082/v1` with the model name `inkling-2-mini`. It has no authentication, so it
listens on localhost only.

### Settings

`~/.config/arcangel/arcangel.env` (written by the installer; restart the service after editing it):

| Setting | What it does |
|---|---|
| `ARCANGEL_MODEL`, `ARCANGEL_CTX`, `ARCANGEL_KV_BITS` | the model file, context length and K/V cache precision (4 or 8) |
| `ARCANGEL_PORT` | API port (default 8082) |
| `ARCANGEL_REASONING` | default reasoning effort: `low` (the model's own default, `high`, is slow); requests can pass `reasoning_effort` |
| `ARCANGEL_OFFLOAD` | offload setup: the layers whose experts stay in RAM |
| `ARCANGEL_SPEC`, `ARCANGEL_SPEC_K` | speculative decoding: `copy` (default for full VRAM) or the path of a DSpark GGUF; guesses per step (default 3) |
| `ARCANGEL_EXPERT_CACHE_MB` | offload setup: VRAM for the cache of frequently used experts (default: what is free) |
| `ARCANGEL_STASH_RAM_GB` | RAM for stashed conversations (default: a quarter of RAM) |
| `ARCANGEL_DEVICE` | which GPU to use on multi-GPU machines (default: the one with the most memory) |

### Good to know

- **Conversations are cached.** The server keeps the current conversation on the GPU and continues it without
  re-reading the prompt, and stashes up to two more in system RAM, so an agent's side requests (titles,
  summaries) do not cost a full re-read.
- **It never spills.** The engine refuses to start rather than overflow into system RAM unexpectedly. If it
  reports too little free VRAM, close other GPU programs or lower the context. A desktop on the same card uses
  0.8-1.2 GB on a B580, and that amount moves around.
- **DSpark** is an optional draft model:
  `ARCANGEL_SPEC=/path/to/dspark-Inkling-2-Mini-Q8_0.gguf` (the current file from Unsloth's repository; the
  `_Old` one does not work). It helps on Lunar Lake and is slower than copy drafting on discrete cards.
  Speculative decoding is switched off automatically with offloading.

The installer adds this provider to `~/.config/opencode/opencode.json` (a backup is kept):

```json
"arcangel": {
  "npm": "@ai-sdk/openai-compatible",
  "name": "ArcAngel",
  "options": { "baseURL": "http://127.0.0.1:8082/v1" },
  "models": {
    "inkling-2-mini": {
      "name": "Inkling 2 Mini (ArcAngel, Q2_K_XL, 96k)",
      "tool_call": true, "reasoning": true,
      "limit": { "context": 98304, "output": 8192 }
    }
  }
}
```

## Docker

The container route needs nothing on the host but Docker (with BuildKit and the Compose plugin), an x86-64
Linux kernel 6.12+ with the `xe` driver and GPU firmware, and an Intel Xe2 GPU. The image includes the GPU
runtime and the checksum-verified tokenizer and chat template; the model is mounted from your disk.

**With the installer:** `./install.sh --docker` (or answer "Docker" when it asks) downloads the model, creates
`.env` with `make .env` and fills in your card's recipe, builds the image, and sets up the `arcangel` service
to run `docker compose up`. An existing `.env` is backed up and only its recipe keys (`MODELS_DIR`, `MODEL`,
`N_CTX`, `KV_BITS`, `INK_OFFLOAD`, `INK_AOT_TARGETS`) are changed.

**By hand:**

1. Create `.env` with your user ID, group ID and render device's group ID:

   ```
   make .env
   ```

   On a multi-GPU host, find your Arc card's render device in `/dev/dri/by-path/` and pass it:
   `make .env RENDER_NODE=/dev/dri/renderD129`. If `.env` already exists, `make` leaves it alone: edit it
   directly. (A `compose.env` from an older checkout can be renamed to `.env`.)

2. Set your card's recipe from [cards.tsv](cards.tsv) in `.env`. The defaults are the B580 full-VRAM recipe
   with `~/models/Inkling-2-Mini-UD-Q2_K_XL.gguf` and `/dev/dri/renderD128`.
   - `MODEL`: the GGUF's filename, relative to `MODELS_DIR` (download it from
     [the model repository](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) and use a copy you trust)
   - `N_CTX`, `KV_BITS`: the recipe's context and cache precision
   - `INK_OFFLOAD`: for the offload recipes, the layer ranges (the template shows the B60's)
   - `INK_AOT_TARGETS`: `bmg_g21` by default (B570, B580, Pro B50, Pro B60); `bmg_g31` for a Pro B70

3. Build and start it:

   ```
   docker compose run --rm --build --service-ports arcangel
   ```

   Compose loads `.env`, builds the image and starts the server in the foreground, with your models directory
   mounted read-only at `/models`. Once the model has loaded, `curl -fsS http://127.0.0.1:8082/health`
   answers. Keep `HOST` unset (or `127.0.0.1`): the API does not authenticate requests. An exported shell
   variable takes precedence over `.env`.

**Diagnostics.** The optional `diagnostics` image adds `ink-probe`, `bench-decode` and `bench-prefill`, with
the same Intel runtime as the server:

```
docker build --target diagnostics -t arcangel:diagnostics .
```

Run it with `--entrypoint /opt/arcangel/build/ink-probe`, your render node passed through and the model mounted
as for the server, and the container path of the GGUF after the image name. For a Pro B70, add
`--build-arg INK_AOT_TARGETS=bmg_g31` to the build.

## Requirements

For the native route the installer checks and installs these (with sudo, after asking):

- Linux 6.12+ with the `xe` (or `i915`) driver, and an Intel Xe2 GPU
- Intel oneAPI DPC++/C++ compiler 2025.1 or newer (tested: 2026.0), Level Zero, and the Intel compute runtime
  with `ocloc`
- CMake 3.24+, Python 3.10+, and 10-20 GB of disk for the model

| Distribution | Native install |
|---|---|
| Arch, CachyOS, Omarchy, EndeavourOS | tested end to end on a B580 |
| Ubuntu 24.04, 25.10 | packages and build tested (container, no GPU); adds Intel's oneAPI repository, and on 24.04 Intel's graphics PPA |
| Fedora 43 | packages and build tested (container, no GPU); adds Intel's oneAPI repository |
| Debian 13 | no Intel GPU runtime packages: use the Docker route (the installer recommends it there) |
| others | install the list above yourself, or use Docker |

Before downloading anything, the installer checks the compiler version, the kernel, the GPU driver and that
Level Zero can see the GPU, so a missing piece shows up in seconds rather than after a 10 GB download.

## How it works

Inkling 2 Mini is a mixture-of-experts model (128 experts, 8 used per token) with looped layers: its 24
layers run as 48 layer visits. ArcAngel is SYCL C++ on Level Zero:

- **Kernels per weight format** for the model's quantization formats (IQ1_S, IQ2_*, Q2_0, NVFP4, MXFP4,
  Q4_K-Q8_0, ...), bit-exact with ggml. Weights are repacked losslessly at load into layouts that suit the GPU.
- **Prompt reading** on the XMX matrix units: grouped expert GEMMs and flash attention.
- **Generation** as a recorded SYCL graph, with int8 (DP4A) experts and attention.
- **K/V cache** in int8 or 4-bit, which is what makes 96k fit on 12 GB.
- **Expert offloading**: pinned host memory, AVX2 CPU kernels running alongside the GPU, a VRAM cache of the
  most-used experts, and the copy engine streaming offloaded experts in for prompts.
- **Speculative decoding**: batched verification of 2-8 drafted tokens in recorded graphs, with expert kernels
  that read each shared expert once; drafts from the conversation (copy drafting) or the DSpark draft model.

[SPEC.md](SPEC.md) walks through the forward pass; [PERF.md](PERF.md) records what made it fast and what did
not; [docs/](docs) holds the speculative-decoding design notes.

<details><summary>Repository layout</summary>

| Path | What |
|---|---|
| `install.sh`, `uninstall.sh`, `cards.tsv` | installer and the per-card recipes |
| `Dockerfile`, `compose.yaml`, `Makefile`, `.env.example` | the Docker route |
| `server/` | OpenAI-compatible server (`inkling_server.py`, `run.sh`) |
| `src/gguf_file.*`, `src/device.*` | GGUF reader, GPU selection |
| `src/quant.h`, `src/quant_tables.h` | the weight formats (tables generated by `tools/gen_tables.py`) |
| `src/model.*` | weights, the 48-visit schedule, repacking, offloading, memory checks |
| `src/engine.*` | naive reference forward pass (f32, one token at a time) |
| `src/kernels.h` | shared kernel helpers: per-format dot products and decoders |
| `src/fast.*` | `FastEngine`: generation (graph replay), K/V cache, snapshots, expert cache |
| `src/prefill.cpp` | prompt reading: XMX GEMMs, grouped experts, flash attention |
| `src/cpu_experts.*` | CPU expert kernels for offloading |
| `src/verify.cpp`, `src/dspark.*` | batched verification, the DSpark draft engine |
| `src/capi.cpp` | C API, built as `build/libinkling.so` |
| `tests/`, `tools/`, `bench/` | correctness tests, benchmarks, request-level and long-context tests |
| `docs/` | design notes |

</details>

## Development

```
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release -DINK_AOT_TARGETS=bmg_g21
cmake --build build
```

`INK_AOT_TARGETS` is `bmg_g21` (B570, B580, Pro B50, Pro B60), `bmg_g31` (Pro B70), `lnl_m` (Arc 140V), or
a comma-separated list. Stop the server before benchmarking: the GPU is usually the desktop's too.

```
M=path/to/Inkling-2-Mini-UD-Q2_K_XL.gguf
export LD_LIBRARY_PATH=/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/tbb/latest/lib
build/bench-prefill $M 4096 8192 profile     # prompt reading; env INK_CHUNK, INK_XMX=0
build/bench-decode $M 48 32768 graph 16000   # tokens, context, mode, start position; env INK_KV=4, INK_OFFLOAD
INK_KV=4 build/test-verify $M 2000 8192      # batched verification vs one-at-a-time, and its timings
build/test-dspark $M path/to/dspark.gguf     # DSpark draft engine checks
```

The correctness tests compare with llama.cpp on the CPU, so they need a llama.cpp build with Inkling 2
support. Configure with `-DINK_LLAMA_TESTS=ON -DINK_LLAMA_DIR=path/to/llama.cpp`, then:

```
build/ref-dump $M ref/p1 "The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is" 2
build/test-forward $M ref/p1 prefill                   # also ref | fast | graph; expect MATCH, KL ~0.001-0.002
tests/format_test.sh build $M ~/arcangel-format-test   # Q8_0 / Q5_K expert formats
```

`ref-dump` must stay CPU-only (`devices={nullptr}`, `op_offload=false`), so it never loads a second copy of
the model onto the GPU. Server tests: `PYTHONPATH=server python -m pytest tests/test_spec_server.py`.

## Contributors

- [@itstxl](https://github.com/itstxl): the engine, installer and recipes
- [@vvoland](https://github.com/vvoland): Docker and Compose packaging, the template sandbox and checksums, richer model info
- [@alpha2beta](https://github.com/alpha2beta): Lunar Lake support, NVFP4 and Q5_K kernel speed-ups, GPU argmax and the incremental output parser, speculative decoding (batched verification, copy drafting, DSpark)
- [@marcellbaan](https://github.com/marcellbaan): the first Docker image and the Arc Pro B60 measurements

## License

MIT, see [LICENSE](LICENSE). The model and tokenizer are Apache-2.0 and are downloaded, not included; see
[NOTICE](NOTICE).
