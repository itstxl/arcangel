#!/usr/bin/env bash
# ArcAngel installer: Inkling 2 Mini on an Intel Arc Battlemage card, as a local OpenAI-compatible server.
#
#   ./install.sh                  interactive: detects the card, asks before changing anything
#   ./install.sh --card b580 -y   non-interactive
#   ./install.sh --list-cards
#
# What it does (each step skips itself when already done, so re-running is safe):
#   1. picks the card profile (cards.tsv): which quant to download, context length, K/V cache bits
#   2. checks the build tools and GPU runtime (on Arch-based systems it can install them with pacman)
#   3. downloads the model from Hugging Face (unsloth/Inkling-2-Mini-GGUF) and the tokenizer and chat
#      template (thinkingmachines/Inkling-2-Mini)
#   4. builds the engine for the card's chip
#   5. writes ~/.config/arcangel/arcangel.env and a systemd user service, `arcangel`
#   6. optionally adds the model to opencode (~/.config/opencode/opencode.json, backed up first)
set -euo pipefail

REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DATA=${XDG_DATA_HOME:-$HOME/.local/share}/arcangel
CONF_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/arcangel
UNIT_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user
OPENCODE_JSON=${XDG_CONFIG_HOME:-$HOME/.config}/opencode/opencode.json
ONEAPI_ROOT=${ONEAPI_ROOT:-/opt/intel/oneapi}
GGUF_REPO=unsloth/Inkling-2-Mini-GGUF
BASE_REPO=thinkingmachines/Inkling-2-Mini
# sha256 of the assets the server was validated with (both Apache-2.0, from $BASE_REPO)
TOKENIZER_SHA=7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6
TEMPLATE_SHA=53a9035f3255945704b5857657cfb00c1165090c0641c4f782a0cbf268cc16dc

card="" model="" port=8082 ctx="" kv="" yes=0 deps=1 service=1 opencode=1 start=1 models_dir="$DATA/models"

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
  cat <<EOF

Options:
  --card ID          card profile (see --list-cards); default: detected
  --model FILE       use a GGUF you already have instead of downloading
  --models-dir DIR   where to download models (default: $DATA/models)
  --ctx N            context length (default: the card's profile)
  --kv-bits 4|8      K/V cache precision (default: the card's profile)
  --port N           server port (default: 8082)
  --no-deps          do not install packages, only check
  --no-service       do not write the systemd service
  --no-start         write the service but do not start it
  --no-opencode      do not touch the opencode config
  -y, --yes          accept the defaults, ask nothing
  --list-cards       show the card profiles and exit
EOF
}

while [ $# -gt 0 ]; do
  case $1 in
    --card) card=$2; shift ;;
    --model) model=$2; shift ;;
    --models-dir) models_dir=$2; shift ;;
    --ctx) ctx=$2; shift ;;
    --kv-bits) kv=$2; shift ;;
    --port) port=$2; shift ;;
    --no-deps) deps=0 ;;
    --no-service) service=0 ;;
    --no-start) start=0 ;;
    --no-opencode) opencode=0 ;;
    -y|--yes) yes=1 ;;
    --list-cards) list=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done

# ------------------------------------------------------------------------------------------------------
# helpers

if [ -t 1 ]; then B=$'\e[1m' G=$'\e[32m' Y=$'\e[33m' R=$'\e[31m' N=$'\e[0m'; else B="" G="" Y="" R="" N=""; fi
step() { echo; echo "${B}==> $*${N}"; }
ok() { echo "  ${G}ok${N} $*"; }
warn() { echo "  ${Y}warning:${N} $*"; }
die() { echo "${R}error:${N} $*" >&2; exit 1; }
ask() {  # ask "question" default(y/n) -> 0 for yes
  local a d=$2
  if [ $yes = 1 ]; then [ "$d" = y ]; return; fi
  read -r -p "  $1 [$( [ "$d" = y ] && echo Y/n || echo y/N )] " a
  a=${a:-$d}
  [[ $a =~ ^[Yy] ]]
}

profiles() { grep -v '^#' "$REPO/cards.tsv" | grep -v '^[[:space:]]*$'; }
field() {  # field ID COLUMN
  profiles | awk -F'\t' -v id="$1" -v c="$2" '$1 == id { print $c }'
}

if [ "${list:-0}" = 1 ]; then
  printf '%-6s %-22s %-8s %-9s %-8s %-6s %s\n' card name chip quant context kv status
  profiles | awk -F'\t' '{ printf "%-6s %-22s %-8s %-9s %-8s %-6s %s\n", $1, $9, $2, $4, $6, $5"-bit", $7 }'
  exit 0
fi

[ "$(id -u)" = 0 ] && die "run this as your user, not root (it asks for sudo when it needs it)"

# ------------------------------------------------------------------------------------------------------
step "1/6 Card"

gpus=$(lspci -nn 2>/dev/null | grep -Ei 'vga|display|3d' | grep -i '8086:' || true)
detected=""
if [ -n "$gpus" ]; then
  while IFS=$'\t' read -r id _ _ _ _ _ _ match _; do
    if echo "$gpus" | grep -Eqi "$match"; then detected=$id; break; fi
  done < <(profiles)
fi
[ -n "$gpus" ] && echo "$gpus" | sed 's/^/  found: /'

if [ -z "$card" ]; then
  if [ $yes = 1 ]; then
    [ -n "$detected" ] || die "could not detect the card: pass --card (see --list-cards)"
    card=$detected
  else
    echo
    i=0; ids=()
    while IFS=$'\t' read -r id chip vram quant kvb n status _ name; do
      i=$((i + 1)); ids+=("$id")
      mark=""; [ "$id" = "$detected" ] && mark=" ${G}<- detected${N}"
      note=""; [ "$status" = untested ] && note=" (untested)"
      printf '  %d) %-22s %-8s %3sk context, %s-bit KV%s%s\n' $i "$name" "$quant" $((n / 1024)) "$kvb" "$note" "$mark"
    done < <(profiles)
    def=1
    for j in "${!ids[@]}"; do [ "${ids[$j]}" = "$detected" ] && def=$((j + 1)); done
    read -r -p "  Which card? [$def] " pick
    pick=${pick:-$def}
    [[ $pick =~ ^[0-9]+$ ]] && [ "$pick" -ge 1 ] && [ "$pick" -le ${#ids[@]} ] || die "no such choice: $pick"
    card=${ids[$((pick - 1))]}
  fi
fi
[ -n "$(field "$card" 1)" ] || die "unknown card '$card' (see --list-cards)"
chip=$(field "$card" 2); quant=$(field "$card" 4); status=$(field "$card" 7); cname=$(field "$card" 9)
ctx=${ctx:-$(field "$card" 6)}; kv=${kv:-$(field "$card" 5)}
[[ $kv =~ ^(4|8)$ ]] || die "--kv-bits must be 4 or 8"
[[ $ctx =~ ^[0-9]+$ ]] && [ "$ctx" -ge 4096 ] || die "--ctx must be a number >= 4096"
gguf_name=Inkling-2-Mini-UD-$quant.gguf
ok "$cname: $gguf_name, $((ctx / 1024))k context, $kv-bit K/V cache, chip $chip"
if [ "$status" = untested ]; then
  warn "this profile has not been run on a real $cname yet; the numbers come from its memory size."
  warn "If the server refuses to start for lack of VRAM, re-run with a smaller --ctx or --kv-bits 4."
fi
if [ -n "$detected" ] && [ "$detected" != "$card" ]; then
  warn "the detected card is '$detected', not '$card'"
fi

# ------------------------------------------------------------------------------------------------------
step "2/6 Build tools and GPU runtime"

missing=()
have() { command -v "$1" >/dev/null 2>&1; }
[ -x "$ONEAPI_ROOT/compiler/latest/bin/icpx" ] || have icpx || missing+=(icpx)
have ocloc || missing+=(ocloc)
ldconfig -p 2>/dev/null | grep -q 'libze_loader.so' || [ -e /usr/lib/libze_loader.so.1 ] || missing+=(level-zero)
ldconfig -p 2>/dev/null | grep -q 'libze_intel_gpu.so' || ls /usr/lib*/libze_intel_gpu.so* >/dev/null 2>&1 || missing+=(gpu-driver)
have cmake || missing+=(cmake)
have python3 || missing+=(python3)
have curl || missing+=(curl)
python3 -c 'import venv, ensurepip' 2>/dev/null || missing+=(python3-venv)

if [ ${#missing[@]} -gt 0 ]; then
  warn "missing: ${missing[*]}"
  if have pacman; then
    pkgs=(intel-oneapi-dpcpp-cpp intel-compute-runtime level-zero-loader cmake ninja python curl)
    echo "  On Arch-based systems these come from: sudo pacman -S --needed ${pkgs[*]}"
    if [ $deps = 1 ] && ask "Install them now with sudo pacman?" y; then
      sudo pacman -S --needed "${pkgs[@]}"
    else
      die "install the packages above and run install.sh again"
    fi
  else
    cat <<EOF
  This installer installs packages only on Arch-based systems. Elsewhere, install:
    - Intel oneAPI DPC++/C++ compiler 2025.1 or newer (icpx), from Intel's oneAPI repository
      https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler-download.html
    - the Intel GPU compute runtime with Level Zero and ocloc (Ubuntu: libze1 libze-intel-gpu1 intel-ocloc,
      Fedora: intel-compute-runtime oneapi-level-zero intel-ocloc)
    - cmake, python3 with venv, curl
  then run install.sh again. (Only Arch has been tested.)
EOF
    exit 1
  fi
fi
[ -f "$ONEAPI_ROOT/setvars.sh" ] || die "no $ONEAPI_ROOT/setvars.sh: set ONEAPI_ROOT to your oneAPI install"
ok "compiler, GPU runtime, cmake, python"
rnode=$(ls /dev/dri/renderD* 2>/dev/null | head -1 || true)
if [ -z "$rnode" ]; then
  warn "no /dev/dri/renderD* device: is the GPU driver (xe or i915) loaded?"
elif ! [ -r "$rnode" ] || ! [ -w "$rnode" ]; then
  warn "no access to $rnode: add yourself to the 'render' group (sudo usermod -aG render \$USER) and log in again"
fi

# ------------------------------------------------------------------------------------------------------
step "3/6 Python environment and model"

venv=$REPO/server/.venv
if [ ! -x "$venv/bin/python" ]; then
  python3 -m venv "$venv"
fi
"$venv/bin/pip" install -q --disable-pip-version-check -r "$REPO/server/requirements.txt"
ok "server environment ($venv)"

assets=$DATA/assets
mkdir -p "$assets"
fetch_asset() {  # NAME SHA
  local f=$assets/$1
  if [ -f "$f" ] && echo "$2  $f" | sha256sum -c --status; then return; fi
  curl -fsSL --retry 3 -o "$f.part" "https://huggingface.co/$BASE_REPO/resolve/main/$1"
  if ! echo "$2  $f.part" | sha256sum -c --status; then
    warn "$1 differs from the version ArcAngel was tested with (the upstream file changed); using it anyway"
  fi
  mv "$f.part" "$f"
}
fetch_asset tokenizer.json $TOKENIZER_SHA
fetch_asset chat_template.jinja $TEMPLATE_SHA
ok "tokenizer and chat template ($assets)"

if [ -z "$model" ]; then
  # reuse a copy that is already on disk (llama.cpp's model folder, the Hugging Face cache, models_dir)
  for d in "$models_dir" "$HOME/.local/share/llama.cpp/models" "$HOME/models" "$HOME/.cache/llama.cpp"; do
    if [ -f "$d/$gguf_name" ]; then model=$d/$gguf_name; break; fi
  done
  hfc=$(ls -d "${HF_HOME:-$HOME/.cache/huggingface}"/hub/models--unsloth--Inkling-2-Mini-GGUF/snapshots/*/"$gguf_name" 2>/dev/null | head -1 || true)
  [ -z "$model" ] && [ -n "$hfc" ] && model=$hfc
fi
size=$(curl -fsSL "https://huggingface.co/api/models/$GGUF_REPO/tree/main" 2>/dev/null |
       python3 -c "import json,sys; print(next((f.get('size',0) for f in json.load(sys.stdin) if f['path']=='$gguf_name'),0))" 2>/dev/null || echo 0)
if [ -n "$model" ]; then
  [ -f "$model" ] || die "no such file: $model"
  if [ "$size" -gt 0 ] && [ "$(stat -c %s "$model")" != "$size" ]; then
    warn "$model is $(stat -c %s "$model") bytes, Hugging Face has $size: incomplete or a different version?"
  fi
  ok "using $model"
else
  mkdir -p "$models_dir"
  free_b=$(df -B1 --output=avail "$models_dir" | tail -1)
  [ "$size" -gt 0 ] && [ "$free_b" -lt $((size + (1 << 30))) ] &&
    die "$gguf_name needs $((size >> 30)) GiB, $models_dir has $((free_b >> 30)) GiB free (use --models-dir)"
  echo "  downloading $gguf_name ($((size >> 20)) MiB) to $models_dir (resumes if interrupted)"
  ask "Download it now?" y || die "nothing downloaded; pass --model FILE to use your own copy"
  "$venv/bin/hf" download "$GGUF_REPO" "$gguf_name" --local-dir "$models_dir" >/dev/null
  model=$models_dir/$gguf_name
  [ "$size" -gt 0 ] && [ "$(stat -c %s "$model")" != "$size" ] && die "download incomplete: run install.sh again to resume"
  ok "$model"
fi

# ------------------------------------------------------------------------------------------------------
step "4/6 Build (kernels compiled ahead of time for $chip; takes a few minutes)"

set +u
# shellcheck disable=SC1091
source "$ONEAPI_ROOT/setvars.sh" --force >/dev/null 2>&1 || true
set -u
have icpx || die "icpx not found after sourcing $ONEAPI_ROOT/setvars.sh"
gen=(); have ninja && gen=(-G Ninja)
build=$REPO/build
if [ -f "$build/CMakeCache.txt" ] && ! grep -q "^INK_AOT_TARGETS:STRING=$chip\$" "$build/CMakeCache.txt"; then
  rm -rf "$build"  # built for another chip
fi
cmake -S "$REPO" -B "$build" "${gen[@]}" -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release \
  -DINK_AOT_TARGETS="$chip" >/dev/null
cmake --build "$build" --target inkling ink-probe bench-decode bench-prefill test-forward -j"$(nproc)" |
  grep -Ev '^\[|^ninja: |^Build succeeded|^Compilation from IR|^$' || true
[ -f "$build/libinkling.so" ] || die "build failed (run: cmake --build $build)"
ok "$build/libinkling.so"

# check the formats and the card's memory against the model (reads the header, moves ~64 MiB through the GPU)
export LD_LIBRARY_PATH=$ONEAPI_ROOT/compiler/latest/lib:$ONEAPI_ROOT/tbb/latest/lib:$ONEAPI_ROOT/umf/latest/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
probe=$("$build/ink-probe" "$model" 2>&1) || {
  echo "$probe" | tail -5 | sed 's/^/    /'
  die "ink-probe failed: unsupported model file or no usable GPU (see above)"
}
echo "$probe" | grep -E '^device:' | sed 's/^/  /'
# VRAM estimate: weights + K/V cache (8 full-attention slots; the sliding-window slots are small) + ~0.45 GiB
# of prompt buffers at 1024-token chunks + the engine's 0.5 GiB margin
python3 - "$probe" "$ctx" "$kv" <<'PY' || true
import re, sys
probe, ctx, kv = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
w = float(re.search(r"^weights: ([0-9.]+) GiB", probe, re.M).group(1))
m = re.search(r"memory ([0-9.]+) GiB total, ([0-9.]+) GiB free", probe)
kvg = ctx * 8 * 2 * 1024 * (kv / 8 + 2 / 32) / 2**30
need = w + kvg + 0.45 + 0.5
print(f"  needs ~{need:.1f} GiB: weights {w:.1f}, {kv}-bit K/V cache {kvg:.1f}, buffers and margin 0.95")
if m:
    total, free = float(m.group(1)), float(m.group(2))
    print(f"  the card has {total:.1f} GiB, {free:.1f} GiB free right now")
    if need > free:
        print("  warning: that is more than is free right now. Fine if ArcAngel or another model server is running"
              " (it frees its memory when stopped); otherwise close other GPU programs, or re-run with a smaller"
              " --ctx or --kv-bits 4")
PY

# ------------------------------------------------------------------------------------------------------
step "5/6 Configuration and service"

mkdir -p "$CONF_DIR"
cat > "$CONF_DIR/arcangel.env" <<EOF
# ArcAngel settings, read by $REPO/server/run.sh. Written by install.sh; edit freely.
ARCANGEL_CARD=$card
ARCANGEL_MODEL=$model
ARCANGEL_ASSETS=$assets
ARCANGEL_CTX=$ctx
ARCANGEL_KV_BITS=$kv
ARCANGEL_PORT=$port
ARCANGEL_REASONING=low
# ARCANGEL_STASH_RAM_GB=8   # host RAM for stashed conversations (default: a quarter of RAM)
# ARCANGEL_DEVICE=1          # pick another Level Zero GPU (default: the one with the most memory)
EOF
ok "$CONF_DIR/arcangel.env"

if [ $service = 1 ] && have systemctl; then
  mkdir -p "$UNIT_DIR"
  cat > "$UNIT_DIR/arcangel.service" <<EOF
[Unit]
Description=ArcAngel: Inkling 2 Mini on Intel Arc, OpenAI API on 127.0.0.1:$port
# Not started at login unless you enable it: it holds most of the card's memory while running.

[Service]
Type=simple
ExecStart=$REPO/server/run.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
EOF
  systemctl --user daemon-reload
  ok "systemd user service 'arcangel' (systemctl --user start|stop|status arcangel)"
  if ask "Start ArcAngel automatically when you log in?" n; then
    systemctl --user enable arcangel >/dev/null 2>&1 && ok "enabled at login"
  fi
fi

# ------------------------------------------------------------------------------------------------------
step "6/6 opencode"

if [ $opencode = 1 ] && { [ -f "$OPENCODE_JSON" ] || have opencode; }; then
  if ask "Add ArcAngel to opencode ($OPENCODE_JSON)?" y; then
    mkdir -p "$(dirname "$OPENCODE_JSON")"
    [ -f "$OPENCODE_JSON" ] && cp "$OPENCODE_JSON" "$OPENCODE_JSON.bak-arcangel"
    if python3 - "$OPENCODE_JSON" "$port" "$ctx" "$cname" <<'EOF'
import json, sys
path, port, ctx, card = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
try:
    cfg = json.load(open(path))
except FileNotFoundError:
    cfg = {"$schema": "https://opencode.ai/config.json"}
cfg.setdefault("provider", {})["arcangel"] = {
    "npm": "@ai-sdk/openai-compatible",
    "name": "ArcAngel",
    "options": {"baseURL": f"http://127.0.0.1:{port}/v1"},
    "models": {"inkling-2-mini": {
        "name": f"Inkling 2 Mini (ArcAngel, {ctx // 1024}k)",
        "tool_call": True, "reasoning": True,
        "limit": {"context": ctx, "output": 8192}}},
}
json.dump(cfg, open(path, "w"), indent=2)
EOF
    then
      ok "provider 'arcangel', model arcangel/inkling-2-mini (backup: $OPENCODE_JSON.bak-arcangel)"
    else
      warn "could not edit $OPENCODE_JSON (comments in it?): add the provider by hand, see README.md"
    fi
  fi
else
  echo "  skipped (opencode not found); any OpenAI-compatible client can use http://127.0.0.1:$port/v1"
fi

# ------------------------------------------------------------------------------------------------------
if [ $service = 1 ] && [ $start = 1 ] && have systemctl; then
  step "Starting"
  if (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null; then
    warn "something already listens on port $port: stop it, or re-run with --port N"
  else
    systemctl --user restart arcangel
    echo "  loading the model (up to a minute)..."
    for _ in $(seq 1 120); do
      curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1 && break
      systemctl --user -q is-active arcangel || break
      sleep 1
    done
    if curl -sf "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      journalctl --user -u arcangel -n 20 --no-pager -o cat | grep 'engine ready' | tail -1 | sed 's/^/  /'
      reply=$(curl -sf "http://127.0.0.1:$port/v1/chat/completions" -H 'Content-Type: application/json' \
        -d '{"messages":[{"role":"user","content":"Reply with just the word: ready"}],"max_tokens":300}' |
        python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"].strip())' 2>/dev/null || true)
      ok "server answers on http://127.0.0.1:$port/v1 (test reply: ${reply:-?})"
    else
      warn "the server did not come up; last log lines:"
      journalctl --user -u arcangel -n 15 --no-pager -o cat | sed 's/^/    /'
      exit 1
    fi
  fi
fi

echo
echo "${B}Done.${N} ArcAngel: $cname, $gguf_name, $((ctx / 1024))k context."
if [ $service = 1 ]; then
  echo "  start/stop:  systemctl --user start arcangel / systemctl --user stop arcangel"
  echo "  logs:        journalctl --user -u arcangel -f"
else
  echo "  start:       $REPO/server/run.sh"
fi
echo "  API:         http://127.0.0.1:$port/v1 (model name: inkling-2-mini)"
echo "  settings:    $CONF_DIR/arcangel.env"
