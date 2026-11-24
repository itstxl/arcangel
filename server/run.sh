#!/bin/bash
# Start the ArcAngel server (OpenAI-compatible API in front of build/libinkling.so).
# Settings come from ~/.config/arcangel/arcangel.env (written by install.sh). Extra arguments go to
# inkling_server.py and override them, e.g. `run.sh --port 9000 --n-ctx 65536`.
here=$(cd "$(dirname "$0")" && pwd)
conf=${ARCANGEL_CONFIG:-${XDG_CONFIG_HOME:-$HOME/.config}/arcangel/arcangel.env}
# shellcheck disable=SC1090
[ -f "$conf" ] && . "$conf"

oneapi=${ONEAPI_ROOT:-/opt/intel/oneapi}
export LD_LIBRARY_PATH=$oneapi/compiler/latest/lib:$oneapi/tbb/latest/lib:$oneapi/umf/latest/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

model=${ARCANGEL_MODEL:-$INK_MODEL}
if [ -z "$model" ]; then
  echo "run.sh: no model: run install.sh, or set ARCANGEL_MODEL=/path/to/Inkling-2-Mini-UD-*.gguf" >&2
  exit 1
fi
args=(--model "$model")
[ -n "$ARCANGEL_ASSETS" ] && args+=(--assets "$ARCANGEL_ASSETS")
[ -n "$ARCANGEL_CTX" ] && args+=(--n-ctx "$ARCANGEL_CTX")
[ -n "$ARCANGEL_KV_BITS" ] && args+=(--kv-bits "$ARCANGEL_KV_BITS")
[ -n "$ARCANGEL_HOST" ] && args+=(--host "$ARCANGEL_HOST")
[ -n "$ARCANGEL_PORT" ] && args+=(--port "$ARCANGEL_PORT")
[ -n "$ARCANGEL_REASONING" ] && args+=(--reasoning-effort "$ARCANGEL_REASONING")
# offload recipe: these layers' routed experts stay in pinned system RAM (see cards.tsv)
[ -n "$ARCANGEL_OFFLOAD" ] && export INK_OFFLOAD=$ARCANGEL_OFFLOAD
[ -n "$ARCANGEL_STASH_RAM_GB" ] && args+=(--stash-ram-gb "$ARCANGEL_STASH_RAM_GB")
exec "$here/.venv/bin/python" -u "$here/inkling_server.py" "${args[@]}" "$@"
