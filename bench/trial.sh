#!/bin/bash
# usage: [LLAMA_DIR=dir] [REQS="code edit"] trial.sh LABEL [llama-server args...]
# Starts llama-server with the Inkling model fully on the GPU plus the given args, waits until healthy,
# runs bench/sbench.py on REQS, then stops the server. Log: bench/logs/LABEL.log
# Refuses to run while the inkling-llama service holds the GPU.
label=$1; shift
here=$(cd "$(dirname "$0")" && pwd)
D=${LLAMA_DIR:-$HOME/.local/share/llama.cpp/vulkan/llama-b12633}
M=$HOME/.local/share/llama.cpp/models/Inkling-2-Mini-UD-Q2_K_XL.gguf
LOG=$here/logs/$label.log
PORT=${PORT:-18130}
if systemctl --user -q is-active inkling-llama; then echo "inkling-llama service is running; stop it first"; exit 1; fi
LD_LIBRARY_PATH=$D $D/llama-server -m "$M" -ngl 99 -fa on -ctk q8_0 -ctv q8_0 --jinja \
  --host 127.0.0.1 --port $PORT "$@" > "$LOG" 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
for i in $(seq 1 240); do
  curl -sf http://127.0.0.1:$PORT/health >/dev/null && break
  kill -0 $pid 2>/dev/null || { echo "== $label: server exited"; grep -E ' E |ErrorOutOf|error' "$LOG" | head -5 | cut -c1-180; exit 1; }
  sleep 1
done
echo "== $label: $*"
grep -E 'compute buffer size|KV buffer size|model buffer size' "$LOG" | sed -E 's/^[0-9.]+ [IWE] //' | cut -c1-90
python3 -I "$here/sbench.py" $PORT ${REQS:-short code edit}
