#!/usr/bin/env bash
# Check the expert-weight formats of the bigger quants on a 12 GB card, against llama.cpp.
#
#   tests/format_test.sh BUILD_DIR Q2_K_XL.gguf WORK_DIR
#
# BUILD_DIR must be configured with -DINK_LLAMA_TESTS=ON (gguf-retype, ref-dump, test-forward). WORK_DIR needs
# ~20 GB of disk (not tmpfs). Makes two copies of UD-Q2_K_XL with a few expert tensors re-quantised:
#   A: Q8_0 experts (as in UD-Q4_K_XL), gate_up and down
#   B: Q5_K experts (as in UD-Q4_K_M), gate_up (Q5_K needs rows that are a multiple of 256: down has 1152)
# then runs llama.cpp on the CPU as the answer key and the engine (prompt and decode paths) against it.
# Stop any server holding the GPU first. Expect MATCH with KL ~0.001-0.002 on every step.
set -euo pipefail
B=$1 M=$2 T=$3
P="The capital of France is Paris. The capital of Germany is Berlin. The capital of Italy is"
mkdir -p "$T"
[ -f "$T/mixA.gguf" ] || "$B/gguf-retype" "$M" "$T/mixA.gguf" blk.0.ffn_down_exps.weight=q8_0 \
  blk.1.ffn_gate_up_exps.weight=q8_0 blk.9.ffn_down_exps.weight=q8_0
[ -f "$T/mixB.gguf" ] || "$B/gguf-retype" "$M" "$T/mixB.gguf" blk.3.ffn_gate_up_exps.weight=q5_K \
  blk.10.ffn_gate_up_exps.weight=q5_K
for x in A B; do
  if [ ! -f "$T/ref$x/tokens.txt" ]; then
    mkdir -p "$T/ref$x"
    "$B/ref-dump" "$T/mix$x.gguf" "$T/ref$x" "$P" 2 > "$T/ref$x.log" 2>&1
  fi
  for m in prefill graph; do
    echo "== mix$x $m"
    "$B/test-forward" "$T/mix$x.gguf" "$T/ref$x" $m 2>&1 | grep -E 'logits:|error'
  done
done
