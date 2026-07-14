#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

SEED_FILE="mnist.seeds.json"
POP_FILE="mnist.pop.json"

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  loom_seed_mnist — download · 80/20 · seed train · heatmap   ║"
echo "╚══════════════════════════════════════════════════════════════╝"

MODE="${LOOM_SEED_MNIST_MODE:-}"
CONT="${LOOM_SEED_MNIST_CONTINUE:-}"

for arg in "$@"; do
  case "$arg" in
    1|warmth|warm|--warmth|-w) MODE=warmth ;;
    2|dna|neat|pop|cluster|--dna|-d) MODE=dna ;;
    3|dna-layer|layer|coord|--dna-layer) MODE=dna-layer ;;
    4|dna-cascade|cascade|hybrid|--dna-cascade) MODE=dna-cascade ;;
    5|micro-fountain|micro|fountain|mega|--micro-fountain) MODE=micro-fountain ;;
    6|sample-fountain|sample|samples|shard|micro-sample|--sample-fountain) MODE=sample-fountain ;;
    continue|--continue|-c) CONT=1; MODE="${MODE:-sample-fountain}" ;;
    fresh|--fresh) rm -f "$SEED_FILE" "$POP_FILE"; CONT= ;;
  esac
done

# Explicit mode + existing seeds → continue that mode (don't silently reload).
if [[ -f "$SEED_FILE" && -n "$MODE" && -z "$CONT" ]]; then
  CONT=1
fi

if [[ -f "$SEED_FILE" && -z "$CONT" && -z "$MODE" ]]; then
  echo
  echo "Saved $SEED_FILE found:"
  echo "  [1] reload only     — no search"
  echo "  [2] continue search — keep seeds, then pick train mode"
  echo "  [3] fresh train     — wipe seeds/pop, then pick train mode"
  echo
  echo "  (train modes incl. [6] sample-fountain appear on the NEXT screen after 2 or 3)"
  echo -n "Choice [1]: "
  read -r ACT || ACT=1
  case "${ACT:-1}" in
    2|continue|c)
      CONT=1
      ;;
    3|fresh|f)
      rm -f "$SEED_FILE" "$POP_FILE"
      ;;
    *)
      echo
      echo "=== reload only ==="
      go run .
      exit 0
      ;;
  esac
fi

if [[ -z "$MODE" ]]; then
  echo
  echo "Train mode:"
  echo "  [1] warmth          — single genome · warm-bit hill-climb"
  echo "  [2] dna             — clustered DNA · all layers at once"
  echo "  [3] dna-layer       — clustered DNA · one layer at a time"
  echo "  [4] dna-cascade     — L0-heavy → expand free-set → warmth"
  echo "  [5] micro-fountain  — per-digit micro → LT → mega"
  echo "  [6] sample-fountain — sample-shard micro → LT → mega  ← try beat ~19.7%"
  echo -n "Choice [6]: "
  read -r CHOICE || CHOICE=6
  case "${CHOICE:-6}" in
    1|warmth) MODE=warmth ;;
    2|dna|DNA|neat) MODE=dna ;;
    3|dna-layer|layer) MODE=dna-layer ;;
    4|dna-cascade|cascade|hybrid) MODE=dna-cascade ;;
    5|micro-fountain|micro|fountain|mega) MODE=micro-fountain ;;
    6|sample-fountain|sample|shard|micro-sample|"") MODE=sample-fountain ;;
    *) MODE=sample-fountain ;;
  esac
fi

export LOOM_SEED_MNIST_MODE="$MODE"
if [[ -n "$CONT" ]]; then
  export LOOM_SEED_MNIST_CONTINUE=1
fi

echo
if [[ -n "${LOOM_SEED_MNIST_CONTINUE:-}" ]]; then
  echo "=== CONTINUE search (mode=$LOOM_SEED_MNIST_MODE) ==="
  go run . --continue --mode="$LOOM_SEED_MNIST_MODE"
else
  echo "=== Pass 1: train (mode=$LOOM_SEED_MNIST_MODE) ==="
  go run . --mode="$LOOM_SEED_MNIST_MODE"
  echo
  echo "=== Pass 2: reload seeds only ==="
  go run .
fi

echo
echo "done"
echo "tips:"
echo "  ./run.sh 6                 # sample-fountain (sample shards → LT → mega)"
echo "  ./run.sh fresh 6           # wipe + sample-fountain"
echo "  ./run.sh 5                 # micro-fountain (per-digit)"
echo "  ./run.sh 4                 # dna-cascade"
echo "  ./run.sh continue          # resume"
