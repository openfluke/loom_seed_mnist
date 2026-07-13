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
    continue|--continue|-c) CONT=1; MODE="${MODE:-dna}" ;;
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
  echo "  [1] reload only"
  echo "  [2] continue search"
  echo "  [3] fresh train (delete seeds + pop, then pick mode)"
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
  echo "  [1] warmth    — single genome · warm-bit hill-climb"
  echo "  [2] dna       — clustered DNA · all layers at once"
  echo "  [3] dna-layer — clustered DNA · one layer at a time"
  echo -n "Choice [1]: "
  read -r CHOICE || CHOICE=1
  case "${CHOICE:-1}" in
    2|dna|DNA|neat) MODE=dna ;;
    3|dna-layer|layer) MODE=dna-layer ;;
    *) MODE=warmth ;;
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
echo "  ./run.sh 3                 # dna-layer (one layer at a time)"
echo "  ./run.sh dna-layer"
echo "  ./run.sh continue          # resume (then pick mode if needed)"
echo "  ./run.sh fresh 3           # wipe + dna-layer"
