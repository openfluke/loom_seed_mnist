#!/usr/bin/env bash
# Build DOCX bundles for loom_seed_mnist: poly seed stack FIRST, then this PoC.
set -euo pipefail
cd "$(dirname "$0")"

stamp="$(date +%Y%m%d_%H%M%S)"
outdir="docs"
mkdir -p "$outdir"

seed_out="$outdir/poly_seed_${stamp}.docx"
proj_out="$outdir/seed_mnist_${stamp}.docx"

seed_latest="$outdir/poly_seed_bundle.docx"
proj_latest="$outdir/seed_mnist_bundle.docx"

if ! python3 -c "import docx" 2>/dev/null; then
  echo "python-docx missing — install with: pip install python-docx" >&2
  exit 1
fi

echo "→ python3 make_docx.py  (poly seeds first, then seed_mnist)"
echo "  1) $seed_out"
echo "  2) $proj_out"
echo

python3 make_docx.py \
  --poly-seeds-output "$seed_out" \
  -o "$proj_out"

cp -f "$seed_out" "$seed_latest"
cp -f "$proj_out" "$proj_latest"

echo
echo "done:"
ls -lh "$seed_out" "$proj_out" "$seed_latest" "$proj_latest"
