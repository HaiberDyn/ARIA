#!/usr/bin/env bash
# Assembles the ARIA spec from module files.
# Output filename is derived automatically from the version in ARIA-MASTER.md.
# Usage: bash spec/modules/build.sh (run from repo root)
set -euo pipefail

# Extract version from ARIA-MASTER.md (e.g. "0.7-draft" from "## Specification 0.7-draft")
VERSION=$(grep -m1 '^## Specification' spec/modules/ARIA-MASTER.md | sed 's/## Specification //')
OUT="spec/ARIA-${VERSION}.md"

echo "<!-- Assembled from spec/modules/ — do not edit directly; edit source modules and rebuild -->" > "$OUT"
for f in spec/modules/ARIA-MASTER.md \
          spec/modules/MOD-C1-TYPES.md \
          spec/modules/MOD-C2-CONTRACT.md \
          spec/modules/MOD-C3-STATE.md \
          spec/modules/MOD-C4-MPA.md \
          spec/modules/MOD-C5-NEGOTIATION.md \
          spec/modules/MOD-C6-ERRORS.md \
          spec/modules/MOD-C7-OBLIGATIONS.md \
          spec/modules/MOD-C8-ABI.md \
          spec/modules/ARIA-TAIL.md; do
  tail -n +2 "$f" >> "$OUT"
done
echo "Built: $OUT ($(wc -l < "$OUT") lines)"
