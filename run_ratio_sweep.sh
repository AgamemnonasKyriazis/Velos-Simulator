#!/usr/bin/env bash
set -euo pipefail

PYTHON=".venv/bin/python"
OPT="optimize_ratios.py"

OUT_DIR="${OUT_DIR:-$HOME/tmp}"
mkdir -p "${OUT_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

# Sweep config (override via env vars)
N_JOBS="${N_JOBS:-64}"
COARSE_STEP="${COARSE_STEP:-0.10}"
FINE_STEP="${FINE_STEP:-0.02}"
RADIUS="${RADIUS:-0.06}"
TOP="${TOP:-20}"

OUT_FILE="${OUT_DIR}/ratio_opt_${N_JOBS}_${TS}.txt"

echo "Running optimization sweep..."
echo "Output file: ${OUT_FILE}"

"${PYTHON}" -u "${OPT}" \
  --n-jobs "${N_JOBS}" \
  --coarse-step "${COARSE_STEP}" \
  --fine-step "${FINE_STEP}" \
  --radius "${RADIUS}" \
  --top "${TOP}" \
  --workers 8 | tee "${OUT_FILE}"\

echo
echo "Done. Best result:"
grep -A8 "^=== BEST RATIOS ===" "${OUT_FILE}" || true
