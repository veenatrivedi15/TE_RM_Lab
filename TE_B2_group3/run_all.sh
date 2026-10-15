#!/usr/bin/env bash
# Reproduces every table and figure.  (~15 min on one CPU core)
set -e
ROOT="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
cd "$ROOT"
PYTHON="${PYTHON:-python3}"
if [ "${CLEAN_WORK:-0}" = "1" ]; then
  rm -rf "$ROOT/work"
fi
"$PYTHON" -m pytest -q tests                                   # integrity + no-leakage tests (fail fast)
mkdir -p work && cp data/modeling_dataset_step1.csv work/ && cd work
cp ../code/*.py ../code/*.js .
cp ../data/T13_external_ICI_cement_PIB.csv .
"$PYTHON" stage1_pipeline.py modeling_dataset_step1.csv      # tables T1-T12, predictions
"$PYTHON" stage1b_external_and_intervals.py                   # external check (T13) + size-stratified intervals (T10c)
"$PYTHON" stage2_figures.py                                   # Figures 1-7
"$PYTHON" stage3_benchmark.py modeling_dataset_step1.csv      # 43-origin benchmark incl. calendar-adjusted baselines (T20-T25 inputs)
"$PYTHON" stage3b_inference.py                                # month-block bootstrap + Diebold-Mariano inference (T20-T25)
"$PYTHON" stage3c_error_sources.py                         # where RF beats persistence (T26-T29c)
"$PYTHON" stage3d_adaptive_conformal.py                     # pooled split conformal + ACI (T30-T32)
"$PYTHON" stage3e_mondrian_conformal.py                     # size-conditional conformal (T33)
"$PYTHON" stage3f_ici_comparison.py                         # panel vs official ICI cement growth (T34-T35)
"$PYTHON" build_spec.py
if command -v node >/dev/null 2>&1 && node -e "require('docx')" >/dev/null 2>&1; then
  node render_docx.js                                      # optional: rebuild the Word manuscript
else
  echo "Skipping manuscript DOCX build: install Node.js dependencies with 'npm install'." >&2
fi
