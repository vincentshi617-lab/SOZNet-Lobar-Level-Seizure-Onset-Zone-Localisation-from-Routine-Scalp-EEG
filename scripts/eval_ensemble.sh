#!/usr/bin/env bash
set -euo pipefail

export SOZNET_DATA_ROOT=${SOZNET_DATA_ROOT:-data/processed}
export SOZNET_ANNOTATION_CSV=${SOZNET_ANNOTATION_CSV:-data/labels.csv}

python -m soznet.evaluate_ensemble \
  --model "${MODEL:-m1}" \
  --model-dir "${MODEL_DIR:-outputs/ensemble}" \
  --output-dir "${OUTPUT_DIR:-outputs/eval}"
