#!/usr/bin/env bash
set -euo pipefail

export SOZNET_DATA_ROOT=${SOZNET_DATA_ROOT:-data/processed}
export SOZNET_ANNOTATION_CSV=${SOZNET_ANNOTATION_CSV:-data/labels.csv}

python -m soznet.train_ensemble \
  --model "${MODEL:-m1}" \
  --num-models "${NUM_MODELS:-3}" \
  --finetune-m "${FINETUNE_M:-3}" \
  --output-dir "${OUTPUT_DIR:-outputs/ensemble}"
