#!/usr/bin/env bash
set -euo pipefail

# Meta-Classifier 학습 스크립트 (MLP 기본)
# 환경변수:
#   META_TYPE (mlp|ridge, 기본 mlp)
#   LOGIT_DIR (기본 ./outputs/fold_logits)
#   OUT_DIR   (기본 ./outputs/meta_features)

META_TYPE="${META_TYPE:-mlp}"
LOGIT_DIR="${LOGIT_DIR:-./outputs/fold_logits}"
OUT_DIR="${OUT_DIR:-./outputs/meta_features}"

if [[ "${META_TYPE}" == "mlp" ]]; then
  python meta/meta_train.py \
    --meta_model_type mlp \
    --hidden_layers 64 32 \
    --dropout 0.2 \
    --epochs 100 \
    --batch_size 32 \
    --lr 0.001 \
    --logit_dir "${LOGIT_DIR}" \
    --output_dir "${OUT_DIR}" \
    --save_model
else
  python meta/meta_train.py \
    --meta_model_type ridge \
    --cv 5 \
    --logit_dir "${LOGIT_DIR}" \
    --output_dir "${OUT_DIR}" \
    --save_model
fi