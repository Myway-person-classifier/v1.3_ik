#!/usr/bin/env bash
set -euo pipefail

# Meta-Classifier 최종 예측 스크립트
# 환경변수:
#   META_TYPE   (mlp|ridge, 기본 mlp)
#   MODEL_PATH  (기본 ./outputs/meta_features/meta_classifier_mlp.pth or .joblib)
#   LOGIT_DIR   (기본 ./outputs/fold_logits)
#   OUT_DIR     (기본 ./outputs/final_predictions)
#   OUT_FILE    (기본 submission.csv)

META_TYPE="${META_TYPE:-mlp}"
LOGIT_DIR="${LOGIT_DIR:-./outputs/fold_logits}"
OUT_DIR="${OUT_DIR:-./outputs/final_predictions}"
OUT_FILE="${OUT_FILE:-submission.csv}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  if [[ "${META_TYPE}" == "ridge" ]]; then
    MODEL_PATH="./outputs/meta_features/meta_classifier_ridge.joblib"
  else
    MODEL_PATH="./outputs/meta_features/meta_classifier_mlp.pth"
  fi
fi

python meta/meta_inference.py \
  --meta_model_type "${META_TYPE}" \
  --model_path "${MODEL_PATH}" \
  --logit_dir "${LOGIT_DIR}" \
  --output_dir "${OUT_DIR}" \
  --output_filename "${OUT_FILE}"