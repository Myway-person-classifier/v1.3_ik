#!/usr/bin/env bash
set -euo pipefail

# Fold별 추론(검증/테스트)용 스크립트
# 이미 학습된 4개 Fold 모델을 사용하여 test.csv에 대한 Logits을 생성합니다.
# 생성된 Logits은 Meta-Classifier의 입력(Meta-Features)으로 사용됩니다.

MODEL_NAME=${1:-"HybridAvsH"}
EMBEDDING_MODEL=${2:-"kykim/funnel-kor-base"}
DATA_DIR=${3:-"./data"}
OUTPUT_DIR=${4:-"./outputs"}

echo "Generating test logits for all 4 folds using model: $MODEL_NAME"

for FOLD in 0 1 2 3; do
  python trainers/fold_trainer.py \
    --predict_only \
    --is_submission True \
    --fold_idx $FOLD \
    --model_name $MODEL_NAME \
    --embedding_model $EMBEDDING_MODEL \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 16 \
    --use_paragraph \
    --save_fold_logits True
done

echo "✅ All 4 folds inference completed."
echo "Now you can run Meta-Classifier inference:"
echo "python meta/meta_inference.py --meta_model_type mlp --logit_dir $OUTPUT_DIR/fold_logits/$MODEL_NAME"
