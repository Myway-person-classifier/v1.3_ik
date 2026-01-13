#!/usr/bin/env bash
set -euo pipefail

# Fold 학습 스크립트
# 사용 예:
#   FOLDS="0 1 2 3" DATA_DIR=./data ./scripts/train_fold.sh
# 환경변수로 조정:
#   MODEL_NAME   (기본 HybridAvsH)
#   EMBEDDING    (기본 kykim/funnel-kor-base)
#   DATA_DIR     (기본 ./data)
#   FOLDS        (기본 "0 1 2 3")
#   EPOCHS       (기본 10)
#   BS           (per_device_train_batch_size, 기본 8)

MODEL_NAME="${MODEL_NAME:-HybridAvsH}"
EMBEDDING="${EMBEDDING:-kykim/funnel-kor-base}"
DATA_DIR="${DATA_DIR:-./data}"
FOLDS="${FOLDS:-0 1 2 3}"
EPOCHS="${EPOCHS:-10}"
BS="${BS:-8}"

for FOLD in ${FOLDS}; do
  python - <<PY
from types import SimpleNamespace
from trainers.fold_trainer import FoldTrainer

fold_idx = int(${FOLD})

cfg = SimpleNamespace(
    # data
    data_dir="${DATA_DIR}",
    is_submission=False,
    is_kfold=True,
    k_fold=4,
    fold_idx=fold_idx,
    val_ratio=0.2,
    use_paragraph=True,
    add_title=False,
    save_dir="baseline",
    save_name=f"fold{fold_idx}",
    # model
    model_name="${MODEL_NAME}",
    embedding_model="${EMBEDDING}",
    num_labels=1,
    num_heads=8,
    num_layers=4,
    dim_feedforward=2048,
    hidden_size=512,
    dropout=0.0,
    # loss
    use_bpr_loss=True,
    bpr_loss_weight=0.25,
    use_infonce_loss=True,
    lambda_cl=0.1,
    temperature=0.07,
    # train
    num_train_epochs=${EPOCHS},
    per_device_train_batch_size=${BS},
    per_device_eval_batch_size=${BS},
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=1e-3,
    logging_steps=10,
    # misc
    split_valid_by_paragraph=False,
    save_fold_logits=True,
    local_rank=-1,
)

FoldTrainer(cfg).train_fold(cfg.fold_idx)
PY
done