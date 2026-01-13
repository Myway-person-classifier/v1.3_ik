# Mut4 v1.3 실행 요약

## 1) 학습 방법

### A. 전체 파이프라인(권장)
```bash
pip install -r requirements.txt

# 4개 Fold 순차 학습 (OOF/Test logits 저장)
for FOLD in 0 1 2 3; do
  python trainers/fold_trainer.py \
    --fold_idx $FOLD \
    --model_name HybridAvsH \
    --embedding_model kykim/funnel-kor-base \
    --use_paragraph \
    --use_infonce_loss True \
    --lambda_cl 0.1 \
    --temperature 0.07 \
    --use_bpr_loss True \
    --bpr_loss_weight 0.25 \
    --data_dir ./data \
    --save_fold_logits True \
    --k_fold 4 --is_kfold True
done

# Meta-Features 생성
python -c "from utils.logit_collector import LogitCollector; LogitCollector('./outputs/fold_logits').save_meta_features('./outputs/meta_features','meta_train.csv')"

# Meta-Classifier 학습 (MLP 예시)
python meta/meta_train.py \
  --meta_model_type mlp \
  --hidden_layers 64 32 \
  --dropout 0.2 \
  --epochs 100 \
  --batch_size 32 \
  --lr 0.001 \
  --logit_dir ./outputs/fold_logits \
  --output_dir ./outputs/meta_features \
  --save_model

# 최종 예측
python meta/meta_inference.py \
  --meta_model_type mlp \
  --model_path ./outputs/meta_features/meta_classifier_mlp.pth \
  --logit_dir ./outputs/fold_logits \
  --output_dir ./outputs/final_predictions \
  --output_filename submission.csv
```

### B. Fold별 단독 학습/추론
- 단일 Fold 학습: `python trainers/fold_trainer.py --fold_idx {0|1|2|3} ...`
- 테스트 추론 시 `--is_submission True`로 실행하여 fold별 test logits 생성.

## 2) 예측값 취합 및 최종 결과
- Fold 0: OOF logits (`outputs/fold_logits/oof/fold0_logits.npy`, labels 포함)
- Fold 1/2/3: Validation logits (`outputs/fold_logits/test/fold{1,2,3}_logits.npy`)
- Meta-Features: `LogitCollector`로 `[N,4]` 배열 생성 후 `meta_train.csv`로 저장.
- Meta-Classifier: MLP 또는 Ridge. 학습 후 `meta_classifier_mlp.pth` 또는 `.joblib` 저장.
- 최종 결과: `outputs/final_predictions/submission.csv`
  - 컬럼: `id, generated, probability`
  - `generated`: 0(Human) / 1(AI), `probability`: AI 확률.

### 2-1) 협업 실행(4명 병렬) 가이드
- 한 명이 먼저 `trainers/fold_trainer.py`를 실행해 `k_fold_split.json`을 생성하고, 그 파일을 모두 공유(동일 split 사용).
- 각자 담당 Fold만 실행(하이퍼파라미터 동일, `--fold_idx`만 다르게):
  - A: `--fold_idx 0 --save_fold_logits True ...`
  - B: `--fold_idx 1 --save_fold_logits True ...`
  - C: `--fold_idx 2 --save_fold_logits True ...`
  - D: `--fold_idx 3 --save_fold_logits True ...`
- 출력 경로 통일: `outputs/fold_{idx}`와 `outputs/fold_logits/{oof,test}`가 한 곳에 모이도록 동일 루트 사용(또는 실행 후 합쳐서 모음).
- 모든 Fold 완료 후 필요한 파일:
  - `fold_logits/oof/fold0_logits.npy`, `fold0_labels.npy`
  - `fold_logits/test/fold1_logits.npy`, `fold2_logits.npy`, `fold3_logits.npy` (labels 포함)
- 위 파일이 모여 있어야 Meta-Features 생성 → Meta-Classifier 학습 → 최종 예측 진행 가능.

## 3) 평가 지표 계산
- 학습/검증 시 `utils/compute_metrics.py`에서 자동 계산:
  - `accuracy`, `f1_score`(macro), `precision`, `recall`, `roc_auc`
- 커스텀 평가:
```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

logits = np.load('outputs/fold_logits/oof/fold0_logits.npy')
labels = np.load('outputs/fold_logits/oof/fold0_labels.npy')
preds = (logits > 0).astype(int)

print("acc", accuracy_score(labels, preds))
print("f1", f1_score(labels, preds, average='macro'))
print("prec", precision_score(labels, preds, average='macro'))
print("recall", recall_score(labels, preds, average='macro'))
print("roc_auc", roc_auc_score(labels, logits))
```

## 4) 필수 데이터/구성 체크리스트
- `data/train.csv` (title, full_text, generated), `data/test.csv`(title, full_text)
- `--save_fold_logits True`로 Fold logits 저장
- `outputs/fold_logits/{oof,test}` 존재 확인 후 Meta-Classifier 학습
- GPU 메모리: Fold 학습 시 4~8GB 권장, 저장공간 0.5~1GB

