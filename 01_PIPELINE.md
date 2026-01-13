# v1.3 전체 구조 및 실행 파이프라인

## 📋 목차
1. [프로젝트 구조](#프로젝트-구조)
2. [필요 데이터셋 구조](#필요-데이터셋-구조)
3. [환경 설정](#환경-설정)
4. [전체 파이프라인 실행](#전체-파이프라인-실행)
5. [Fold별 학습 방법](#fold별-학습-방법)
6. [Meta-Classifier 학습 및 추론](#meta-classifier-학습-및-추론)
7. [협업 실행 가이드](#협업-실행-가이드)
8. [평가 지표 확인](#평가-지표-확인)
9. [트러블슈팅](#트러블슈팅)

---

## 프로젝트 구조

### 디렉토리 구조
```
mut4/
├── data/                          # 데이터 처리 코드
│   ├── __init__.py
│   ├── get_dataset.py            # K-Fold 데이터 분할
│   ├── text_dataset.py           # 데이터셋 클래스
│   ├── text_collator.py          # 배치 전처리
│   └── meta_dataset.py           # Meta-Features 생성
│
├── models/                        # 모델 정의
│   ├── AvsHModel.py              # SKKUAI 문단 계층 모델
│   ├── hybrid_model.py           # AvsH + InfoNCE 통합
│   ├── gemma3_seqcls_infonce.py  # Gemma3 InfoNCE 모델
│   └── qwen3_seqcls_infonce.py   # Qwen3 InfoNCE 모델
│
├── trainers/                      # 학습 로직
│   ├── hybrid_trainer.py         # 통합 Trainer (BCE + BPR + InfoNCE)
│   └── fold_trainer.py           # Fold별 학습 관리
│
├── meta/                          # Meta-Learning
│   ├── meta_classifier.py        # MLP/Ridge Meta-Classifier
│   ├── meta_train.py             # Meta-Classifier 학습
│   └── meta_inference.py         # 최종 예측
│
├── utils/                         # 유틸리티
│   ├── arguments.py              # 하이퍼파라미터 설정
│   ├── losses.py                 # Loss 함수 (BPR + InfoNCE)
│   ├── compute_metrics.py        # 평가 지표
│   ├── logit_collector.py        # Fold Logits 수집
│   └── evaluate_folds.py         # 평가 및 시각화
│
├── scripts/                       # 실행 스크립트
│   ├── train_fold.sh             # Fold별 학습
│   ├── meta_train.sh             # Meta-Classifier 학습
│   └── meta_inference.sh         # 최종 예측
│
├── outputs/                        # 출력 디렉토리
│   ├── fold_0/                   # Fold 0 학습 결과
│   ├── fold_1/                   # Fold 1 학습 결과
│   ├── fold_2/                   # Fold 2 학습 결과
│   ├── fold_3/                   # Fold 3 학습 결과
│   ├── fold_logits/              # Fold별 Logits
│   │   ├── oof/                  # OOF Logits (Fold 0)
│   │   └── test/                 # Test Logits (Fold 1,2,3)
│   ├── meta_features/            # Meta-Features 데이터셋
│   ├── final_predictions/        # 최종 예측 결과
│   └── evaluation/               # 평가 결과 및 시각화
│
└── constants_phase4/              # K-Fold Split 저장
    └── k_fold_split.json         # 4-Fold 데이터 분할 정보
```

### 데이터 흐름도
```
[Main Dataset]
      │
      ▼
[Paragraph Decomposition]  (최대 10문단)
      │
      ▼
┌─────────────────────────────────────┐
│  4-Fold Training (Base Models)     │
│  - Fold 0: Train → OOF Logits       │
│  - Fold 1,2,3: Train → Test Logits  │
└─────────────────────────────────────┘
      │
      ▼
[Meta-Features Dataset]
  Shape: [Num_Samples] × 4
  - Column 0: OOF Logits (Fold 0)
  - Column 1-3: Test Logits (Fold 1,2,3)
      │
      ▼
[Meta-Classifier Training]
  - MLP 또는 Ridge
  - Input: [Num_Samples] × 4
  - Output: 최종 예측 (0: Human, 1: AI)
      │
      ▼
[Final Prediction & Submission]
```
```
┌─────────────────────────────────────┐
│  모델 선택 (model_name="HybridAvsH") │
└─────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────┐
│  4-Fold Cross-Validation            │
│  (모든 Fold에서 동일한 모델 사용)    │
└─────────────────────────────────────┘
              │
    ┌─────────┼─────────┐
    ▼         ▼         ▼
┌──────┐  ┌──────┐  ┌──────┐
│Fold 0│  │Fold 1│  │Fold 3│
│Logits│  │Logits│  │Logits│
└──────┘  └──────┘  └──────┘
    │         │         │
    └─────────┼─────────┘
              ▼
    ┌──────────────────┐
    │ Meta-Features    │
    │ [N, 4]           │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ Meta-Classifier  │
    │ (MLP or Ridge)    │
    └──────────────────┘
              │
              ▼
    ┌──────────────────┐
    │ Final Prediction │
    └──────────────────┘
```
---

## 필요 데이터셋 구조

### 디렉토리 구조
```
mut4/
├── data/
│   ├── train.csv                  # 학습 데이터
│   └── test.csv                   # 테스트 데이터 (선택적)
```

### 데이터 형식

#### train.csv
**필수 컬럼**:
- `full_text`: 전체 문서 내용 (str, 문단은 `\n`으로 구분)
- `generated`: 레이블 (int, 0: Human, 1: AI)

**선택 컬럼**:
- `title`: 문서 제목 (str, 없으면 빈 문자열로 처리)

**예시**:
```csv
title,full_text,generated
"문서 1","첫 번째 문단입니다.\n두 번째 문단입니다.\n세 번째 문단입니다.",0
"문서 2","이것은 AI가 생성한 문서입니다.\n여러 문단으로 구성되어 있습니다.",1
```

**참고**: `augmented_merged_reviews.csv`와 같이 `title` 컬럼이 없는 경우도 자동 처리됩니다.

#### test.csv (추론용)
**필수 컬럼**:
- `full_text`: 전체 문서 내용 (str)

**선택 컬럼**:
- `title`: 문서 제목 (str)
- `ID` 또는 `id`: 샘플 ID

---

## 환경 설정

### requirements.txt 설치
```bash
pip install -r requirements.txt
```

### 주요 의존성
- `torch>=2.0.0`
- `transformers>=4.51.0`
- `numpy>=1.24.0`
- `pandas>=1.5.0`
- `scikit-learn>=1.3.0`
- `matplotlib>=3.5.0` (시각화용)
- `seaborn>=0.12.0` (시각화용)

### GPU 설정 (선택적)
- CUDA 11.8 이상 권장
- PyTorch는 CUDA 버전에 맞게 설치 필요

---

## 전체 파이프라인 실행

### 완전 자동화 실행 (권장)

#### Step 1: Fold별 학습 (4개 Fold)
```bash
# 순차 실행
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
    --num_train_epochs 10 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 3e-5 \
    --data_dir ./data \
    --save_fold_logits True \
    --k_fold 4 --is_kfold True
done
```

#### Step 2: Meta-Features 생성
```bash
python -c "from utils.logit_collector import LogitCollector; LogitCollector('./outputs/fold_logits').save_meta_features('./outputs/meta_features','meta_train.csv')"
```

#### Step 3: Meta-Classifier 학습
```bash
# MLP 옵션
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

# 또는 Ridge 옵션
python meta/meta_train.py \
  --meta_model_type ridge \
  --cv 5 \
  --logit_dir ./outputs/fold_logits \
  --output_dir ./outputs/meta_features \
  --save_model
```

#### Step 4: 최종 예측
```bash
python meta/meta_inference.py \
  --meta_model_type mlp \
  --model_path ./outputs/meta_features/meta_classifier_mlp.pth \
  --logit_dir ./outputs/fold_logits \
  --output_dir ./outputs/final_predictions \
  --output_filename submission.csv
```

### 실행 순서 요약
```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 준비
# data/train.csv 또는 data/augmented_merged_reviews.csv 준비

# 3. Fold별 학습 (4개 Fold)
python trainers/fold_trainer.py --fold_idx 0 --save_fold_logits True ...
python trainers/fold_trainer.py --fold_idx 1 --save_fold_logits True ...
python trainers/fold_trainer.py --fold_idx 2 --save_fold_logits True ...
python trainers/fold_trainer.py --fold_idx 3 --save_fold_logits True ...

# 4. Meta-Features 생성
python -c "from utils.logit_collector import LogitCollector; LogitCollector('./outputs/fold_logits').save_meta_features()"

# 5. Meta-Classifier 학습
python meta/meta_train.py --meta_model_type mlp --save_model

# 6. 최종 예측
python meta/meta_inference.py --meta_model_type mlp
```

---

## Fold별 학습 방법

### K-Fold Split 자동 생성
- 첫 실행 시 `constants_phase4/k_fold_split.json` 자동 생성
- `random_state=42`로 고정되어 동일한 split 보장
- 한 번 생성되면 모든 Fold에서 재사용

### 단일 Fold 학습
```bash
python trainers/fold_trainer.py \
  --fold_idx 0 \  # 0, 1, 2, 3 중 선택
  --model_name HybridAvsH \
  --embedding_model kykim/funnel-kor-base \
  --use_paragraph \
  --use_infonce_loss True \
  --lambda_cl 0.1 \
  --use_bpr_loss True \
  --data_dir ./data \
  --save_fold_logits True \
  --k_fold 4 --is_kfold True
```

### 모든 Fold 순차 학습
```bash
for FOLD in 0 1 2 3; do
  python trainers/fold_trainer.py --fold_idx $FOLD ...
done
```

### 병렬 실행 (4개 GPU 사용 시)
```bash
# 각 터미널에서 실행
CUDA_VISIBLE_DEVICES=0 python trainers/fold_trainer.py --fold_idx 0 ...
CUDA_VISIBLE_DEVICES=1 python trainers/fold_trainer.py --fold_idx 1 ...
CUDA_VISIBLE_DEVICES=2 python trainers/fold_trainer.py --fold_idx 2 ...
CUDA_VISIBLE_DEVICES=3 python trainers/fold_trainer.py --fold_idx 3 ...
```

### 학습 결과 저장 위치
```
outputs/
├── fold_0/
│   ├── best_model/          # 학습된 모델
│   └── ...
├── fold_1/
│   ├── best_model/
│   └── ...
├── fold_2/
│   ├── best_model/
│   └── ...
├── fold_3/
│   ├── best_model/
│   └── ...
└── fold_logits/
    ├── oof/
    │   ├── fold0_logits.npy    # Fold 0 Validation Logits
    │   └── fold0_labels.npy    # Fold 0 Validation Labels
    └── test/
        ├── fold1_logits.npy    # Fold 1 Validation Logits
        ├── fold1_labels.npy
        ├── fold2_logits.npy    # Fold 2 Validation Logits
        ├── fold2_labels.npy
        ├── fold3_logits.npy    # Fold 3 Validation Logits
        └── fold3_labels.npy
```

---

## Meta-Classifier 학습 및 추론

### Meta-Features 생성
4개 Fold의 Logits를 취합하여 Meta-Features 데이터셋 생성:

```python
from utils.logit_collector import LogitCollector

collector = LogitCollector('./outputs/fold_logits')
meta_features, labels = collector.collect_logits()
print(f"Meta-features shape: {meta_features.shape}")  # [N, 4]
```

**Meta-Features 구조**:
- Shape: `[Num_Samples, 4]`
- Column 0: OOF Logits (Fold 0 validation)
- Column 1: Test Logits (Fold 1 validation)
- Column 2: Test Logits (Fold 2 validation)
- Column 3: Test Logits (Fold 3 validation)

### Meta-Classifier 학습

**MLP Meta-Classifier**:
```bash
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
```

**Ridge Meta-Classifier**:
```bash
python meta/meta_train.py \
  --meta_model_type ridge \
  --cv 5 \
  --logit_dir ./outputs/fold_logits \
  --output_dir ./outputs/meta_features \
  --save_model
```

### 최종 예측 생성
```bash
python meta/meta_inference.py \
  --meta_model_type mlp \
  --model_path ./outputs/meta_features/meta_classifier_mlp.pth \
  --logit_dir ./outputs/fold_logits \
  --output_dir ./outputs/final_predictions \
  --output_filename submission.csv
```

### 최종 결과 파일 형식

**submission.csv**:
```csv
id,generated,probability
0,0,0.234
1,1,0.876
2,0,0.445
3,1,0.912
...
```

**컬럼 설명**:
- `id`: 샘플 인덱스 (0부터 시작)
- `generated`: 예측 레이블 (0: Human, 1: AI)
- `probability`: AI일 확률 (0.0 ~ 1.0)

---

## 협업 실행 가이드

### 4명 병렬 실행 방법

#### Step 1: K-Fold Split 생성 및 공유
- 한 명이 먼저 `trainers/fold_trainer.py`를 실행하여 `constants_phase4/k_fold_split.json` 생성
- 이 파일을 다른 팀원들과 공유 (동일한 split 사용)

#### Step 2: 각자 다른 Fold 학습
각 팀원은 `--fold_idx`만 다르게 설정하여 실행:

```bash
# 팀원 A (Fold 0 담당)
python trainers/fold_trainer.py \
  --fold_idx 0 \
  --save_fold_logits True \
  --model_name HybridAvsH \
  --embedding_model kykim/funnel-kor-base \
  --use_paragraph \
  --use_infonce_loss True \
  --lambda_cl 0.1 \
  --use_bpr_loss True \
  --data_dir ./data \
  --k_fold 4 --is_kfold True

# 팀원 B (Fold 1 담당)
python trainers/fold_trainer.py \
  --fold_idx 1 \
  --save_fold_logits True \
  ... (나머지 인자는 동일)

# 팀원 C, D도 동일하게 Fold 2, 3 담당
```

**주의사항**:
- `MODEL_NAME`, `EMBEDDING_MODEL`, `USE_INFONCE` 등 다른 하이퍼파라미터는 모든 팀원이 동일하게 설정
- `--fold_idx`만 다르게 설정

#### Step 3: 출력 경로 통일
- 모든 Fold의 학습 결과는 `outputs/fold_{idx}` 및 `outputs/fold_logits/{oof,test}` 경로에 저장
- 모든 팀원이 동일한 `outputs/` 루트 디렉토리를 사용하거나, 각자 학습 완료 후 생성된 `outputs/fold_logits` 내용을 한 곳으로 모음

#### Step 4: Meta-Learning 진행
모든 4개 Fold의 학습이 완료되고 `outputs/fold_logits`에 다음 파일들이 모두 모이면:
- `fold_logits/oof/fold0_logits.npy`, `fold0_labels.npy`
- `fold_logits/test/fold1_logits.npy`, `fold2_logits.npy`, `fold3_logits.npy` (labels 포함)

한 명이 `scripts/meta_train.sh` 및 `scripts/meta_inference.sh`를 실행하여 최종 예측을 생성합니다.

---

## 평가 지표 확인

### 방법 1: 학습 중 자동 평가
- `utils/compute_metrics.py`에서 각 Epoch마다 자동 계산
- 출력: `accuracy`, `f1_score`, `precision`, `recall`, `roc_auc`

### 방법 2: Fold별 종합 평가 리포트
```bash
# Fold별 평가만
python utils/evaluate_folds.py \
  --logit_dir ./outputs/fold_logits \
  --output_dir ./outputs/evaluation

# Meta-Classifier 포함 평가
python utils/evaluate_folds.py \
  --logit_dir ./outputs/fold_logits \
  --meta_model_path ./outputs/meta_features/meta_classifier_mlp.pth \
  --meta_model_type mlp \
  --output_dir ./outputs/evaluation
```

**출력 파일**:
- `fold_metrics.csv`: Fold별 상세 지표
- `evaluation_summary.json`: 종합 요약
- `metrics_comparison.png`: 성능 비교 차트
- `roc_curves.png`: ROC 곡선 비교
- `metrics_table.png`: 성능 테이블

### 방법 3: Python 코드로 직접 확인
```python
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from utils.logit_collector import LogitCollector

# Fold logits 로드
collector = LogitCollector('./outputs/fold_logits')
meta_features, labels = collector.collect_logits()

# Fold별 평가
for fold_idx in range(4):
    logits = meta_features[:, fold_idx]
    preds = (logits > 0).astype(int)
    labels_binary = (labels > 0.5).astype(int)
    
    acc = accuracy_score(labels_binary, preds)
    f1 = f1_score(labels_binary, preds, average='macro')
    prec = precision_score(labels_binary, preds, average='macro')
    recall = recall_score(labels_binary, preds, average='macro')
    auc = roc_auc_score(labels_binary, logits)
    
    print(f"Fold {fold_idx}: Acc={acc:.4f}, F1={f1:.4f}, Prec={prec:.4f}, Recall={recall:.4f}, AUC={auc:.4f}")
```

---

## 예상 소요 시간

| 단계 | 작업 | 예상 시간 (단일 GPU) | 예상 시간 (4개 GPU 병렬) |
|------|------|---------------------|------------------------|
| Step 1 | Fold별 학습 (4개) | 20-28시간 | 8-10시간 |
| Step 2 | Meta-Features 생성 | 10분 | 10분 |
| Step 3 | Meta-Classifier 학습 | 30분-1시간 | 30분-1시간 |
| Step 4 | 최종 예측 | 10분 | 10분 |
| **총계** | | **21-30시간** | **9-12시간** |

---

## 트러블슈팅

### 문제: Logits가 저장되지 않음
- **해결**: `--save_fold_logits True` 인자 확인
- **해결**: `outputs/fold_logits/` 디렉토리 권한 확인

### 문제: Meta-Features 생성 실패
- **해결**: 4개 Fold 모두 학습 완료 확인
- **해결**: `fold_logits/oof/fold0_logits.npy` 파일 존재 확인

### 문제: Meta-Classifier 학습 실패
- **해결**: Meta-Features CSV 파일 확인
- **해결**: Labels와 Logits 개수 일치 확인

### 문제: K-Fold Split 불일치
- **해결**: `constants_phase4/k_fold_split.json` 파일 공유 확인
- **해결**: 모든 팀원이 동일한 split 파일 사용 확인

### 문제: GPU 메모리 부족
- **해결**: `--per_device_train_batch_size` 줄이기 (예: 8 → 4)
- **해결**: `--gradient_accumulation_steps` 늘리기 (예: 2 → 4)

---

## 주의사항

1. **메모리**: 각 Fold 학습 시 약 4-8GB GPU 메모리 필요
2. **저장공간**: Logits 저장 시 약 500MB-1GB 필요
3. **데이터 일관성**: 모든 Fold에서 동일한 데이터 분할 사용 (K-Fold split은 자동 저장됨)
4. **모델 저장**: 각 Fold의 best model은 `outputs/fold_{idx}/best_model/`에 저장
5. **데이터 형식**: `title` 컬럼이 없어도 자동 처리됨 (빈 문자열로 처리)

