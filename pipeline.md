# v1.3 파이프라인 실행 가이드

## 📋 목차
1. [필요 데이터셋 구조](#필요-데이터셋-구조)
2. [환경 설정 (requirements.txt)](#환경-설정-requirementstxt)
3. [모델 학습 구조 (Fold별)](#모델-학습-구조-fold별)
4. [예측값 취합 및 Meta-Classifier](#예측값-취합-및-meta-classifier)
5. [최종 예측값 결과 형식](#최종-예측값-결과-형식)
6. [전체 파이프라인 실행](#전체-파이프라인-실행)

---

## 필요 데이터셋 구조

### 디렉토리 구조
```
mut4/
├── data/
│   ├── train.csv       # 학습 데이터
│   └── test.csv        # 테스트 데이터 (선택적)
```

### 데이터 형식

#### train.csv
필수 컬럼:
- `title`: 문서 제목 (str)
- `full_text`: 전체 문서 내용 (str, 문단은 `\n`으로 구분)
- `generated`: 레이블 (int, 0: Human, 1: AI)

예시:
```csv
title,full_text,generated
"문서 1","첫 번째 문단입니다.\n두 번째 문단입니다.\n세 번째 문단입니다.",0
"문서 2","이것은 AI가 생성한 문서입니다.\n여러 문단으로 구성되어 있습니다.",1
```

#### test.csv (추론용)
필수 컬럼:
- `title`: 문서 제목 (str)
- `full_text`: 전체 문서 내용 (str)

예시:
```csv
title,full_text
"테스트 문서 1","테스트 내용입니다.\n여러 문단이 있습니다."
```

---

## 환경 설정 (requirements.txt)

```txt
# Core dependencies
torch>=2.0.0
transformers>=4.51.0
numpy>=1.24.0
pandas>=1.5.0
scikit-learn>=1.3.0

# Tokenizer and models
sentencepiece>=0.1.99
protobuf>=3.20.0

# Utilities
tqdm>=4.65.0
joblib>=1.3.0

# Optional: for better performance
accelerate>=0.20.0
datasets>=2.14.0
```

### 설치 방법
```bash
pip install -r requirements.txt
```

### GPU 설정 (선택적)
- CUDA 11.8 이상 권장
- PyTorch는 CUDA 버전에 맞게 설치 필요
```bash
# CUDA 11.8 예시
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

---

## 모델 학습 구조 (Fold별)

### 전체 학습 프로세스

v1.3은 **4-Fold Cross-Validation** 구조를 사용합니다:

```
[전체 데이터]
    │
    ├─ KFold(n_splits=4, random_state=42)
    │
    ├─ Fold 0: Train → Validation (OOF Logits 생성)
    ├─ Fold 1: Train → Validation (Test Logits 생성)
    ├─ Fold 2: Train → Validation (Test Logits 생성)
    └─ Fold 3: Train → Validation (Test Logits 생성)
```

### Fold별 학습 과정

#### Step 1: 단일 Fold 학습

**Python 스크립트 실행**:
```bash
python trainers/fold_trainer.py \
    --fold_idx 0 \
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
    --k_fold 4 \
    --is_kfold True
```

**주요 인자 설명**:
- `--fold_idx`: Fold 번호 (0, 1, 2, 3)
- `--model_name`: 모델 타입 (`AvsHModel`, `HybridAvsH`, `Gemma3InfoNCE`, `Qwen3InfoNCE`)
- `--embedding_model`: 백본 모델 (`kykim/funnel-kor-base`, `kykim/bert-kor-base` 등)
- `--use_infonce_loss`: InfoNCE Loss 사용 여부
- `--lambda_cl`: InfoNCE Loss 가중치 (기본값: 0.1)
- `--save_fold_logits`: Fold Logits 저장 여부 (Meta-Learning에 필요)

#### Step 2: 모든 Fold 학습

**방법 1: 순차 실행** (권장)
```bash
for FOLD in 0 1 2 3; do
    python trainers/fold_trainer.py \
        --fold_idx ${FOLD} \
        --model_name HybridAvsH \
        --embedding_model kykim/funnel-kor-base \
        --use_paragraph \
        --use_infonce_loss True \
        --lambda_cl 0.1 \
        --data_dir ./data \
        --save_fold_logits True \
        --k_fold 4 \
        --is_kfold True
done
```

**방법 2: Python 스크립트 사용**
```python
from trainers.fold_trainer import train_all_folds
from utils.arguments import get_arguments

args = get_arguments()
args.model_name = 'HybridAvsH'
args.use_infonce_loss = True
args.save_fold_logits = True
args.k_fold = 4
args.is_kfold = True

train_all_folds(args)
```

**방법 3: 병렬 실행** (4개 GPU 사용 시)
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

## 예측값 취합 및 Meta-Classifier

### Step 1: Meta-Features 생성

4개 Fold의 Logits를 취합하여 Meta-Features 데이터셋 생성:

```bash
python -c "
from utils.logit_collector import LogitCollector

collector = LogitCollector('./outputs/fold_logits')
collector.save_meta_features('./outputs/meta_features', 'meta_train.csv')
"
```

**또는 Python 스크립트**:
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

### Step 2: Meta-Classifier 학습

Meta-Classifier는 Fold 예측값을 입력으로 받아 최종 예측을 수행합니다.

**MLP Meta-Classifier 학습**:
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

**Ridge Meta-Classifier 학습**:
```bash
python meta/meta_train.py \
    --meta_model_type ridge \
    --cv 5 \
    --logit_dir ./outputs/fold_logits \
    --output_dir ./outputs/meta_features \
    --save_model
```

**주요 인자**:
- `--meta_model_type`: `mlp` 또는 `ridge`
- `--hidden_layers`: MLP hidden layer 크기 (MLP만)
- `--epochs`: 학습 에폭 수 (MLP만)
- `--save_model`: 모델 저장 여부

**학습 결과**:
- 모델 저장 위치: `./outputs/meta_features/meta_classifier_mlp.pth` (또는 `.joblib`)
- 평가 결과: ROC-AUC, Accuracy, F1-Score 출력

### Step 3: Meta-Classifier 구조

**MLP 구조**:
```
Input (4 features: Fold Logits)
  ↓
Hidden Layer 1 (64 units, ReLU, Dropout=0.2)
  ↓
Hidden Layer 2 (32 units, ReLU, Dropout=0.2)
  ↓
Output (1 unit, Sigmoid)
```

**Ridge 구조**:
```
Ridge Regression with L2 Regularization
- Cross-Validation으로 최적 alpha 선택
- Alpha 범위: 10^-4 ~ 10^2
```

---

## 최종 예측값 결과 형식

### Step 1: Test 데이터에 대한 Fold별 예측

각 Fold 모델로 Test 데이터 예측:

```bash
# Fold별 추론 (각 Fold마다 실행)
for FOLD in 0 1 2 3; do
    python trainers/fold_trainer.py \
        --fold_idx ${FOLD} \
        --is_submission True \
        --data_dir ./data \
        --model_name HybridAvsH \
        --embedding_model kykim/funnel-kor-base \
        --use_paragraph \
        # ... 기타 인자
done
```

### Step 2: Test Logits 취합

```python
from utils.logit_collector import collect_test_logits

test_logits = collect_test_logits('./outputs/fold_logits')
print(f"Test logits shape: {test_logits.shape}")  # [N_test, 4]
```

### Step 3: Meta-Classifier로 최종 예측

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

## 전체 파이프라인 실행

### 완전 자동화 스크립트

**run_pipeline.py** (예시):
```python
#!/usr/bin/env python
"""
v1.3 전체 파이프라인 실행 스크립트
"""

import os
from trainers.fold_trainer import train_all_folds
from utils.arguments import get_arguments
from utils.logit_collector import LogitCollector
from meta.meta_train import main as train_meta
from meta.meta_inference import main as infer_meta

def main():
    print("="*60)
    print("v1.3 Pipeline: Training and Inference")
    print("="*60)
    
    # Step 1: Fold별 학습
    print("\n[Step 1/4] Training 4 Folds...")
    args = get_arguments()
    args.model_name = 'HybridAvsH'
    args.use_infonce_loss = True
    args.lambda_cl = 0.1
    args.save_fold_logits = True
    args.k_fold = 4
    args.is_kfold = True
    args.use_paragraph = True
    
    train_all_folds(args)
    
    # Step 2: Meta-Features 생성
    print("\n[Step 2/4] Creating Meta-Features...")
    collector = LogitCollector('./outputs/fold_logits')
    collector.save_meta_features('./outputs/meta_features', 'meta_train.csv')
    
    # Step 3: Meta-Classifier 학습
    print("\n[Step 3/4] Training Meta-Classifier...")
    import sys
    sys.argv = [
        'meta_train.py',
        '--meta_model_type', 'mlp',
        '--hidden_layers', '64', '32',
        '--epochs', '100',
        '--save_model'
    ]
    train_meta()
    
    # Step 4: 최종 예측
    print("\n[Step 4/4] Generating Final Predictions...")
    sys.argv = [
        'meta_inference.py',
        '--meta_model_type', 'mlp',
        '--output_filename', 'submission.csv'
    ]
    infer_meta()
    
    print("\n" + "="*60)
    print("✅ Pipeline Complete!")
    print("Final predictions saved to: ./outputs/final_predictions/submission.csv")
    print("="*60)

if __name__ == "__main__":
    main()
```

### 실행 순서 요약

```bash
# 1. 환경 설정
pip install -r requirements.txt

# 2. 데이터 준비
# data/train.csv, data/test.csv 준비

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

## 예상 소요 시간

| 단계 | 작업 | 예상 시간 (단일 GPU) | 예상 시간 (4개 GPU 병렬) |
|------|------|---------------------|------------------------|
| Step 1 | Fold별 학습 (4개) | 20-28시간 | 8-10시간 |
| Step 2 | Meta-Features 생성 | 10분 | 10분 |
| Step 3 | Meta-Classifier 학습 | 30분-1시간 | 30분-1시간 |
| Step 4 | 최종 예측 | 10분 | 10분 |
| **총계** | | **21-30시간** | **9-12시간** |

---

## 주의사항

1. **메모리**: 각 Fold 학습 시 약 4-8GB GPU 메모리 필요
2. **저장공간**: Logits 저장 시 약 500MB-1GB 필요
3. **데이터 일관성**: 모든 Fold에서 동일한 데이터 분할 사용 (K-Fold split은 자동 저장됨)
4. **모델 저장**: 각 Fold의 best model은 `outputs/fold_{idx}/best_model/`에 저장

---

## 트러블슈팅

### 문제: Logits가 저장되지 않음
- 해결: `--save_fold_logits True` 인자 확인
- 해결: `outputs/fold_logits/` 디렉토리 권한 확인

### 문제: Meta-Features 생성 실패
- 해결: 4개 Fold 모두 학습 완료 확인
- 해결: `fold_logits/oof/fold0_logits.npy` 파일 존재 확인

### 문제: Meta-Classifier 학습 실패
- 해결: Meta-Features CSV 파일 확인
- 해결: Labels와 Logits 개수 일치 확인

---

## 참고 자료

- **구현 가이드**: `v1.3_implementation_guide.md`
- **모델 요약**: `v1.3_models_summary.md`
- **파일 구조**: `v1.3_file_structure.md`

