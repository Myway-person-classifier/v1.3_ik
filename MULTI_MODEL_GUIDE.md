# 여러 모델 동시 사용 가이드

## 개요

이제 여러 모델을 동시에 사용하여 더 강력한 앙상블을 만들 수 있습니다.

**핵심 원칙**:
- ✅ 모든 모델이 **동일한 Fold split**을 사용 (k_fold_split.json)
- ✅ 각 모델별로 logits가 별도 디렉토리에 저장
- ✅ Meta-Classifier가 모든 모델의 logits를 결합하여 최종 예측

---

## 구조

### 디렉토리 구조

```
outputs/fold_logits/
├── HybridAvsH/          # 모델 1
│   ├── oof/fold0_logits.npy
│   └── test/fold1_logits.npy, fold2_logits.npy, fold3_logits.npy
├── Gemma3InfoNCE/        # 모델 2
│   ├── oof/fold0_logits.npy
│   └── test/fold1_logits.npy, fold2_logits.npy, fold3_logits.npy
└── Qwen3InfoNCE/         # 모델 3
    ├── oof/fold0_logits.npy
    └── test/fold1_logits.npy, fold2_logits.npy, fold3_logits.npy
```

### Meta-Features 구조

**단일 모델** (기존):
- Shape: [N, 4]
- Columns: [Fold0, Fold1, Fold2, Fold3]

**여러 모델** (새로운):
- Shape: [N, 4×M] (M = 모델 수)
- Columns: [Model1_Fold0, Model1_Fold1, Model1_Fold2, Model1_Fold3,
             Model2_Fold0, Model2_Fold1, Model2_Fold2, Model2_Fold3, ...]

예: 3개 모델 사용 시
- Shape: [N, 12]
- Columns: [HybridAvsH_fold0, HybridAvsH_fold1, ..., Qwen3InfoNCE_fold3]

---

## 사용 방법

### 방법 1: train_multiple_models 함수 사용 (권장)

```python
from trainers.fold_trainer import train_multiple_models
from types import SimpleNamespace
import os

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'

# 기본 설정
base_cfg = SimpleNamespace(
    data_dir=f'{PROJECT_ROOT}/data',
    is_submission=False,
    is_kfold=True,
    k_fold=4,
    val_ratio=0.2,
    use_paragraph=True,
    add_title=False,
    save_dir="v1.3_fold",
    # model (model_name은 train_multiple_models에서 설정됨)
    embedding_model="kykim/funnel-kor-base",
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
    num_train_epochs=10,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=3e-5,
    warmup_ratio=0.1,
    weight_decay=1e-3,
    logging_steps=10,
    max_length=256,
    # misc
    split_valid_by_paragraph=False,
    save_fold_logits=True,
    local_rank=-1,
)

# 여러 모델 학습
model_names = ['HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE']
model_logit_dirs = train_multiple_models(model_names, base_cfg)

print("\n모든 모델 학습 완료!")
```

### 방법 2: 수동으로 각 모델 학습

```python
from trainers.fold_trainer import train_all_folds
from types import SimpleNamespace
import copy

base_cfg = SimpleNamespace(
    # ... 기본 설정 ...
)

# 모델 1
cfg1 = copy.deepcopy(base_cfg)
cfg1.model_name = 'HybridAvsH'
train_all_folds(cfg1)

# 모델 2
cfg2 = copy.deepcopy(base_cfg)
cfg2.model_name = 'Gemma3InfoNCE'
train_all_folds(cfg2)

# 모델 3
cfg3 = copy.deepcopy(base_cfg)
cfg3.model_name = 'Qwen3InfoNCE'
train_all_folds(cfg3)
```

---

## Meta-Classifier 학습

### 여러 모델 사용 시

```python
import os
from meta.meta_train import main
import sys

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

# 여러 모델 지정
sys.argv = [
    'meta_train.py',
    '--meta_model_type', 'mlp',
    '--model_names', 'HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE',  # 여러 모델
    '--hidden_layers', '128', '64', '32',  # 더 큰 모델 (입력 차원 증가)
    '--dropout', '0.2',
    '--epochs', '100',
    '--batch_size', '32',
    '--lr', '0.001',
    '--save_model'
]

main()
```

**중요**: 
- `--model_names`로 학습에 사용한 모델들을 지정
- `input_dim`이 자동으로 계산됨 (모델 수 × 4)
- 예: 3개 모델 → input_dim = 12

### 단일 모델 사용 시 (기존 방식)

```python
sys.argv = [
    'meta_train.py',
    '--meta_model_type', 'mlp',
    # model_names 생략 (단일 모델 모드)
    '--hidden_layers', '64', '32',
    '--dropout', '0.2',
    '--epochs', '100',
    '--batch_size', '32',
    '--lr', '0.001',
    '--save_model'
]
```

---

## Meta-Classifier 추론

### 여러 모델 사용 시

```python
import os
from meta.meta_inference import main
import sys

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.environ['DATA_DIR'] = f'{PROJECT_ROOT}/data'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

sys.argv = [
    'meta_inference.py',
    '--meta_model_type', 'mlp',
    '--model_names', 'HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE',  # 학습 시와 동일
    '--hidden_layers', '128', '64', '32',  # 학습 시와 동일
    '--dropout', '0.2',  # 학습 시와 동일
    '--output_filename', 'submission.csv'
]

main()
```

**중요**: 
- `--model_names`는 학습 시와 **반드시 동일**해야 함
- `--hidden_layers`, `--dropout`도 학습 시와 동일해야 함
- `input_dim`은 자동으로 계산됨

---

## 전체 파이프라인 예시

```python
# ============================================
# 1. 여러 모델 학습
# ============================================

from trainers.fold_trainer import train_multiple_models
from types import SimpleNamespace
import os

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'

base_cfg = SimpleNamespace(
    data_dir=f'{PROJECT_ROOT}/data',
    is_kfold=True,
    k_fold=4,
    save_dir="v1.3_fold",
    model_name="HybridAvsH",  # train_multiple_models에서 덮어씀
    embedding_model="kykim/funnel-kor-base",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    max_length=256,
    # ... 나머지 설정
)

# 3개 모델 학습
model_names = ['HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE']
train_multiple_models(model_names, base_cfg)

# ============================================
# 2. Meta-Classifier 학습
# ============================================

from meta.meta_train import main
import sys

os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

sys.argv = [
    'meta_train.py',
    '--meta_model_type', 'mlp',
    '--model_names', 'HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE',
    '--hidden_layers', '128', '64', '32',  # 입력 차원 12에 맞게 조정
    '--dropout', '0.2',
    '--epochs', '100',
    '--batch_size', '32',
    '--lr', '0.001',
    '--save_model'
]

main()

# ============================================
# 3. 최종 추론
# ============================================

from meta.meta_inference import main
import sys

os.environ['DATA_DIR'] = f'{PROJECT_ROOT}/data'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

sys.argv = [
    'meta_inference.py',
    '--meta_model_type', 'mlp',
    '--model_names', 'HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE',  # 학습 시와 동일
    '--hidden_layers', '128', '64', '32',  # 학습 시와 동일
    '--dropout', '0.2',  # 학습 시와 동일
    '--output_filename', 'submission.csv'
]

main()
```

---

## 주의사항

### 1. Fold Split 일관성

✅ **올바른 사용**:
- 모든 모델이 같은 `save_dir` 사용 → 같은 `k_fold_split.json` 사용
- 모든 모델이 같은 데이터 split으로 학습

❌ **잘못된 사용**:
- 모델마다 다른 `save_dir` 사용 → 다른 split 사용
- 일관성 없는 결과

### 2. 모델 순서

✅ **중요**: Meta-Classifier 학습과 추론 시 `model_names` 순서가 **반드시 동일**해야 함

```python
# 학습 시
model_names = ['HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE']

# 추론 시 (동일한 순서)
model_names = ['HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE']  # ✅
# model_names = ['Gemma3InfoNCE', 'HybridAvsH', 'Qwen3InfoNCE']  # ❌ 순서 다름
```

### 3. Meta-Classifier 크기 조정

여러 모델 사용 시 입력 차원이 증가하므로 Meta-Classifier도 조정:

```python
# 단일 모델: input_dim=4
hidden_layers = [64, 32]

# 3개 모델: input_dim=12
hidden_layers = [128, 64, 32]  # 더 큰 모델 권장
```

### 4. 메모리 및 시간

- 여러 모델 사용 시 학습 시간이 모델 수만큼 증가
- Meta-Classifier 입력 차원 증가로 메모리 사용량 증가
- 필요에 따라 모델 수 조정

---

## 성능 비교

### 단일 모델 vs 여러 모델

**단일 모델**:
- 빠른 학습
- 간단한 구조
- Meta-Classifier 입력: [N, 4]

**여러 모델**:
- 더 강력한 앙상블
- 다양한 모델의 장점 결합
- Meta-Classifier 입력: [N, 4×M]
- 학습 시간 증가

---

## 문제 해결

### 문제: "Dimension mismatch" 에러

**원인**: 학습 시와 추론 시 `model_names` 순서가 다름

**해결**: `model_names` 순서를 학습 시와 동일하게 맞춤

### 문제: "No logits found" 에러

**원인**: 모델 학습이 완료되지 않음

**해결**: 모든 모델의 모든 Fold 학습 완료 확인

### 문제: Meta-Classifier 성능 저하

**원인**: 입력 차원 증가에 비해 모델이 작음

**해결**: `hidden_layers` 크기 증가 (예: [128, 64, 32])

---

## 요약

1. ✅ **여러 모델 동시 사용 가능**
2. ✅ **모든 모델이 동일한 Fold split 사용**
3. ✅ **각 모델별 logits 별도 저장**
4. ✅ **Meta-Classifier가 자동으로 결합**
5. ⚠️ **모델 순서 일관성 유지 필수**

