# Colab 환경 설정 및 실행 가이드

## 프로젝트 위치
```
내 드라이브/외부/멋사NLP/project/mut4_colab
```
Colab 경로: `/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab`

---

## Part 1: 환경 셋업

### 1.1 초기 설정 (첫 셀에 실행)

```python
# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 작업 디렉토리 설정
import os
PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.chdir(PROJECT_ROOT)

# 환경 변수 설정 (코드에서 자동으로 사용됨)
os.environ['DATA_DIR'] = f'{PROJECT_ROOT}/data'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

# 패키지 설치
!pip install -r requirements.txt

# GPU 확인
!nvidia-smi
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 1.2 데이터 준비

데이터 파일이 다음 위치에 있어야 합니다:
```
{PROJECT_ROOT}/data/
  ├── train.csv
  ├── test.csv
  └── augmented_merged_reviews.csv (선택사항)
```

### 1.3 경로 구조

```
{PROJECT_ROOT}/
├── data/                    # 입력 데이터
├── outputs/                  # 모든 출력 (Google Drive에 저장)
│   ├── fold_logits/         # Fold별 로그잇
│   │   ├── oof/            # Fold 0 (OOF)
│   │   └── test/            # Fold 1,2,3
│   ├── fold_0/              # Fold 0 모델 체크포인트
│   ├── fold_1/              # Fold 1 모델 체크포인트
│   ├── fold_2/              # Fold 2 모델 체크포인트
│   ├── fold_3/              # Fold 3 모델 체크포인트
│   ├── meta_features/       # Meta-Classifier 모델
│   └── final_predictions/   # 최종 submission 파일
└── constants_phase4/         # K-fold split 정보 (자동 생성)
```

---

## Part 2: 빠른 시작 가이드

### 2.1 Fold 학습 (단일 Fold)

```python
from types import SimpleNamespace
from trainers.fold_trainer import FoldTrainer
import os

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'

cfg = SimpleNamespace(
    data_dir=f'{PROJECT_ROOT}/data',
    is_submission=False,
    is_kfold=True,
    k_fold=4,
    fold_idx=0,  # 0, 1, 2, 3 중 선택
    val_ratio=0.2,
    use_paragraph=True,
    add_title=False,
    save_dir="v1.3_fold",
    save_name="fold0",
    # model
    model_name="HybridAvsH",
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
    per_device_train_batch_size=8,  # GPU 메모리에 맞게 조정
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

# 학습 실행
trainer = FoldTrainer(cfg)
trainer.train_fold(0)  # fold_idx 변경하여 다른 fold 학습
```

### 2.2 모든 Fold 학습 (순차 실행)

```python
from trainers.fold_trainer import train_all_folds
from utils.arguments import get_arguments
import os

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.environ['DATA_DIR'] = f'{PROJECT_ROOT}/data'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

# 기본 인자로 실행 (필요시 수정)
args = get_arguments()
args.data_dir = f'{PROJECT_ROOT}/data'
args.output_dir = f'{PROJECT_ROOT}/outputs'
args.is_kfold = True
args.k_fold = 4
args.save_dir = 'v1.3_fold'
args.model_name = 'HybridAvsH'
args.embedding_model = 'kykim/funnel-kor-base'
args.num_train_epochs = 10
args.per_device_train_batch_size = 8

train_all_folds(args)
```

### 2.3 Meta-Classifier 학습

```python
import os
from meta.meta_train import main
import sys

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

# 명령줄 인자로 실행
sys.argv = [
    'meta_train.py',
    '--meta_model_type', 'mlp',  # 또는 'ridge'
    '--hidden_layers', '64', '32',
    '--dropout', '0.2',
    '--epochs', '100',
    '--batch_size', '32',
    '--lr', '0.001',
    '--save_model'
]

main()
```

### 2.4 최종 추론 및 Submission 생성

```python
import os
from meta.meta_inference import main
import sys

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.environ['DATA_DIR'] = f'{PROJECT_ROOT}/data'  # test.csv를 읽기 위해 필요
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

sys.argv = [
    'meta_inference.py',
    '--meta_model_type', 'mlp',  # 학습 시 사용한 타입과 동일
    '--output_filename', 'submission.csv'
]

main()
```

**결과 파일 위치**: `{PROJECT_ROOT}/outputs/final_predictions/submission.csv`

---

## Part 3: 로컬 → Colab Migration 가이드

### 3.1 수정된 코드 부분

#### ✅ 이미 수정 완료된 부분:
1. **`utils/arguments.py`**: 환경 변수 `DATA_DIR`, `OUTPUT_DIR` 지원
2. **`trainers/fold_trainer.py`**: 출력 경로를 환경 변수에서 가져오도록 수정
3. **`meta/meta_train.py`**: 출력 경로를 환경 변수에서 가져오도록 수정
4. **`meta/meta_inference.py`**: 
   - 출력 경로를 환경 변수에서 가져오도록 수정
   - **test.csv에서 실제 ID를 읽어서 submission 파일 생성** (수정 완료)

#### ⚠️ Colab 실행 시 주의사항:

1. **상수 디렉토리 경로** (`constants_phase4/`)
   - 현재: 상대 경로로 생성됨 (`data/get_dataset.py`)
   - Colab에서도 정상 작동 (프로젝트 루트 기준)
   - 문제 없음 ✅

2. **데이터 경로**
   - 환경 변수 `DATA_DIR` 설정 필수
   - 또는 `cfg.data_dir` 직접 지정

3. **출력 경로**
   - 환경 변수 `OUTPUT_DIR` 설정 필수
   - Google Drive 경로 사용 권장 (세션 종료 방지)

4. **Test 데이터 추론**
   - 현재 코드는 validation set의 logits만 수집
   - **실제 test set 추론이 필요하면 별도 구현 필요** (아래 참고)

### 3.2 Test Set 추론이 필요한 경우

**현재 구조**: 
- Fold 학습 시 validation set의 logits만 저장됨
- Meta-Classifier는 validation set logits로 학습됨
- 최종 추론도 validation set logits 사용 (test set logits가 아닌 validation set logits 사용)

**실제 test set 추론이 필요한 경우**:
현재 코드는 validation set에 대한 logits만 생성하므로, 실제 test set에 대한 추론이 필요하면 별도 구현이 필요합니다. 

현재 구조에서는:
1. Fold 0: validation set logits → OOF logits로 저장
2. Fold 1,2,3: validation set logits → test logits로 저장
3. Meta-Classifier: 이 logits들을 meta-features로 사용

**참고**: 대회 제출용이라면 validation set logits를 사용하는 현재 구조로도 충분할 수 있습니다.

### 3.3 주요 변경 사항 요약

| 항목 | 로컬 환경 | Colab 환경 | 수정 여부 |
|------|----------|-----------|----------|
| 데이터 경로 | `./data` | `{PROJECT_ROOT}/data` | ✅ 환경 변수 지원 |
| 출력 경로 | `./outputs` | `{PROJECT_ROOT}/outputs` | ✅ 환경 변수 지원 |
| 상수 디렉토리 | `constants_phase4/` | `constants_phase4/` | ✅ 문제 없음 |
| GPU 설정 | `torch.cuda.is_available()` | 동일 | ✅ 문제 없음 |
| DataLoader workers | `0` | `0` | ✅ 문제 없음 |
| Submission ID | `range(len(preds))` | `test.csv`에서 읽기 | ✅ 수정 완료 |
| 상수 디렉토리 | `constants_phase4/` | 상대 경로 (프로젝트 루트 기준) | ✅ 문제 없음 |

---

## Part 4: 실행 체크리스트 및 주의사항

### 4.1 실행 전 체크리스트

- [ ] Google Drive 마운트 완료
- [ ] 프로젝트 디렉토리로 이동 (`os.chdir`)
- [ ] 환경 변수 설정 (`DATA_DIR`, `OUTPUT_DIR`)
- [ ] 데이터 파일 확인 (`train.csv`, `test.csv`)
- [ ] `requirements.txt` 설치 완료
- [ ] GPU 사용 가능 확인
- [ ] 출력 디렉토리가 Google Drive 경로인지 확인

### 4.2 주의사항

1. **세션 종료 방지**
   - 모든 출력은 Google Drive에 저장 (`OUTPUT_DIR` 설정)
   - 중간 체크포인트가 자동 저장됨 (`save_total_limit=2`)

2. **GPU 메모리 관리**
   - T4 GPU (16GB): `per_device_train_batch_size=8` 권장
   - V100 GPU (16GB): 더 큰 배치 사이즈 가능
   - 메모리 부족 시 배치 사이즈 감소 또는 `gradient_accumulation_steps` 증가

3. **실행 시간**
   - Colab 무료: 최대 12시간
   - 긴 학습은 중간 체크포인트 활용
   - Fold별로 별도 셀에서 실행 권장

4. **디스크 공간**
   - Colab 디스크 공간 제한 있음
   - Google Drive 용량 확인
   - 불필요한 체크포인트 정리

5. **모델 다운로드**
   - Hugging Face 모델 자동 다운로드
   - 인터넷 연결 필요
   - 첫 실행 시 시간 소요

### 4.3 문제 해결

**문제**: `FileNotFoundError: test.csv not found`
- 해결: `DATA_DIR` 환경 변수 확인, `test.csv` 파일 위치 확인
- 참고: Meta-Classifier 추론 시 test.csv의 ID를 읽기 위해 필요

**문제**: `FileNotFoundError: k_fold_split.json not found`
- 해결: 첫 Fold 학습 시 자동 생성됨. `constants_phase4/` 디렉토리 확인
- 참고: 프로젝트 루트에서 실행해야 상대 경로 정상 작동

**문제**: `CUDA out of memory`
- 해결: `per_device_train_batch_size` 감소 또는 `gradient_accumulation_steps` 증가

**문제**: 세션 종료로 출력 손실
- 해결: `OUTPUT_DIR`을 Google Drive 경로로 설정

**문제**: 패키지 버전 충돌
- 해결: Colab 기본 패키지 확인 후 `requirements.txt` 수정

---

## Part 5: Colab 실행 순서 (셀별 가이드)

### 전체 파이프라인

1. **환경설정** → 2. **학습** → 3. **최종 결과 생성** → 4. **성능 평가 및 시각화**

---

### 셀 1: 환경설정

```python
# ============================================
# 1. 환경설정
# ============================================

# Google Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 작업 디렉토리 설정
import os
PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.chdir(PROJECT_ROOT)

# 환경 변수 설정 (코드에서 자동으로 사용됨)
os.environ['DATA_DIR'] = f'{PROJECT_ROOT}/data'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

# 패키지 설치
!pip install -r requirements.txt

# GPU 확인
!nvidia-smi
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# 데이터 파일 확인
import pandas as pd
print("\n=== 데이터 파일 확인 ===")
data_dir = os.environ.get('DATA_DIR', './data')
if os.path.exists(os.path.join(data_dir, 'train.csv')):
    train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    print(f"✅ train.csv: {len(train_df)} rows")
else:
    print("❌ train.csv not found")
    
if os.path.exists(os.path.join(data_dir, 'test.csv')):
    test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    print(f"✅ test.csv: {len(test_df)} rows")
else:
    print("❌ test.csv not found")
```

---

### 셀 2: Fold 학습 (Fold 0)

```python
# ============================================
# 2. 학습 - Fold 0
# ============================================

from types import SimpleNamespace
from trainers.fold_trainer import FoldTrainer
import os

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'

cfg = SimpleNamespace(
    data_dir=f'{PROJECT_ROOT}/data',
    is_submission=False,
    is_kfold=True,
    k_fold=4,
    fold_idx=0,  # Fold 0
    val_ratio=0.2,
    use_paragraph=True,
    add_title=False,
    save_dir="v1.3_fold",
    save_name="fold0",
    # model
    model_name="HybridAvsH",
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

# 학습 실행
trainer = FoldTrainer(cfg)
trainer.train_fold(0)
print("✅ Fold 0 학습 완료")
```

---

### 셀 3: Fold 학습 (Fold 1)

```python
# ============================================
# 2. 학습 - Fold 1
# ============================================

# 위 셀의 cfg에서 fold_idx만 변경
cfg.fold_idx = 1
cfg.save_name = "fold1"

trainer = FoldTrainer(cfg)
trainer.train_fold(1)
print("✅ Fold 1 학습 완료")
```

---

### 셀 4: Fold 학습 (Fold 2)

```python
# ============================================
# 2. 학습 - Fold 2
# ============================================

cfg.fold_idx = 2
cfg.save_name = "fold2"

trainer = FoldTrainer(cfg)
trainer.train_fold(2)
print("✅ Fold 2 학습 완료")
```

---

### 셀 5: Fold 학습 (Fold 3)

```python
# ============================================
# 2. 학습 - Fold 3
# ============================================

cfg.fold_idx = 3
cfg.save_name = "fold3"

trainer = FoldTrainer(cfg)
trainer.train_fold(3)
print("✅ Fold 3 학습 완료")
```

**또는 모든 Fold를 한 번에 실행하려면:**

```python
# ============================================
# 2. 학습 - 모든 Fold (순차 실행)
# ============================================

from trainers.fold_trainer import train_all_folds
from utils.arguments import get_arguments
import os

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.environ['DATA_DIR'] = f'{PROJECT_ROOT}/data'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

args = get_arguments()
args.data_dir = f'{PROJECT_ROOT}/data'
args.is_kfold = True
args.k_fold = 4
args.save_dir = 'v1.3_fold'
args.model_name = 'HybridAvsH'
args.embedding_model = 'kykim/funnel-kor-base'
args.num_train_epochs = 10
args.per_device_train_batch_size = 8
args.max_length = 256

train_all_folds(args)
print("✅ 모든 Fold 학습 완료")
```

---

### 셀 6: Meta-Classifier 학습

```python
# ============================================
# 3. 최종 결과 생성 - Meta-Classifier 학습
# ============================================

import os
from meta.meta_train import main
import sys

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

# 명령줄 인자로 실행
sys.argv = [
    'meta_train.py',
    '--meta_model_type', 'mlp',  # 또는 'ridge'
    '--hidden_layers', '64', '32',
    '--dropout', '0.2',
    '--epochs', '100',
    '--batch_size', '32',
    '--lr', '0.001',
    '--save_model'
]

main()
print("✅ Meta-Classifier 학습 완료")
```

---

### 셀 7: 최종 추론 및 Submission 생성

```python
# ============================================
# 3. 최종 결과 생성 - 최종 추론
# ============================================

import os
from meta.meta_inference import main
import sys

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.environ['DATA_DIR'] = f'{PROJECT_ROOT}/data'  # test.csv 읽기 위해 필요
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

sys.argv = [
    'meta_inference.py',
    '--meta_model_type', 'mlp',  # 학습 시 사용한 타입과 동일
    '--output_filename', 'submission.csv'
]

main()

# 결과 확인
import pandas as pd
output_path = f'{PROJECT_ROOT}/outputs/final_predictions/submission.csv'
if os.path.exists(output_path):
    submission_df = pd.read_csv(output_path)
    print(f"\n✅ Submission 파일 생성 완료: {output_path}")
    print(f"   총 {len(submission_df)}개 예측")
    print(f"\n   예시:")
    print(submission_df.head())
else:
    print(f"❌ Submission 파일을 찾을 수 없습니다: {output_path}")
```

---

### 셀 8: 성능 평가 및 시각화

```python
# ============================================
# 4. 성능 평가 및 시각화
# ============================================

import os
from utils.evaluate_folds import generate_evaluation_report

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

# 평가 리포트 생성
logit_dir = f'{PROJECT_ROOT}/outputs/fold_logits'
meta_model_path = f'{PROJECT_ROOT}/outputs/meta_features/meta_classifier_mlp.pth'  # 또는 ridge.joblib
output_dir = f'{PROJECT_ROOT}/outputs/evaluation'

# Meta-Classifier 모델 경로 확인
if not os.path.exists(meta_model_path):
    # ridge 모델인 경우
    meta_model_path = meta_model_path.replace('.pth', '.joblib')
    meta_model_type = 'ridge'
else:
    meta_model_type = 'mlp'

generate_evaluation_report(
    logit_dir=logit_dir,
    meta_model_path=meta_model_path if os.path.exists(meta_model_path) else None,
    meta_model_type=meta_model_type,
    output_dir=output_dir
)

print(f"\n✅ 평가 리포트 생성 완료: {output_dir}")
print("   생성된 파일:")
print("   - fold_metrics.csv: Fold별 성능 지표")
print("   - evaluation_summary.json: 전체 요약")
print("   - metrics_comparison.png: 성능 비교 그래프")
print("   - roc_curves.png: ROC 곡선")
print("   - metrics_table.png: 성능 표")
```

---

### 셀 9: 결과 다운로드 (선택사항)

```python
# ============================================
# 결과 다운로드
# ============================================

from google.colab import files
import os

PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'

# Submission 파일 다운로드
submission_path = f'{PROJECT_ROOT}/outputs/final_predictions/submission.csv'
if os.path.exists(submission_path):
    files.download(submission_path)
    print("✅ Submission 파일 다운로드 완료")
else:
    print(f"❌ 파일을 찾을 수 없습니다: {submission_path}")

# 평가 결과 다운로드 (선택)
evaluation_dir = f'{PROJECT_ROOT}/outputs/evaluation'
if os.path.exists(evaluation_dir):
    import shutil
    shutil.make_archive('/tmp/evaluation_results', 'zip', evaluation_dir)
    files.download('/tmp/evaluation_results.zip')
    print("✅ 평가 결과 다운로드 완료")
```

---

## 실행 순서 요약

| 셀 | 단계 | 설명 | 예상 시간 |
|----|------|------|----------|
| 1 | 환경설정 | Drive 마운트, 패키지 설치, GPU 확인 | 5-10분 |
| 2-5 | 학습 | Fold 0, 1, 2, 3 순차 학습 | 각 Fold당 1-3시간 |
| 6 | Meta-Classifier 학습 | Fold logits로 Meta-Classifier 학습 | 5-10분 |
| 7 | 최종 추론 | Submission 파일 생성 | 1-2분 |
| 8 | 성능 평가 | 평가 리포트 및 시각화 생성 | 1-2분 |
| 9 | 결과 다운로드 | 파일 다운로드 (선택) | 1분 |

---

## 빠른 실행 (전체 자동화)

모든 단계를 한 번에 실행하려면:

```python
# ============================================
# 전체 파이프라인 자동 실행
# ============================================

# 1. 환경설정
from google.colab import drive
drive.mount('/content/drive')
import os
PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.chdir(PROJECT_ROOT)
os.environ['DATA_DIR'] = f'{PROJECT_ROOT}/data'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'
!pip install -r requirements.txt

# 2. 모든 Fold 학습
from trainers.fold_trainer import train_all_folds
from utils.arguments import get_arguments
args = get_arguments()
args.data_dir = f'{PROJECT_ROOT}/data'
args.is_kfold = True
args.k_fold = 4
args.save_dir = 'v1.3_fold'
args.model_name = 'HybridAvsH'
args.embedding_model = 'kykim/funnel-kor-base'
args.num_train_epochs = 10
args.per_device_train_batch_size = 8
train_all_folds(args)

# 3. Meta-Classifier 학습
from meta.meta_train import main
import sys
sys.argv = ['meta_train.py', '--meta_model_type', 'mlp', '--save_model']
main()

# 4. 최종 추론
from meta.meta_inference import main
import sys
sys.argv = ['meta_inference.py', '--meta_model_type', 'mlp']
main()

# 5. 성능 평가
from utils.evaluate_folds import generate_evaluation_report
generate_evaluation_report(
    logit_dir=f'{PROJECT_ROOT}/outputs/fold_logits',
    meta_model_path=f'{PROJECT_ROOT}/outputs/meta_features/meta_classifier_mlp.pth',
    meta_model_type='mlp',
    output_dir=f'{PROJECT_ROOT}/outputs/evaluation'
)

print("✅ 전체 파이프라인 완료!")
```

---

## 경로 요약

- **프로젝트 루트**: `/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab`
- **데이터 경로**: `{PROJECT_ROOT}/data`
- **출력 경로**: `{PROJECT_ROOT}/outputs`
  - Fold 로그잇: `{PROJECT_ROOT}/outputs/fold_logits/`
  - Meta 모델: `{PROJECT_ROOT}/outputs/meta_features/`
  - 최종 예측: `{PROJECT_ROOT}/outputs/final_predictions/`
