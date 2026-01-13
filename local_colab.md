# 로컬(mut4) vs Colab(mut4_colab) 프로젝트 차이점 상세 분석

## 📋 목차
1. [전체 개요](#전체-개요)
2. [아키텍처 비교](#아키텍처-비교)
3. [코드 차이점 상세](#코드-차이점-상세)
4. [기능 차이점](#기능-차이점)
5. [파일 구조 차이](#파일-구조-차이)
6. [통일 방안](#통일-방안)

---

## 전체 개요

### 프로젝트 정보
- **mut4 (로컬)**: 로컬 환경에서 실행하는 v1.3 아키텍처 프로젝트
- **mut4_colab**: Google Colab 환경에서 실행하는 v1.3 아키텍처 프로젝트

### 핵심 결론
✅ **두 프로젝트 모두 v1.3 아키텍처를 사용합니다**
- 4-Fold Cross-Validation 구조
- Meta-Classifier (MLP/Ridge) 사용
- Fold별 logits 수집 및 메타 특징 생성

### 주요 차이점 요약
1. **모델 사용 방식**: 단일 모델 vs 멀티 모델 앙상블
2. **환경 변수 지원**: Colab 경로 설정을 위한 환경 변수 지원
3. **test.csv ID 읽기**: submission 파일 생성 시 실제 ID 사용
4. **문서**: Colab 전용 가이드 문서

---

## 아키텍처 비교

### v1.3 아키텍처 구조
```
[Base Models] → [4-Fold Training] → [Meta-Features] → [Meta-Classifier] → [Final Prediction]
```

### 두 프로젝트 모두 v1.3 아키텍처 준수

#### 공통점
- ✅ 4-Fold Cross-Validation 구조
- ✅ Meta-Classifier (MLP/Ridge) 사용
- ✅ Fold별 logits 수집 및 메타 특징 생성
- ✅ 동일한 모델 아키텍처 (HybridAvsH, AvsHModel 등)

#### 차이점
| 항목 | mut4 (로컬) | mut4_colab |
|------|------------|------------|
| **아키텍처 버전** | v1.3 기본 | v1.3 확장 |
| **모델 수** | 단일 모델만 | 여러 모델 가능 |
| **Meta-Classifier 입력** | `[N, 4]` | `[N, 4×M]` (M=모델 수) |

---

## 코드 차이점 상세

### 1. `utils/arguments.py`

#### mut4 (로컬)
```python
parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
# output_dir 인자 없음
```

#### mut4_colab
```python
# 환경 변수 지원 추가
default_data_dir = os.environ.get('DATA_DIR', './data')
default_output_dir = os.environ.get('OUTPUT_DIR', './outputs')

parser.add_argument('--data_dir', type=str, default=default_data_dir, 
                    help='Data directory (can be set via DATA_DIR env var)')
parser.add_argument('--output_dir', type=str, default=default_output_dir, 
                    help='Output directory (can be set via OUTPUT_DIR env var)')
```

**차이점**:
- ✅ 환경 변수 `DATA_DIR`, `OUTPUT_DIR` 지원
- ✅ `--output_dir` 인자 추가

---

### 2. `trainers/fold_trainer.py`

#### mut4 (로컬)
```python
def __init__(self, args):
    self.args = args
    self.logit_dir = "./outputs/fold_logits"  # 하드코딩된 경로
    os.makedirs(f"{self.logit_dir}/oof", exist_ok=True)
    os.makedirs(f"{self.logit_dir}/test", exist_ok=True)

def train_fold(self, fold_idx: int):
    # ...
    output_dir = f"./outputs/fold_{fold_idx}"  # 하드코딩된 경로
    os.makedirs(output_dir, exist_ok=True)
```

#### mut4_colab
```python
def __init__(self, args):
    self.args = args
    # 환경 변수 지원
    base_output_dir = getattr(args, 'output_dir', None) or os.environ.get('OUTPUT_DIR', './outputs')
    
    # 여러 모델 지원: 모델별로 logit 디렉토리 분리
    model_name = getattr(args, 'model_name', 'default')
    self.logit_dir = os.path.join(base_output_dir, "fold_logits", model_name)
    os.makedirs(f"{self.logit_dir}/oof", exist_ok=True)
    os.makedirs(f"{self.logit_dir}/test", exist_ok=True)

def train_fold(self, fold_idx: int):
    # ...
    # 환경 변수 지원
    base_output_dir = getattr(self.args, 'output_dir', None) or os.environ.get('OUTPUT_DIR', './outputs')
    output_dir = os.path.join(base_output_dir, f"fold_{fold_idx}")
    os.makedirs(output_dir, exist_ok=True)
```

**차이점**:
- ✅ 환경 변수 `OUTPUT_DIR` 지원
- ✅ 모델별 logit 디렉토리 분리 (`fold_logits/{model_name}/`)
- ✅ 멀티 모델 앙상블 지원을 위한 구조

**디렉토리 구조 차이**:
```
# mut4 (로컬)
outputs/
└── fold_logits/
    ├── oof/
    └── test/

# mut4_colab
outputs/
└── fold_logits/
    ├── HybridAvsH/
    │   ├── oof/
    │   └── test/
    ├── Gemma3InfoNCE/
    │   ├── oof/
    │   └── test/
    └── Qwen3InfoNCE/
        ├── oof/
        └── test/
```

---

### 3. `meta/meta_train.py`

#### mut4 (로컬)
```python
parser.add_argument('--logit_dir', type=str, default='./outputs/fold_logits', 
                    help='Directory containing fold logits')
parser.add_argument('--output_dir', type=str, default='./outputs/meta_features', 
                    help='Output directory for meta-features and model')
# model_names 인자 없음 (단일 모델만 지원)
```

#### mut4_colab
```python
# 환경 변수 지원
default_output_dir = os.environ.get('OUTPUT_DIR', './outputs')
default_logit_dir = os.path.join(default_output_dir, 'fold_logits')
default_meta_output_dir = os.path.join(default_output_dir, 'meta_features')

parser.add_argument('--logit_dir', type=str, default=default_logit_dir, 
                    help='Directory containing fold logits (can be set via OUTPUT_DIR env var)')
parser.add_argument('--output_dir', type=str, default=default_meta_output_dir, 
                    help='Output directory for meta-features and model (can be set via OUTPUT_DIR env var)')
parser.add_argument('--model_names', type=str, nargs='+', default=None, 
                    help='List of model names for multi-model ensemble (e.g., HybridAvsH Gemma3InfoNCE). If None, single model mode.')
```

**차이점**:
- ✅ 환경 변수 `OUTPUT_DIR` 지원
- ✅ `--model_names` 인자 추가 (멀티 모델 앙상블 지원)
- ✅ 여러 모델의 logits를 결합하여 Meta-Classifier 학습

**사용 예시**:
```bash
# mut4 (로컬) - 단일 모델
python meta/meta_train.py --meta_model_type mlp

# mut4_colab - 멀티 모델
python meta/meta_train.py --meta_model_type mlp \
    --model_names HybridAvsH Gemma3InfoNCE Qwen3InfoNCE \
    --hidden_layers 128 64 32  # 입력 차원 증가 (12 = 3×4)
```

---

### 4. `meta/meta_inference.py`

#### mut4 (로컬)
```python
parser.add_argument('--logit_dir', type=str, default='./outputs/fold_logits', ...)
parser.add_argument('--output_dir', type=str, default='./outputs/final_predictions', ...)
# model_names, input_dim 인자 없음
# test.csv ID 읽기 로직 없음 (range(len(preds)) 사용)
```

#### mut4_colab
```python
# 환경 변수 지원
default_output_dir = os.environ.get('OUTPUT_DIR', './outputs')
default_logit_dir = os.path.join(default_output_dir, 'fold_logits')
default_final_output_dir = os.path.join(default_output_dir, 'final_predictions')

parser.add_argument('--logit_dir', type=str, default=default_logit_dir, ...)
parser.add_argument('--output_dir', type=str, default=default_final_output_dir, ...)
parser.add_argument('--model_names', type=str, nargs='+', default=None, 
                    help='List of model names for multi-model ensemble (must match training)')
parser.add_argument('--input_dim', type=int, default=None, 
                    help='Input dimension for MLP (auto-detected if model_names provided, otherwise 4)')

# test.csv에서 실제 ID를 읽어서 submission 파일 생성
# (코드 내부에서 test.csv의 ID 컬럼 사용)
```

**차이점**:
- ✅ 환경 변수 `OUTPUT_DIR` 지원
- ✅ `--model_names` 인자 추가
- ✅ `--input_dim` 인자 추가 (자동 계산 가능)
- ✅ test.csv에서 실제 ID 읽기 (submission 파일 생성 시)

**submission 파일 생성 차이**:
```python
# mut4 (로컬)
submission_df = pd.DataFrame({
    'id': range(len(preds)),  # 0부터 시작하는 인덱스
    'generated': preds,
    'probability': probs
})

# mut4_colab
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
submission_df = pd.DataFrame({
    'id': test_df['id'].values,  # test.csv의 실제 ID
    'generated': preds,
    'probability': probs
})
```

---

### 5. `utils/logit_collector.py`

#### mut4 (로컬)
```python
class LogitCollector:
    """
    Collects fold logits and creates meta-features dataset
    
    Meta-features structure:
    - Shape: [Num_Samples] × 4
    - Column 0: OOF Logits (Fold 0 validation)
    - Column 1-3: Test Logits (Fold 1, 2, 3 validation)
    """
    
    def __init__(self, logit_dir: str = "./outputs/fold_logits"):
        self.logit_dir = logit_dir
        # 단일 모델만 지원
```

#### mut4_colab
```python
class LogitCollector:
    """
    Collects fold logits and creates meta-features dataset
    Supports both single model and multiple models
    """
    
    def __init__(self, logit_dir: str = "./outputs/fold_logits"):
        self.logit_dir = logit_dir
        # 멀티 모델 지원 (model_names 파라미터 추가)
    
    def collect_logits(self, model_names: Optional[List[str]] = None):
        """
        Collect logits for single or multiple models
        
        If model_names is None: single model mode (shape: [N, 4])
        If model_names is provided: multi-model mode (shape: [N, 4×M])
        """
```

**차이점**:
- ✅ `model_names` 파라미터 추가
- ✅ 여러 모델의 logits를 결합하는 로직 추가
- ✅ Meta-Features shape: `[N, 4]` → `[N, 4×M]` (M=모델 수)

---

## 기능 차이점

### 1. 멀티 모델 앙상블 지원

#### mut4 (로컬)
- ❌ 단일 모델만 지원
- Meta-Classifier 입력: `[N, 4]` (단일 모델의 4개 Fold logits)

#### mut4_colab
- ✅ 여러 모델 동시 사용 가능
- Meta-Classifier 입력: `[N, 4×M]` (M개 모델 × 4개 Fold)

**사용 예시**:
```python
# mut4_colab에서 3개 모델 사용
model_names = ['HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE']

# 각 모델별로 4-Fold 학습
for model_name in model_names:
    train_all_folds(cfg, model_name=model_name)

# Meta-Classifier 학습 (입력 차원: 12 = 3×4)
python meta/meta_train.py \
    --model_names HybridAvsH Gemma3InfoNCE Qwen3InfoNCE \
    --hidden_layers 128 64 32
```

---

### 2. 환경 변수 지원

#### mut4 (로컬)
- ❌ 환경 변수 지원 없음
- 하드코딩된 경로 사용 (`./outputs`, `./data`)

#### mut4_colab
- ✅ `DATA_DIR`, `OUTPUT_DIR` 환경 변수 지원
- Colab에서 Google Drive 경로 사용 가능

**사용 예시**:
```python
# Colab에서
import os
PROJECT_ROOT = '/content/drive/MyDrive/외부/멋사NLP/project/mut4_colab'
os.environ['DATA_DIR'] = f'{PROJECT_ROOT}/data'
os.environ['OUTPUT_DIR'] = f'{PROJECT_ROOT}/outputs'

# 코드에서 자동으로 환경 변수 사용
# arguments.py에서 default_data_dir = os.environ.get('DATA_DIR', './data')
```

---

### 3. test.csv ID 읽기

#### mut4 (로컬)
- ❌ test.csv ID 읽기 없음
- submission 파일의 ID: `range(len(preds))` (0부터 시작)

#### mut4_colab
- ✅ test.csv에서 실제 ID 읽기
- submission 파일의 ID: test.csv의 실제 ID 값

**차이점**:
```python
# mut4 (로컬)
submission_df = pd.DataFrame({
    'id': range(len(preds)),  # [0, 1, 2, 3, ...]
    'generated': preds,
    'probability': probs
})

# mut4_colab
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
submission_df = pd.DataFrame({
    'id': test_df['id'].values,  # test.csv의 실제 ID
    'generated': preds,
    'probability': probs
})
```

---

## 파일 구조 차이

### mut4 (로컬)에만 있는 파일
- 없음 (기본 구조만 있음)

### mut4_colab에만 있는 파일

#### 1. `COLAB_SETUP.md`
- Colab 환경 설정 및 실행 가이드
- Google Drive 마운트 방법
- 셀별 실행 가이드
- 경로 설정 방법

#### 2. `AVAILABLE_MODELS.md`
- 사용 가능한 모델 목록
- 모델별 특징 비교
- 모델 선택 가이드

#### 3. `MULTI_MODEL_GUIDE.md`
- 여러 모델 동시 사용 가이드
- 멀티 모델 앙상블 구조
- 사용 예시 및 주의사항

#### 4. `PARAMETER.md`
- 파라미터 조정 가이드
- 모든 파라미터 상세 설명
- 조정 전략 및 예시

### 공통 파일
- `01_PIPELINE.md`: 거의 동일 (mut4_colab에 데이터 흐름도 추가)
- `02_MODELS.md`: 동일
- `03_IMPLEMENTATION.md`: 동일

---

## requirements.txt 차이

### mut4 (로컬)
```txt
# 모든 패키지 명시 (전체 버전 포함)
torch==2.7.1
transformers==4.53.2
numpy==2.0.2
pandas==2.3.0
# ... 모든 의존성 명시
```

### mut4_colab
```txt
# Colab 기본 패키지 제외
# Colab에 기본 설치: torch, numpy, pandas, matplotlib, seaborn, scikit-learn, scipy 등

# 실제로 필요한 패키지만 포함
transformers==4.53.2
datasets==4.0.0
accelerate==1.12.0
# ... Colab에 없는 패키지만 명시
```

**차이점**:
- mut4: 모든 패키지 명시 (로컬 환경에서 완전한 재현성)
- mut4_colab: Colab 기본 패키지 제외 (중복 설치 방지)

---

## 통일 방안

로컬 프로젝트(mut4)를 mut4_colab과 통일하려면 다음 수정이 필요합니다.

### 1. `utils/arguments.py` 수정

```python
# 환경 변수 지원 추가
default_data_dir = os.environ.get('DATA_DIR', './data')
default_output_dir = os.environ.get('OUTPUT_DIR', './outputs')

parser.add_argument('--data_dir', type=str, default=default_data_dir, 
                    help='Data directory (can be set via DATA_DIR env var)')
parser.add_argument('--output_dir', type=str, default=default_output_dir, 
                    help='Output directory (can be set via OUTPUT_DIR env var)')
```

### 2. `trainers/fold_trainer.py` 수정

```python
def __init__(self, args):
    self.args = args
    # 환경 변수 지원
    base_output_dir = getattr(args, 'output_dir', None) or os.environ.get('OUTPUT_DIR', './outputs')
    
    # 단일 모델 모드 (멀티 모델 지원은 선택적)
    # 옵션 1: 단일 모델 모드 (기존과 동일)
    self.logit_dir = os.path.join(base_output_dir, "fold_logits")
    # 옵션 2: 멀티 모델 모드 (확장)
    # model_name = getattr(args, 'model_name', 'default')
    # self.logit_dir = os.path.join(base_output_dir, "fold_logits", model_name)
    
    os.makedirs(f"{self.logit_dir}/oof", exist_ok=True)
    os.makedirs(f"{self.logit_dir}/test", exist_ok=True)

def train_fold(self, fold_idx: int):
    # ...
    base_output_dir = getattr(self.args, 'output_dir', None) or os.environ.get('OUTPUT_DIR', './outputs')
    output_dir = os.path.join(base_output_dir, f"fold_{fold_idx}")
    os.makedirs(output_dir, exist_ok=True)
```

### 3. `meta/meta_train.py` 수정

```python
# 환경 변수 지원 추가
default_output_dir = os.environ.get('OUTPUT_DIR', './outputs')
default_logit_dir = os.path.join(default_output_dir, 'fold_logits')
default_meta_output_dir = os.path.join(default_output_dir, 'meta_features')

parser.add_argument('--logit_dir', type=str, default=default_logit_dir, ...)
parser.add_argument('--output_dir', type=str, default=default_meta_output_dir, ...)

# 멀티 모델 지원 (선택적)
parser.add_argument('--model_names', type=str, nargs='+', default=None, 
                    help='List of model names for multi-model ensemble. If None, single model mode.')
```

### 4. `meta/meta_inference.py` 수정

```python
# 환경 변수 지원 추가
default_output_dir = os.environ.get('OUTPUT_DIR', './outputs')
default_logit_dir = os.path.join(default_output_dir, 'fold_logits')
default_final_output_dir = os.path.join(default_output_dir, 'final_predictions')

parser.add_argument('--logit_dir', type=str, default=default_logit_dir, ...)
parser.add_argument('--output_dir', type=str, default=default_final_output_dir, ...)

# 멀티 모델 지원 및 test.csv ID 읽기 추가
parser.add_argument('--model_names', type=str, nargs='+', default=None, ...)
parser.add_argument('--input_dim', type=int, default=None, ...)

# test.csv에서 ID 읽는 로직 추가
data_dir = os.environ.get('DATA_DIR', './data')
test_df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
submission_df = pd.DataFrame({
    'id': test_df['id'].values,  # 실제 ID
    'generated': preds,
    'probability': probs
})
```

### 5. `utils/logit_collector.py` 수정 (멀티 모델 지원 시)

```python
def collect_logits(self, model_names: Optional[List[str]] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Collect logits for single or multiple models
    
    Args:
        model_names: List of model names. If None, single model mode.
    
    Returns:
        Tuple of (meta_features, labels)
        - Single model: meta_features shape [N, 4]
        - Multi model: meta_features shape [N, 4×M] (M=len(model_names))
    """
    if model_names is None:
        # 단일 모델 모드 (기존 로직)
        # ...
    else:
        # 멀티 모델 모드
        # 각 모델별 logits 수집 후 결합
        # ...
```

---

## 요약

### 핵심 차이점

1. **모델 사용 방식**
   - mut4: 단일 모델만 (v1.3 기본)
   - mut4_colab: 멀티 모델 앙상블 지원 (v1.3 확장)

2. **환경 변수 지원**
   - mut4: 하드코딩된 경로
   - mut4_colab: `DATA_DIR`, `OUTPUT_DIR` 환경 변수 지원

3. **test.csv ID 읽기**
   - mut4: 인덱스 사용 (0부터 시작)
   - mut4_colab: 실제 ID 사용

4. **문서**
   - mut4_colab에 Colab 전용 가이드 문서 추가

### v1.3 아키텍처 준수

✅ **두 프로젝트 모두 v1.3 아키텍처를 완전히 준수합니다**
- 4-Fold Cross-Validation
- Meta-Classifier 사용
- Fold별 logits 수집

### 통일 권장사항

환경 설정 차이(로컬 vs Colab)를 제외하고, 위 수정사항을 적용하면 두 프로젝트를 완전히 통일할 수 있습니다. 특히:
- 환경 변수 지원 추가 (로컬에서도 유용)
- test.csv ID 읽기 (더 정확한 submission)
- 멀티 모델 앙상블 지원 (선택적, 필요시 사용)

---

## 참고사항

- 환경 설정 차이(로컬 vs Colab)는 유지하는 것이 좋습니다
- 멀티 모델 앙상블은 선택적 기능이므로, 단일 모델만 사용해도 문제없습니다
- 환경 변수 지원은 로컬에서도 유용합니다 (경로 변경 용이)

