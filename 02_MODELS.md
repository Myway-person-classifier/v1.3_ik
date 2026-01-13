# v1.3 모델 설명

## 📋 목차
1. [모델 구조 개요](#모델-구조-개요)
2. [Base Models (v1.2 구조)](#base-models-v12-구조)
3. [Hybrid Model 구성](#hybrid-model-구성)
4. [Meta-Classifier](#meta-classifier)
5. [모델 설정 파라미터](#모델-설정-파라미터)
6. [모델 선택 가이드](#모델-선택-가이드)
7. [권장 구성](#권장-구성)

---

## 모델 구조 개요

v1.3은 **2단계 모델 구조**를 사용합니다:
1. **Base Models (v1.2)**: 4-Fold로 각각 학습
2. **Meta-Classifier**: Fold별 예측값을 통합하여 최종 예측

```
[Base Models] → [4-Fold Training] → [Meta-Features] → [Meta-Classifier] → [Final Prediction]
```

---

## Base Models (v1.2 구조)

### 1. SKKUAI AvsHModel (문단 계층 구조)

**모델 아키텍처**:
```
문서 → 문단 분할 (최대 10개) 
  → Paragraph Encoder (LLM Backbone) 
  → Transformer Encoder (AvsH) 
  → Classifier (nn.Linear)
```

**Embedding Backbone 옵션**:
- `kykim/funnel-kor-base` (기본값, 110M 파라미터)
  - **특징**: 한국어 특화, 효율적
  - **용도**: Phase 1, 2, 3에서 주로 사용
  
- `kykim/bert-kor-base` (대안)
  - **특징**: 한국어 BERT 기반
  - **용도**: Phase 3, 4에서 사용

**AvsHModel 구성 요소**:
- **Paragraph Encoder**: LLM Backbone (Frozen 또는 LoRA)
  - Hidden Size: Backbone 모델에 따라 자동 설정
  - Max Paragraphs: 10개
  - Max Length: 256-512 토큰

- **Transformer Encoder**: 문단 간 관계 학습
  - Layers: 4 (기본값)
  - Heads: 8 (기본값)
  - Feedforward Dim: 2048 (기본값)
  - Dropout: 0.0 (기본값)

- **Learnable CLS Token**: 전역 문맥 표현

- **Classifier**: 
  - Output: Binary (0: Human, 1: AI)

**Loss Functions**:
- BCE Loss (기본)
- BPR Loss (선택적, `--use_bpr_loss`)
- InfoNCE Loss (선택적, v1.3에서 추가)

---

### 2. AIGT InfoNCE Loss Models

AIGT 프로젝트에서 제공하는 **Contrastive Learning** 지원 모델들:

#### 2.1 Gemma3-12B (InfoNCE)

**모델 정보**:
- **모델명**: `google/gemma-3-12b-it`
- **HuggingFace**: https://huggingface.co/google/gemma-3-12b-it
- **파라미터**: 12B
- **특징**: InfoNCE Loss로 Contrastive Learning 지원

**구현 파일**:
- `models/gemma3_seqcls_infonce.py`

**주요 기능**:
- Standard Loss (BCE/CE/MSE)
- Contrastive Loss (InfoNCE) with temperature
- Adversarial Training (선택적)

**Loss 계산**:
```python
Total Loss = Standard Loss + λ_cl × InfoNCE Loss
```

#### 2.2 Qwen3-14B (InfoNCE)

**모델 정보**:
- **모델명**: `Qwen/Qwen3-14B`
- **HuggingFace**: https://huggingface.co/Qwen/Qwen3-14B
- **파라미터**: 14B
- **특징**: InfoNCE Loss로 Contrastive Learning 지원

**구현 파일**:
- `models/qwen3_seqcls_infonce.py`

**주요 기능**:
- Standard Loss (BCE/CE/MSE)
- Contrastive Loss (InfoNCE) with temperature
- Adversarial Training (선택적)

---

## Hybrid Model 구성

v1.3에서는 **AvsHModel + InfoNCE Loss**를 결합합니다:

### HybridAvsHModel 구조

```python
class HybridAvsHModel(AvsHModel):
    """
    AvsHModel에 InfoNCE Loss를 추가한 통합 모델
    """
    
    # AvsHModel 구조 상속
    - Embedding Backbone (Funnel/BERT)
    - Transformer Encoder
    - Classifier
    
    # 추가 기능
    - Contrastive Learning 지원
    - InfoNCE Loss 계산
```

**선택 가능한 구성**:

| 구성 | Embedding Backbone | InfoNCE Loss | 추천 용도 |
|------|-------------------|--------------|-----------|
| **Option 1** | `kykim/funnel-kor-base` | ❌ 없음 | 빠른 학습, 경량화 |
| **Option 2** | `kykim/funnel-kor-base` | ✅ Gemma3/Qwen3 | 균형잡힌 성능 |
| **Option 3** | `kykim/bert-kor-base` | ✅ Gemma3/Qwen3 | 높은 성능 |

---

## Meta-Classifier

4-Fold 학습 후 **Fold별 예측값(Logits)**을 메타 특징으로 사용:

### 메타 특징 데이터셋

```
Shape: [Num_Samples] × 4

Column 0: OOF Logits (Fold 0, Validation 데이터)
Column 1: Test Logits (Fold 1, Validation 데이터)
Column 2: Test Logits (Fold 2, Validation 데이터)
Column 3: Test Logits (Fold 3, Validation 데이터)
```

### Meta-Classifier 옵션

#### Option 1: MLP (Multi-Layer Perceptron)

**구조**:
```
Input (4 features: Fold Logits)
  ↓
Hidden Layer 1 (64 units, ReLU, Dropout=0.2)
  ↓
Hidden Layer 2 (32 units, ReLU, Dropout=0.2)
  ↓
Output (1 unit, Sigmoid)
```

**특징**:
- 비선형 관계 학습 가능
- 복잡한 패턴 캡처

#### Option 2: Ridge Regression (L2 정규화)

**구조**:
```
Ridge Regression with L2 Regularization
- Cross-Validation으로 최적 alpha 선택
- Alpha 범위: 10^-4 ~ 10^2
```

**특징**:
- 빠른 학습
- 과적합 방지
- 해석 가능

**선택 기준**:
- **MLP**: 복잡한 Fold 간 상호작용 학습 필요 시
- **Ridge**: 빠른 학습 및 안정적인 성능 필요 시

---

## 모델 설정 파라미터

### AvsHModel 파라미터

```python
# Embedding Backbone
embedding_model = "kykim/funnel-kor-base"  # 또는 "kykim/bert-kor-base"

# Transformer Encoder
num_layers = 4              # Transformer Encoder 레이어 수
num_heads = 8               # Attention 헤드 수
dim_feedforward = 2048      # Feedforward 차원
dropout = 0.0               # Dropout 비율

# 학습 파라미터
max_length = 256            # 문단당 최대 토큰 수
max_paragraphs = 10         # 문서당 최대 문단 수
learning_rate = 3e-5        # 학습률
num_train_epochs = 10       # 에폭 수
```

### InfoNCE Loss 파라미터

```python
# InfoNCE Loss 활성화
use_infonce_loss = True     # InfoNCE Loss 사용 여부
lambda_cl = 0.1             # InfoNCE Loss 가중치
temperature = 0.07          # Temperature 파라미터

# Contrastive Learning
contrastive_labels = labels # 대조 학습용 레이블
```

### Meta-Classifier 파라미터

```python
# MLP 옵션
meta_model_type = "mlp"
hidden_layers = [64, 32]    # Hidden layer 크기
dropout = 0.2               # Dropout 비율
activation = "relu"         # 활성화 함수

# Ridge 옵션
meta_model_type = "ridge"
alphas = np.logspace(-4, 2, 25)  # 정규화 강도 범위
cv = 5                            # Cross-validation folds
```

---

## 모델별 메모리 사용량

| 모델 구성 | GPU 메모리 (학습) | GPU 메모리 (추론) | 배치 크기 (권장) |
|-----------|-------------------|-------------------|------------------|
| **Funnel + AvsH** | 4-6 GB | 2-3 GB | 16-32 |
| **BERT + AvsH** | 5-7 GB | 3-4 GB | 16-32 |
| **Funnel + AvsH + InfoNCE** | 6-8 GB | 3-4 GB | 8-16 |
| **BERT + AvsH + InfoNCE** | 7-10 GB | 4-5 GB | 8-16 |

**참고**: InfoNCE Loss는 Contrastive Learning을 위해 추가 메모리가 필요합니다.

---

## 모델 선택 가이드

### 질문 1: 학습 시간이 제한적인가?
- ✅ **Yes** → 구성 1 (경량화)
- ❌ **No** → 다음 질문

### 질문 2: 최고 성능이 필요한가?
- ✅ **Yes** → 구성 3 (고성능)
- ❌ **No** → 구성 2 (균형)

### 질문 3: GPU 메모리가 충분한가?
- ✅ **Yes (≥16GB)** → BERT Backbone + InfoNCE
- ❌ **No (<16GB)** → Funnel Backbone (InfoNCE 선택적)

### 질문 4: 복잡한 패턴 학습이 필요한가?
- ✅ **Yes** → MLP Meta-Classifier
- ❌ **No** → Ridge Meta-Classifier

---

## 권장 구성

### 구성 1: 경량화 (학습 시간 최소화)

```
Base Model:
  - Embedding: kykim/funnel-kor-base
  - InfoNCE Loss: ❌ 없음
  - Loss: BCE + BPR

Meta-Classifier:
  - Type: Ridge Regression
  - 장점: 빠른 학습, 안정적 성능
```

**예상 학습 시간**: 6-8시간 (4-Fold 병렬)

### 구성 2: 균형 (성능-시간 균형)

```
Base Model:
  - Embedding: kykim/funnel-kor-base
  - InfoNCE Loss: ✅ Gemma3/Qwen3
  - Loss: BCE + BPR + InfoNCE (λ_cl=0.1)

Meta-Classifier:
  - Type: MLP (2 hidden layers)
  - 장점: 비선형 관계 학습
```

**예상 학습 시간**: 8-10시간 (4-Fold 병렬)

### 구성 3: 고성능 (최대 정확도)

```
Base Model:
  - Embedding: kykim/bert-kor-base
  - InfoNCE Loss: ✅ Gemma3/Qwen3
  - Loss: BCE + BPR + InfoNCE (λ_cl=0.2)

Meta-Classifier:
  - Type: MLP (2-3 hidden layers)
  - 장점: 복잡한 패턴 학습
```

**예상 학습 시간**: 10-12시간 (4-Fold 병렬)

---

## 전체 모델 요약 테이블

### Base Models (Fold별 학습)

| 모델 타입 | 모델명 | 파라미터 | 용도 | Loss |
|-----------|--------|----------|------|------|
| **AvsHModel** | `kykim/funnel-kor-base` | 110M | Embedding Backbone | BCE + BPR |
| **AvsHModel** | `kykim/bert-kor-base` | 110M | Embedding Backbone | BCE + BPR |
| **Gemma3** | `google/gemma-3-12b-it` | 12B | InfoNCE Loss (선택) | BCE + InfoNCE |
| **Qwen3** | `Qwen/Qwen3-14B` | 14B | InfoNCE Loss (선택) | BCE + InfoNCE |

### Meta-Classifier

| 분류기 타입 | 입력 차원 | 출력 | 용도 |
|-------------|-----------|------|------|
| **MLP** | 4 (Fold Logits) | 1 (Binary) | 최종 예측 |
| **Ridge** | 4 (Fold Logits) | 1 (Binary) | 최종 예측 |

---

## 요약

### v1.3에서 사용하는 모델

1. **Base Model (v1.2 구조)**: 4개 Fold에서 각각 학습
   - AvsHModel (문단 계층 구조)
   - Embedding Backbone: Funnel/BERT
   - Optional: InfoNCE Loss (Gemma3/Qwen3)

2. **Meta-Classifier**: Fold 예측값 통합
   - MLP 또는 Ridge Regression
   - Input: 4개 Fold Logits
   - Output: 최종 예측 (Binary)

### 권장 구성
- **균형형**: Funnel + InfoNCE + MLP
- **경량형**: Funnel + Ridge
- **고성능형**: BERT + InfoNCE + MLP

