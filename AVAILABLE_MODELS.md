# 사용 가능한 모델 목록

## 멀티 모델로 사용 가능한 모델

다음 모델들을 멀티 모델 앙상블에 사용할 수 있습니다:

### 1. **HybridAvsH** (권장)
- **설명**: AvsHModel + InfoNCE Loss 통합 모델
- **특징**:
  - 문단 단위 계층적 구조
  - Contrastive Learning (InfoNCE Loss) 지원
  - BPR Loss 지원
- **사용 예시**:
  ```python
  model_name = "HybridAvsH"
  ```

### 2. **AvsHModel**
- **설명**: 기본 AvsHModel (SKKUAI)
- **특징**:
  - 문단 단위 계층적 구조
  - Transformer Encoder 사용
- **사용 예시**:
  ```python
  model_name = "AvsHModel"
  ```

### 3. **Gemma3InfoNCE**
- **설명**: Gemma3 기반 모델 + InfoNCE Loss
- **특징**:
  - Gemma3 사전 학습 모델 사용
  - Contrastive Learning 지원
- **사용 예시**:
  ```python
  model_name = "Gemma3InfoNCE"
  ```

### 4. **Qwen3InfoNCE**
- **설명**: Qwen3 기반 모델 + InfoNCE Loss
- **특징**:
  - Qwen3 사전 학습 모델 사용
  - Contrastive Learning 지원
- **사용 예시**:
  ```python
  model_name = "Qwen3InfoNCE"
  ```

---

## 멀티 모델 조합 예시

### 예시 1: 2개 모델 조합
```python
model_names = ['HybridAvsH', 'Gemma3InfoNCE']
# Meta-Classifier 입력 차원: 2 × 4 = 8
```

### 예시 2: 3개 모델 조합
```python
model_names = ['HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE']
# Meta-Classifier 입력 차원: 3 × 4 = 12
```

### 예시 3: 4개 모델 조합 (모든 모델)
```python
model_names = ['HybridAvsH', 'AvsHModel', 'Gemma3InfoNCE', 'Qwen3InfoNCE']
# Meta-Classifier 입력 차원: 4 × 4 = 16
```

---

## 모델별 특징 비교

| 모델 | 계층 구조 | InfoNCE | BPR Loss | 권장 사용 |
|------|----------|---------|----------|----------|
| HybridAvsH | ✅ | ✅ | ✅ | ⭐⭐⭐ (가장 권장) |
| AvsHModel | ✅ | ❌ | ✅ | ⭐⭐ |
| Gemma3InfoNCE | ❌ | ✅ | ❌ | ⭐⭐ |
| Qwen3InfoNCE | ❌ | ✅ | ❌ | ⭐⭐ |

---

## 사용 권장사항

### 단일 모델 사용 시
- **HybridAvsH** 권장 (가장 강력한 성능)

### 멀티 모델 사용 시
- **2개 모델**: `['HybridAvsH', 'Gemma3InfoNCE']` 또는 `['HybridAvsH', 'Qwen3InfoNCE']`
- **3개 모델**: `['HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE']`
- **4개 모델**: 모든 모델 사용 (시간이 많이 걸림)

### 조합 전략
1. **다양성**: 서로 다른 아키텍처 조합
   - 예: `HybridAvsH` (계층적) + `Gemma3InfoNCE` (표준)
2. **성능**: 성능이 좋은 모델 우선
   - 예: `HybridAvsH` + `Gemma3InfoNCE`
3. **시간**: 학습 시간 고려
   - 2-3개 모델이 일반적으로 좋은 균형

---

## 주의사항

1. **모델 순서**: Meta-Classifier 학습과 추론 시 모델 순서가 **반드시 동일**해야 함
2. **입력 차원**: 모델 수에 따라 Meta-Classifier 입력 차원 자동 계산
   - 1개 모델: 4
   - 2개 모델: 8
   - 3개 모델: 12
   - 4개 모델: 16
3. **학습 시간**: 모델 수만큼 학습 시간 증가
4. **메모리**: Meta-Classifier 입력 차원 증가로 메모리 사용량 증가

---

## 전체 사용 예시

```python
from trainers.fold_trainer import train_multiple_models
from types import SimpleNamespace

# 기본 설정
base_cfg = SimpleNamespace(
    data_dir='./data',
    is_kfold=True,
    k_fold=4,
    save_dir="v1.3_fold",
    embedding_model="kykim/funnel-kor-base",
    num_train_epochs=10,
    per_device_train_batch_size=8,
    max_length=256,
    # ... 나머지 설정
)

# 멀티 모델 학습
model_names = ['HybridAvsH', 'Gemma3InfoNCE', 'Qwen3InfoNCE']
train_multiple_models(model_names, base_cfg)

# Meta-Classifier 학습
# --model_names HybridAvsH Gemma3InfoNCE Qwen3InfoNCE
# 입력 차원: 12 (자동 계산)
```

