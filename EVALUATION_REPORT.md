# 학습 및 평가 리포트 형식 가이드

이 문서는 프로젝트에서 출력되는 모든 학습 및 평가 리포트의 형식을 정리합니다.

---

## 📊 1. Fold 학습 중 리포트 (각 Epoch)

### 출력 위치
- `trainers/fold_trainer.py` → `HybridTrainer` → Transformers `Trainer`
- `utils/compute_metrics.py`의 `compute_metrics` 함수에서 계산

### 출력 형식
각 Epoch마다 Transformers Trainer가 자동으로 출력합니다:

```
{'eval_loss': 0.6523, 
 'eval_accuracy': 0.5800, 
 'eval_f1_score': 0.5700, 
 'eval_precision': 0.6000, 
 'eval_recall': 0.5800, 
 'eval_roc_auc': 0.6500, 
 'epoch': 1.0}
```

### 포함 지표
- `eval_loss`: 검증 손실
- `eval_accuracy`: 정확도
- `eval_f1_score`: F1-Score (macro average)
- `eval_precision`: Precision (macro average)
- `eval_recall`: Recall (macro average)
- `eval_roc_auc`: ROC-AUC Score

### 특징
- 딕셔너리 형식으로 출력
- 숫자 지표만 표시
- 클래스별 상세 분석 없음

---

## 📋 2. Fold 학습 완료 후 리포트

### 출력 위치
- `trainers/fold_trainer.py` → `train_fold()` 메서드

### 출력 형식
```
============================================================
Training Fold 0
============================================================

... (학습 과정) ...

Collecting logits for Fold 0...
Saved OOF logits to ./outputs/fold_logits/oof/fold0_logits.npy
✅ Fold 0 completed successfully
```

### 특징
- 간단한 완료 메시지
- Logit 저장 경로 표시
- 상세 지표 없음

---

## 🎯 3. Meta-Classifier 학습 후 리포트

### 출력 위치
- `meta/meta_train.py` → `main()` 함수의 평가 부분

### 출력 형식

#### 3.1 기본 지표
```
4. Evaluating meta-classifier...
ROC-AUC: 0.6500
Accuracy: 0.5800
F1-Score: 0.5700
```

#### 3.2 상세 리포트 (Classification Report)
```
📋 상세 리포트 (0: Human, 1: AI)
------------------------------------------------------------
              precision    recall  f1-score   support

       Human       0.63      0.39      0.48      1000
          AI       0.56      0.77      0.65      1000

    accuracy                           0.58      2000
   macro avg       0.60      0.58      0.57      2000
weighted avg       0.60      0.58      0.57      2000
```

#### 3.3 Confusion Matrix
```
------------------------------------------------------------
🔍 Confusion Matrix
[[391 609]
 [226 774]]
( [TN FP]
  [FN TP] )
```

### 포함 정보
- **기본 지표**: ROC-AUC, Accuracy, F1-Score
- **클래스별 지표**: Human/AI 각각의 precision, recall, f1-score, support
- **평균 지표**: macro avg, weighted avg
- **Confusion Matrix**: 
  - TN (True Negative): 391 - 실제 Human, 예측 Human
  - FP (False Positive): 609 - 실제 Human, 예측 AI
  - FN (False Negative): 226 - 실제 AI, 예측 Human
  - TP (True Positive): 774 - 실제 AI, 예측 AI

---

## 📈 4. 종합 평가 리포트 (Fold별 + Meta-Classifier)

### 출력 위치
- `utils/evaluate_folds.py` → `generate_evaluation_report()` 함수

### 실행 방법
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

### 출력 형식

#### 4.1 Fold별 요약 지표 (DataFrame)
```
1. Evaluating all folds...
   fold      type  accuracy  f1_score  precision  recall  roc_auc
fold_0       OOF     0.5800    0.5700     0.6000  0.5800   0.6500
fold_1      Test     0.5900    0.5800     0.6100  0.5900   0.6600
fold_2      Test     0.5850    0.5750     0.6050  0.5850   0.6550
fold_3      Test     0.5750    0.5650     0.5950  0.5750   0.6450
```

#### 4.2 Fold별 상세 리포트
각 Fold마다 다음 형식으로 출력:

```
============================================================
Fold별 상세 리포트
============================================================

fold_0 (OOF):

📋 상세 리포트 (0: Human, 1: AI)
------------------------------------------------------------
              precision    recall  f1-score   support

       Human       0.63      0.39      0.48      1000
          AI       0.56      0.77      0.65      1000

    accuracy                           0.58      2000
   macro avg       0.60      0.58      0.57      2000
weighted avg       0.60      0.58      0.57      2000

🔍 Confusion Matrix
[[391 609]
 [226 774]]
( [TN FP]
  [FN TP] )

fold_1 (Test):
... (동일한 형식) ...
```

#### 4.3 Meta-Classifier 평가 (선택적)
Meta-Classifier 모델 경로가 제공된 경우:

```
2. Evaluating meta-classifier...

📋 상세 리포트 (0: Human, 1: AI)
------------------------------------------------------------
              precision    recall  f1-score   support

       Human       0.65      0.42      0.51      2000
          AI       0.58      0.78      0.67      2000

    accuracy                           0.60      4000
   macro avg       0.62      0.60      0.59      4000
weighted avg       0.62      0.60      0.59      4000

------------------------------------------------------------
🔍 Confusion Matrix
[[840 1160]
 [440 1560]]
( [TN FP]
  [FN TP] )

Meta-Classifier Metrics:
  accuracy: 0.6000
  f1_score: 0.5900
  precision: 0.6200
  recall: 0.6000
  roc_auc: 0.6800
```

### 생성되는 파일
- `fold_metrics.csv`: Fold별 상세 지표 (CSV 형식)
- `evaluation_summary.json`: 종합 요약 (JSON 형식)
- `metrics_comparison.png`: 성능 비교 차트
- `roc_curves.png`: ROC 곡선 비교
- `metrics_table.png`: 성능 테이블

---

## 📝 리포트 형식 비교

| 리포트 유형 | 기본 지표 | 클래스별 분석 | Confusion Matrix | 출력 위치 |
|-----------|---------|------------|-----------------|----------|
| **학습 중 (Epoch)** | ✅ | ❌ | ❌ | 콘솔 (Trainer) |
| **Fold 완료** | ❌ | ❌ | ❌ | 콘솔 (간단 메시지) |
| **Meta-Classifier** | ✅ | ✅ | ✅ | 콘솔 |
| **종합 평가** | ✅ | ✅ | ✅ | 콘솔 + 파일 |

---

## 🔍 Confusion Matrix 해석 가이드

### 형식
```
[[TN FP]
 [FN TP]]
```

### 의미
- **TN (True Negative)**: 실제 Human(0), 예측 Human(0) - 정확한 음성 예측
- **FP (False Positive)**: 실제 Human(0), 예측 AI(1) - 잘못된 양성 예측 (Type I Error)
- **FN (False Negative)**: 실제 AI(1), 예측 Human(0) - 잘못된 음성 예측 (Type II Error)
- **TP (True Positive)**: 실제 AI(1), 예측 AI(1) - 정확한 양성 예측

### 성능 해석
- **높은 TN, TP**: 좋은 성능
- **높은 FP**: Human을 AI로 잘못 분류 (False Alarm)
- **높은 FN**: AI를 Human으로 잘못 분류 (Missed Detection)

---

## 💡 사용 팁

1. **학습 중 모니터링**: Epoch별 기본 지표로 학습 진행 상황 확인
2. **Fold별 분석**: 종합 평가 리포트로 각 Fold의 성능 차이 확인
3. **최종 평가**: Meta-Classifier 학습 후 상세 리포트로 최종 성능 확인
4. **비교 분석**: `evaluation_summary.json`과 시각화 파일로 전체 성능 비교

---

## 📌 참고사항

- 모든 리포트는 콘솔에 출력됩니다
- 종합 평가 리포트는 추가로 파일로도 저장됩니다
- Classification Report와 Confusion Matrix는 클래스별 상세 분석을 제공합니다
- 숫자 지표만 필요한 경우 기본 지표만 확인하면 됩니다

