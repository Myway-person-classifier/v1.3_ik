# 파라미터 조정 가이드

학습, 추론, 평가, 결과 분석 시 조정 가능한 모든 파라미터를 정리한 문서입니다.

---

## Part 1: Fold 학습 파라미터

### 1.1 학습 기본 파라미터

#### `num_train_epochs` (기본값: 10)
- **설명**: 전체 데이터셋을 몇 번 반복 학습할지 설정
- **영향**:
  - **낮게 설정 (5-10)**: 빠른 학습, 과소적합 가능성
  - **적절한 값 (10-20)**: 일반적으로 좋은 성능
  - **높게 설정 (20-50)**: 더 나은 성능 가능, 과적합 위험, 학습 시간 증가
- **권장 범위**: 10-20
- **조정 팁**:
  - Early stopping이 활성화되어 있어 과적합 방지됨
  - Validation loss가 증가하기 시작하면 자동으로 중단
  - 더 높은 epoch는 더 큰 데이터셋이나 복잡한 모델에 유용
- **예시**:
  ```python
  cfg.num_train_epochs = 15  # 기본값 10에서 증가
  ```

#### `learning_rate` (기본값: 3e-5)
- **설명**: 모델 가중치 업데이트 속도
- **영향**:
  - **너무 높음 (1e-4 이상)**: 학습 불안정, 발산 가능
  - **적절한 값 (1e-5 ~ 5e-5)**: 안정적인 학습
  - **너무 낮음 (1e-6 이하)**: 학습 속도 느림, 지역 최적해에 갇힐 수 있음
- **권장 범위**: 1e-5 ~ 5e-5
- **조정 팁**:
  - Transformer 모델에는 보통 작은 learning rate 사용
  - 큰 모델일수록 작은 learning rate 권장
  - Learning rate scheduler와 함께 사용 (warmup_ratio)
- **예시**:
  ```python
  cfg.learning_rate = 5e-5  # 더 빠른 학습
  cfg.learning_rate = 1e-5  # 더 안정적인 학습
  ```

#### `warmup_ratio` (기본값: 0.1)
- **설명**: 전체 학습의 몇 %를 warmup에 사용할지 (learning rate 점진적 증가)
- **영향**:
  - **낮게 (0.05)**: 빠른 warmup, 초기 불안정 가능
  - **적절한 값 (0.1-0.2)**: 안정적인 학습 시작
  - **높게 (0.3 이상)**: 너무 느린 시작, 학습 시간 낭비
- **권장 범위**: 0.1-0.2
- **조정 팁**:
  - 큰 모델이나 작은 데이터셋에서는 더 긴 warmup 유용
  - Transformer 모델에는 일반적으로 0.1 사용
- **예시**:
  ```python
  cfg.warmup_ratio = 0.15  # 더 긴 warmup
  ```

#### `weight_decay` (기본값: 1e-3)
- **설명**: L2 정규화 강도 (과적합 방지)
- **영향**:
  - **낮게 (1e-4)**: 약한 정규화, 과적합 가능
  - **적절한 값 (1e-3 ~ 1e-2)**: 적절한 정규화
  - **높게 (1e-1 이상)**: 너무 강한 정규화, 성능 저하
- **권장 범위**: 1e-4 ~ 1e-2
- **조정 팁**:
  - 데이터가 적을수록 높은 weight_decay 권장
  - 모델이 복잡할수록 높은 weight_decay 유용
- **예시**:
  ```python
  cfg.weight_decay = 5e-3  # 더 강한 정규화
  ```

---

### 1.2 배치 사이즈 및 메모리 관련

#### `per_device_train_batch_size` (기본값: 8)
- **설명**: GPU당 학습 배치 사이즈
- **영향**:
  - **작게 (4)**: 메모리 절약, 더 많은 업데이트, 학습 안정적
  - **적절한 값 (8-16)**: 메모리와 성능의 균형
  - **크게 (32 이상)**: 빠른 학습, 메모리 부족 가능
- **권장 범위**: GPU 메모리에 따라 조정
  - T4 (16GB): 8-16
  - V100 (16GB): 16-32
  - A100 (40GB): 32-64
- **조정 팁**:
  - GPU 메모리 부족 시 감소
  - `gradient_accumulation_steps`와 함께 조정하여 effective batch size 유지
- **예시**:
  ```python
  cfg.per_device_train_batch_size = 16  # 더 큰 배치
  # 메모리 부족 시
  cfg.per_device_train_batch_size = 4
  cfg.gradient_accumulation_steps = 4  # effective batch size 유지
  ```

#### `per_device_eval_batch_size` (기본값: 8)
- **설명**: GPU당 평가 배치 사이즈
- **영향**:
  - 평가 시에는 메모리 사용량이 적으므로 학습보다 크게 설정 가능
- **권장 범위**: 학습 배치 사이즈의 1-2배
- **예시**:
  ```python
  cfg.per_device_eval_batch_size = 16  # 평가는 더 큰 배치 가능
  ```

#### `gradient_accumulation_steps` (기본값: 2)
- **설명**: 몇 개의 배치를 누적한 후 가중치 업데이트할지
- **영향**:
  - **Effective batch size = batch_size × gradient_accumulation_steps**
  - 메모리 부족 시 배치 사이즈를 줄이고 이 값을 늘려서 effective batch size 유지
- **권장 범위**: 1-4
- **조정 팁**:
  - 메모리 부족 시 배치 사이즈를 줄이고 이 값을 증가
  - 너무 크면 업데이트가 느려짐
- **예시**:
  ```python
  cfg.per_device_train_batch_size = 4
  cfg.gradient_accumulation_steps = 4  # effective batch size = 16 유지
  ```

---

### 1.3 데이터 관련 파라미터

#### `max_length` (기본값: 256)
- **설명**: 입력 시퀀스의 최대 길이 (토큰 수)
- **영향**:
  - **짧게 (128)**: 빠른 학습, 긴 텍스트 정보 손실
  - **적절한 값 (256-512)**: 대부분의 텍스트 처리 가능
  - **길게 (512-1024)**: 더 많은 정보, 메모리 사용 증가, 학습 시간 증가
- **권장 범위**: 256-512
- **조정 팁**:
  - 데이터의 평균 길이 확인 후 설정
  - 긴 텍스트가 많으면 증가
  - 메모리 부족 시 감소
- **예시**:
  ```python
  cfg.max_length = 512  # 더 긴 텍스트 처리
  ```

#### `use_paragraph` (기본값: True)
- **설명**: 문단 텍스트 사용 여부
- **영향**:
  - True: 더 많은 컨텍스트 정보 사용
  - False: 제목 등 다른 필드 사용
- **조정 팁**: 데이터 구조에 따라 결정
- **예시**:
  ```python
  cfg.use_paragraph = True  # 문단 텍스트 사용
  ```

#### `add_title` (기본값: False)
- **설명**: 제목을 입력에 추가할지 여부
- **영향**:
  - True: 제목 정보 추가로 성능 향상 가능
  - False: 기본 설정
- **조정 팁**: 제목이 유용한 정보를 제공하는지 확인
- **예시**:
  ```python
  cfg.add_title = True  # 제목 추가
  ```

---

### 1.4 모델 아키텍처 파라미터

#### `model_name` (기본값: 'HybridAvsH')
- **설명**: 사용할 모델 타입
- **옵션**: `'AvsHModel'`, `'HybridAvsH'`, `'Gemma3InfoNCE'`, `'Qwen3InfoNCE'`
- **영향**:
  - 각 모델은 다른 아키텍처와 성능 특성
  - `HybridAvsH`: 기본 권장 모델
- **예시**:
  ```python
  cfg.model_name = 'HybridAvsH'  # 기본값
  cfg.model_name = 'Gemma3InfoNCE'  # 다른 모델 시도
  ```

#### `embedding_model` (기본값: 'kykim/funnel-kor-base')
- **설명**: 사용할 사전 학습된 모델 (Hugging Face)
- **영향**:
  - 다른 모델은 다른 성능과 특징
  - 한국어 모델: `kykim/funnel-kor-base`, `klue/roberta-base` 등
  - 영어 모델: `bert-base-uncased`, `distilbert-base-uncased` 등
- **권장 모델**:
  - 한국어: `kykim/funnel-kor-base`, `klue/roberta-base`
  - 영어: `bert-base-uncased`, `distilbert-base-uncased`
- **예시**:
  ```python
  cfg.embedding_model = 'kykim/funnel-kor-base'  # 기본값
  cfg.embedding_model = 'klue/roberta-base'  # 다른 모델 시도
  ```

#### `num_heads` (기본값: 8)
- **설명**: Attention head 수
- **영향**:
  - **적게 (4)**: 단순한 모델, 빠른 학습
  - **적절한 값 (8)**: 일반적으로 좋은 성능
  - **많게 (16)**: 더 복잡한 모델, 메모리 사용 증가
- **권장 범위**: 4-16
- **조정 팁**: 모델 크기와 데이터 복잡도에 따라 조정
- **예시**:
  ```python
  cfg.num_heads = 16  # 더 많은 attention heads
  ```

#### `num_layers` (기본값: 4)
- **설명**: Transformer 레이어 수
- **영향**:
  - **적게 (2-3)**: 단순한 모델, 빠른 학습
  - **적절한 값 (4-6)**: 일반적으로 좋은 성능
  - **많게 (8-12)**: 더 복잡한 모델, 메모리 사용 증가, 과적합 위험
- **권장 범위**: 4-8
- **조정 팁**: 데이터가 복잡할수록 더 많은 레이어 유용
- **예시**:
  ```python
  cfg.num_layers = 6  # 더 깊은 모델
  ```

#### `hidden_size` (기본값: 512)
- **설명**: Hidden layer 크기
- **영향**:
  - **작게 (256)**: 메모리 절약, 표현력 제한
  - **적절한 값 (512)**: 일반적으로 좋은 성능
  - **크게 (1024)**: 더 강한 표현력, 메모리 사용 증가
- **권장 범위**: 256-1024
- **조정 팁**: 모델 크기와 성능의 트레이드오프
- **예시**:
  ```python
  cfg.hidden_size = 768  # 더 큰 hidden size
  ```

#### `dim_feedforward` (기본값: 2048)
- **설명**: Feedforward network 크기
- **영향**:
  - **작게 (1024)**: 메모리 절약
  - **적절한 값 (2048)**: 일반적으로 좋은 성능
  - **크게 (4096)**: 더 강한 표현력, 메모리 사용 증가
- **권장 범위**: 1024-4096
- **조정 팁**: `hidden_size`의 2-4배가 일반적
- **예시**:
  ```python
  cfg.dim_feedforward = 3072  # 더 큰 feedforward
  ```

#### `dropout` (기본값: 0.0)
- **설명**: Dropout 비율 (과적합 방지)
- **영향**:
  - **0.0**: Dropout 없음
  - **0.1-0.3**: 적절한 정규화
  - **0.5 이상**: 너무 강한 정규화, 성능 저하
- **권장 범위**: 0.0-0.3
- **조정 팁**: 과적합이 발생하면 증가
- **예시**:
  ```python
  cfg.dropout = 0.1  # Dropout 추가
  ```

---

### 1.5 Loss 함수 관련 파라미터

#### `use_bpr_loss` (기본값: True)
- **설명**: BPR (Bayesian Personalized Ranking) Loss 사용 여부
- **영향**:
  - True: 순위 학습에 유용, 일반적으로 좋은 성능
  - False: 기본 BCE Loss 사용
- **조정 팁**: 순위 기반 태스크에 유용
- **예시**:
  ```python
  cfg.use_bpr_loss = True  # BPR Loss 사용
  ```

#### `bpr_loss_weight` (기본값: 0.25)
- **설명**: BPR Loss의 가중치 (다른 loss와의 균형)
- **영향**:
  - **낮게 (0.1)**: BPR Loss 영향 적음
  - **적절한 값 (0.25-0.5)**: 균형잡힌 학습
  - **높게 (0.7 이상)**: BPR Loss에 과도하게 의존
- **권장 범위**: 0.1-0.5
- **예시**:
  ```python
  cfg.bpr_loss_weight = 0.3  # BPR Loss 가중치 증가
  ```

#### `use_infonce_loss` (기본값: True)
- **설명**: InfoNCE Loss (Contrastive Learning) 사용 여부
- **영향**:
  - True: 대조 학습으로 표현력 향상 가능
  - False: 기본 Loss만 사용
- **조정 팁**: 유사한 샘플을 구분하는 태스크에 유용
- **예시**:
  ```python
  cfg.use_infonce_loss = True  # InfoNCE Loss 사용
  ```

#### `lambda_cl` (기본값: 0.1)
- **설명**: InfoNCE Loss의 가중치
- **영향**:
  - **낮게 (0.05)**: Contrastive learning 영향 적음
  - **적절한 값 (0.1-0.2)**: 균형잡힌 학습
  - **높게 (0.3 이상)**: Contrastive learning에 과도하게 의존
- **권장 범위**: 0.05-0.2
- **예시**:
  ```python
  cfg.lambda_cl = 0.15  # Contrastive learning 가중치 증가
  ```

#### `temperature` (기본값: 0.07)
- **설명**: InfoNCE Loss의 temperature 파라미터
- **영향**:
  - **낮게 (0.05)**: 더 sharp한 분포, 어려운 negative 샘플에 집중
  - **적절한 값 (0.07-0.1)**: 일반적으로 좋은 성능
  - **높게 (0.2 이상)**: 더 부드러운 분포, 학습 어려움
- **권장 범위**: 0.05-0.1
- **예시**:
  ```python
  cfg.temperature = 0.1  # 더 부드러운 분포
  ```

---

### 1.6 기타 학습 파라미터

#### `logging_steps` (기본값: 10)
- **설명**: 몇 step마다 로그를 출력할지
- **영향**:
  - **낮게 (5)**: 더 자주 로그 출력, 학습 속도 약간 감소
  - **적절한 값 (10-50)**: 적절한 모니터링
  - **높게 (100)**: 로그 출력 적음
- **권장 범위**: 10-50
- **예시**:
  ```python
  cfg.logging_steps = 20  # 더 자주 로그 출력
  ```

#### `k_fold` (기본값: 4)
- **설명**: K-Fold Cross-Validation의 K 값
- **영향**:
  - **작게 (3)**: 빠른 학습, 덜 안정적인 평가
  - **적절한 값 (4-5)**: 일반적으로 좋은 균형
  - **크게 (10)**: 더 안정적인 평가, 학습 시간 증가
- **권장 범위**: 4-5
- **조정 팁**: 데이터 크기와 시간에 따라 조정
- **예시**:
  ```python
  cfg.k_fold = 5  # 5-Fold 사용
  ```

---

## Part 2: Meta-Classifier 학습 파라미터

### 2.1 모델 타입 선택

#### `meta_model_type` (기본값: 'mlp')
- **설명**: Meta-Classifier 모델 타입
- **옵션**: `'mlp'`, `'ridge'`
- **영향**:
  - **MLP**: 비선형 관계 학습 가능, 더 복잡한 패턴 포착
  - **Ridge**: 선형 모델, 빠른 학습, 해석 가능
- **조정 팁**:
  - MLP: 더 복잡한 관계가 있을 때
  - Ridge: 빠른 실험, 해석이 필요할 때
- **예시**:
  ```python
  sys.argv = ['meta_train.py', '--meta_model_type', 'mlp']  # MLP 사용
  sys.argv = ['meta_train.py', '--meta_model_type', 'ridge']  # Ridge 사용
  ```

---

### 2.2 MLP 파라미터

#### `hidden_layers` (기본값: [64, 32])
- **설명**: MLP의 hidden layer 크기 리스트
- **영향**:
  - **작게 ([32, 16])**: 단순한 모델, 빠른 학습
  - **적절한 값 ([64, 32], [128, 64])**: 일반적으로 좋은 성능
  - **크게 ([256, 128, 64])**: 더 복잡한 모델, 과적합 위험
- **권장 범위**: [32, 16] ~ [128, 64]
- **조정 팁**:
  - 입력 차원이 4이므로 너무 큰 모델은 과적합 위험
  - 점진적으로 감소하는 구조 권장
- **예시**:
  ```python
  sys.argv = ['meta_train.py', '--hidden_layers', '128', '64']  # 더 큰 모델
  sys.argv = ['meta_train.py', '--hidden_layers', '32', '16']  # 더 작은 모델
  ```

#### `dropout` (기본값: 0.2)
- **설명**: MLP의 Dropout 비율
- **영향**:
  - **낮게 (0.1)**: 약한 정규화
  - **적절한 값 (0.2-0.3)**: 적절한 정규화
  - **높게 (0.5)**: 강한 정규화, 성능 저하 가능
- **권장 범위**: 0.1-0.3
- **조정 팁**: 과적합이 발생하면 증가
- **예시**:
  ```python
  sys.argv = ['meta_train.py', '--dropout', '0.3']  # 더 강한 정규화
  ```

#### `activation` (기본값: 'relu')
- **설명**: 활성화 함수
- **옵션**: `'relu'`, `'tanh'`
- **영향**:
  - **ReLU**: 일반적으로 좋은 성능, 빠른 학습
  - **Tanh**: 부드러운 출력, 특정 경우에 유용
- **조정 팁**: 대부분의 경우 ReLU 권장
- **예시**:
  ```python
  sys.argv = ['meta_train.py', '--activation', 'relu']  # ReLU 사용
  sys.argv = ['meta_train.py', '--activation', 'tanh']  # Tanh 사용
  ```

#### `epochs` (기본값: 100)
- **설명**: MLP 학습 epoch 수
- **영향**:
  - **낮게 (50)**: 빠른 학습, 과소적합 가능
  - **적절한 값 (100-200)**: 일반적으로 좋은 성능
  - **높게 (300 이상)**: 과적합 위험, 학습 시간 증가
- **권장 범위**: 100-200
- **조정 팁**: Early stopping이 없으므로 주의 필요
- **예시**:
  ```python
  sys.argv = ['meta_train.py', '--epochs', '150']  # 더 많은 epoch
  ```

#### `batch_size` (기본값: 32)
- **설명**: MLP 학습 배치 사이즈
- **영향**:
  - **작게 (16)**: 더 많은 업데이트, 안정적인 학습
  - **적절한 값 (32-64)**: 일반적으로 좋은 성능
  - **크게 (128)**: 빠른 학습, 메모리 사용 증가
- **권장 범위**: 32-64
- **예시**:
  ```python
  sys.argv = ['meta_train.py', '--batch_size', '64']  # 더 큰 배치
  ```

#### `lr` (기본값: 0.001)
- **설명**: MLP 학습률
- **영향**:
  - **낮게 (0.0001)**: 느린 학습, 안정적
  - **적절한 값 (0.001-0.01)**: 일반적으로 좋은 성능
  - **높게 (0.1)**: 불안정한 학습 가능
- **권장 범위**: 0.001-0.01
- **조정 팁**: 학습이 불안정하면 감소
- **예시**:
  ```python
  sys.argv = ['meta_train.py', '--lr', '0.005']  # 더 높은 학습률
  ```

---

### 2.3 Ridge 파라미터

#### `alphas` (기본값: None, 자동)
- **설명**: Ridge 정규화 강도 리스트 (CV로 선택)
- **영향**:
  - None: 자동으로 적절한 값 선택
  - 수동 지정: 특정 범위 테스트 가능
- **권장**: None (자동 선택)
- **예시**:
  ```python
  sys.argv = ['meta_train.py', '--alphas', '0.1', '1.0', '10.0']  # 수동 지정
  ```

#### `cv` (기본값: 5)
- **설명**: Ridge Cross-Validation fold 수
- **영향**:
  - **작게 (3)**: 빠른 학습, 덜 안정적
  - **적절한 값 (5)**: 일반적으로 좋은 성능
  - **크게 (10)**: 더 안정적, 학습 시간 증가
- **권장 범위**: 5-10
- **예시**:
  ```python
  sys.argv = ['meta_train.py', '--cv', '10']  # 더 많은 CV fold
  ```

---

## Part 3: 추론 파라미터

### 3.1 Meta-Classifier 추론

#### `meta_model_type` (기본값: 'mlp')
- **설명**: 사용할 Meta-Classifier 타입 (학습 시와 동일해야 함)
- **옵션**: `'mlp'`, `'ridge'`
- **주의**: 학습 시 사용한 타입과 반드시 일치해야 함
- **예시**:
  ```python
  sys.argv = ['meta_inference.py', '--meta_model_type', 'mlp']  # MLP 모델 사용
  ```

#### `hidden_layers` (기본값: [64, 32])
- **설명**: MLP 모델의 hidden layer 크기 (학습 시와 동일해야 함)
- **주의**: 학습 시 사용한 값과 반드시 일치해야 함
- **예시**:
  ```python
  sys.argv = ['meta_inference.py', '--hidden_layers', '64', '32']  # 학습 시와 동일
  ```

#### `dropout` (기본값: 0.2)
- **설명**: MLP 모델의 dropout (학습 시와 동일해야 함)
- **주의**: 학습 시 사용한 값과 반드시 일치해야 함
- **예시**:
  ```python
  sys.argv = ['meta_inference.py', '--dropout', '0.2']  # 학습 시와 동일
  ```

#### `output_filename` (기본값: 'submission.csv')
- **설명**: 생성할 submission 파일명
- **영향**: 파일명만 변경, 내용은 동일
- **예시**:
  ```python
  sys.argv = ['meta_inference.py', '--output_filename', 'submission_v2.csv']
  ```

---

## Part 4: 평가 파라미터

### 4.1 평가 리포트 생성

#### `logit_dir` (기본값: './outputs/fold_logits')
- **설명**: Fold logits가 저장된 디렉토리
- **영향**: 평가할 logits 위치 지정
- **예시**:
  ```python
  generate_evaluation_report(
      logit_dir=f'{PROJECT_ROOT}/outputs/fold_logits',
      ...
  )
  ```

#### `meta_model_path` (기본값: None)
- **설명**: Meta-Classifier 모델 경로
- **영향**: Meta-Classifier 평가 포함 여부
- **예시**:
  ```python
  generate_evaluation_report(
      meta_model_path=f'{PROJECT_ROOT}/outputs/meta_features/meta_classifier_mlp.pth',
      ...
  )
  ```

#### `meta_model_type` (기본값: 'mlp')
- **설명**: Meta-Classifier 타입
- **옵션**: `'mlp'`, `'ridge'`
- **예시**:
  ```python
  generate_evaluation_report(
      meta_model_type='mlp',
      ...
  )
  ```

#### `output_dir` (기본값: './outputs/evaluation')
- **설명**: 평가 결과 저장 디렉토리
- **영향**: 평가 리포트 저장 위치
- **예시**:
  ```python
  generate_evaluation_report(
      output_dir=f'{PROJECT_ROOT}/outputs/evaluation',
      ...
  )
  ```

---

## Part 5: 파라미터 조정 전략

### 5.1 성능 향상을 위한 조정 순서

1. **기본 파라미터 조정** (가장 영향 큼)
   - `num_train_epochs`: 10 → 15-20
   - `learning_rate`: 3e-5 → 5e-5 (안정적이면)
   - `per_device_train_batch_size`: 메모리 허용 범위에서 증가

2. **모델 아키텍처 조정**
   - `embedding_model`: 다른 사전 학습 모델 시도
   - `num_layers`: 4 → 6
   - `hidden_size`: 512 → 768

3. **데이터 관련 조정**
   - `max_length`: 256 → 512 (긴 텍스트가 많으면)
   - `add_title`: True로 변경

4. **Loss 함수 조정**
   - `bpr_loss_weight`: 0.25 → 0.3
   - `lambda_cl`: 0.1 → 0.15

5. **Meta-Classifier 조정**
   - `hidden_layers`: [64, 32] → [128, 64]
   - `epochs`: 100 → 150

### 5.2 메모리 부족 시 조정

```python
# 배치 사이즈 감소
cfg.per_device_train_batch_size = 4
cfg.per_device_eval_batch_size = 8

# Gradient accumulation 증가
cfg.gradient_accumulation_steps = 4

# 모델 크기 감소
cfg.hidden_size = 256
cfg.dim_feedforward = 1024
cfg.num_layers = 3

# 시퀀스 길이 감소
cfg.max_length = 128
```

### 5.3 학습 속도 향상을 위한 조정

```python
# 배치 사이즈 증가
cfg.per_device_train_batch_size = 16

# Epoch 감소 (Early stopping 활용)
cfg.num_train_epochs = 8

# 시퀀스 길이 감소
cfg.max_length = 128

# 모델 크기 감소
cfg.num_layers = 3
cfg.hidden_size = 256
```

### 5.4 과적합 방지를 위한 조정

```python
# 정규화 강화
cfg.weight_decay = 5e-3
cfg.dropout = 0.1

# Epoch 조정 (Early stopping 활용)
cfg.num_train_epochs = 10  # Early stopping이 자동으로 중단

# 모델 크기 감소
cfg.num_layers = 3
cfg.hidden_size = 256
```

---

## Part 6: 파라미터 조정 예시

### 예시 1: 빠른 실험 (시간 절약)

```python
cfg = SimpleNamespace(
    # ... 기본 설정 ...
    num_train_epochs=5,  # 빠른 학습
    per_device_train_batch_size=16,  # 큰 배치
    max_length=128,  # 짧은 시퀀스
    num_layers=3,  # 작은 모델
    hidden_size=256,
)
```

### 예시 2: 최고 성능 추구

```python
cfg = SimpleNamespace(
    # ... 기본 설정 ...
    num_train_epochs=20,  # 더 많은 epoch
    learning_rate=5e-5,  # 약간 높은 학습률
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # Effective batch size = 32
    max_length=512,  # 긴 시퀀스
    num_layers=6,  # 더 깊은 모델
    hidden_size=768,
    dim_feedforward=3072,
    embedding_model='klue/roberta-base',  # 다른 모델
)
```

### 예시 3: 메모리 제한 환경

```python
cfg = SimpleNamespace(
    # ... 기본 설정 ...
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # Effective batch size 유지
    max_length=128,
    num_layers=3,
    hidden_size=256,
    dim_feedforward=1024,
    num_heads=4,
)
```

### 예시 4: Meta-Classifier 고급 설정

```python
# MLP 고급 설정
sys.argv = [
    'meta_train.py',
    '--meta_model_type', 'mlp',
    '--hidden_layers', '128', '64', '32',  # 더 깊은 모델
    '--dropout', '0.3',  # 더 강한 정규화
    '--epochs', '200',  # 더 많은 epoch
    '--batch_size', '64',  # 큰 배치
    '--lr', '0.005',  # 높은 학습률
    '--save_model'
]
```

---

## Part 7: 파라미터 조정 체크리스트

### 학습 전 확인사항

- [ ] `num_train_epochs`: 데이터 크기와 시간에 맞게 설정
- [ ] `learning_rate`: 모델 크기에 맞게 설정
- [ ] `per_device_train_batch_size`: GPU 메모리에 맞게 설정
- [ ] `max_length`: 데이터의 평균 길이 확인 후 설정
- [ ] `embedding_model`: 태스크에 적합한 모델 선택

### 학습 중 모니터링

- [ ] Validation loss가 감소하는지 확인
- [ ] GPU 메모리 사용량 확인
- [ ] 학습 속도 확인
- [ ] Early stopping이 적절히 작동하는지 확인

### 학습 후 평가

- [ ] Fold별 성능 차이 확인
- [ ] 과적합 여부 확인
- [ ] Meta-Classifier 성능 확인
- [ ] 최종 submission 성능 확인

---

## 주의사항

1. **파라미터 조정은 한 번에 하나씩**: 여러 파라미터를 동시에 변경하면 어떤 것이 효과적인지 알기 어려움
2. **기본값에서 시작**: 기본값은 일반적으로 잘 작동하도록 설정됨
3. **Validation 성능 확인**: 학습 중 validation 성능을 모니터링하여 과적합 방지
4. **메모리 제한 확인**: GPU 메모리를 초과하지 않도록 주의
5. **시간 제약 고려**: Colab 세션 시간 제한을 고려하여 파라미터 조정

