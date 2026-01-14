import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def get_metric(args):
    
    def compute_metrics(eval_preds):
        metric = dict()

        # 예측값과 레이블 추출
        logits = eval_preds.predictions  # shape: [B] or [B, 1]
        labels = eval_preds.label_ids    # shape: [B] or [B, 1]
        
        # Ensure logits and labels are 1D arrays
        logits = np.asarray(logits)
        labels = np.asarray(labels)
        
        # Flatten if needed (handle [B, 1] -> [B])
        if logits.ndim > 1:
            logits = logits.flatten()
        if labels.ndim > 1:
            labels = labels.flatten()

        # ✅ 디버깅: 원본 라벨 확인
        labels_raw = labels.copy()
        print(f"\n[DEBUG compute_metrics] Raw labels - dtype: {labels_raw.dtype}, shape: {labels_raw.shape}")
        print(f"[DEBUG compute_metrics] Raw labels - min: {labels_raw.min()}, max: {labels_raw.max()}, unique: {np.unique(labels_raw)}")
        if len(labels_raw) > 0:
            unique_labels, counts = np.unique(labels_raw, return_counts=True)
            print(f"[DEBUG compute_metrics] Raw label distribution: {dict(zip(unique_labels, counts))}")
        
        # ✅ 디버깅: Logits 확인
        print(f"[DEBUG compute_metrics] Logits - dtype: {logits.dtype}, shape: {logits.shape}")
        print(f"[DEBUG compute_metrics] Logits - min: {logits.min():.4f}, max: {logits.max():.4f}, mean: {logits.mean():.4f}, std: {logits.std():.4f}")
        if len(logits) > 0:
            print(f"[DEBUG compute_metrics] Logits near 0: {(np.abs(logits) < 0.1).sum()} / {len(logits)} ({(np.abs(logits) < 0.1).sum() / len(logits) * 100:.1f}%)")

        # 라벨 변환: 이미 정수인지 확인
        if labels_raw.dtype == float or (labels_raw.min() < 0 or labels_raw.max() > 1):
            # 라벨이 float이거나 범위 밖이면 변환
            labels = (labels_raw > 0.5).astype(int)
            print(f"[DEBUG compute_metrics] Converted labels from float/out-of-range to int")
        else:
            # 이미 정수면 그대로 사용
            labels = labels_raw.astype(int)
            print(f"[DEBUG compute_metrics] Labels already integers, using as-is")
        
        # ✅ 변환 후 라벨 확인
        unique_labels_final, counts_final = np.unique(labels, return_counts=True)
        print(f"[DEBUG compute_metrics] Final labels - unique: {unique_labels_final}, distribution: {dict(zip(unique_labels_final, counts_final))}")
        
        if len(unique_labels_final) < 2:
            print(f"⚠️ WARNING: Only one class in labels! This will cause ROC-AUC = 0.5")

        # v1.3: Save fold logits for meta-learning
        if hasattr(args, 'save_fold_logits') and args.save_fold_logits:
            fold_idx = getattr(args, 'fold_idx', 0)
            model_name = getattr(args, 'model_name', 'default')
            base_output_dir = getattr(args, 'output_dir', './outputs')
            logit_dir = os.path.join(base_output_dir, "fold_logits", model_name)
            
            os.makedirs(f"{logit_dir}/oof", exist_ok=True)
            os.makedirs(f"{logit_dir}/test", exist_ok=True)
            
            if args.local_rank == 0 or args.local_rank == -1:
                if fold_idx == 0:
                    # OOF Logits (Fold 0)
                    np.save(f"{logit_dir}/oof/fold0_logits.npy", logits)
                    np.save(f"{logit_dir}/oof/fold0_labels.npy", labels)
                    print(f"Saved OOF logits to {logit_dir}/oof/fold0_logits.npy")
                else:
                    # Test Logits (Fold 1, 2, 3)
                    np.save(f"{logit_dir}/test/fold{fold_idx}_logits.npy", logits)
                    np.save(f"{logit_dir}/test/fold{fold_idx}_labels.npy", labels)
                    print(f"Saved test logits to {logit_dir}/test/fold{fold_idx}_logits.npy")

        if args.is_submission:             #save logits for submission
            if args.local_rank == 0 or args.local_rank == -1:
                os.makedirs(f'./ckpt/{args.save_dir}', exist_ok=True)
                np.save(f'./ckpt/{args.save_dir}/{args.save_name}.npy', logits)
                np.save(f'./ckpt/{args.save_dir}/{args.save_name}_label.npy', labels)
        
        # logits를 확률값으로 변환 (sigmoid)
        # BCEWithLogitsLoss를 사용하므로 sigmoid를 적용해야 함
        # Numerical stability를 위해 clip 적용
        logits_clipped = np.clip(logits, -500, 500)  # overflow 방지
        probs = 1 / (1 + np.exp(-logits_clipped))  # sigmoid: 1 / (1 + exp(-x))
        
        # ✅ 디버깅: 확률 분포 확인
        print(f"[DEBUG compute_metrics] Probs - min: {probs.min():.4f}, max: {probs.max():.4f}, mean: {probs.mean():.4f}, std: {probs.std():.4f}")
        if len(probs) > 0:
            near_05 = (np.abs(probs - 0.5) < 0.01).sum()
            print(f"[DEBUG compute_metrics] Probs near 0.5 (±0.01): {near_05} / {len(probs)} ({near_05 / len(probs) * 100:.1f}%)")
            print(f"[DEBUG compute_metrics] Probs < 0.1: {(probs < 0.1).sum()}, > 0.9: {(probs > 0.9).sum()}")
        
        # accuracy 계산: 확률 > 0.5이면 1, 아니면 0
        preds = (probs > 0.5).astype(int)

        # Check if labels are dummy (e.g. all 0s or all same)
        unique_labels_check = np.unique(labels)
        is_dummy_labels = len(unique_labels_check) < 2

        if is_dummy_labels and args.is_submission:
            print(f"⚠️ Dummy labels detected during submission. Setting dummy metrics.")
            metric['accuracy'] = 1.0
            metric['f1_score'] = 0.0
            metric['roc_auc'] = 0.5
        else:
            metric['accuracy'] = accuracy_score(labels, preds)
            
            # ✅ 디버깅: 예측 분포 확인
            unique_preds, pred_counts = np.unique(preds, return_counts=True)
            print(f"[DEBUG compute_metrics] Predictions distribution: {dict(zip(unique_preds, pred_counts))}")  

            # f1 score 계산
            metric['f1_score'] = f1_score(labels, preds, average='macro')

            # precision 계산
            metric['precision'] = precision_score(labels, preds, average='macro', zero_division=0)

            # recall 계산
            metric['recall'] = recall_score(labels, preds, average='macro', zero_division=0)

            # roc_auc score 계산 (확률값 사용)
            try:
                if is_dummy_labels:
                    # 클래스가 하나만 있는 경우
                    print(f"⚠️ WARNING: Only {len(unique_labels_check)} unique label(s): {unique_labels_check}. ROC-AUC set to 0.5")
                    metric['roc_auc'] = 0.5
                else:
                    metric['roc_auc'] = roc_auc_score(labels, probs)
                    print(f"[DEBUG compute_metrics] ✅ ROC-AUC calculated: {metric['roc_auc']:.4f}")
            except ValueError as e:
                # 예외 처리
                print(f"⚠️ ERROR: ROC AUC calculation failed: {e}. Using default value 0.5")
                metric['roc_auc'] = 0.5
        
        return metric

    return compute_metrics

