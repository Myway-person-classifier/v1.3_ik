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

        labels = (labels > 0.5).astype(int)

        # v1.3: Save fold logits for meta-learning
        if hasattr(args, 'save_fold_logits') and args.save_fold_logits:
            fold_idx = getattr(args, 'fold_idx', 0)
            logit_dir = "./outputs/fold_logits"
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

        if args.is_submission or args.split_valid_by_paragraph:             #save logits for submission
            if args.local_rank == 0 or args.local_rank == -1:
                os.makedirs(f'./ckpt/{args.save_dir}', exist_ok=True)
                np.save(f'./ckpt/{args.save_dir}/{args.save_name}.npy', logits)
                np.save(f'./ckpt/{args.save_dir}/{args.save_name}_label.npy', labels)
            metric['accuracy'] = 1.0
        else:
            # logits를 확률값으로 변환 (sigmoid)
            # BCEWithLogitsLoss를 사용하므로 sigmoid를 적용해야 함
            # Numerical stability를 위해 clip 적용
            logits_clipped = np.clip(logits, -500, 500)  # overflow 방지
            probs = 1 / (1 + np.exp(-logits_clipped))  # sigmoid: 1 / (1 + exp(-x))
            
            # accuracy 계산: 확률 > 0.5이면 1, 아니면 0
            preds = (probs > 0.5).astype(int)
            metric['accuracy'] = accuracy_score(labels, preds)  

            # f1 score 계산
            metric['f1_score'] = f1_score(labels, preds, average='macro')

            # precision 계산
            metric['precision'] = precision_score(labels, preds, average='macro', zero_division=0)

            # recall 계산
            metric['recall'] = recall_score(labels, preds, average='macro', zero_division=0)

            # roc_auc score 계산 (확률값 사용)
            try:
                # roc_auc_score는 확률값(0~1)을 기대함
                # labels에 0과 1이 모두 있어야 함
                if len(np.unique(labels)) < 2:
                    # 클래스가 하나만 있는 경우
                    metric['roc_auc'] = 0.5
                else:
                    metric['roc_auc'] = roc_auc_score(labels, probs)
            except ValueError as e:
                # 예외 처리
                print(f"Warning: ROC AUC calculation failed: {e}. Using default value 0.5")
                metric['roc_auc'] = 0.5
        
        return metric

    return compute_metrics

