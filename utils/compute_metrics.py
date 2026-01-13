import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score


def get_metric(args):
    
    def compute_metrics(eval_preds):
        metric = dict()

        # 예측값과 레이블 추출
        logits = eval_preds.predictions  # shape: [B]
        labels = eval_preds.label_ids    # shape: [B]

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
            # accuracy 계산 # 1 if >0.5 else 0
            preds = np.argmax(logits, axis=1) if logits.ndim > 1 else (logits > 0.).astype(int)
            metric['accuracy'] = accuracy_score(labels, preds)  

            # f1 score 계산
            metric['f1_score'] = f1_score(labels, preds, average='macro')

            # precision 계산
            metric['precision'] = precision_score(labels, preds, average='macro')

            # recall 계산
            metric['recall'] = recall_score(labels, preds, average='macro')

            # roc_auc score 계산
            metric['roc_auc'] = roc_auc_score(labels, logits)
        
        return metric

    return compute_metrics

