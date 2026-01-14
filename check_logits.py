"""
저장된 logits 파일을 확인하여 문제 진단
"""
import os
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

def check_logits(logit_dir="./outputs/fold_logits"):
    """저장된 logits와 labels를 확인"""
    
    # 모델별로 확인
    base_dir = logit_dir
    if not os.path.exists(base_dir):
        print(f"❌ Logit directory not found: {logit_dir}")
        print(f"\n🔍 Searching for logits in common locations...")
        # 일반적인 경로들 확인
        common_paths = [
            "./outputs/fold_logits",
            "../outputs/fold_logits",
            "outputs/fold_logits",
            os.path.join(os.getcwd(), "outputs", "fold_logits")
        ]
        for path in common_paths:
            if os.path.exists(path):
                print(f"  Found: {path}")
                base_dir = path
                break
        else:
            print(f"  No logit directory found in common locations.")
            return
    
    print(f"📂 Checking directory: {base_dir}")
    
    # 디렉토리 구조 확인
    if not os.path.isdir(base_dir):
        print(f"❌ Not a directory: {base_dir}")
        return
    
    try:
        items = os.listdir(base_dir)
    except Exception as e:
        print(f"❌ Error reading directory: {e}")
        return
    
    # oof, test가 직접 있으면 (구버전 구조)
    if 'oof' in items or 'test' in items:
        print(f"\n📁 Found legacy structure (oof/test directly in base_dir)")
        check_single_model(base_dir)
    else:
        # 모델별 디렉토리 확인 (신버전 구조: fold_logits/{model_name}/oof/)
        print(f"\n📁 Found model-based structure")
        model_dirs = [d for d in items if os.path.isdir(os.path.join(base_dir, d))]
        
        if not model_dirs:
            print(f"❌ No model directories found in {base_dir}")
            print(f"   Available items: {items}")
            return
        
        for model_name in sorted(model_dirs):
            model_path = os.path.join(base_dir, model_name)
            print(f"\n{'='*60}")
            print(f"Model: {model_name}")
            print(f"{'='*60}")
            check_single_model(model_path)

def check_single_model(logit_dir):
    """단일 모델의 logits 확인"""
    
    print(f"  Checking: {logit_dir}")
    
    # OOF logits (Fold 0)
    oof_dir = os.path.join(logit_dir, "oof")
    oof_logits_path = os.path.join(oof_dir, "fold0_logits.npy")
    oof_labels_path = os.path.join(oof_dir, "fold0_labels.npy")
    
    # 경로 정규화 (Windows/Linux 호환)
    oof_logits_path = os.path.normpath(oof_logits_path)
    oof_labels_path = os.path.normpath(oof_labels_path)
    
    if os.path.exists(oof_logits_path):
        print(f"\n📊 Fold 0 (OOF) Logits:")
        logits = np.load(oof_logits_path)
        print(f"  Shape: {logits.shape}")
        print(f"  Min: {logits.min():.4f}, Max: {logits.max():.4f}")
        print(f"  Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
        print(f"  Near 0 (<0.1): {(np.abs(logits) < 0.1).sum()} / {len(logits)} ({(np.abs(logits) < 0.1).sum() / len(logits) * 100:.1f}%)")
        
        # 확률로 변환
        probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        print(f"\n  📈 Probabilities:")
        print(f"    Min: {probs.min():.4f}, Max: {probs.max():.4f}")
        print(f"    Mean: {probs.mean():.4f}, Std: {probs.std():.4f}")
        print(f"    Near 0.5 (±0.01): {(np.abs(probs - 0.5) < 0.01).sum()} / {len(probs)} ({(np.abs(probs - 0.5) < 0.01).sum() / len(probs) * 100:.1f}%)")
        print(f"    < 0.1: {(probs < 0.1).sum()}, > 0.9: {(probs > 0.9).sum()}")
        
        # 예측
        preds = (probs > 0.5).astype(int)
        unique_preds, counts = np.unique(preds, return_counts=True)
        print(f"    Predictions: {dict(zip(unique_preds, counts))}")
        
        # Labels 확인
        if os.path.exists(oof_labels_path):
            labels = np.load(oof_labels_path)
            print(f"\n  🏷️  Labels:")
            print(f"    Shape: {labels.shape}")
            print(f"    Dtype: {labels.dtype}")
            print(f"    Min: {labels.min()}, Max: {labels.max()}")
            unique_labels, label_counts = np.unique(labels, return_counts=True)
            print(f"    Unique: {unique_labels}")
            print(f"    Distribution: {dict(zip(unique_labels, label_counts))}")
            
            # 지표 계산
            if len(unique_labels) >= 2:
                try:
                    roc_auc = roc_auc_score(labels, probs)
                    acc = accuracy_score(labels, preds)
                    print(f"\n  📊 Metrics:")
                    print(f"    ROC-AUC: {roc_auc:.4f}")
                    print(f"    Accuracy: {acc:.4f}")
                except Exception as e:
                    print(f"\n  ⚠️  Error calculating metrics: {e}")
            else:
                print(f"\n  ⚠️  WARNING: Only {len(unique_labels)} unique label(s)! ROC-AUC cannot be calculated.")
    else:
        print(f"❌ OOF logits not found: {oof_logits_path}")
        # 디렉토리 구조 확인
        if os.path.exists(logit_dir):
            print(f"   Available in {logit_dir}:")
            try:
                items = os.listdir(logit_dir)
                for item in items:
                    item_path = os.path.join(logit_dir, item)
                    if os.path.isdir(item_path):
                        print(f"     📁 {item}/")
                        try:
                            subitems = os.listdir(item_path)
                            for subitem in subitems[:5]:  # 처음 5개만
                                print(f"        - {subitem}")
                            if len(subitems) > 5:
                                print(f"        ... and {len(subitems) - 5} more")
                        except:
                            pass
                    else:
                        print(f"     📄 {item}")
            except Exception as e:
                print(f"   Error listing directory: {e}")
    
    # Test logits (Fold 1, 2, 3)
    test_dir = os.path.join(logit_dir, "test")
    test_dir = os.path.normpath(test_dir)
    if os.path.exists(test_dir):
        for fold_idx in [1, 2, 3]:
            test_logits_path = os.path.normpath(os.path.join(test_dir, f"fold{fold_idx}_logits.npy"))
            test_labels_path = os.path.normpath(os.path.join(test_dir, f"fold{fold_idx}_labels.npy"))
            
            if os.path.exists(test_logits_path):
                print(f"\n📊 Fold {fold_idx} (Test) Logits:")
                logits = np.load(test_logits_path)
                print(f"  Shape: {logits.shape}")
                print(f"  Min: {logits.min():.4f}, Max: {logits.max():.4f}")
                print(f"  Mean: {logits.mean():.4f}, Std: {logits.std():.4f}")
                
                probs = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
                print(f"  Probs - Mean: {probs.mean():.4f}, Near 0.5: {(np.abs(probs - 0.5) < 0.01).sum() / len(probs) * 100:.1f}%")
                
                if os.path.exists(test_labels_path):
                    labels = np.load(test_labels_path)
                    unique_labels = np.unique(labels)
                    print(f"  Labels - Unique: {unique_labels}, Count: {len(unique_labels)} classes")
                    
                    if len(unique_labels) >= 2:
                        try:
                            roc_auc = roc_auc_score(labels, probs)
                            print(f"  ROC-AUC: {roc_auc:.4f}")
                        except:
                            pass

if __name__ == "__main__":
    import sys
    logit_dir = sys.argv[1] if len(sys.argv) > 1 else "./outputs/fold_logits"
    check_logits(logit_dir)

