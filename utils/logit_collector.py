"""
LogitCollector: Collects and organizes fold logits for meta-learning
Creates meta-features dataset from fold predictions
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple


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
        self.oof_dir = os.path.join(logit_dir, "oof")
        self.test_dir = os.path.join(logit_dir, "test")
    
    def collect_logits(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Collect all fold logits and create meta-features
        
        Returns:
            Tuple of (meta_features, labels)
            - meta_features: [N, 4] array of logits
            - labels: [N] array of labels (if available)
        """
        # Load OOF logits (Fold 0)
        oof_logits_path = os.path.join(self.oof_dir, "fold0_logits.npy")
        if not os.path.exists(oof_logits_path):
            raise FileNotFoundError(
                f"OOF logits not found at {oof_logits_path}. "
                "Please train Fold 0 first."
            )
        
        oof_logits = np.load(oof_logits_path)
        print(f"Loaded OOF logits: {oof_logits.shape}")
        
        # Load test logits (Fold 1, 2, 3)
        test_logits_list = []
        for fold_idx in [1, 2, 3]:
            test_logits_path = os.path.join(self.test_dir, f"fold{fold_idx}_logits.npy")
            if not os.path.exists(test_logits_path):
                raise FileNotFoundError(
                    f"Test logits for fold {fold_idx} not found at {test_logits_path}. "
                    "Please train all folds first."
                )
            
            test_logits = np.load(test_logits_path)
            test_logits_list.append(test_logits)
            print(f"Loaded test logits for fold {fold_idx}: {test_logits.shape}")
        
        # Stack to create meta-features: [N, 4]
        meta_features = np.stack([oof_logits] + test_logits_list, axis=1)
        print(f"Meta-features shape: {meta_features.shape}")
        
        # Load labels if available
        oof_labels_path = os.path.join(self.oof_dir, "fold0_labels.npy")
        labels = None
        if os.path.exists(oof_labels_path):
            labels = np.load(oof_labels_path)
            print(f"Loaded labels: {labels.shape}")
        else:
            print("Warning: Labels not found. Meta-classifier will use test labels only.")
        
        return meta_features, labels
    
    def save_meta_features(
        self, 
        output_dir: str = "./outputs/meta_features",
        filename: str = "meta_train.csv"
    ):
        """
        Save meta-features as CSV
        
        Args:
            output_dir: Output directory
            filename: Output filename
        """
        os.makedirs(output_dir, exist_ok=True)
        
        meta_features, labels = self.collect_logits()
        
        # Create DataFrame
        df = pd.DataFrame(
            meta_features,
            columns=[f'fold_{i}_logits' for i in range(4)]
        )
        
        if labels is not None:
            df['label'] = labels
        
        # Save
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved meta-features to {output_path}")
        
        return df


def collect_test_logits(logit_dir: str = "./outputs/fold_logits") -> np.ndarray:
    """
    Collect test logits from all folds for final prediction
    
    Args:
        logit_dir: Directory containing fold logits
        
    Returns:
        Test logits: [N, 4] array
    """
    collector = LogitCollector(logit_dir)
    
    # For test prediction, we use all 4 folds' test predictions
    # In practice, you'd run inference on test set for each fold
    test_logits_list = []
    
    # Load test logits from all folds
    for fold_idx in range(4):
        test_logits_path = os.path.join(logit_dir, "test", f"fold{fold_idx}_logits.npy")
        if os.path.exists(test_logits_path):
            test_logits = np.load(test_logits_path)
            test_logits_list.append(test_logits)
        else:
            # Fallback to OOF if test not available
            oof_logits_path = os.path.join(logit_dir, "oof", f"fold{fold_idx}_logits.npy")
            if os.path.exists(oof_logits_path):
                test_logits = np.load(oof_logits_path)
                test_logits_list.append(test_logits)
            else:
                raise FileNotFoundError(f"No logits found for fold {fold_idx}")
    
    # Stack: [N, 4]
    test_logits = np.stack(test_logits_list, axis=1)
    return test_logits
