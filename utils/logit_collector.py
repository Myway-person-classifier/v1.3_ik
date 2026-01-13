"""
LogitCollector: Collects and organizes fold logits for meta-learning
Creates meta-features dataset from fold predictions
Supports both single model and multiple models
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List


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
        num_features = meta_features.shape[1]
        df = pd.DataFrame(
            meta_features,
            columns=[f'fold_{i}_logits' for i in range(num_features)]
        )
        
        if labels is not None:
            df['label'] = labels
        
        # Save
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved meta-features to {output_path}")
        
        return df


class MultiModelLogitCollector:
    """
    Collects logits from multiple models and creates combined meta-features
    
    Meta-features structure for M models:
    - Shape: [Num_Samples] × (4 × M)
    - Columns: [Model1_Fold0, Model1_Fold1, ..., Model1_Fold3, 
                 Model2_Fold0, Model2_Fold1, ..., Model2_Fold3, ...]
    """
    
    def __init__(self, base_logit_dir: str, model_names: List[str]):
        """
        Args:
            base_logit_dir: Base directory containing model subdirectories
            model_names: List of model names to collect logits from
        """
        self.base_logit_dir = base_logit_dir
        self.model_names = model_names
    
    def collect_logits(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Collect logits from all models and create combined meta-features
        
        Returns:
            Tuple of (meta_features, labels)
            - meta_features: [N, 4×M] array where M is number of models
            - labels: [N] array of labels (if available)
        """
        all_model_features = []
        labels = None
        
        for model_name in self.model_names:
            model_logit_dir = os.path.join(self.base_logit_dir, model_name)
            print(f"\nCollecting logits from {model_name}...")
            
            collector = LogitCollector(model_logit_dir)
            model_features, model_labels = collector.collect_logits()
            
            # Shape: [N, 4]
            all_model_features.append(model_features)
            
            # Use labels from first model (all should have same labels)
            if labels is None and model_labels is not None:
                labels = model_labels
        
        # Concatenate: [N, 4×M]
        combined_features = np.concatenate(all_model_features, axis=1)
        print(f"\nCombined meta-features shape: {combined_features.shape}")
        print(f"  - {len(self.model_names)} models × 4 folds = {combined_features.shape[1]} features")
        
        return combined_features, labels
    
    def save_meta_features(
        self,
        output_dir: str = "./outputs/meta_features",
        filename: str = "meta_train_multimodel.csv"
    ):
        """
        Save combined meta-features as CSV
        
        Args:
            output_dir: Output directory
            filename: Output filename
        """
        os.makedirs(output_dir, exist_ok=True)
        
        meta_features, labels = self.collect_logits()
        
        # Create column names: model_fold format
        columns = []
        for model_name in self.model_names:
            for fold_idx in range(4):
                columns.append(f'{model_name}_fold{fold_idx}')
        
        # Create DataFrame
        df = pd.DataFrame(meta_features, columns=columns)
        
        if labels is not None:
            df['label'] = labels
        
        # Save
        output_path = os.path.join(output_dir, filename)
        df.to_csv(output_path, index=False)
        print(f"Saved combined meta-features to {output_path}")
        
        return df


def collect_test_logits(logit_dir: str = "./outputs/fold_logits", model_names: Optional[List[str]] = None) -> np.ndarray:
    """
    Collect test logits from all folds for final prediction
    
    Args:
        logit_dir: Directory containing fold logits (or base directory for multiple models)
        model_names: List of model names (if None, single model mode)
        
    Returns:
        Test logits: [N, 4] array (single model) or [N, 4×M] array (multiple models)
    """
    if model_names is None:
        # Single model mode
        collector = LogitCollector(logit_dir)
        
        # For test prediction, we use all 4 folds' test predictions
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
    else:
        # Multiple models mode
        all_model_logits = []
        for model_name in model_names:
            model_logit_dir = os.path.join(logit_dir, model_name)
            model_logits = collect_test_logits(model_logit_dir, model_names=None)
            all_model_logits.append(model_logits)
        
        # Concatenate: [N, 4×M]
        combined_logits = np.concatenate(all_model_logits, axis=1)
        return combined_logits
