"""
MetaDataset: Dataset for meta-features
Loads fold logits and creates meta-features dataset
"""

import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from utils.logit_collector import LogitCollector


class MetaDataset(Dataset):
    """
    Dataset for meta-learning using fold logits
    
    Features:
    - Shape: [Num_Samples] × 4
    - Column 0: OOF Logits (Fold 0)
    - Column 1-3: Test Logits (Fold 1, 2, 3)
    """
    
    def __init__(self, logit_dir: str = "./outputs/fold_logits", labels=None):
        """
        Args:
            logit_dir: Directory containing fold logits
            labels: Optional labels array (if None, will load from logit_dir)
        """
        self.logit_dir = logit_dir
        
        # Collect logits
        collector = LogitCollector(logit_dir)
        self.meta_features, self.labels = collector.collect_logits()
        
        # Override labels if provided
        if labels is not None:
            self.labels = labels
        
        print(f"MetaDataset initialized with {len(self.meta_features)} samples")
        print(f"Meta-features shape: {self.meta_features.shape}")
    
    def __len__(self):
        return len(self.meta_features)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with:
                - 'features': [4] array of fold logits
                - 'label': scalar label (if available)
        """
        item = {
            'features': self.meta_features[idx].astype(np.float32),
        }
        
        if self.labels is not None:
            item['label'] = float(self.labels[idx])
        
        return item


def load_meta_features_csv(csv_path: str):
    """
    Load meta-features from CSV file
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Tuple of (meta_features, labels)
    """
    df = pd.read_csv(csv_path)
    
    # Extract fold logits columns
    # ✅ Support both single model (fold_0_logits) and multi-model (ModelName_fold0)
    fold_cols = [col for col in df.columns if 'fold' in col and col != 'label']
    
    # Ensure consistent ordering if possible
    # For single model, it will be fold_0_logits, fold_1_logits...
    # For multi-model, it will be Model1_fold0, Model1_fold1...
    fold_cols = sorted(fold_cols)
    
    meta_features = df[fold_cols].values.astype(np.float32)
    
    # Extract labels if available
    labels = None
    if 'label' in df.columns:
        labels = df['label'].values
    
    return meta_features, labels
