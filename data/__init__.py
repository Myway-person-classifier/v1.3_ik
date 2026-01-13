"""
Data processing module for v1.3
Contains dataset loading, collation, and meta-features generation
"""

# Lazy imports to avoid circular dependencies
__all__ = [
    'get_dataset',
    'TextDataset',
    'TextCollator',
    'MetaDataset',
    'load_meta_features_csv',
]

