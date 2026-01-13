"""
Meta-Classifier Inference Script
Generates final predictions from test logits
"""

import os
import argparse
import numpy as np
import pandas as pd
from utils.logit_collector import collect_test_logits
from meta.meta_classifier import MetaClassifier
import torch


def main():
    parser = argparse.ArgumentParser(description="Meta-Classifier Inference")
    
    parser.add_argument(
        '--meta_model_type',
        type=str,
        default='mlp',
        choices=['mlp', 'ridge'],
        help='Type of meta-classifier'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to trained meta-classifier model'
    )
    parser.add_argument(
        '--logit_dir',
        type=str,
        default='./outputs/fold_logits',
        help='Directory containing fold logits'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/final_predictions',
        help='Output directory for final predictions'
    )
    parser.add_argument(
        '--output_filename',
        type=str,
        default='submission.csv',
        help='Output filename'
    )
    
    # MLP arguments (for loading)
    parser.add_argument(
        '--hidden_layers',
        type=int,
        nargs='+',
        default=[64, 32],
        help='Hidden layer sizes for MLP (must match training)'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.2,
        help='Dropout rate for MLP'
    )
    parser.add_argument(
        '--activation',
        type=str,
        default='relu',
        help='Activation function for MLP'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Meta-Classifier Inference")
    print("="*60)
    
    # Load model
    if args.model_path is None:
        model_path = os.path.join(
            './outputs/meta_features',
            f"meta_classifier_{args.meta_model_type}.pth"
        )
        if args.meta_model_type == 'ridge':
            model_path = model_path.replace('.pth', '.joblib')
    else:
        model_path = args.model_path
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Please train the meta-classifier first."
        )
    
    print(f"\n1. Loading meta-classifier from {model_path}...")
    
    if args.meta_model_type == 'mlp':
        classifier = MetaClassifier(
            model_type='mlp',
            input_dim=4,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
            activation=args.activation
        )
        classifier.model.load_state_dict(torch.load(model_path))
        classifier.model.eval()
    else:
        import joblib
        ridge_model = joblib.load(model_path)
        classifier = MetaClassifier(model_type='ridge')
        classifier.model.model = ridge_model
        classifier.model.is_fitted = True
    
    # Load test logits
    print("\n2. Loading test logits...")
    test_logits = collect_test_logits(args.logit_dir)
    print(f"Test logits shape: {test_logits.shape}")
    
    # Predict
    print("\n3. Generating predictions...")
    probs = classifier.predict_proba(test_logits)
    preds = classifier.predict(test_logits)
    
    print(f"Predictions shape: {preds.shape}")
    print(f"Probability range: [{probs[:, 1].min():.4f}, {probs[:, 1].max():.4f}]")
    
    # Save predictions
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, args.output_filename)
    
    # Create submission file
    # Assuming test set has the same number of samples
    df = pd.DataFrame({
        'id': range(len(preds)),
        'generated': preds,
        'probability': probs[:, 1]
    })
    
    df.to_csv(output_path, index=False)
    print(f"\n4. Predictions saved to {output_path}")
    
    print("\n" + "="*60)
    print("Meta-Classifier Inference Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
