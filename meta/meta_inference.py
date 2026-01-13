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
from typing import Optional, List


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
    # Colab 환경 지원: 환경 변수로 기본 경로 설정 가능
    default_output_dir = os.environ.get('OUTPUT_DIR', './outputs')
    default_logit_dir = os.path.join(default_output_dir, 'fold_logits')
    default_final_output_dir = os.path.join(default_output_dir, 'final_predictions')
    
    parser.add_argument(
        '--logit_dir',
        type=str,
        default=default_logit_dir,
        help='Directory containing fold logits (can be set via OUTPUT_DIR env var)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=default_final_output_dir,
        help='Output directory for final predictions (can be set via OUTPUT_DIR env var)'
    )
    parser.add_argument(
        '--output_filename',
        type=str,
        default='submission.csv',
        help='Output filename'
    )
    parser.add_argument(
        '--model_names',
        type=str,
        nargs='+',
        default=None,
        help='List of model names for multi-model ensemble (must match training). If None, single model mode.'
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
        '--input_dim',
        type=int,
        default=None,
        help='Input dimension for MLP (auto-detected if model_names provided, otherwise 4)'
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
        default_output_dir = os.environ.get('OUTPUT_DIR', './outputs')
        meta_features_dir = os.path.join(default_output_dir, 'meta_features')
        model_path = os.path.join(
            meta_features_dir,
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
    
    # Determine input dimension
    if args.input_dim is not None:
        input_dim = args.input_dim
    elif args.model_names and len(args.model_names) > 1:
        input_dim = len(args.model_names) * 4  # M models × 4 folds
        print(f"Multi-model mode detected: {args.model_names} → input_dim={input_dim}")
    else:
        # 단일 모델 모드: logit_dir에서 자동 감지
        if os.path.exists(args.logit_dir):
            model_dirs = [d for d in os.listdir(args.logit_dir) 
                         if os.path.isdir(os.path.join(args.logit_dir, d)) 
                         and os.path.exists(os.path.join(args.logit_dir, d, 'oof'))]
            
            if len(model_dirs) == 1:
                # 단일 모델 디렉토리 발견
                model_name = model_dirs[0]
                actual_logit_dir = os.path.join(args.logit_dir, model_name)
                print(f"Single model mode: Found {model_name}")
                print(f"  Using logit_dir: {actual_logit_dir}")
                args.logit_dir = actual_logit_dir  # 경로 업데이트
            elif len(model_dirs) > 1:
                raise ValueError(
                    f"Multiple model directories found. Please specify --model_names."
                )
        
        input_dim = 4  # Single model: 4 folds
        print(f"Single model mode → input_dim={input_dim}")
    
    if args.meta_model_type == 'mlp':
        classifier = MetaClassifier(
            model_type='mlp',
            input_dim=input_dim,  # Dynamic input dimension
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
    test_logits = collect_test_logits(args.logit_dir, model_names=args.model_names)
    print(f"Test logits shape: {test_logits.shape}")
    
    # Verify dimension match
    if test_logits.shape[1] != input_dim:
        raise ValueError(
            f"Dimension mismatch: test_logits has {test_logits.shape[1]} features, "
            f"but meta-classifier expects {input_dim} features. "
            f"Check model_names argument."
        )
    
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
    # Load test.csv to get actual IDs
    data_dir = os.environ.get('DATA_DIR', './data')
    test_csv_path = os.path.join(data_dir, 'test.csv')
    
    if os.path.exists(test_csv_path):
        test_df = pd.read_csv(test_csv_path)
        if 'id' in test_df.columns:
            test_ids = test_df['id'].values
        else:
            test_ids = range(len(test_df))
        print(f"Loaded {len(test_ids)} test IDs from {test_csv_path}")
    else:
        # Fallback: use range if test.csv not found
        print(f"Warning: test.csv not found at {test_csv_path}, using range(len(preds))")
        test_ids = range(len(preds))
    
    # Ensure lengths match
    if len(test_ids) != len(preds):
        print(f"Warning: Test IDs length ({len(test_ids)}) != predictions length ({len(preds)})")
        print(f"Using range(len(preds)) as fallback")
        test_ids = range(len(preds))
    
    df = pd.DataFrame({
        'id': test_ids,
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
