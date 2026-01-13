"""
Meta-Classifier Training Script
Trains MLP or Ridge classifier on fold logits
"""

import os
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from utils.logit_collector import LogitCollector
from meta.meta_classifier import MetaClassifier


def main():
    parser = argparse.ArgumentParser(description="Train Meta-Classifier")
    
    # Model arguments
    parser.add_argument(
        '--meta_model_type',
        type=str,
        default='mlp',
        choices=['mlp', 'ridge'],
        help='Type of meta-classifier (mlp or ridge)'
    )
    
    # MLP arguments
    parser.add_argument(
        '--hidden_layers',
        type=int,
        nargs='+',
        default=[64, 32],
        help='Hidden layer sizes for MLP'
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
        choices=['relu', 'tanh'],
        help='Activation function for MLP'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs for MLP'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32,
        help='Batch size for MLP training'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate for MLP'
    )
    
    # Ridge arguments
    parser.add_argument(
        '--alphas',
        type=float,
        nargs='+',
        default=None,
        help='Alpha values for Ridge (default: auto)'
    )
    parser.add_argument(
        '--cv',
        type=int,
        default=5,
        help='Cross-validation folds for Ridge'
    )
    
    # Data arguments
    parser.add_argument(
        '--logit_dir',
        type=str,
        default='./outputs/fold_logits',
        help='Directory containing fold logits'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs/meta_features',
        help='Output directory for meta-features and model'
    )
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='Save trained model'
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("Meta-Classifier Training")
    print("="*60)
    
    # Collect logits
    print("\n1. Collecting fold logits...")
    collector = LogitCollector(args.logit_dir)
    meta_features, labels = collector.collect_logits()
    
    print(f"Meta-features shape: {meta_features.shape}")
    print(f"Labels shape: {labels.shape if labels is not None else None}")
    
    if labels is None:
        raise ValueError("Labels not found. Cannot train meta-classifier.")
    
    # Create meta-classifier
    print(f"\n2. Creating {args.meta_model_type} meta-classifier...")
    
    if args.meta_model_type == 'mlp':
        classifier = MetaClassifier(
            model_type='mlp',
            input_dim=4,
            hidden_layers=args.hidden_layers,
            dropout=args.dropout,
            activation=args.activation
        )
    else:
        classifier = MetaClassifier(
            model_type='ridge',
            alphas=args.alphas,
            cv=args.cv
        )
    
    # Train
    print("\n3. Training meta-classifier...")
    
    if args.meta_model_type == 'mlp':
        classifier.fit(
            meta_features,
            labels,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            verbose=True
        )
    else:
        classifier.fit(meta_features, labels)
        print(f"Best alpha: {classifier.model.model.alpha_}")
    
    # Evaluate
    print("\n4. Evaluating meta-classifier...")
    probs = classifier.predict_proba(meta_features)
    preds = classifier.predict(meta_features)
    
    auc = roc_auc_score(labels, probs[:, 1])
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Save model
    if args.save_model:
        os.makedirs(args.output_dir, exist_ok=True)
        model_path = os.path.join(args.output_dir, f"meta_classifier_{args.meta_model_type}.pth")
        
        if args.meta_model_type == 'mlp':
            torch.save(classifier.model.state_dict(), model_path)
        else:
            # For sklearn models, use joblib
            import joblib
            joblib.dump(classifier.model.model, model_path.replace('.pth', '.joblib'))
        
        print(f"\n5. Model saved to {model_path}")
    
    # Save meta-features CSV
    print("\n6. Saving meta-features...")
    collector.save_meta_features(args.output_dir, "meta_train.csv")
    
    print("\n" + "="*60)
    print("Meta-Classifier Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
