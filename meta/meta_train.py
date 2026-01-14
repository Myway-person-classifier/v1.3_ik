"""
Meta-Classifier Training Script
Trains MLP or Ridge classifier on fold logits
"""

import os
import argparse
import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    classification_report, confusion_matrix
)
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
    # Colab 환경 지원: 환경 변수로 기본 경로 설정 가능
    default_output_dir = os.environ.get('OUTPUT_DIR', './outputs')
    default_logit_dir = os.path.join(default_output_dir, 'fold_logits')
    default_meta_output_dir = os.path.join(default_output_dir, 'meta_features')
    
    parser.add_argument(
        '--logit_dir',
        type=str,
        default=default_logit_dir,
        help='Directory containing fold logits (can be set via OUTPUT_DIR env var)'
    )
    parser.add_argument(
        '--model_names',
        type=str,
        nargs='+',
        default=None,
        help='List of model names for multi-model ensemble (e.g., HybridAvsH Gemma3InfoNCE). If None, single model mode.'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default=default_meta_output_dir,
        help='Output directory for meta-features and model (can be set via OUTPUT_DIR env var)'
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
    
    # Multi-model or single model mode
    if args.model_names and len(args.model_names) > 1:
        print(f"Multi-model mode: {args.model_names}")
        from utils.logit_collector import MultiModelLogitCollector
        collector = MultiModelLogitCollector(args.logit_dir, args.model_names)
        meta_features, labels = collector.collect_logits()
        input_dim = len(args.model_names) * 4  # M models × 4 folds
    else:
        print("Single model mode")
        # 단일 모델일 때: logit_dir이 base directory인 경우 model_name 하위 디렉토리 찾기
        if os.path.exists(args.logit_dir):
            # logit_dir이 base directory인 경우 (outputs/fold_logits)
            # 하위에 모델 디렉토리가 있는지 확인
            model_dirs = [d for d in os.listdir(args.logit_dir) 
                         if os.path.isdir(os.path.join(args.logit_dir, d)) 
                         and os.path.exists(os.path.join(args.logit_dir, d, 'oof'))]
            
            if len(model_dirs) == 1:
                # 단일 모델 디렉토리 발견
                model_name = model_dirs[0]
                actual_logit_dir = os.path.join(args.logit_dir, model_name)
                print(f"  Found single model directory: {model_name}")
                print(f"  Using logit_dir: {actual_logit_dir}")
                collector = LogitCollector(actual_logit_dir)
            elif len(model_dirs) == 0:
                # 기존 구조 (logit_dir 바로 아래에 oof/test)
                print(f"  Using legacy structure: {args.logit_dir}")
                collector = LogitCollector(args.logit_dir)
            else:
                # 여러 모델 디렉토리 발견 - 에러
                raise ValueError(
                    f"Multiple model directories found in {args.logit_dir}: {model_dirs}. "
                    f"Please specify --model_names to use multi-model mode."
                )
        else:
            # 기존 구조
            collector = LogitCollector(args.logit_dir)
        
        meta_features, labels = collector.collect_logits()
        input_dim = 4  # 4 folds
    
    print(f"Meta-features shape: {meta_features.shape}")
    print(f"Labels shape: {labels.shape if labels is not None else None}")
    print(f"Input dimension for Meta-Classifier: {input_dim}")
    
    if labels is None:
        raise ValueError("Labels not found. Cannot train meta-classifier.")
    
    # Create meta-classifier
    print(f"\n2. Creating {args.meta_model_type} meta-classifier...")
    
    if args.meta_model_type == 'mlp':
        classifier = MetaClassifier(
            model_type='mlp',
            input_dim=input_dim,  # Dynamic input dimension
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
    
    # Convert labels to binary if needed
    labels_binary = (labels > 0.5).astype(int) if labels.dtype != int else labels.astype(int)
    
    auc = roc_auc_score(labels_binary, probs[:, 1])
    acc = accuracy_score(labels_binary, preds)
    f1 = f1_score(labels_binary, preds)
    
    print(f"ROC-AUC: {auc:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Detailed classification report
    print("\n" + "📋 상세 리포트 (0: Human, 1: AI)")
    print("-" * 60)
    print(classification_report(labels_binary, preds, target_names=['Human', 'AI']))
    
    # Confusion matrix
    print("\n" + "-" * 60)
    print("🔍 Confusion Matrix")
    cm = confusion_matrix(labels_binary, preds)
    print(cm)
    print("( [TN FP]")
    print("  [FN TP] )")
    
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
