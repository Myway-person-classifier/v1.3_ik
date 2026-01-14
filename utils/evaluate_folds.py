"""
Fold별 평가 지표 계산 및 시각화
각 Fold와 Meta-Classifier의 성능을 비교 분석
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import json


def load_fold_results(logit_dir: str = "./outputs/fold_logits") -> Dict:
    """
    Fold별 logits와 labels 로드
    
    Returns:
        Dictionary with fold results
    """
    results = {}
    
    # Fold 0 (OOF)
    oof_logits_path = os.path.join(logit_dir, "oof", "fold0_logits.npy")
    oof_labels_path = os.path.join(logit_dir, "oof", "fold0_labels.npy")
    
    if os.path.exists(oof_logits_path):
        results['fold_0'] = {
            'logits': np.load(oof_logits_path),
            'labels': np.load(oof_labels_path) if os.path.exists(oof_labels_path) else None,
            'type': 'OOF'
        }
    
    # Fold 1, 2, 3 (Test)
    for fold_idx in [1, 2, 3]:
        test_logits_path = os.path.join(logit_dir, "test", f"fold{fold_idx}_logits.npy")
        test_labels_path = os.path.join(logit_dir, "test", f"fold{fold_idx}_labels.npy")
        
        if os.path.exists(test_logits_path):
            results[f'fold_{fold_idx}'] = {
                'logits': np.load(test_logits_path),
                'labels': np.load(test_labels_path) if os.path.exists(test_labels_path) else None,
                'type': 'Test'
            }
    
    return results


def calculate_metrics(logits: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    평가 지표 계산
    
    Args:
        logits: 예측 logits
        labels: 실제 레이블
        
    Returns:
        Dictionary of metrics
    """
    preds = (logits > 0).astype(int)
    labels_binary = (labels > 0.5).astype(int) if labels.dtype != int else labels
    
    metrics = {
        'accuracy': accuracy_score(labels_binary, preds),
        'f1_score': f1_score(labels_binary, preds, average='macro'),
        'precision': precision_score(labels_binary, preds, average='macro'),
        'recall': recall_score(labels_binary, preds, average='macro'),
        'roc_auc': roc_auc_score(labels_binary, logits)
    }
    
    return metrics


def evaluate_all_folds(logit_dir: str = "./outputs/fold_logits") -> pd.DataFrame:
    """
    모든 Fold 평가 지표 계산
    
    Returns:
        DataFrame with fold metrics
    """
    results = load_fold_results(logit_dir)
    metrics_list = []
    
    for fold_name, fold_data in results.items():
        if fold_data['labels'] is not None:
            metrics = calculate_metrics(fold_data['logits'], fold_data['labels'])
            metrics['fold'] = fold_name
            metrics['type'] = fold_data['type']
            metrics_list.append(metrics)
    
    df = pd.DataFrame(metrics_list)
    return df


def evaluate_meta_classifier(
    meta_model_path: str,
    logit_dir: str = "./outputs/fold_logits",
    meta_model_type: str = 'mlp'
) -> Dict[str, float]:
    """
    Meta-Classifier 평가
    
    Args:
        meta_model_path: Meta-Classifier 모델 경로
        logit_dir: Logits 디렉토리
        meta_model_type: 'mlp' or 'ridge'
        
    Returns:
        Dictionary of metrics
    """
    from utils.logit_collector import LogitCollector
    from meta.meta_classifier import MetaClassifier
    import torch
    
    # Load meta-features
    collector = LogitCollector(logit_dir)
    meta_features, labels = collector.collect_logits()
    
    if labels is None:
        raise ValueError("Labels not found for meta-classifier evaluation")
    
    # Load meta-classifier
    if meta_model_type == 'mlp':
        classifier = MetaClassifier(
            model_type='mlp',
            input_dim=4,
            hidden_layers=[64, 32],
            dropout=0.2
        )
        classifier.model.load_state_dict(torch.load(meta_model_path))
        classifier.model.eval()
    else:
        import joblib
        ridge_model = joblib.load(meta_model_path)
        classifier = MetaClassifier(model_type='ridge')
        classifier.model.model = ridge_model
        classifier.model.is_fitted = True
    
    # Predict
    probs = classifier.predict_proba(meta_features)
    preds = classifier.predict(meta_features)
    
    # Calculate metrics
    labels_binary = (labels > 0.5).astype(int) if labels.dtype != int else labels.astype(int)
    metrics = {
        'accuracy': accuracy_score(labels_binary, preds),
        'f1_score': f1_score(labels_binary, preds, average='macro'),
        'precision': precision_score(labels_binary, preds, average='macro'),
        'recall': recall_score(labels_binary, preds, average='macro'),
        'roc_auc': roc_auc_score(labels_binary, probs[:, 1])
    }
    
    # Print detailed report
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
    
    return metrics


def plot_metrics_comparison(
    fold_metrics_df: pd.DataFrame,
    meta_metrics: Dict[str, float] = None,
    output_dir: str = "./outputs/evaluation"
):
    """
    Fold별 및 Meta-Classifier 성능 비교 시각화
    
    Args:
        fold_metrics_df: Fold별 metrics DataFrame
        meta_metrics: Meta-Classifier metrics (optional)
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Metrics Comparison Bar Plot
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx]
        
        # Fold metrics
        fold_data = fold_metrics_df.groupby('fold')[metric].mean()
        x_pos = np.arange(len(fold_data))
        bars = ax.bar(x_pos, fold_data.values, alpha=0.7, label='Fold Average')
        
        # Meta-Classifier (if available)
        if meta_metrics and metric in meta_metrics:
            ax.axhline(y=meta_metrics[metric], color='r', linestyle='--', 
                      linewidth=2, label='Meta-Classifier')
        
        ax.set_xlabel('Fold')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Comparison')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(fold_data.index)
        ax.legend()
        ax.grid(alpha=0.3)
    
    # Remove empty subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. ROC Curves
    fig, ax = plt.subplots(figsize=(10, 8))
    
    results = load_fold_results()
    for fold_name, fold_data in results.items():
        if fold_data['labels'] is not None:
            labels_binary = (fold_data['labels'] > 0.5).astype(int)
            fpr, tpr, _ = roc_curve(labels_binary, fold_data['logits'])
            auc = roc_auc_score(labels_binary, fold_data['logits'])
            ax.plot(fpr, tpr, label=f'{fold_name} (AUC={auc:.3f})', linewidth=2)
    
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves - Fold Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Metrics Summary Table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for _, row in fold_metrics_df.iterrows():
        table_data.append([
            row['fold'],
            f"{row['accuracy']:.4f}",
            f"{row['f1_score']:.4f}",
            f"{row['precision']:.4f}",
            f"{row['recall']:.4f}",
            f"{row['roc_auc']:.4f}"
        ])
    
    if meta_metrics:
        table_data.append([
            'Meta-Classifier',
            f"{meta_metrics['accuracy']:.4f}",
            f"{meta_metrics['f1_score']:.4f}",
            f"{meta_metrics['precision']:.4f}",
            f"{meta_metrics['recall']:.4f}",
            f"{meta_metrics['roc_auc']:.4f}"
        ])
    
    table = ax.table(
        cellText=table_data,
        colLabels=['Model', 'Accuracy', 'F1-Score', 'Precision', 'Recall', 'ROC-AUC'],
        cellLoc='center',
        loc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    plt.title('Evaluation Metrics Summary', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'metrics_table.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualizations saved to {output_dir}")


def generate_evaluation_report(
    logit_dir: str = "./outputs/fold_logits",
    meta_model_path: str = None,
    meta_model_type: str = 'mlp',
    output_dir: str = "./outputs/evaluation"
):
    """
    전체 평가 리포트 생성
    
    Args:
        logit_dir: Logits 디렉토리
        meta_model_path: Meta-Classifier 모델 경로 (optional)
        meta_model_type: 'mlp' or 'ridge'
        output_dir: 출력 디렉토리
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("Evaluation Report Generation")
    print("="*60)
    
    # 1. Evaluate all folds
    print("\n1. Evaluating all folds...")
    fold_metrics_df = evaluate_all_folds(logit_dir)
    print(fold_metrics_df.to_string(index=False))
    
    # Print detailed reports for each fold
    print("\n" + "="*60)
    print("Fold별 상세 리포트")
    print("="*60)
    results = load_fold_results(logit_dir)
    for fold_name, fold_data in results.items():
        if fold_data['labels'] is not None:
            print(f"\n{fold_name} ({fold_data['type']}):")
            logits = fold_data['logits']
            labels = fold_data['labels']
            preds = (logits > 0).astype(int)
            labels_binary = (labels > 0.5).astype(int) if labels.dtype != int else labels.astype(int)
            
            print("\n📋 상세 리포트 (0: Human, 1: AI)")
            print("-" * 60)
            print(classification_report(labels_binary, preds, target_names=['Human', 'AI']))
            
            print("\n🔍 Confusion Matrix")
            cm = confusion_matrix(labels_binary, preds)
            print(cm)
            print("( [TN FP]")
            print("  [FN TP] )")
    
    # Save fold metrics
    fold_metrics_path = os.path.join(output_dir, 'fold_metrics.csv')
    fold_metrics_df.to_csv(fold_metrics_path, index=False)
    print(f"\nFold metrics saved to {fold_metrics_path}")
    
    # 2. Evaluate meta-classifier (if available)
    meta_metrics = None
    if meta_model_path and os.path.exists(meta_model_path):
        print("\n2. Evaluating meta-classifier...")
        try:
            meta_metrics = evaluate_meta_classifier(meta_model_path, logit_dir, meta_model_type)
            print("Meta-Classifier Metrics:")
            for metric, value in meta_metrics.items():
                print(f"  {metric}: {value:.4f}")
        except Exception as e:
            print(f"Warning: Could not evaluate meta-classifier: {e}")
    
    # 3. Generate visualizations
    print("\n3. Generating visualizations...")
    plot_metrics_comparison(fold_metrics_df, meta_metrics, output_dir)
    
    # 4. Save summary JSON
    summary = {
        'fold_metrics': fold_metrics_df.to_dict('records'),
        'meta_metrics': meta_metrics,
        'average_fold_metrics': fold_metrics_df[['accuracy', 'f1_score', 'precision', 'recall', 'roc_auc']].mean().to_dict()
    }
    
    summary_path = os.path.join(output_dir, 'evaluation_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")
    
    print("\n" + "="*60)
    print("Evaluation Report Complete!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Folds and Meta-Classifier")
    parser.add_argument('--logit_dir', type=str, default='./outputs/fold_logits',
                       help='Directory containing fold logits')
    parser.add_argument('--meta_model_path', type=str, default=None,
                       help='Path to meta-classifier model')
    parser.add_argument('--meta_model_type', type=str, default='mlp',
                       choices=['mlp', 'ridge'],
                       help='Type of meta-classifier')
    parser.add_argument('--output_dir', type=str, default='./outputs/evaluation',
                       help='Output directory for evaluation results')
    
    args = parser.parse_args()
    
    generate_evaluation_report(
        logit_dir=args.logit_dir,
        meta_model_path=args.meta_model_path,
        meta_model_type=args.meta_model_type,
        output_dir=args.output_dir
    )

