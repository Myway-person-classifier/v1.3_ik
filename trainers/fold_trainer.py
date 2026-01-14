"""
FoldTrainer: Manages fold-based training and logit generation
Handles 4-Fold cross-validation training with logit collection
"""

import os
import numpy as np
from typing import Tuple, Optional
import torch
from transformers import TrainingArguments, Trainer

from data.get_dataset import get_dataset
from data.text_collator import TextCollator
from trainers.hybrid_trainer import HybridTrainer
from utils.compute_metrics import get_metric
from utils.arguments import get_arguments
from transformers import AutoTokenizer


class FoldTrainer:
    """
    Manages fold-based training and logit collection for v1.3 architecture
    
    Features:
    - 4-Fold cross-validation training
    - OOF (Out-of-Fold) logits collection (Fold 0)
    - Test logits collection (Fold 1, 2, 3)
    - Automatic logit saving
    """
    
    def __init__(self, args):
        self.args = args
        # Colab 환경 지원: 환경 변수 또는 args에서 출력 경로 가져오기
        base_output_dir = getattr(args, 'output_dir', None) or os.environ.get('OUTPUT_DIR', './outputs')
        
        # 여러 모델 지원: 모델별로 logit 디렉토리 분리
        model_name = getattr(args, 'model_name', 'default')
        self.logit_dir = os.path.join(base_output_dir, "fold_logits", model_name)
        os.makedirs(f"{self.logit_dir}/oof", exist_ok=True)
        os.makedirs(f"{self.logit_dir}/test", exist_ok=True)
    
    def train_fold(self, fold_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Train a single fold and collect logits
        
        Args:
            fold_idx: Index of the fold to train (0-3)
            
        Returns:
            Tuple of (logits, labels) for the validation/test set
        """
        if getattr(self.args, 'predict_only', False):
            return self.predict_fold(fold_idx)

        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx}")
        print(f"{'='*60}\n")
        
        # ... (rest of the existing train_fold code)
    
    def predict_fold(self, fold_idx: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Run inference only for a single fold
        """
        print(f"\n{'='*60}")
        print(f"Inference only: Fold {fold_idx}")
        print(f"{'='*60}\n")
        
        self.args.fold_idx = fold_idx
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.embedding_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get datasets
        datasets = get_dataset(self.args, tokenizer)
        val_dataset = datasets['val']
        
        print(f"✅ Dataset loaded: size={len(val_dataset)}")
        
        # Create collator
        collator = TextCollator(self.args, tokenizer)
        
        # Load model
        model = self._load_model()
        
        # Load weights
        base_output_dir = getattr(self.args, 'output_dir', None) or os.environ.get('OUTPUT_DIR', './outputs')
        model_path = os.path.join(base_output_dir, f"fold_{fold_idx}", "best_model")
        
        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}")
            # Try to load state dict if it's a pytorch model, otherwise use from_pretrained
            if hasattr(model, 'load_state_dict'):
                try:
                    # HuggingFace models might need special handling
                    if hasattr(model, 'from_pretrained'):
                        model = model.__class__.from_pretrained(model_path)
                    else:
                        state_dict = torch.load(os.path.join(model_path, "pytorch_model.bin"), map_location='cpu')
                        model.load_state_dict(state_dict)
                except Exception as e:
                    print(f"Warning: Could not load weights: {e}")
        
        # Training arguments (needed for Trainer initialization)
        training_args = TrainingArguments(
            output_dir="./tmp",
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            fp16=torch.cuda.is_available(),
            report_to=None,
        )
        
        # Create trainer
        trainer = HybridTrainer(
            args_original=self.args,
            model=model,
            args=training_args,
            data_collator=collator,
        )
        
        # Collect logits
        save_type = 'test' if self.args.is_submission else 'oof'
        logits, labels = self._collect_logits(trainer, val_dataset, collator, fold_idx, save_type=save_type)
        
        return logits, labels

    def _load_model(self):
        """Load model based on args.model_name"""
        if self.args.model_name == 'AvsHModel' or self.args.model_name == 'HybridAvsH':
            from models.AvsHModel import AvsHModel
            if self.args.model_name == 'HybridAvsH':
                from models.hybrid_model import HybridAvsHModel
                model = HybridAvsHModel(self.args)
            else:
                model = AvsHModel(self.args)
        elif self.args.model_name == 'Gemma3InfoNCE':
            from models.gemma3_seqcls_infonce import Gemma3ForSequenceClassification
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                self.args.embedding_model,
                num_labels=self.args.num_labels
            )
            model = Gemma3ForSequenceClassification.from_pretrained(
                self.args.embedding_model,
                config=config
            )
        elif self.args.model_name == 'Qwen3InfoNCE':
            from models.qwen3_seqcls_infonce import Qwen3ForSequenceClassificationCL
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(
                self.args.embedding_model,
                num_labels=self.args.num_labels
            )
            model = Qwen3ForSequenceClassificationCL.from_pretrained(
                self.args.embedding_model,
                config=config
            )
        else:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(
                self.args.embedding_model,
                num_labels=self.args.num_labels
            )
        
        return model
    
    def _collect_logits(
        self, 
        trainer: Trainer, 
        dataset, 
        collator,
        fold_idx: int,
        save_type: str = 'oof'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect logits from validation/test dataset
        
        Args:
            trainer: Trained trainer
            dataset: Validation/test dataset
            collator: Data collator
            fold_idx: Fold index
            save_type: 'oof' or 'test'
            
        Returns:
            Tuple of (logits, labels)
        """
        trainer.model.eval()
        all_logits = []
        all_labels = []
        
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=collator,
            shuffle=False
        )
        
        with torch.no_grad():
            for batch in dataloader:
                # Move to device
                device = next(trainer.model.parameters()).device
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Get predictions
                # For AvsH models, use model directly instead of compute_loss to avoid label issues
                is_avsh_model = (
                    self.args.model_name == 'AvsHModel' or 
                    self.args.model_name == 'HybridAvsH'
                )
                
                if is_avsh_model:
                    # For AvsH models - get logits directly
                    with torch.no_grad():
                        total_logits, _ = trainer.model(**batch)
                        logits = total_logits.view(-1).cpu().numpy()
                        labels = batch.get('total_labels', batch.get('labels', None))
                        if labels is not None:
                            labels = labels.cpu().numpy()
                else:
                    # Standard models
                    outputs = trainer.model(**batch)
                    logits = outputs.logits.view(-1).cpu().numpy()
                    labels = batch.get('labels', None)
                    if labels is not None:
                        labels = labels.cpu().numpy()
                
                all_logits.append(logits)
                if labels is not None:
                    all_labels.append(labels)
        
        # Concatenate
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0) if all_labels else None
        
        # Save logits
        target_dir = os.path.join(self.logit_dir, save_type)
        os.makedirs(target_dir, exist_ok=True)
        
        logit_path = f"{target_dir}/fold{fold_idx}_logits.npy"
        label_path = f"{target_dir}/fold{fold_idx}_labels.npy"
        
        np.save(logit_path, logits)
        if labels is not None:
            np.save(label_path, labels)
        
        print(f"Saved {save_type.upper()} logits to {logit_path}")
        
        return logits, labels


def train_all_folds(args):
    """
    Train all 4 folds sequentially
    
    Args:
        args: Training arguments
    """
    fold_trainer = FoldTrainer(args)
    
    for fold_idx in range(4):
        try:
            logits, labels = fold_trainer.train_fold(fold_idx)
            print(f"✅ Fold {fold_idx} completed successfully")
        except Exception as e:
            print(f"❌ Fold {fold_idx} failed with error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "="*60)
    print("All folds completed!")
    print("="*60)


def train_multiple_models(model_names, base_args):
    """
    Train multiple models sequentially, each with all 4 folds
    
    All models use the same Fold split (k_fold_split.json) to ensure
    consistent data splits across models.
    
    Args:
        model_names: List of model names to train (e.g., ['HybridAvsH', 'Gemma3InfoNCE'])
        base_args: Base configuration (will be copied for each model)
    
    Returns:
        Dictionary mapping model_name to logit directory
    """
    import copy
    
    print("="*60)
    print(f"Training {len(model_names)} models: {model_names}")
    print("="*60)
    print("Note: All models will use the same Fold split for consistency.\n")
    
    model_logit_dirs = {}
    
    for model_idx, model_name in enumerate(model_names):
        print(f"\n{'='*60}")
        print(f"Model {model_idx+1}/{len(model_names)}: {model_name}")
        print(f"{'='*60}\n")
        
        # Copy args and set model name
        args = copy.deepcopy(base_args)
        args.model_name = model_name
        
        # Train all folds for this model
        fold_trainer = FoldTrainer(args)
        
        for fold_idx in range(4):
            try:
                logits, labels = fold_trainer.train_fold(fold_idx)
                print(f"✅ {model_name} - Fold {fold_idx} completed")
            except Exception as e:
                print(f"❌ {model_name} - Fold {fold_idx} failed: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        # Store logit directory
        base_output_dir = getattr(args, 'output_dir', None) or os.environ.get('OUTPUT_DIR', './outputs')
        model_logit_dirs[model_name] = os.path.join(base_output_dir, "fold_logits", model_name)
        print(f"\n✅ {model_name} training completed!")
        print(f"   Logits saved to: {model_logit_dirs[model_name]}\n")
    
    print("\n" + "="*60)
    print(f"All {len(model_names)} models completed!")
    print("="*60)
    print("\nLogit directories:")
    for model_name, logit_dir in model_logit_dirs.items():
        print(f"  - {model_name}: {logit_dir}")
    
    return model_logit_dirs


def main():
    """
    Main entry point for fold training
    Can be used to train a single fold or all folds
    """
    args = get_arguments()
    
    # If fold_idx is specified, train only that fold
    if args.fold_idx is not None and args.fold_idx >= 0:
        fold_trainer = FoldTrainer(args)
        logits, labels = fold_trainer.train_fold(args.fold_idx)
        print(f"✅ Fold {args.fold_idx} completed successfully")
    else:
        # Train all folds
        train_all_folds(args)


if __name__ == "__main__":
    main()
