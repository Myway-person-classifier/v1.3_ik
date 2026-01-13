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
        self.logit_dir = "./outputs/fold_logits"
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
        print(f"\n{'='*60}")
        print(f"Training Fold {fold_idx}")
        print(f"{'='*60}\n")
        
        # Set fold index in args
        self.args.fold_idx = fold_idx
        self.args.is_kfold = True
        self.args.k_fold = 4
        
        # Set save_dir to ensure consistent k_fold_split.json location
        # This ensures all folds use the same split file (constants_phase4/k_fold_split.json)
        if not hasattr(self.args, 'save_dir') or self.args.save_dir == 'baseline':
            self.args.save_dir = 'v1.3_fold'  # Use v1.3_fold to get constants_phase4
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.args.embedding_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Get datasets
        train_dataset, val_dataset = get_dataset(self.args, tokenizer)
        
        # Create collator
        collator = TextCollator(self.args, tokenizer)
        
        # Load model
        model = self._load_model()
        
        # Training arguments
        output_dir = f"./outputs/fold_{fold_idx}"
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.args.num_train_epochs,
            per_device_train_batch_size=self.args.per_device_train_batch_size,
            per_device_eval_batch_size=self.args.per_device_eval_batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            learning_rate=self.args.learning_rate,
            warmup_ratio=self.args.warmup_ratio,
            weight_decay=self.args.weight_decay,
            logging_steps=self.args.logging_steps,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="roc_auc",
            greater_is_better=True,
            save_total_limit=2,
            fp16=torch.cuda.is_available(),
            dataloader_num_workers=0,
            report_to=None,
        )
        
        # Create trainer
        compute_metrics = get_metric(self.args)
        
        trainer = HybridTrainer(
            args_original=self.args,
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model(f"{output_dir}/best_model")
        
        # Collect logits
        print(f"\nCollecting logits for Fold {fold_idx}...")
        logits, labels = self._collect_logits(trainer, val_dataset, collator, fold_idx)
        
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
        fold_idx: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Collect logits from validation/test dataset
        
        Args:
            trainer: Trained trainer
            dataset: Validation/test dataset
            collator: Data collator
            fold_idx: Fold index
            
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
        if fold_idx == 0:
            # OOF logits (Fold 0 validation set)
            np.save(f"{self.logit_dir}/oof/fold0_logits.npy", logits)
            if labels is not None:
                np.save(f"{self.logit_dir}/oof/fold0_labels.npy", labels)
            print(f"Saved OOF logits to {self.logit_dir}/oof/fold0_logits.npy")
        else:
            # Test logits (Fold 1, 2, 3 validation sets)
            np.save(f"{self.logit_dir}/test/fold{fold_idx}_logits.npy", logits)
            if labels is not None:
                np.save(f"{self.logit_dir}/test/fold{fold_idx}_labels.npy", labels)
            print(f"Saved test logits to {self.logit_dir}/test/fold{fold_idx}_logits.npy")
        
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
