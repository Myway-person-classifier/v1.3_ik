"""
HybridTrainer: Trainer with BCE + BPR + InfoNCE Loss support
Extends TextTrainer from SKKUAI with InfoNCE Loss from AIGT
"""

from torch import nn
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers.trainer import Trainer
import torch
import inspect

from utils.losses import BPRLoss, compute_infonce_loss


class HybridTrainer(Trainer):
    """
    HybridTrainer extends Trainer with support for:
    - BCE Loss (standard)
    - BPR Loss (SKKUAI)
    - InfoNCE Loss (AIGT)
    """
    
    def __init__(self, args_original, **kwargs):
        super().__init__(**kwargs)
        self.args_original = args_original
        
        # Initialize loss functions
        self.loss_fn_list = [('bce', nn.BCEWithLogitsLoss(), 1.0)]
        
        if args_original.use_bpr_loss:
            self.loss_fn_list.append(
                ('bpr', BPRLoss(), args_original.bpr_loss_weight)
            )
            print("Using BPR loss in addition to BCEWithLogitsLoss")
        
        # InfoNCE Loss will be handled in compute_loss
        self.use_infonce_loss = getattr(args_original, 'use_infonce_loss', False)
        if self.use_infonce_loss:
            print(f"Using InfoNCE Loss with lambda_cl={getattr(args_original, 'lambda_cl', 0.1)}")
    
    def compute_loss(self, model, inputs, return_outputs=False, *args, **kwargs):
        """
        Compute loss with support for BCE, BPR, and InfoNCE Loss
        
        Args:
            model: The model to compute loss for
            inputs: Input dictionary
            return_outputs: Whether to return outputs
            
        Returns:
            Total loss (and optionally outputs)
        """
        # Check if model is AvsHModel or HybridAvsHModel
        is_avsh_model = (
            self.args_original.model_name == 'AvsHModel' or 
            self.args_original.model_name == 'HybridAvsH' or
            hasattr(model, '__class__') and 'AvsH' in model.__class__.__name__
        )
        
        if is_avsh_model:
            return self._compute_loss_avsh(model, inputs, return_outputs)
        else:
            return self._compute_loss_standard(model, inputs, return_outputs)
    
    def _compute_loss_avsh(self, model, inputs, return_outputs=False):
        """Compute loss for AvsHModel or HybridAvsHModel"""
        # Make a copy to avoid mutating original inputs
        inputs = inputs.copy()
        
        total_label = inputs.pop("total_labels", None)
        total_label = total_label.float() if total_label is not None else None
        paragraph_label = inputs.pop("paragraph_labels", None)
        paragraph_label = paragraph_label.float() if paragraph_label is not None else None
        
        # Check if model supports InfoNCE Loss
        # For AvsH models, contrastive_labels should be same as total_label if not provided
        contrastive_labels = inputs.pop("contrastive_labels", None)
        if contrastive_labels is None and total_label is not None and self.use_infonce_loss:
            # Use total_label as contrastive_labels if not provided
            contrastive_labels = total_label
        lambda_cl = getattr(self.args_original, 'lambda_cl', 0.1)
        
        # Check if model forward supports contrastive_labels
        forward_sig = inspect.signature(model.forward)
        has_contrastive = 'contrastive_labels' in forward_sig.parameters
        
        if self.use_infonce_loss and has_contrastive and contrastive_labels is not None:
            # Use HybridAvsHModel with InfoNCE Loss
            outputs = model(
                **inputs,
                contrastive_labels=contrastive_labels,
                lambda_cl=lambda_cl
            )
            if len(outputs) == 3:
                total_logits, paragraph_logits, cl_loss = outputs
            else:
                total_logits, paragraph_logits = outputs
                cl_loss = None
        else:
            # Standard AvsHModel forward
            total_logits, paragraph_logits = model(**inputs)
            cl_loss = None
        
        if paragraph_label is None:
            paragraph_label = torch.zeros_like(
                paragraph_logits, 
                dtype=torch.float, 
                device=paragraph_logits.device
            )
        
        total_loss = torch.tensor(0.0, device=total_logits.device)
        
        if not self.args_original.split_valid_by_paragraph:
            for loss_name, loss_fn, weight in self.loss_fn_list:
                if loss_name == 'bce':
                    # Process paragraph and total labels
                    flat_paragraph_logits = paragraph_logits.view(-1)
                    flat_paragraph_label = paragraph_label.view(-1)
                    flat_total_logits = total_logits.view(-1)
                    flat_total_label = total_label.view(-1)
                    
                    # Remove padding
                    flat_paragraph_logits = flat_paragraph_logits[
                        flat_paragraph_label != -1
                    ]
                    flat_paragraph_label = flat_paragraph_label[
                        flat_paragraph_label != -1
                    ]
                    
                    if total_label is not None:
                        total_loss += loss_fn(flat_total_logits, flat_total_label) * weight
                    if len(flat_paragraph_logits) > 0:
                        total_loss += loss_fn(
                            flat_paragraph_logits, 
                            flat_paragraph_label
                        ) * weight
                else:
                    # BPR Loss
                    if total_label is not None:
                        total_loss += loss_fn(
                            total_logits.view(-1), 
                            total_label.view(-1)
                        ) * weight
                    # BPR for paragraphs (flatten and remove padding)
                    flat_para_logits = paragraph_logits.view(-1)
                    flat_para_labels = paragraph_label.view(-1)
                    valid_mask = flat_para_labels != -1
                    if valid_mask.sum() > 0:
                        total_loss += loss_fn(
                            flat_para_logits[valid_mask],
                            flat_para_labels[valid_mask]
                        ) * weight
        
        # Add InfoNCE Loss if computed
        if cl_loss is not None:
            total_loss += lambda_cl * cl_loss
        
        label = total_label if total_label is not None else torch.zeros_like(
            total_logits.view(-1), dtype=torch.float, device=total_logits.device
        )
        logits = total_logits.view(-1)
        
        if return_outputs:
            return total_loss, logits, label
        return total_loss
    
    def _compute_loss_standard(self, model, inputs, return_outputs=False):
        """Compute loss for standard models (Gemma3, Qwen3, etc.)"""
        # Make a copy to avoid mutating original inputs
        inputs = inputs.copy()
        
        label = inputs.pop("labels", None)
        label = label.float() if label is not None else None
        
        # Check for InfoNCE Loss support
        contrastive_labels = inputs.pop("contrastive_labels", None)
        # Use label as contrastive_labels if not provided
        if contrastive_labels is None and label is not None and self.use_infonce_loss:
            contrastive_labels = label
        lambda_cl = getattr(self.args_original, 'lambda_cl', 0.1)
        temperature = getattr(self.args_original, 'temperature', 0.07)
        
        # Check if model supports InfoNCE
        forward_sig = inspect.signature(model.forward)
        has_contrastive = 'contrastive_labels' in forward_sig.parameters
        
        if has_contrastive and contrastive_labels is not None:
            # Pass contrastive_labels to model
            output = model(
                **inputs,
                labels=label,
                contrastive_labels=contrastive_labels,
                lambda_cl=lambda_cl,
                temperature=temperature
            )
            logits = output.logits.view(-1)
            total_loss = output.loss
        else:
            # Standard forward
            output = model(**inputs, labels=label)
            logits = output.logits.view(-1)
            total_loss = output.loss
        
        if label is None:
            label = torch.zeros_like(logits, dtype=torch.float, device=logits.device)
        
        if return_outputs:
            return total_loss, logits, label
        return total_loss
    
    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        *args, **kwargs
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Prediction step for evaluation"""
        model.eval()
        
        with torch.no_grad():
            eval_loss, pred, label = self.compute_loss(
                model, inputs, return_outputs=True
            )
        
        if self.args_original.split_valid_by_paragraph:
            pred = pred.view(1, -1)
            label = label.view(1, -1)
        
        return (eval_loss, pred, label)


# For backward compatibility
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl


class StopAfterEpoch(TrainerCallback):
    """Callback to stop training after specified number of epochs"""
    
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs
    
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs
    ) -> TrainerControl:
        if state.epoch >= self.max_epochs:
            control.should_training_stop = True
        return control
