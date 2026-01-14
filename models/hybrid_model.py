"""
HybridAvsHModel: AvsHModel with InfoNCE Loss support
Combines SKKUAI AvsHModel with AIGT InfoNCE Loss functionality
"""

import torch
from torch import nn
from transformers import AutoModel
from models.AvsHModel import AvsHModel, get_check_parameters


def compute_infonce_loss(features, labels, temperature=0.07):
    """
    InfoNCE (NT-Xent) Loss for contrastive learning
    
    Args:
        features: [B, D] tensor of embeddings
        labels: [B] tensor of class labels (0 or 1)
        temperature: Temperature parameter for softmax scaling
        
    Returns:
        InfoNCE loss value
    """
    # L2 normalize features for cosine similarity
    f = nn.functional.normalize(features.float(), dim=-1)
    
    # Compute similarity matrix
    sim = torch.matmul(f, f.T)  # [B, B]
    sim = sim / temperature  # Scale by temperature
    
    # Create positive/negative masks
    labels = labels.view(-1)
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # Same class -> True
    pos_mask.fill_diagonal_(False)  # Remove self-similarity
    
    # Suppress self-similarity
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9
    
    # Compute InfoNCE loss
    log_prob = sim - sim.logsumexp(dim=1, keepdim=True)
    loss = -(log_prob * pos_mask).sum() / pos_mask.sum().clamp_min(1)
    
    return loss


class HybridAvsHModel(AvsHModel):
    """
    HybridAvsHModel extends AvsHModel with InfoNCE Loss support
    
    This model combines:
    - AvsHModel's paragraph-level hierarchical structure
    - InfoNCE Loss for contrastive learning (from AIGT)
    """
    
    def __init__(self, args):
        super().__init__(args)
        self.use_infonce_loss = getattr(args, 'use_infonce_loss', False)
        self.temperature = getattr(args, 'temperature', 0.07)
        
        if self.use_infonce_loss:
            print(f"HybridAvsHModel: InfoNCE Loss enabled with temperature={self.temperature}")
    
    def forward(self, input_ids, attention_mask=None, chunk_size=12, 
                contrastive_labels=None, lambda_cl=0.1, **kwargs):
        """
        Forward pass with optional InfoNCE Loss
        
        Args:
            input_ids: [B, P, L] tensor
            attention_mask: [B, P, L] tensor
            chunk_size: Number of paragraphs to process at once
            contrastive_labels: [B] tensor of labels for contrastive learning
            lambda_cl: Weight for InfoNCE Loss
            **kwargs: Additional arguments
            
        Returns:
            If contrastive_labels is provided:
                (total_logits, paragraph_logits, cl_loss)
            Otherwise:
                (total_logits, paragraph_logits)
        """
        # Call parent forward to get logits
        total_logits, paragraph_logits = super().forward(
            input_ids, attention_mask, chunk_size, **kwargs
        )
        
        # Compute InfoNCE Loss if requested
        cl_loss = None
        if self.use_infonce_loss and contrastive_labels is not None:
            # Get CLS embeddings for contrastive learning
            # Extract embeddings before classifier layer using helper method
            batch_size = total_logits.size(0)
            if batch_size > 1:  # Need at least 2 samples for contrastive learning
                try:
                    # Try to get CLS embeddings
                    features = self.get_cls_embeddings(input_ids, attention_mask, chunk_size)
                    cl_loss = compute_infonce_loss(
                        features,  # [B, D]
                        contrastive_labels,
                        temperature=self.temperature
                    )
                except Exception as e:
                    # Fallback: Use logits as features (less optimal but works)
                    print(f"Warning: Could not extract CLS embeddings, using logits: {e}")
                    features = total_logits.view(-1)  # [B]
                    cl_loss = compute_infonce_loss(
                        features.unsqueeze(1),  # [B, 1] - expand to 2D
                        contrastive_labels,
                        temperature=self.temperature
                    )
        
        if cl_loss is not None:
            return total_logits, paragraph_logits, cl_loss
        else:
            return total_logits, paragraph_logits
    
    def get_cls_embeddings(self, input_ids, attention_mask=None, chunk_size=12):
        """
        Extract CLS token embeddings for contrastive learning
        
        This method extracts the CLS embeddings before the classifier layer
        """
        batch_size, num_paragraphs, seq_length = input_ids.size()
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        cls_embedding = []
        for start in range(0, num_paragraphs, chunk_size):
            end = min(start + chunk_size, num_paragraphs)
            input_chunk = input_ids[:, start:end, :].contiguous().view(-1, seq_length)
            attn_chunk = attention_mask[:, start:end, :].contiguous().view(-1, seq_length)
            
            outputs = self.embedding_model(
                input_ids=input_chunk,
                attention_mask=attn_chunk
            )
            cls_chunk = outputs.last_hidden_state[:, 0, :].view(batch_size, end - start, -1)
            cls_embedding.append(cls_chunk)
        
        cls_embedding = torch.cat(cls_embedding, dim=1)  # [B, P, D]
        
        # Create learnable CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, D]
        hidden_states = torch.cat([cls_tokens, cls_embedding], dim=1)  # [B, 1+P, D]
        
        # Create paragraph-level mask
        para_mask = (attention_mask.view(batch_size, num_paragraphs, seq_length)
                                .sum(dim=2) > 0).long()
        cls_mask = torch.ones(batch_size, 1, device=input_ids.device, dtype=para_mask.dtype)
        attention_mask_para = torch.cat([cls_mask, para_mask], dim=1)  # [B, 1+P]
        
        # Pass through transformer encoder
        encoded = self.transformer_encoder(
            hidden_states,
            src_key_padding_mask=~attention_mask_para.bool()
        )
        
        # Return CLS token embedding (before classifier)
        return encoded[:, 0, :]  # [B, D]
