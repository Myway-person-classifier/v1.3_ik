"""
Loss Functions: BPR Loss and InfoNCE Loss
Combines SKKUAI BPR Loss with AIGT InfoNCE Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BPRLoss(nn.Module):
    """
    Bayesian Personalized Ranking (BPR) loss
    
    Args:
        logits: 1-D tensor [batch] - each sample's prediction score
        labels: 1-D tensor [batch] - binary (1 = positive, 0 = negative)
    
    Note:
        The mini-batch must contain at least one positive and one negative sample.
    """
    
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps  # Small value to prevent log(0)
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Separate positive and negative samples
        pos_scores = logits[labels == 1]  # [P]
        neg_scores = logits[labels == 0]  # [N]
        
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            # If batch doesn't contain both classes, return zero loss
            return logits.new_tensor(0.0, requires_grad=True)
        
        # Compute differences between all positive-negative pairs: s_i - s_j
        diff = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)  # [P, N]
        
        # BPR: -log σ(s_i - s_j)
        loss = -torch.log(torch.sigmoid(diff) + self.eps).mean()
        return loss


def compute_infonce_loss(features, labels, temperature=0.07):
    """
    InfoNCE (NT-Xent) Loss for contrastive learning
    
    Args:
        features: [B, D] tensor of embeddings
        labels: [B] tensor of class labels (0 or 1)
        temperature: Temperature parameter for softmax scaling (default: 0.07)
        
    Returns:
        InfoNCE loss value
        
    Reference:
        Based on AIGT project's implementation
    """
    # L2 normalize features for cosine similarity
    f = F.normalize(features.float(), dim=-1)
    
    # Compute similarity matrix
    sim = torch.matmul(f, f.T)  # [B, B] cosine similarities
    sim = sim / temperature  # Scale by temperature
    
    # Create positive/negative masks
    labels = labels.view(-1)
    pos_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # Same class -> True
    pos_mask.fill_diagonal_(False)  # Remove self-similarity
    
    # Suppress self-similarity in similarity matrix
    sim = sim - torch.eye(sim.size(0), device=sim.device) * 1e9
    
    # Compute InfoNCE loss
    # log_prob = log(exp(sim) / sum(exp(sim)))
    log_prob = sim - sim.logsumexp(dim=1, keepdim=True)
    
    # Average over positive pairs
    loss = -(log_prob * pos_mask).sum() / pos_mask.sum().clamp_min(1)
    
    return loss


# Alias for backward compatibility
BPRloss = BPRLoss
