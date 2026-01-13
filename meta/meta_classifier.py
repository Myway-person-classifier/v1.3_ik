"""
MetaClassifier: MLP or Ridge Regression for meta-learning
Takes fold logits as input and produces final predictions
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifierCV
from typing import Optional, Union


class MLPClassifierTorch(nn.Module):
    """
    Multi-Layer Perceptron for meta-classification
    
    Architecture:
    Input (4 features) → Hidden Layer 1 (64 units, ReLU, Dropout)
                       → Hidden Layer 2 (32 units, ReLU, Dropout)
                       → Output (1 unit, Sigmoid)
    """
    
    def __init__(
        self,
        input_dim: int = 4,
        hidden_layers: list = [64, 32],
        dropout: float = 0.2,
        activation: str = 'relu'
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, hidden_dim))
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            else:
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze(-1)
    
    def fit(self, X, y, epochs=100, batch_size=32, lr=0.001, verbose=True):
        """
        Train the MLP classifier
        
        Args:
            X: [N, 4] array of meta-features
            y: [N] array of labels
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            verbose: Whether to print training progress
        """
        self.train()
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y)
        
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True
        )
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloader):.4f}")
    
    def predict_proba(self, X):
        """Predict probabilities"""
        self.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            probs = self(X_tensor).numpy()
        # Return in format [prob_class_0, prob_class_1]
        return np.stack([1 - probs, probs], axis=1)
    
    def predict(self, X):
        """Predict binary labels"""
        probs = self.predict_proba(X)
        return (probs[:, 1] > 0.5).astype(int)


class RidgeMetaClassifier:
    """
    Ridge Regression meta-classifier using sklearn
    
    Uses cross-validation to find optimal alpha (regularization strength)
    """
    
    def __init__(self, alphas=None, cv=5, scoring='roc_auc'):
        """
        Args:
            alphas: List of alpha values to try (default: logspace(-4, 2, 25))
            cv: Number of cross-validation folds
            scoring: Scoring metric
        """
        if alphas is None:
            alphas = np.logspace(-4, 2, 25)
        
        self.model = RidgeClassifierCV(
            alphas=alphas,
            cv=cv,
            scoring=scoring
        )
        self.is_fitted = False
    
    def fit(self, X, y):
        """Train the Ridge classifier"""
        self.model.fit(X, y)
        self.is_fitted = True
    
    def predict_proba(self, X):
        """Predict probabilities"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # RidgeClassifierCV doesn't have predict_proba, so we use decision_function
        # and convert to probabilities using sigmoid
        decision = self.model.decision_function(X)
        probs = 1 / (1 + np.exp(-decision))  # Sigmoid
        
        # Return in format [prob_class_0, prob_class_1]
        return np.stack([1 - probs, probs], axis=1)
    
    def predict(self, X):
        """Predict binary labels"""
        return self.model.predict(X)


class MetaClassifier:
    """
    Wrapper for meta-classifiers (MLP or Ridge)
    """
    
    def __init__(self, model_type: str = 'mlp', **kwargs):
        """
        Args:
            model_type: 'mlp' or 'ridge'
            **kwargs: Additional arguments for the classifier
        """
        self.model_type = model_type
        
        if model_type == 'mlp':
            self.model = MLPClassifierTorch(**kwargs)
        elif model_type == 'ridge':
            self.model = RidgeMetaClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'mlp' or 'ridge'.")
    
    def fit(self, X, y, **kwargs):
        """Train the meta-classifier"""
        self.model.fit(X, y, **kwargs)
    
    def predict_proba(self, X):
        """Predict probabilities"""
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """Predict binary labels"""
        return self.model.predict(X)
