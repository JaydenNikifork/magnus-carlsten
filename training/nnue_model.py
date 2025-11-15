"""
Simple 768-feature NNUE model for chess position evaluation.
Architecture: 768 -> 512 -> 64 -> 1
"""

import torch
import torch.nn as nn


class SimpleNNUE(nn.Module):
    """
    Simplified NNUE for hackathon use.
    Input: 768 features (12 piece types × 64 squares)
    Output: Single centipawn evaluation
    """
    
    def __init__(self, h1=512, h2=64):
        super().__init__()
        self.fc1 = nn.Linear(768, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, 1)
        self.relu = nn.ReLU()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training"""
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape [batch_size, 768] - binary feature vectors
        
        Returns:
            Tensor of shape [batch_size, 1] - evaluation in normalized units
        """
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    # Test the model
    model = SimpleNNUE(h1=512, h2=64)
    
    # Create dummy input (batch of 4 positions)
    dummy_input = torch.randn(4, 768)
    
    # Forward pass
    output = model(dummy_input)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output: {output[:2].flatten().tolist()}")

