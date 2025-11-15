#!/usr/bin/env python3
"""
Export PyTorch NNUE model to TorchScript format for C++ loading
"""
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
from nnue_model import SimpleNNUE

def export_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
    output_path = os.path.join(os.path.dirname(__file__), 'model_traced.pt')
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model = SimpleNNUE(
        h1=checkpoint['config']['h1'],
        h2=checkpoint['config']['h2']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model architecture: 768 → {checkpoint['config']['h1']} → {checkpoint['config']['h2']} → 1")
    
    dummy_input = torch.zeros(1, 768, dtype=torch.float32)
    
    print("Tracing model...")
    traced_model = torch.jit.trace(model, dummy_input)
    
    print(f"Saving traced model to: {output_path}")
    traced_model.save(output_path)
    
    test_output = traced_model(dummy_input)
    print(f"Test output: {test_output.item()}")
    
    with open(os.path.join(os.path.dirname(__file__), 'model_config.txt'), 'w') as f:
        f.write(f"{checkpoint['config']['max_cp']}\n")
    
    print(f"✓ Model exported successfully!")
    print(f"  Max CP: ±{checkpoint['config']['max_cp']}")
    print(f"  Output file: {output_path}")

if __name__ == "__main__":
    export_model()

