# src/ Directory

This directory contains the deployed model ready for use in your chess engine.

## Files

- **`model.pt`** - The best trained NNUE model (automatically copied after training)

## How It Gets Here

The model is automatically copied to this directory when training completes:

```bash
cd training
python3 train.py data.jsonl --max-lines 500000 --epochs 15
# After training, model.pt will be in ../src/
```

## Manual Deployment

If you want to deploy a different checkpoint:

```bash
cd training
bash deploy_model.sh models/nnue_epoch10.pt
```

Or manually:
```bash
cp training/models/nnue_best.pt src/model.pt
```

## Using the Model

### In Python:

```python
import torch
from training.nnue_model import SimpleNNUE
from training.features import board_to_features

# Load model
checkpoint = torch.load('src/model.pt', map_location='cpu')
model = SimpleNNUE(
    h1=checkpoint['config']['h1'],
    h2=checkpoint['config']['h2']
)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Evaluate a position
import chess
board = chess.Board()
features = board_to_features(board).unsqueeze(0)

with torch.no_grad():
    eval_normalized = model(features).item()
    eval_cp = int(eval_normalized * checkpoint['config']['max_cp'])

print(f"Position eval: {eval_cp:+d} centipawns")
```

### In Your Chess Engine:

```python
# engine.py
from training.nnue_model import SimpleNNUE
from training.features import board_to_features
import torch
import chess

class ChessEngine:
    def __init__(self):
        checkpoint = torch.load('src/model.pt', map_location='cpu')
        self.model = SimpleNNUE(
            h1=checkpoint['config']['h1'],
            h2=checkpoint['config']['h2']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.max_cp = checkpoint['config']['max_cp']
    
    def evaluate(self, board):
        """Evaluate position in centipawns"""
        features = board_to_features(board).unsqueeze(0)
        with torch.no_grad():
            normalized = self.model(features).item()
        return int(normalized * self.max_cp)
    
    def search(self, board, depth=5):
        """Your minimax/alpha-beta search here"""
        # ... use self.evaluate(board) for leaf nodes
        pass
```

## Model Information

Check the loaded model's details:

```python
import torch

checkpoint = torch.load('src/model.pt')
print(f"Epoch: {checkpoint['epoch']}")
print(f"Val Loss: {checkpoint['val_loss']:.4f}")
print(f"Architecture: 768 -> {checkpoint['config']['h1']} -> {checkpoint['config']['h2']} -> 1")
print(f"Max CP: ±{checkpoint['config']['max_cp']}")
```

## File Size

The model.pt file should be approximately:
- With h1=256, h2=32: ~1.5 MB
- With h1=512, h2=64: ~2.5 MB
- With h1=1024, h2=128: ~6 MB

Small enough for git, fast loading, quick inference!

