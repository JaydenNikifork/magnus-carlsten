# Quick Start Guide - 36-Hour Hackathon Plan

## Hour-by-Hour Timeline

### Hours 0-2: Setup & Data Acquisition ✅

```bash
# 1. Setup environment
cd training
chmod +x setup.sh
bash setup.sh

# 2. Download Lichess data
# Option A: Use existing dataset (if provided)
# Option B: Download from https://database.lichess.org/
#   - Look for files with evaluations
#   - You need JSON format with FEN + eval fields

# 3. Verify data format
head -n 1 data/yourdata.jsonl
# Should show: {"fen": "...", "evals": [...]}
```

### Hours 2-4: Quick Training Test

```bash
# Train on sample data to verify everything works
python3 train.py data/sample.jsonl --epochs 2 --batch-size 64

# If successful, you'll see:
# - Training and validation loss decreasing
# - Model saved to models/nnue_best.pt
```

### Hours 4-10: Full NNUE Training

```bash
# Train on real data (500k-2M positions recommended)
python3 train.py data/lichess_data.jsonl \
    --epochs 15 \
    --batch-size 1024 \
    --h1 512 \
    --h2 64 \
    --lr 0.001 \
    --max-cp 1000 \
    --output-dir models

# This should take 15-30 minutes with good hardware
# Monitor the validation loss - should converge around 0.01-0.03
```

**While training runs, work on:**
- Chess engine integration (minimax/alpha-beta)
- UI/demo features
- Testing infrastructure

### Hours 10-16: Integration

Create your chess engine that uses the NNUE:

```python
# engine.py - Simple example
import chess
import torch
from features import board_to_features
from nnue_model import SimpleNNUE

class ChessEngine:
    def __init__(self, model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model = SimpleNNUE(
            h1=checkpoint['config']['h1'],
            h2=checkpoint['config']['h2']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        self.max_cp = checkpoint['config']['max_cp']
    
    def evaluate(self, board):
        features = board_to_features(board).unsqueeze(0)
        with torch.no_grad():
            normalized = self.model(features).item()
        return int(normalized * self.max_cp)
    
    def minimax(self, board, depth, alpha, beta, maximizing):
        if depth == 0 or board.is_game_over():
            return self.evaluate(board)
        
        if maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth-1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth-1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
    
    def best_move(self, board, depth=5):
        best = None
        best_val = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            val = self.minimax(board, depth-1, float('-inf'), float('inf'), False)
            board.pop()
            if val > best_val:
                best_val = val
                best = move
        return best, best_val

# Usage
engine = ChessEngine('models/nnue_best.pt')
board = chess.Board()
move, score = engine.best_move(board, depth=5)
print(f"Best move: {move}, Score: {score}cp")
```

### Hours 16-24: Optimization & Features

**Improvements to add:**
1. **Move ordering** (checks/captures first)
2. **Quiescence search** (finish captures)
3. **Opening book** (lichess popular lines)
4. **Time management**
5. **Evaluation UI** (show position eval bar)

### Hours 24-32: Testing & Polish

```bash
# Test against Stockfish
python3 play_stockfish.py  # Create this script

# Play games and verify:
# - No illegal moves
# - Reasonable evaluations
# - Decent tactical play
# - No crashes
```

### Hours 32-36: Demo Prep

**Must have for demo:**
1. ✅ Working bot that makes legal moves
2. ✅ Can play full game vs. human or Stockfish
3. ✅ Shows evaluation for each position
4. ✅ Clean UI (terminal or web)

**Nice to have:**
1. Eval bar visualization
2. "Best line" display
3. Training loss graphs
4. Live retraining demo

## Quick Tests

### Test 1: Feature Encoding
```bash
python3 features.py
# Should show 32 active features for starting position
```

### Test 2: Model Forward Pass
```bash
python3 nnue_model.py
# Should show model parameters and sample output
```

### Test 3: Data Loading
```bash
python3 dataset.py data/sample.jsonl
# Should load and parse positions
```

### Test 4: Evaluation
```bash
echo "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" | \
    python3 evaluate.py models/nnue_best.pt
# Should output centipawn evaluation
```

## Troubleshooting

**Training loss stuck at ~0.5:**
- Data might be low quality or mislabeled
- Try reducing learning rate: `--lr 0.0005`

**Out of memory:**
- Reduce batch size: `--batch-size 512` or `--batch-size 256`
- Use smaller model: `--h1 256 --h2 32`

**Model makes random moves:**
- Check if training completed successfully
- Verify model file loaded correctly
- Test eval on known positions

**Slow inference:**
- Use CPU for NNUE (it's fast enough)
- Reduce search depth
- Add move ordering to prune tree better

## Expected Results

With 500k-2M training positions:
- **Validation loss**: 0.015-0.030 (lower is better)
- **Playing strength** (eval only): ~1800 Elo
- **With search** (depth 5-6): ~2000-2100 Elo

## Files to Submit

```
magnus-carlsten/
├── training/
│   ├── nnue_model.py       # Your model
│   ├── features.py         # Feature encoding
│   ├── train.py            # Training script
│   ├── evaluate.py         # Evaluation
│   └── models/
│       └── nnue_best.pt    # Trained weights
├── engine.py               # Chess engine with minimax
├── play.py                 # Interactive play script
├── ui.py                   # Demo interface
└── README.md               # Documentation
```

## Demo Script

```
1. Show starting position eval: ~+20cp
2. Make a move (e.g., e4): eval changes to ~+35cp
3. Play 10 moves showing evaluations
4. Show it beating Stockfish at 1400 Elo
5. (Bonus) Show training loss graph
6. (Bonus) Retrain on new game in real-time
```

Good luck! 🚀

