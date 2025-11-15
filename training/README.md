# Magnus Carlsen - Chess NNUE Trainer

Simple 768-feature NNUE (Efficiently Updatable Neural Network) for chess position evaluation.

## Architecture

- **Input**: 768 binary features (12 piece types × 64 squares)
- **Hidden layers**: 768 → 512 → 64 → 1
- **Output**: Centipawn evaluation from white's perspective
- **Training**: Supervised learning on Lichess evaluation data

## Installation

```bash
cd training

# Install dependencies (already done if venv is activated)
pip install torch chess
```

## Quick Start

### 1. Prepare Training Data

Your data should be in JSON Lines format (one JSON object per line):

```json
{
  "fen": "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -",
  "evals": [
    {
      "pvs": [{"cp": 311, "line": "..."}],
      "depth": 36
    }
  ]
}
```

### 2. Train the Model

```bash
python train.py data.jsonl --epochs 15 --batch-size 1024
```

**Training options:**
```bash
python train.py data.jsonl \
    --epochs 15 \
    --batch-size 1024 \
    --h1 512 \
    --h2 64 \
    --lr 0.001 \
    --max-cp 1000 \
    --output-dir models
```

Expected training time (with fast hardware):
- 500k positions: ~5-10 minutes
- 2M positions: ~15-20 minutes
- 5M positions: ~30-40 minutes

### 3. Evaluate Positions

**Evaluate FEN strings:**
```bash
# Single position
echo "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" | \
    python evaluate.py models/nnue_best.pt

# Multiple positions
cat positions.fen | python evaluate.py models/nnue_best.pt
```

**Evaluate PGN:**
```bash
cat game.pgn | python evaluate.py models/nnue_best.pt --pgn
```

Output format:
```
Move | Evaluation (cp)
------------------------------
Start |    +20
1    |    +35 (e4)
2    |    +28 (e5)
...
```

## File Structure

```
training/
├── nnue_model.py       # Model architecture
├── features.py         # Feature encoding (FEN → 768-dim vector)
├── dataset.py          # Data loading and preprocessing
├── train.py            # Training script
├── evaluate.py         # Evaluation/inference script
├── hardwaretest.py     # Hardware benchmark
└── models/             # Saved model checkpoints
    ├── nnue_epoch1.pt
    ├── nnue_epoch2.pt
    └── nnue_best.pt    # Best model (lowest validation loss)
```

## Advanced Usage

### Test Feature Encoding

```bash
python features.py
```

### Test Dataset Loading

```bash
python dataset.py data.jsonl
```

### Training with Different Architectures

**Larger model (more capacity):**
```bash
python train.py data.jsonl --h1 1024 --h2 128
```

**Smaller model (faster inference):**
```bash
python train.py data.jsonl --h1 256 --h2 32
```

### Resume Training

```bash
# Load a checkpoint and continue training
# (TODO: implement resume functionality)
```

### Model Ensemble

Train multiple models with different seeds and average their predictions:

```bash
for seed in 1 2 3 4 5; do
    python train.py data.jsonl --output-dir models/seed${seed}
done

# Average predictions in your evaluation script
```

## Model Checkpoints

Each checkpoint contains:
- `model_state_dict`: Model weights
- `config`: Architecture configuration (h1, h2, max_cp)
- `epoch`: Training epoch number
- `val_loss`: Validation loss

Load a checkpoint:
```python
import torch
from nnue_model import SimpleNNUE

checkpoint = torch.load('models/nnue_best.pt')
model = SimpleNNUE(h1=checkpoint['config']['h1'], 
                   h2=checkpoint['config']['h2'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Performance Tips

1. **Batch size**: Increase if you have GPU memory (1024, 2048, 4096)
2. **Data augmentation**: Flip board colors during training (TODO)
3. **Learning rate**: Use cosine annealing (already implemented)
4. **Gradient clipping**: Prevents exploding gradients (already implemented)
5. **Multiple training runs**: Train 3-5 models and ensemble

## Expected Strength

With proper training data:
- **768 → 512 → 64 → 1**: 1800-1900 Elo
- **768 → 1024 → 128 → 1**: 1900-2000 Elo
- With search (minimax depth 5-6): Add ~200-300 Elo

## Troubleshooting

**Out of memory during training:**
- Reduce batch size: `--batch-size 512` or `--batch-size 256`
- Use smaller model: `--h1 256 --h2 32`

**Training loss not decreasing:**
- Check data quality (are evals reasonable?)
- Reduce learning rate: `--lr 0.0005`
- Increase training data

**Model outputs random values:**
- Check feature encoding (should sum to ~32 for normal positions)
- Verify data format matches expected JSON structure
- Ensure FEN strings are valid

## TODO / Future Improvements

- [ ] Data augmentation (color flip, board mirroring)
- [ ] King-relative feature encoding (1536 or 98k features)
- [ ] WDL (Win/Draw/Loss) head alongside centipawn value
- [ ] Policy head for move ordering
- [ ] Incremental feature updates during search
- [ ] Self-play data generation
- [ ] Model quantization (int8/int16) for faster inference
- [ ] Integration with chess engine (minimax/alpha-beta)

## References

- [Stockfish NNUE](https://github.com/official-stockfish/Stockfish)
- [NNUE paper](https://arxiv.org/abs/2109.01872)
- [Lichess database](https://database.lichess.org/)

## License

MIT License - feel free to use for your hackathon!

