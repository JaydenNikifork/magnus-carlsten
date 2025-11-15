# Streaming Mode - Training on Large Datasets

## Overview

The training pipeline now supports two modes:

### 1. **In-Memory Mode** (Default)
- Loads all positions into RAM at startup
- **Pros**: Fastest training speed (~100k pos/sec)
- **Cons**: High memory usage (~500MB per 500k positions)
- **Use when**: Dataset fits comfortably in RAM

### 2. **Streaming Mode** (New!)
- Only stores file offsets in memory (~8 bytes per position)
- Reads and parses positions on-demand during training
- **Pros**: Very low memory usage (~40MB for 5M positions)
- **Cons**: Slower training (disk I/O overhead, ~50-80k pos/sec)
- **Use when**: Training on full 60GB dataset or limited RAM

## Usage

### Enable Streaming Mode

Simply add the `--streaming` flag:

```bash
# Train on full dataset with streaming
python3 training/train.py data/lichess_evals.jsonl --streaming

# Train on full dataset with streaming and other parameters
python3 training/train.py data/lichess_evals.jsonl \
    --streaming \
    --batch-size 2048 \
    --epochs 20 \
    --lr 1e-3
```

### Without Streaming (Default)

```bash
# Train on subset (loads into RAM)
python3 training/train.py data/lichess_evals.jsonl --max-lines 500000
```

## Memory Usage Comparison

| Dataset Size | In-Memory Mode | Streaming Mode |
|--------------|----------------|----------------|
| 500k positions | ~500 MB | ~4 MB |
| 2M positions | ~2 GB | ~16 MB |
| 10M positions | ~10 GB | ~80 MB |
| 50M positions | ~50 GB | ~400 MB |

## Performance Comparison

### In-Memory Mode
- **Indexing**: 5-10 seconds for 500k positions
- **Training speed**: ~100k positions/second
- **Best for**: Quick iteration, < 5M positions

### Streaming Mode
- **Indexing**: 30-60 seconds for 5M positions (builds offset index)
- **Training speed**: ~50-80k positions/second (disk I/O limited)
- **Best for**: Full dataset training, >5M positions

## How It Works

### In-Memory Mode
1. Reads entire file
2. Parses all JSON
3. Validates all FENs
4. Stores all features in memory
5. Fast random access during training

### Streaming Mode
1. **Index Phase**: Scans file, stores only byte offsets of valid positions
2. **Training Phase**: When batch needs position N:
   - Seeks to offset N in file
   - Reads that line only
   - Parses JSON on-the-fly
   - Converts to features
   - Returns to PyTorch

## Example: Full 60GB Dataset

```bash
# Index + train on entire dataset
python3 training/train.py data/lichess_db_eval.jsonl \
    --streaming \
    --batch-size 4096 \
    --epochs 10 \
    --lr 1e-3 \
    --h1 512 \
    --h2 64
```

**Expected behavior**:
- Indexing: 2-5 minutes (builds offset index)
- Memory usage: ~400 MB (just offsets)
- Training: 3-5 hours per epoch
- Total: ~30-50 hours for 10 epochs

## Tips for Hackathon

### Quick Development (First 12 hours)
```bash
# Use in-memory mode with subset
python3 training/train.py data/eval.jsonl --max-lines 500000 --epochs 15
```

### Final Training (Last 12 hours)
```bash
# Switch to streaming mode, train overnight on full dataset
python3 training/train.py data/eval.jsonl --streaming --epochs 5
```

## Technical Details

### Index Structure
- Array of 64-bit integers (file offsets)
- One offset per valid position
- ~8 bytes per position
- Example: 5M positions = 40MB index

### DataLoader Compatibility
- Works with PyTorch's DataLoader
- Supports shuffling (shuffles index, not file)
- Supports multi-worker data loading (each worker opens file handle)
- Thread-safe file access

### Performance Tuning

**Increase batch size** when streaming:
```bash
--batch-size 4096  # Amortizes file I/O overhead
```

**Use SSD** for best streaming performance:
- HDD: ~20-30k pos/sec
- SSD: ~50-80k pos/sec
- NVMe: ~80-100k pos/sec

**Adjust workers** (if supported):
```python
# In code, modify DataLoader:
DataLoader(dataset, batch_size=4096, num_workers=4)
```

## Backward Compatibility

All existing code works unchanged:
- No `--streaming` flag = in-memory mode (default)
- `--max-lines` works with both modes
- All other parameters unchanged

