# Training on Modal

This guide explains how to train your NNUE chess model on Modal's cloud GPUs.

## Setup

1. Install Modal:
```bash
pip install modal
```

2. Authenticate with Modal:
```bash
modal token new
```

## Upload Training Data

First, upload your training data to the Modal volume:

```bash
modal volume put training-data trainingData/lichess_db_eval.jsonl /lichess_db_eval.jsonl
```

Or for smaller test data:
```bash
modal volume put training-data trainingData/small_data.jsonl /small_data.jsonl
```

## Run Training

Train your model on Modal's A10G GPU:

```bash
modal run training/train.py /lichess_db_eval.jsonl --epochs 20 --batch-size 1024
```

### Training Options

```bash
modal run training/train.py <data_file> [options]

Required:
  data_file              Path to data file (relative to /data volume)

Optional:
  --output-dir PATH      Output directory (default: /models)
  --h1 INT              First hidden layer size (default: 512)
  --h2 INT              Second hidden layer size (default: 64)
  --batch-size INT      Batch size (default: 1024)
  --epochs INT          Number of epochs (default: 15)
  --lr FLOAT            Learning rate (default: 1e-3)
  --weight-decay FLOAT  Weight decay (default: 1e-4)
  --max-cp INT          Max centipawn clamp (default: 1000)
  --max-lines INT       Max lines to read (default: None/all)
  --streaming           Use streaming mode (low memory)
  --stream-buffer INT   Buffer size for streaming (default: None)
  --device STR          Device to use (default: auto-detect)
```

### Example Commands

Train on small dataset for testing:
```bash
modal run training/train.py /small_data.jsonl --epochs 5 --max-lines 10000
```

Full training run:
```bash
modal run training/train.py /lichess_db_eval.jsonl --epochs 20 --batch-size 2048 --lr 1e-3
```

Streaming mode for huge datasets:
```bash
modal run training/train.py /lichess_db_eval.jsonl --streaming --stream-buffer 50000 --epochs 15
```

## Download Trained Models

After training completes, download your models:

```bash
modal volume get trained-models /nnue_best.pt models/nnue_best.pt
modal volume get trained-models /nnue_epoch20.pt models/nnue_epoch20.pt
```

Or download all models:
```bash
modal volume get trained-models / models/
```

## Copy Model for Deployment

After downloading, copy the best model to your src directory:

```bash
cp training/models/nnue_best.pt src/model.pt
```

## Monitor Training

You can view your Modal app and logs at:
https://modal.com/apps/jayden-40324/magnus-carlsten-training

## Volumes

This setup uses two Modal volumes:

1. **training-data**: Stores your training datasets
2. **trained-models**: Stores model checkpoints and outputs

List files in a volume:
```bash
modal volume ls training-data
modal volume ls trained-models
```

## GPU Options

The default configuration uses an A10G GPU. To use a different GPU, modify `train.py`:

```python
@app.function(
    gpu="A100",  # or "T4", "A10G", "H100", etc.
    ...
)
```

## Troubleshooting

**Error: Volume not found**
- Volumes are created automatically on first run. If you get an error, ensure you're authenticated with `modal token new`

**Out of memory**
- Reduce `--batch-size`
- Use `--streaming` mode
- Reduce `--max-lines` for testing

**Slow training**
- Ensure you're using GPU (check logs for "Using device: cuda")
- Increase batch size if GPU memory allows
- Don't use `--streaming` unless memory-constrained

## Cost

Modal charges based on GPU usage. Approximate costs:
- A10G: ~$1.10/hour
- Training typically takes 2-6 hours depending on dataset size

Check current pricing at: https://modal.com/pricing

