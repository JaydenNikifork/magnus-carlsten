"""
Training script for the NNUE model - runs on Modal.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SubsetRandomSampler
import time
import os
import sys
import shutil
import modal

from nnue_model import SimpleNNUE
from dataset import create_dataloader

app = modal.App("magnus-carlsten-training")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.1.0",
        "tensorboard",
        "python-chess",
    )
    .add_local_file("nnue_model.py", "/root/nnue_model.py")
    .add_local_file("dataset.py", "/root/dataset.py")
    .add_local_file("features.py", "/root/features.py")
)

training_volume = modal.Volume.from_name("training-data", create_if_missing=True)
models_volume = modal.Volume.from_name("trained-models", create_if_missing=True)


class OrderedSampler(torch.utils.data.Sampler):
    """Sampler that yields indices in the provided order."""
    def __init__(self, indices):
        self.indices = list(indices)

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch, shuffle_chunks=False, shuffle_within_chunk=False):
    """Train for one epoch.

    If the provided DataLoader wraps a streaming dataset (Subset over a
    streaming LichessEvalDatasetStreaming with `buffer_lines` set), iterate
    chunk-by-chunk to preserve locality: for each chunk create a temporary
    DataLoader (SubsetRandomSampler) that yields indices inside that chunk
    sequentially (optionally shuffled within-chunk). This keeps each chunk
    resident in the dataset buffer while its samples are consumed and avoids
    repeated chunk load/unload thrash.
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    # Detect if train_loader.dataset is a Subset wrapping the original dataset
    ds_wrapper = getattr(train_loader, 'dataset', None)
    orig_dataset = getattr(ds_wrapper, 'dataset', None)
    indices = getattr(ds_wrapper, 'indices', None)

    # If we have a streaming dataset with buffer_lines configured, do chunked iteration
    if orig_dataset is not None and getattr(orig_dataset, 'buffer_lines', None):
        buffer_lines = orig_dataset.buffer_lines

        # Group training indices by chunk id
        from collections import defaultdict
        groups = defaultdict(list)
        for idx in indices:
            chunk_id = idx // buffer_lines
            groups[chunk_id].append(idx)

        # Determine chunk order. By default do NOT shuffle chunks to preserve
        # streaming locality; set shuffle_chunks=True to randomize chunk order.
        import random
        if shuffle_chunks:
            chunk_order = list(groups.keys())
            random.shuffle(chunk_order)
        else:
            # Use sorted chunk order to ensure sequential/monotonic access
            chunk_order = sorted(groups.keys())

        for chunk_id in chunk_order:
            chunk_indices = groups[chunk_id]
            if shuffle_within_chunk:
                random.shuffle(chunk_indices)

            # Logging about chunk
            batch_size = getattr(train_loader, 'batch_size', 1)
            num_samples = len(chunk_indices)
            expected_batches = (num_samples + batch_size - 1) // batch_size
            print(f"Processing chunk {chunk_id}: {num_samples} samples -> ~{expected_batches} batches (batch_size={batch_size})")

            # Choose sampler: ordered or random within chunk
            if shuffle_within_chunk:
                sampler = SubsetRandomSampler(chunk_indices)
            else:
                sampler = OrderedSampler(chunk_indices)

            # Create a small DataLoader over the original dataset limited to this chunk's indices
            chunk_loader = DataLoader(
                orig_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=getattr(train_loader, 'num_workers', 0),
                pin_memory=getattr(train_loader, 'pin_memory', False)
            )

            # Track per-chunk batches so we can print more frequent progress
            chunk_batch_counter = 0
            for batch_idx, (features, targets) in enumerate(chunk_loader):
                features = features.to(device)
                targets = targets.to(device)

                predictions = model(features)
                loss = loss_fn(predictions, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1
                chunk_batch_counter += 1

                # Print frequent progress inside the chunk (every 10 batches)
                if chunk_batch_counter % 10 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"  Chunk {chunk_id} progress: batch {chunk_batch_counter}/{expected_batches} (global batches {num_batches}) - avg loss {avg_loss:.4f}")

                # Also preserve a coarser global checkpoint (every 100 batches)
                if num_batches % 100 == 0:
                    avg_loss = total_loss / num_batches
                    print(f"  Batches {num_batches}: Loss = {avg_loss:.4f}")

            # Finished processing this chunk
            print(f"Finished chunk {chunk_id}: processed {chunk_batch_counter} batches, global batches {num_batches}")

    else:
        # Fallback: original behavior (in-memory or non-streaming)
        for batch_idx, (features, targets) in enumerate(train_loader):
            features = features.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(features)
            loss = loss_fn(predictions, targets)

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 100 == 0:
                avg_loss = total_loss / num_batches
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {avg_loss:.4f}")

    return total_loss / max(1, num_batches)


def validate(model, val_loader, loss_fn, device):
    """Validate the model.

    If `val_loader` wraps a streaming dataset, iterate chunk-by-chunk
    (sequentially) to preserve locality and avoid repeated random chunk
    loads during validation.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    # Detect if val_loader.dataset is a Subset wrapping the original dataset
    ds_wrapper = getattr(val_loader, 'dataset', None)
    orig_dataset = getattr(ds_wrapper, 'dataset', None)
    indices = getattr(ds_wrapper, 'indices', None)

    with torch.no_grad():
        if orig_dataset is not None and getattr(orig_dataset, 'buffer_lines', None):
            # Chunk-aware validation: process chunks sequentially
            buffer_lines = orig_dataset.buffer_lines
            from collections import defaultdict
            groups = defaultdict(list)
            for idx in indices:
                chunk_id = idx // buffer_lines
                groups[chunk_id].append(idx)

            # iterate chunks in sorted order for determinism
            for chunk_id in sorted(groups.keys()):
                chunk_indices = groups[chunk_id]
                print(f"Validation: processing chunk {chunk_id} ({len(chunk_indices)} samples)")

                # Use ordered sampler for validation to preserve locality
                sampler = OrderedSampler(chunk_indices)
                chunk_loader = DataLoader(
                    orig_dataset,
                    batch_size=getattr(val_loader, 'batch_size', 1),
                    sampler=sampler,
                    num_workers=getattr(val_loader, 'num_workers', 0),
                    pin_memory=getattr(val_loader, 'pin_memory', False)
                )

                for features, targets in chunk_loader:
                    features = features.to(device)
                    targets = targets.to(device)

                    predictions = model(features)
                    loss = loss_fn(predictions, targets)

                    total_loss += loss.item()
                    num_batches += 1

        else:
            # fallback to the original behavior
            for features, targets in val_loader:
                features = features.to(device)
                targets = targets.to(device)

                predictions = model(features)
                loss = loss_fn(predictions, targets)

                total_loss += loss.item()
                num_batches += 1

    return total_loss / max(1, num_batches)


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=86400,
    volumes={
        "/data": training_volume,
        "/models": models_volume
    }
)
def train_nnue(
    data_file,
    output_dir="/models",
    h1=512,
    h2=64,
    batch_size=1024,
    epochs=15,
    lr=1e-3,
    weight_decay=1e-4,
    max_cp=1000,
    max_lines=None,
    streaming=False,
    buffer_lines=None,
    device=None
):
    """
    Main training function running on Modal.
    
    Args:
        data_file: Path to JSON lines file with training data (should be in /data volume)
        output_dir: Directory to save model checkpoints (defaults to /models volume)
        h1: Size of first hidden layer
        h2: Size of second hidden layer
        batch_size: Training batch size
        epochs: Number of training epochs
        lr: Learning rate
        weight_decay: L2 regularization
        max_cp: Clamp centipawn values
        max_lines: Maximum number of lines to read from data file (None = read all)
        streaming: If True, use streaming mode (low memory). If False, load all into RAM (faster)
        device: Device to train on (None = auto-detect)
    """
    
    # Setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Using device: {device}")
    print(f"\nModel configuration:")
    print(f"  Architecture: 768 -> {h1} -> {h2} -> 1")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {lr}")
    print(f"  Weight decay: {weight_decay}")
    print(f"  Max CP: ±{max_cp}")
    print(f"  Epochs: {epochs}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    print(f"\nLoading data from {data_file}...")
    if streaming:
        print("⚡ Streaming mode enabled - low memory usage, reads on-demand")
    train_loader, val_loader = create_dataloader(
        data_file,
        batch_size=batch_size,
        max_cp=max_cp,
        max_lines=max_lines,
        streaming=streaming,
        buffer_lines=buffer_lines
    )
    
    # Create model
    print("\nInitializing model...")
    model = SimpleNNUE(h1=h1, h2=h2).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        # Train
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, loss_fn, device)
        
        # Update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss:   {val_loss:.4f}")
        print(f"  LR:         {current_lr:.6f}")
        print(f"  Time:       {epoch_time:.1f}s")
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"nnue_epoch{epoch}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': {
                'h1': h1,
                'h2': h2,
                'max_cp': max_cp,
            }
        }, checkpoint_path)
        print(f"  Saved: {checkpoint_path}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(output_dir, "nnue_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'h1': h1,
                    'h2': h2,
                    'max_cp': max_cp,
                }
            }, best_path)
            print(f"  *** New best model! Val loss: {val_loss:.4f} ***")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*60)
    
    models_volume.commit()
    
    return model


@app.local_entrypoint()
def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NNUE chess evaluation model on Modal")
    parser.add_argument("data_file", help="Path to JSON lines data file (relative to /data volume)")
    parser.add_argument("--output-dir", default="/models", help="Output directory")
    parser.add_argument("--h1", type=int, default=512, help="First hidden layer size")
    parser.add_argument("--h2", type=int, default=64, help="Second hidden layer size")
    parser.add_argument("--batch-size", type=int, default=1024, help="Batch size")
    parser.add_argument("--epochs", type=int, default=15, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--max-cp", type=int, default=1000, help="Max centipawn clamp")
    parser.add_argument("--max-lines", type=int, default=None, help="Maximum number of lines to read from data file")
    parser.add_argument("--streaming", action="store_true", help="Use streaming mode (low memory, slower) instead of loading all into RAM")
    parser.add_argument("--stream-buffer", type=int, default=None, help="(streaming) number of lines to keep in-memory as a rolling buffer (larger = more RAM, less I/O)")
    parser.add_argument("--device", default=None, help="Device (cuda/cpu)")
    
    args = parser.parse_args()
    
    print("🚀 Starting training on Modal...")
    print(f"   Data file: {args.data_file}")
    print(f"   GPU: A100-80GB")
    print(f"   Epochs: {args.epochs}")
    
    train_nnue.remote(
        data_file=args.data_file,
        output_dir=args.output_dir,
        h1=args.h1,
        h2=args.h2,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        max_cp=args.max_cp,
        max_lines=args.max_lines,
        streaming=args.streaming,
        buffer_lines=args.stream_buffer,
        device=args.device
    )
    
    print("\n✅ Training complete! Models saved to Modal volume 'trained-models'")


