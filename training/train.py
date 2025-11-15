"""
Training script for the NNUE model.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time
import os
import sys
import shutil

from nnue_model import SimpleNNUE
from dataset import create_dataloader


def train_epoch(model, train_loader, optimizer, loss_fn, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
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
    
    return total_loss / num_batches


def validate(model, val_loader, loss_fn, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for features, targets in val_loader:
            features = features.to(device)
            targets = targets.to(device)
            
            predictions = model(features)
            loss = loss_fn(predictions, targets)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches


def train_nnue(
    data_file,
    output_dir="models",
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
    Main training function.
    
    Args:
        data_file: Path to JSON lines file with training data
        output_dir: Directory to save model checkpoints
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
    
    # Copy best model to src/ directory for deployment
    best_model_path = os.path.join(output_dir, "nnue_best.pt")
    src_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "src")
    deployment_path = os.path.join(src_dir, "model.pt")
    
    if os.path.exists(best_model_path):
        # Create src directory if it doesn't exist
        os.makedirs(src_dir, exist_ok=True)
        
        # Copy the best model
        shutil.copy2(best_model_path, deployment_path)
        print(f"\n✅ Copied best model to: {deployment_path}")
        print(f"   Ready for deployment!")
    else:
        print(f"\n⚠️  Warning: Best model not found at {best_model_path}")
    
    return model


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train NNUE chess evaluation model")
    parser.add_argument("data_file", help="Path to JSON lines data file")
    parser.add_argument("--output-dir", default="models", help="Output directory")
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
    
    train_nnue(
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

