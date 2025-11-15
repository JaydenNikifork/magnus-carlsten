#!/bin/bash
# Manual model deployment script
# Use this if you want to copy a specific model checkpoint to src/

set -e

# Default values
SOURCE_MODEL="models/nnue_best.pt"
DEST_DIR="../src"
DEST_NAME="model.pt"

# Parse arguments
if [ "$1" != "" ]; then
    SOURCE_MODEL=$1
fi

if [ "$2" != "" ]; then
    DEST_DIR=$2
fi

echo "=========================================="
echo "Model Deployment Script"
echo "=========================================="
echo ""
echo "Source: $SOURCE_MODEL"
echo "Destination: $DEST_DIR/$DEST_NAME"
echo ""

# Check if source exists
if [ ! -f "$SOURCE_MODEL" ]; then
    echo "❌ Error: Source model not found: $SOURCE_MODEL"
    exit 1
fi

# Create destination directory
mkdir -p "$DEST_DIR"

# Copy model
cp "$SOURCE_MODEL" "$DEST_DIR/$DEST_NAME"

echo "✅ Model deployed successfully!"
echo ""
echo "Model info:"
python3 -c "
import torch
ckpt = torch.load('$SOURCE_MODEL', map_location='cpu')
print(f\"  Epoch: {ckpt.get('epoch', 'N/A')}\")
print(f\"  Val Loss: {ckpt.get('val_loss', 'N/A'):.4f}\")
print(f\"  Config: {ckpt.get('config', {})}\")
"

echo ""
echo "Ready to use in your chess engine!"

