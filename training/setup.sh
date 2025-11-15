#!/bin/bash
# Setup script for NNUE training environment

set -e

echo "======================================"
echo "Magnus Carlsen NNUE Setup"
echo "======================================"

# Check Python version
echo -e "\n1. Checking Python version..."
python3 --version

# Install dependencies
echo -e "\n2. Installing dependencies..."
echo "   Installing python-chess..."
pip3 install python-chess --break-system-packages || pip3 install python-chess

echo "   Checking PyTorch..."
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')" || {
    echo "   PyTorch not found. Please install it manually:"
    echo "   pip3 install torch"
    exit 1
}

# Test imports
echo -e "\n3. Testing imports..."
python3 -c "
import torch
import chess
print('✓ All imports successful')
print(f'  - PyTorch: {torch.__version__}')
print(f'  - python-chess: {chess.__version__}')
print(f'  - CUDA available: {torch.cuda.is_available()}')
"

# Test feature encoding
echo -e "\n4. Testing feature encoding..."
python3 features.py

# Create necessary directories
echo -e "\n5. Creating directories..."
mkdir -p models
mkdir -p data
echo "   Created: models/, data/"

# Create sample data
echo -e "\n6. Creating sample training data..."
cat > data/sample.jsonl << 'EOF'
{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "evals": [{"pvs": [{"cp": 20}], "depth": 20}]}
{"fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", "evals": [{"pvs": [{"cp": 35}], "depth": 20}]}
{"fen": "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "evals": [{"pvs": [{"cp": 28}], "depth": 20}]}
EOF
echo "   Created: data/sample.jsonl (3 positions)"

# Test model
echo -e "\n7. Testing model..."
python3 nnue_model.py

echo -e "\n======================================"
echo "Setup complete!"
echo "======================================"
echo ""
echo "Next steps:"
echo "1. Get training data (Lichess format) and put it in data/"
echo "2. Train: python3 train.py data/yourdata.jsonl"
echo "3. Evaluate: echo 'FEN' | python3 evaluate.py models/nnue_best.pt"
echo ""
echo "Quick test with sample data:"
echo "  python3 train.py data/sample.jsonl --epochs 2"

