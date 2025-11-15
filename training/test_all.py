#!/usr/bin/env python3
"""
Comprehensive test suite for NNUE implementation.
Verifies all components work correctly.
"""

import sys
import os


def test_imports():
    """Test that all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import torch
        print(f"  ✓ PyTorch {torch.__version__}")
    except ImportError:
        print("  ✗ PyTorch not found")
        return False
    
    try:
        import chess
        print(f"  ✓ python-chess {chess.__version__}")
    except ImportError:
        print("  ✗ python-chess not found")
        print("    Install with: pip3 install python-chess")
        return False
    
    return True


def test_model():
    """Test model creation and forward pass"""
    print("\nTesting model...")
    
    try:
        from nnue_model import SimpleNNUE
        import torch
        
        model = SimpleNNUE(h1=512, h2=64)
        dummy_input = torch.randn(4, 768)
        output = model(dummy_input)
        
        assert output.shape == (4, 1), f"Wrong output shape: {output.shape}"
        assert not torch.isnan(output).any(), "Model outputs NaN"
        
        print(f"  ✓ Model created ({sum(p.numel() for p in model.parameters()):,} parameters)")
        print(f"  ✓ Forward pass works")
        return True
    
    except Exception as e:
        print(f"  ✗ Model test failed: {e}")
        return False


def test_features():
    """Test feature encoding"""
    print("\nTesting feature encoding...")
    
    try:
        from features import board_to_features, fen_to_features
        import chess
        import torch
        
        # Test starting position
        board = chess.Board()
        features = board_to_features(board)
        
        assert features.shape == (768,), f"Wrong feature shape: {features.shape}"
        assert features.sum().item() == 32, f"Wrong number of pieces: {features.sum().item()}"
        assert ((features == 0) | (features == 1)).all(), "Features should be binary"
        
        # Test FEN conversion
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"
        features = fen_to_features(fen)
        assert features.sum().item() == 32, "FEN encoding failed"
        
        print(f"  ✓ Feature encoding works")
        print(f"  ✓ Starting position: 32 pieces")
        print(f"  ✓ Binary features verified")
        return True
    
    except Exception as e:
        print(f"  ✗ Feature test failed: {e}")
        return False


def test_dataset():
    """Test dataset loading"""
    print("\nTesting dataset...")
    
    try:
        from dataset import LichessEvalDataset
        import json
        import tempfile
        
        # Create temporary test file
        test_data = [
            {
                "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                "evals": [{"pvs": [{"cp": 20}], "depth": 20}]
            },
            {
                "fen": "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
                "evals": [{"pvs": [{"cp": 35}], "depth": 20}]
            },
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
            temp_file = f.name
        
        try:
            dataset = LichessEvalDataset(temp_file)
            assert len(dataset) == 2, f"Wrong dataset size: {len(dataset)}"
            
            features, target = dataset[0]
            assert features.shape == (768,), f"Wrong feature shape: {features.shape}"
            assert target.shape == (1,), f"Wrong target shape: {target.shape}"
            
            print(f"  ✓ Dataset loading works")
            print(f"  ✓ Loaded {len(dataset)} positions")
            return True
        
        finally:
            os.unlink(temp_file)
    
    except Exception as e:
        print(f"  ✗ Dataset test failed: {e}")
        return False


def test_integration():
    """Test full pipeline"""
    print("\nTesting integration...")
    
    try:
        from nnue_model import SimpleNNUE
        from features import board_to_features
        import chess
        import torch
        
        # Create model
        model = SimpleNNUE(h1=256, h2=32)
        model.eval()
        
        # Create position
        board = chess.Board()
        features = board_to_features(board).unsqueeze(0)
        
        # Evaluate
        with torch.no_grad():
            output = model(features)
        
        eval_cp = int(output.item() * 1000)
        
        print(f"  ✓ Full pipeline works")
        print(f"  ✓ Starting position eval: {eval_cp:+d}cp")
        return True
    
    except Exception as e:
        print(f"  ✗ Integration test failed: {e}")
        return False


def test_device():
    """Test CUDA availability"""
    print("\nTesting compute devices...")
    
    try:
        import torch
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"    {torch.cuda.device_count()} GPU(s) detected")
        else:
            print(f"  ℹ CUDA not available (CPU only)")
        
        # Test MPS (Apple Silicon)
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  ✓ Apple Silicon GPU (MPS) available")
        
        return True
    
    except Exception as e:
        print(f"  ✗ Device test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("NNUE Implementation Test Suite")
    print("="*60)
    
    tests = [
        test_imports,
        test_device,
        test_model,
        test_features,
        test_dataset,
        test_integration,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(results)
    total = len(results)
    
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("\n✅ All tests passed! Ready for training.")
        print("\nNext steps:")
        print("  1. Get Lichess data: data/lichess_data.jsonl")
        print("  2. Train: python3 train.py data/lichess_data.jsonl")
        print("  3. Evaluate: echo 'FEN' | python3 evaluate.py models/nnue_best.pt")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed")
        print("\nTroubleshooting:")
        print("  - Run: bash setup.sh")
        print("  - Install python-chess: pip3 install python-chess")
        print("  - Check Python version: python3 --version")
        return 1


if __name__ == "__main__":
    sys.exit(main())

