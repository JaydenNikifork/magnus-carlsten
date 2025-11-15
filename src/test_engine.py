#!/usr/bin/env python3
"""
Test script to verify the C++ engine integration works correctly.
"""

import subprocess
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))

def test_cpp_engine():
    print("Testing C++ engine subprocess communication...")
    
    engine_path = os.path.join(os.path.dirname(__file__), 'build', 'chess_engine')
    
    if not os.path.exists(engine_path):
        print(f"ERROR: Engine not found at {engine_path}")
        print("Please run: ./build_engine.sh")
        return False
    
    process = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    ready = process.stdout.readline().strip()
    print(f"Engine response: {ready}")
    
    if ready != "READY":
        print("ERROR: Engine did not start properly")
        return False
    
    print("✓ Engine started successfully")
    
    starting_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    
    print("\nTesting POSITION command...")
    process.stdin.write(f"POSITION {starting_fen}\n")
    process.stdin.flush()
    response = process.stdout.readline().strip()
    print(f"Response: {response[:80]}...")
    
    if not response.startswith("NORMAL"):
        print("ERROR: Expected NORMAL position")
        return False
    print("✓ POSITION command works")
    
    print("\nTesting MAKEMOVE command...")
    process.stdin.write(f"MAKEMOVE {starting_fen} e2e4\n")
    process.stdin.flush()
    response = process.stdout.readline().strip()
    print(f"Response: {response}")
    
    if "e4" not in response.lower():
        print("ERROR: Move was not applied correctly")
        return False
    print("✓ MAKEMOVE command works")
    
    print("\nTesting EVAL command...")
    process.stdin.write(f"EVAL {starting_fen}\n")
    process.stdin.flush()
    
    try:
        import torch
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
        from nnue_model import SimpleNNUE
        from features import board_to_features
        import chess
        
        model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
        checkpoint = torch.load(model_path, map_location='cpu')
        model = SimpleNNUE(h1=checkpoint['config']['h1'], h2=checkpoint['config']['h2'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        board = chess.Board(starting_fen)
        features = board_to_features(board).unsqueeze(0)
        with torch.no_grad():
            normalized = model(features).item()
        score = int(normalized * checkpoint['config']['max_cp'])
        
        response_score = score
        process.stdin.write(f"{response_score}\n")
        process.stdin.flush()
        
        print(f"Sent evaluation: {response_score}")
        print("✓ EVAL command works")
        
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return False
    
    process.stdin.write("QUIT\n")
    process.stdin.flush()
    process.wait(timeout=5)
    
    print("\n" + "="*50)
    print("✓ All tests passed!")
    print("="*50)
    print("\nThe C++ engine is working correctly and can communicate with Python.")
    return True

if __name__ == "__main__":
    success = test_cpp_engine()
    sys.exit(0 if success else 1)

