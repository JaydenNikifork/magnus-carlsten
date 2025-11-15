from .utils import chess_manager, GameContext
from chess import Move
import torch
import subprocess
import os
import sys
import chess

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'training'))
from nnue_model import SimpleNNUE
from features import board_to_features

model = None
max_cp = None
engine_process = None

def load_model():
    global model, max_cp
    model_path = os.path.join(os.path.dirname(__file__), 'model.pt')
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location='cpu')
    model = SimpleNNUE(
        h1=checkpoint['config']['h1'],
        h2=checkpoint['config']['h2']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    max_cp = checkpoint['config']['max_cp']
    print(f"✓ Model loaded successfully!")
    print(f"  Architecture: 768 → {checkpoint['config']['h1']} → {checkpoint['config']['h2']} → 1")
    print(f"  Max centipawns: ±{max_cp}")
    print(f"  Training epoch: {checkpoint.get('epoch', 'unknown')}")
    if 'val_loss' in checkpoint:
        print(f"  Validation loss: {checkpoint['val_loss']:.4f}")
    print()

def evaluate_position(board: chess.Board) -> int:
    global model, max_cp
    features = board_to_features(board).unsqueeze(0)
    with torch.no_grad():
        normalized = model(features).item()
    return int(normalized * max_cp)

def start_cpp_engine():
    global engine_process
    engine_path = os.path.join(os.path.dirname(__file__), 'build', 'chess_engine')
    
    print(f"Starting C++ engine from: {engine_path}")
    
    if not os.path.exists(engine_path):
        raise FileNotFoundError(
            f"C++ engine not found at {engine_path}. "
            "Please run ./build_engine.sh first."
        )
    
    engine_process = subprocess.Popen(
        [engine_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    ready = engine_process.stdout.readline().strip()
    if ready != "READY":
        raise RuntimeError(f"Engine failed to start. Got: {ready}")
    
    print("✓ C++ engine started successfully!")
    print()

def handle_engine_request(request: str) -> str:
    parts = request.split(maxsplit=1)
    command = parts[0]
    
    if command == "EVAL":
        fen = parts[1]
        board = chess.Board(fen)
        score = evaluate_position(board)
        return str(score)
    
    elif command == "POSITION":
        fen = parts[1]
        board = chess.Board(fen)
        
        if board.is_game_over():
            result = board.result()
            return f"TERMINAL {result}"
        else:
            legal_moves = [move.uci() for move in board.legal_moves]
            return f"NORMAL {' '.join(legal_moves)}"
    
    elif command == "MAKEMOVE":
        parts = request.split()
        fen = ' '.join(parts[1:-1])
        move_uci = parts[-1]
        board = chess.Board(fen)
        board.push_uci(move_uci)
        return board.fen()
    
    return "ERROR"

load_model()
start_cpp_engine()


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    global engine_process
    
    print("=" * 60)
    print("Searching for best move...")
    print(f"Current position FEN: {ctx.board.fen()}")
    print(f"Move number: {ctx.board.fullmove_number}")
    print(f"Turn: {'White' if ctx.board.turn else 'Black'}")
    print(f"Time left: {ctx.timeLeft}ms")
    
    current_eval = evaluate_position(ctx.board)
    print(f"Current position evaluation: {current_eval:+d} centipawns")
    
    fen = ctx.board.fen()
    engine_process.stdin.write(f"SEARCH {fen}\n")
    engine_process.stdin.flush()
    
    request_count = 0
    
    while True:
        line = engine_process.stdout.readline().strip()
        
        if line.startswith("EVAL") or line.startswith("POSITION") or line.startswith("MAKEMOVE"):
            request_count += 1
            response = handle_engine_request(line)
            engine_process.stdin.write(response + "\n")
            engine_process.stdin.flush()
        elif line.startswith("BESTMOVE"):
            parts = line.split()
            best_move_uci = parts[1]
            score = int(parts[2]) if len(parts) > 2 else 0
            
            print(f"\n{'=' * 60}")
            print(f"Search complete!")
            print(f"Best move: {best_move_uci}")
            print(f"Expected score after move: {score:+d} centipawns")
            print(f"Score change: {score - current_eval:+d} centipawns")
            print(f"Total C++ requests handled: {request_count}")
            print(f"{'=' * 60}\n")
            
            legal_moves = list(ctx.board.legal_moves)
            move_probs = {}
            
            for move in legal_moves:
                if move.uci() == best_move_uci:
                    move_probs[move] = 0.9
                else:
                    move_probs[move] = 0.1 / (len(legal_moves) - 1) if len(legal_moves) > 1 else 0.0
            
            ctx.logProbabilities(move_probs)
            
            return Move.from_uci(best_move_uci)
        else:
            if line:
                print(f"[C++ Engine] {line}")


@chess_manager.reset
def reset_func(ctx: GameContext):
    print("=" * 60)
    print("🎮 New game started!")
    print("=" * 60)
    print()
