from .utils import chess_manager, GameContext
from chess import Move
import subprocess
import os

engine_process = None

def start_cpp_engine():
    global engine_process
    
    engine_path = os.path.join(os.path.dirname(__file__), 'build', 'chess_engine_onnx')
    model_path = os.path.join(os.path.dirname(__file__), 'model.onnx')
    config_path = os.path.join(os.path.dirname(__file__), 'model_config.txt')
    
    print("=" * 60)
    print("🚀 Starting Chess Engine - ONNX Runtime")
    print("=" * 60)
    print(f"Engine: {engine_path}")
    print(f"Model: {model_path}")
    print()
    print("Performance:")
    print("  • 20-30% faster than LibTorch")
    print("  • ~40-80ms per move")
    print("  • ~10MB binary")
    print("  • Easy setup!")
    print("=" * 60)
    print()
    
    if not os.path.exists(engine_path):
        raise FileNotFoundError(
            f"ONNX engine not found at {engine_path}.\n"
            "Please run: ./build_onnx.sh"
        )
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"ONNX model not found at {model_path}.\n"
            "Please run: python3 export_model_onnx.py"
        )
    
    engine_process = subprocess.Popen(
        [engine_path, model_path, config_path],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )
    
    ready = engine_process.stdout.readline().strip()
    if ready != "READY":
        raise RuntimeError(f"Engine failed to start. Got: {ready}")
    
    print("✓ ONNX Runtime engine started successfully!")
    print()

start_cpp_engine()


@chess_manager.entrypoint
def test_func(ctx: GameContext):
    global engine_process
    
    print("=" * 60)
    print("Searching for best move...")
    print(f"Position: {ctx.board.fen()[:50]}...")
    print(f"Move: {ctx.board.fullmove_number}, Turn: {'White' if ctx.board.turn else 'Black'}")
    print(f"Time left: {ctx.timeLeft}ms")
    print("=" * 60)
    
    fen = ctx.board.fen()
    engine_process.stdin.write(f"SEARCH {fen}\n")
    engine_process.stdin.flush()
    
    while True:
        line = engine_process.stdout.readline().strip()
        
        if line.startswith("BESTMOVE"):
            parts = line.split()
            best_move_uci = parts[1]
            score = int(parts[2]) if len(parts) > 2 else 0
            
            print()
            print("=" * 60)
            print("Search complete!")
            print(f"Best move: {best_move_uci}")
            print(f"Score: {score:+d} centipawns")
            print("=" * 60)
            print()
            
            legal_moves = list(ctx.board.legal_moves)
            move_probs = {}
            
            for move in legal_moves:
                if move.uci() == best_move_uci:
                    move_probs[move] = 0.9
                else:
                    move_probs[move] = 0.1 / (len(legal_moves) - 1) if len(legal_moves) > 1 else 0.0
            
            ctx.logProbabilities(move_probs)
            
            return Move.from_uci(best_move_uci)


@chess_manager.reset
def reset_func(ctx: GameContext):
    print("=" * 60)
    print("🎮 New game started!")
    print("=" * 60)
    print()
