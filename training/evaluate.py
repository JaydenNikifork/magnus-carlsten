"""
Evaluate chess positions using trained NNUE model.
Reads FEN or PGN from stdin and outputs evaluations.
"""

import torch
import sys
import chess
import chess.pgn
import io

from nnue_model import SimpleNNUE
from features import board_to_features, fen_to_features


def load_model(checkpoint_path, device='cpu'):
    """Load a trained NNUE model"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint.get('config', {})
    h1 = config.get('h1', 512)
    h2 = config.get('h2', 64)
    max_cp = config.get('max_cp', 1000)
    
    model = SimpleNNUE(h1=h1, h2=h2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, max_cp


def evaluate_position(model, board, max_cp=1000):
    """
    Evaluate a chess position.
    
    Args:
        model: Trained NNUE model
        board: chess.Board object
        max_cp: Max centipawn value used during training
    
    Returns:
        Evaluation in centipawns from white's perspective
    """
    features = board_to_features(board).unsqueeze(0)  # Add batch dimension
    
    with torch.no_grad():
        normalized_eval = model(features).item()
    
    # Convert back to centipawns
    cp = int(normalized_eval * max_cp)
    
    return cp


def evaluate_fen(model, fen, max_cp=1000):
    """Evaluate a FEN string"""
    try:
        board = chess.Board(fen)
        return evaluate_position(model, board, max_cp)
    except Exception as e:
        raise ValueError(f"Invalid FEN: {fen}") from e


def process_pgn(model, pgn_text, max_cp=1000):
    """
    Process a PGN and evaluate each position.
    
    Returns list of (move_number, move, evaluation) tuples
    """
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    
    if game is None:
        raise ValueError("Could not parse PGN")
    
    results = []
    board = game.board()
    
    # Evaluate starting position
    eval_cp = evaluate_position(model, board, max_cp)
    results.append((0, "start", eval_cp))
    
    # Evaluate each position after a move
    for move_num, move in enumerate(game.mainline_moves(), 1):
        board.push(move)
        eval_cp = evaluate_position(model, board, max_cp)
        results.append((move_num, board.san(move), eval_cp))
    
    return results


def main():
    """
    Main evaluation loop.
    Reads from stdin and outputs evaluations.
    
    Usage:
        echo "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" | python evaluate.py model.pt
        echo "[PGN game]" | python evaluate.py model.pt --pgn
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate chess positions with NNUE")
    parser.add_argument("model", help="Path to trained model checkpoint (.pt file)")
    parser.add_argument("--pgn", action="store_true", help="Input is PGN format")
    parser.add_argument("--device", default="cpu", help="Device to run on")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Load model
    if args.verbose:
        print(f"Loading model from {args.model}...", file=sys.stderr)
    
    model, max_cp = load_model(args.model, device=args.device)
    
    if args.verbose:
        print(f"Model loaded. Max CP: ±{max_cp}", file=sys.stderr)
        print(f"Reading from stdin...", file=sys.stderr)
        print("-" * 60, file=sys.stderr)
    
    # Read from stdin
    if args.pgn:
        # Read entire PGN
        pgn_text = sys.stdin.read()
        
        try:
            results = process_pgn(model, pgn_text, max_cp)
            
            print("Move | Evaluation (cp)")
            print("-" * 30)
            for move_num, move, eval_cp in results:
                if move_num == 0:
                    print(f"{'Start':<4} | {eval_cp:+6}")
                else:
                    print(f"{move_num:<4} | {eval_cp:+6} ({move})")
        
        except Exception as e:
            print(f"Error processing PGN: {e}", file=sys.stderr)
            sys.exit(1)
    
    else:
        # Read FEN strings line by line
        for line_num, line in enumerate(sys.stdin, 1):
            line = line.strip()
            
            if not line:
                continue
            
            try:
                eval_cp = evaluate_fen(model, line, max_cp)
                print(f"{eval_cp:+6} | {line}")
            
            except Exception as e:
                print(f"Error on line {line_num}: {e}", file=sys.stderr)
                continue


if __name__ == "__main__":
    main()

