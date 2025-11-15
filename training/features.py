"""
Feature encoding for chess positions.
Converts FEN strings to 768-dimensional binary vectors.
"""

import chess
import torch


def board_to_features(board):
    """
    Convert a chess board to a 768-dimensional binary feature vector.
    
    Encoding: 12 piece types × 64 squares = 768 features
    - Features 0-63: White pawns
    - Features 64-127: White knights
    - Features 128-191: White bishops
    - Features 192-255: White rooks
    - Features 256-319: White queens
    - Features 320-383: White kings
    - Features 384-447: Black pawns
    - Features 448-511: Black knights
    - Features 512-575: Black bishops
    - Features 576-639: Black rooks
    - Features 640-703: Black queens
    - Features 704-767: Black kings
    
    Args:
        board: chess.Board object
    
    Returns:
        torch.Tensor of shape [768] with binary values (0 or 1)
    """
    features = torch.zeros(768, dtype=torch.float32)
    
    # Mapping from piece type to feature offset
    piece_to_idx = {
        (chess.PAWN, chess.WHITE): 0,
        (chess.KNIGHT, chess.WHITE): 64,
        (chess.BISHOP, chess.WHITE): 128,
        (chess.ROOK, chess.WHITE): 192,
        (chess.QUEEN, chess.WHITE): 256,
        (chess.KING, chess.WHITE): 320,
        (chess.PAWN, chess.BLACK): 384,
        (chess.KNIGHT, chess.BLACK): 448,
        (chess.BISHOP, chess.BLACK): 512,
        (chess.ROOK, chess.BLACK): 576,
        (chess.QUEEN, chess.BLACK): 640,
        (chess.KING, chess.BLACK): 704,
    }
    
    # Iterate through all squares
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            # Get the base index for this piece type and color
            base_idx = piece_to_idx[(piece.piece_type, piece.color)]
            # Add the square index
            feature_idx = base_idx + square
            features[feature_idx] = 1.0
    
    return features


def fen_to_features(fen):
    """
    Convert a FEN string to a 768-dimensional feature vector.
    
    Args:
        fen: FEN string
    
    Returns:
        torch.Tensor of shape [768]
    """
    try:
        board = chess.Board(fen)
        return board_to_features(board)
    except Exception as e:
        raise ValueError(f"Invalid FEN string: {fen}") from e


def batch_boards_to_features(boards):
    """
    Convert a list of boards to a batch of feature tensors.
    
    Args:
        boards: List of chess.Board objects
    
    Returns:
        torch.Tensor of shape [batch_size, 768]
    """
    return torch.stack([board_to_features(board) for board in boards])


def batch_fens_to_features(fens):
    """
    Convert a list of FEN strings to a batch of feature tensors.
    
    Args:
        fens: List of FEN strings
    
    Returns:
        torch.Tensor of shape [batch_size, 768]
    """
    return torch.stack([fen_to_features(fen) for fen in fens])


if __name__ == "__main__":
    # Test feature encoding
    
    # Test 1: Starting position
    board = chess.Board()
    features = board_to_features(board)
    print(f"Starting position:")
    print(f"  Total active features: {features.sum().item():.0f} (should be 32)")
    print(f"  Feature vector shape: {features.shape}")
    
    # Test 2: After 1. e4
    board.push_san("e4")
    features = board_to_features(board)
    print(f"\nAfter 1. e4:")
    print(f"  Total active features: {features.sum().item():.0f} (should be 32)")
    
    # Test 3: Specific FEN
    fen = "2bq1rk1/pr3ppn/1p2p3/7P/2pP1B1P/2P5/PPQ2PB1/R3R1K1 w - -"
    features = fen_to_features(fen)
    print(f"\nCustom position (from example):")
    print(f"  Total active features: {features.sum().item():.0f}")
    
    # Test 4: Batch processing
    fens = [
        chess.Board().fen(),
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1").fen(),
    ]
    batch_features = batch_fens_to_features(fens)
    print(f"\nBatch processing:")
    print(f"  Batch shape: {batch_features.shape}")
    print(f"  Active features per position: {batch_features.sum(dim=1).tolist()}")

