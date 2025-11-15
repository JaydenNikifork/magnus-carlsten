"""
Quick reference for the alpha-beta chess engine implementation.
"""

# =============================================================================
# COMMUNICATION FLOW
# =============================================================================
# 
# 1. Game starts → Python loads model and starts C++ subprocess
# 2. Move needed → Python sends: "SEARCH <fen>"
# 3. C++ begins search:
#    For each position in search tree:
#      C++ → Python: "POSITION <fen>"
#      Python → C++: "NORMAL e2e4 d2d4 ..." or "TERMINAL 1-0"
#      
#      C++ → Python: "MAKEMOVE <fen> e2e4"
#      Python → C++: "<new_fen>"
#      
#      C++ → Python: "EVAL <fen>"
#      Python → C++: "150" (centipawns)
# 
# 4. C++ completes search → "BESTMOVE e2e4 150"
# 5. Python returns move to chess manager
#
# =============================================================================

# =============================================================================
# ALPHA-BETA PSEUDOCODE
# =============================================================================
# 
# function alphaBeta(position, depth, alpha, beta, maximizingPlayer):
#     if depth == 0 or game_over(position):
#         return evaluate(position)
#     
#     if maximizingPlayer:
#         maxEval = -∞
#         for each move in position.legal_moves:
#             eval = alphaBeta(position.after(move), depth-1, alpha, beta, False)
#             maxEval = max(maxEval, eval)
#             alpha = max(alpha, eval)
#             if beta <= alpha:
#                 break  # Beta cutoff
#         return maxEval
#     else:
#         minEval = +∞
#         for each move in position.legal_moves:
#             eval = alphaBeta(position.after(move), depth-1, alpha, beta, True)
#             minEval = min(minEval, eval)
#             beta = min(beta, eval)
#             if beta <= alpha:
#                 break  # Alpha cutoff
#         return minEval
#
# =============================================================================

# =============================================================================
# KEY CONSTANTS
# =============================================================================

MAX_DEPTH = 4           # Search depth (in chess_engine.cpp)
INF = 1000000           # Infinity value
MATE_SCORE = 100000     # Checkmate score

# =============================================================================
# QUICK BUILD COMMANDS
# =============================================================================

# Build engine:
# $ cd src && ./build_engine.sh

# Or manually:
# $ cd src && make

# Test:
# $ cd src && python3 test_engine.py

# =============================================================================
# DEBUGGING TIPS
# =============================================================================

# 1. C++ engine logs to stderr (std::cerr)
# 2. Python can print to stdout normally
# 3. Check C++ is running: ps aux | grep chess_engine
# 4. Manual test C++ engine:
#    $ ./build/chess_engine
#    READY
#    POSITION rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1
#    (then type the response)

# =============================================================================

