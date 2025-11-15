# Chess Engine with C++ Alpha-Beta Search

This implementation combines a PyTorch NNUE model with a C++ minimax search engine using alpha-beta pruning.

## Architecture

- **C++ Engine** (`chess_engine.cpp`): Implements minimax search with alpha-beta pruning
- **Python Bridge** (`main.py`): Loads the PyTorch model and handles position evaluation
- **Communication**: Subprocess communication via stdin/stdout

## How It Works

1. **Python** loads the NNUE model (`model.pt`) and starts the C++ engine as a subprocess
2. **C++ engine** performs minimax search with alpha-beta pruning (depth 4)
3. For each position to evaluate, C++ sends requests to Python:
   - `POSITION <fen>` - Get legal moves and check if terminal
   - `MAKEMOVE <fen> <move>` - Apply a move and get new FEN
   - `EVAL <fen>` - Evaluate position using the NNUE model
4. Python responds with the requested information
5. C++ returns the best move to Python

## Protocol

### C++ → Python Commands

- `POSITION <fen>` → `NORMAL <move1> <move2> ...` or `TERMINAL <result>`
- `MAKEMOVE <fen> <move_uci>` → `<new_fen>`
- `EVAL <fen>` → `<score_in_centipawns>`

### Python → C++ Commands

- `SEARCH <fen>` → C++ performs search and returns `BESTMOVE <move_uci> <score>`

## Building the Engine

```bash
cd src
./build_engine.sh
```

This creates `src/build/chess_engine` which is automatically called by `main.py`.

## Requirements

- **C++ compiler** with C++17 support (g++, clang++)
- **PyTorch** (already in training/requirements.txt)
- **python-chess** (already in training/requirements.txt)

## Configuration

You can adjust the search depth in `chess_engine.cpp`:

```cpp
const int MAX_DEPTH = 4;  // Increase for stronger play (but slower)
```

## Usage

The engine is automatically used when you run `main.py`:

```python
from src.main import chess_manager, test_func

# The C++ engine is started automatically
# test_func() will use alpha-beta search with the NNUE model
```

## Performance

- **Search depth**: 4 ply (2 full moves)
- **Alpha-beta pruning**: Reduces search tree significantly
- **NNUE evaluation**: Fast PyTorch inference (~1ms per position)
- **Typical move time**: 1-5 seconds depending on position complexity

## Files

- `chess_engine.cpp` - C++ minimax implementation
- `main.py` - Python integration with chess manager
- `build_engine.sh` - Build script
- `Makefile` - Build configuration
- `model.pt` - Trained NNUE model

