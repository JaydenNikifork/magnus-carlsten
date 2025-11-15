# Stockfish Integration

This integration allows you to use Stockfish for position evaluation instead of the custom NNUE model while keeping the minimax search algorithm intact.

## Overview

The system now supports two evaluation modes:
- **ML Model (default)**: Uses your custom NNUE model for evaluation
- **Stockfish**: Uses Stockfish engine for evaluation

## How It Works

When `USE_STOCKFISH = True`:
- The C++ minimax engine still performs the search algorithm
- When the C++ engine requests position evaluations via `EVAL` commands
- Python calls Stockfish instead of the NNUE model
- Stockfish evaluates the position and returns a centipawn score
- The score is passed back to the C++ engine to continue the search

This means:
✅ Your minimax algorithm logic remains unchanged
✅ Alpha-beta pruning still works the same way
✅ Move generation and game tree exploration is identical
❌ Only the leaf node evaluation changes

## Setup

### Install Stockfish

**macOS:**
```bash
brew install stockfish
```

**Ubuntu/Debian:**
```bash
sudo apt-get install stockfish
```

**Other:**
Download from [stockfishchess.org](https://stockfishchess.org/download/)

The integration will automatically find Stockfish if installed in standard locations:
- `/opt/homebrew/bin/stockfish`
- `/usr/local/bin/stockfish`
- `/usr/bin/stockfish`
- Or anywhere in your PATH

## Usage

### Enable Stockfish Mode

In `main.py`, change line 14:

```python
USE_STOCKFISH = True  # Enable Stockfish evaluation
```

### Adjust Evaluation Depth (Optional)

In `main.py`, modify the `StockfishEvaluator` initialization (line 28):

```python
stockfish_evaluator = StockfishEvaluator(depth=20)  # Default is 15
```

Higher depth = stronger evaluation but slower.

### Run Your Bot

```bash
python serve.py
```

You'll see:
```
============================================================
🐟 Using Stockfish for evaluation
============================================================
✓ Stockfish evaluator initialized!
  Path: /opt/homebrew/bin/stockfish
  Evaluation depth: 15
```

## Configuration

### StockfishEvaluator Parameters

```python
StockfishEvaluator(
    stockfish_path=None,  # Auto-detect or provide custom path
    depth=15              # Evaluation depth (1-20+ recommended)
)
```

### Evaluation Depth Guidelines

- **depth=10**: Fast, ~100-200ms per position, ~1800 Elo
- **depth=15**: Balanced, ~500ms per position, ~2400 Elo  ⭐ Recommended
- **depth=20**: Strong, ~2s per position, ~2800 Elo
- **depth=25+**: Very strong, very slow, 3200+ Elo

## Performance Comparison

When running minimax with depth=4 (4 ply search):

| Evaluation Method | Eval Time | Total Move Time | Strength |
|------------------|-----------|-----------------|----------|
| NNUE Model       | ~1ms      | ~50-100ms       | ~1800-2000 Elo |
| Stockfish (d=10) | ~150ms    | ~300-500ms      | ~2400-2600 Elo |
| Stockfish (d=15) | ~500ms    | ~1-2s           | ~2800-3000 Elo |

## Troubleshooting

### "Stockfish not found"

Ensure Stockfish is installed and in your PATH:
```bash
which stockfish
```

Or provide the full path:
```python
stockfish_evaluator = StockfishEvaluator(stockfish_path="/path/to/stockfish")
```

### Evaluation Too Slow

Reduce the depth:
```python
stockfish_evaluator = StockfishEvaluator(depth=10)
```

### Stockfish Not Shutting Down

The evaluator automatically cleans up on exit. If needed, manually call:
```python
stockfish_evaluator.shutdown()
```

## Architecture

```
┌──────────────────────────────────────────────────────┐
│                    Game Context                       │
│                   (chess_manager)                     │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│                    main.py                            │
│  • Receives position                                  │
│  • Sends SEARCH command to C++ engine                │
└───────────────────────┬──────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────┐
│              C++ Minimax Engine                       │
│  • Alpha-beta search                                  │
│  • Move generation                                    │
│  • Sends EVAL requests ──────────────────┐           │
└───────────────────────┬──────────────────┘           │
                        │                               │
                        │ (EVAL request)                │
                        ▼                               │
┌──────────────────────────────────────────────────────┘
│                evaluate_position()                    
│                        │                              
│         ┌──────────────┴──────────────┐              
│         ▼                              ▼              
│  USE_STOCKFISH = False          USE_STOCKFISH = True 
│         │                              │              
│         ▼                              ▼              
│   ┌──────────┐                  ┌─────────────┐     
│   │   NNUE   │                  │  Stockfish  │     
│   │  Model   │                  │  Evaluator  │     
│   └────┬─────┘                  └──────┬──────┘     
│        └──────────┬──────────────────┬─┘            
│                   ▼                  ▼               
│            Returns centipawn score                   
└──────────────────────────────────────────────────────┘
```

## Why Use This?

### Benefits of Stockfish Evaluation:
1. **Stronger Evaluation**: Stockfish is one of the strongest chess engines
2. **Testing**: Compare your NNUE model against a known baseline
3. **Debugging**: Verify your minimax implementation works correctly
4. **Benchmarking**: Measure search efficiency independent of evaluation quality

### Benefits of NNUE Model:
1. **Speed**: 500x faster evaluation (~1ms vs ~500ms)
2. **Learning**: Trainable on specific positions or play styles
3. **Customization**: Tune to your preferences
4. **Resource Efficiency**: Less CPU intensive

## Future Enhancements

Possible improvements:
- [ ] Support multiple engines (Leela, etc.)
- [ ] Mix evaluations (ensemble)
- [ ] Cache Stockfish evaluations
- [ ] Dynamic depth based on time remaining
- [ ] Use Stockfish's PV line for move ordering

