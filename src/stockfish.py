import subprocess
import os
import chess
from typing import Optional

class StockfishEvaluator:
    def __init__(self, stockfish_path: Optional[str] = None, depth: int = 15):
        self.depth = depth
        self.process = None
        
        if stockfish_path is None:
            stockfish_path = self._find_stockfish()
        
        if stockfish_path is None:
            raise FileNotFoundError(
                "Stockfish not found. Please install Stockfish or provide path.\n"
                "Install with: brew install stockfish (macOS) or apt-get install stockfish (Linux)"
            )
        
        self.stockfish_path = stockfish_path
        self._start_engine()
    
    def _find_stockfish(self) -> Optional[str]:
        common_paths = [
            '/opt/homebrew/bin/stockfish',
            '/usr/local/bin/stockfish',
            '/usr/bin/stockfish',
            '/usr/games/stockfish',
        ]
        
        for path in common_paths:
            if os.path.exists(path):
                return path
        
        try:
            result = subprocess.run(
                ['which', 'stockfish'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        return None
    
    def _start_engine(self):
        self.process = subprocess.Popen(
            [self.stockfish_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
        
        self._send_command("uci")
        self._wait_for("uciok")
        self._send_command("isready")
        self._wait_for("readyok")
        
        print(f"✓ Stockfish evaluator initialized!")
        print(f"  Path: {self.stockfish_path}")
        print(f"  Evaluation depth: {self.depth}")
        print()
    
    def _send_command(self, command: str):
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()
    
    def _wait_for(self, expected: str) -> str:
        while True:
            line = self.process.stdout.readline().strip()
            if line.startswith(expected):
                return line
    
    def evaluate(self, board: chess.Board) -> int:
        self._send_command(f"position fen {board.fen()}")
        self._send_command(f"go depth {self.depth}")
        
        score_cp = 0
        
        while True:
            line = self.process.stdout.readline().strip()
            
            if line.startswith("info") and "score cp" in line:
                parts = line.split()
                try:
                    cp_index = parts.index("cp") + 1
                    score_cp = int(parts[cp_index])
                except (ValueError, IndexError):
                    pass
            
            elif line.startswith("info") and "score mate" in line:
                parts = line.split()
                try:
                    mate_index = parts.index("mate") + 1
                    mate_in = int(parts[mate_index])
                    score_cp = 10000 * (1 if mate_in > 0 else -1)
                except (ValueError, IndexError):
                    pass
            
            elif line.startswith("bestmove"):
                break
        
        return score_cp
    
    def shutdown(self):
        if self.process:
            self._send_command("quit")
            self.process.wait(timeout=2)
            self.process = None
    
    def __del__(self):
        try:
            self.shutdown()
        except:
            pass

