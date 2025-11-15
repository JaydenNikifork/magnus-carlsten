"""
Dataset for loading and processing Lichess evaluation data.
Supports the JSON format with FEN and evaluations.
Supports both in-memory and streaming modes.
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
import chess
from features import fen_to_features


class LichessEvalDatasetStreaming(Dataset):
    """
    Streaming dataset for Lichess positions - loads data on-demand.
    Only stores file offsets in memory, reads positions when needed.
    Memory usage: ~8 bytes per position (just the offset)
    """
    
    def __init__(self, json_file, max_cp=1000, use_depth=None, max_lines=None):
        """
        Args:
            json_file: Path to JSON lines file
            max_cp: Clamp centipawn values
            use_depth: If specified, only use evals at this depth
            max_lines: Maximum number of lines to index
        """
        self.json_file = json_file
        self.max_cp = max_cp
        self.use_depth = use_depth
        self.max_lines = max_lines
        
        print(f"Building streaming index for {json_file}...")
        if max_lines:
            print(f"  (limiting to {max_lines:,} lines)")
        
        # Build index of valid positions (stores only file offsets)
        self.valid_offsets = []
        self._build_index()
        
        print(f"Indexed {len(self.valid_offsets):,} valid positions")
        print(f"Memory usage: ~{len(self.valid_offsets) * 8 / 1024 / 1024:.1f} MB (offsets only)")
    
    def _build_index(self):
        """Build index of file offsets for valid positions"""
        with open(self.json_file, 'rb') as f:
            offset = 0
            line_num = 0
            
            for line in f:
                line_num += 1
                
                # Check max_lines limit
                if self.max_lines and line_num > self.max_lines:
                    print(f"  Reached max_lines limit ({self.max_lines:,})")
                    break
                
                # Progress update
                if line_num % 10000 == 0:
                    valid_count = len(self.valid_offsets)
                    if self.max_lines:
                        remaining = self.max_lines - line_num
                        print(f"  Progress: {line_num:,}/{self.max_lines:,} lines "
                              f"({valid_count:,} valid positions, {remaining:,} remaining)")
                    else:
                        print(f"  Progress: {line_num:,} lines ({valid_count:,} valid positions)")
                
                # Try to parse and validate
                try:
                    obj = json.loads(line.decode('utf-8').strip())
                    fen = obj.get('fen')
                    evals = obj.get('evals', [])
                    
                    if not fen or not evals:
                        offset += len(line)
                        continue
                    
                    # Quick validation
                    cp = self._extract_cp_quick(evals)
                    if cp is None:
                        offset += len(line)
                        continue
                    
                    # Store offset of this valid position
                    self.valid_offsets.append(offset)
                    
                except:
                    pass
                
                offset += len(line)
    
    def _extract_cp_quick(self, evals):
        """Quick CP extraction for indexing (minimal validation)"""
        if not evals:
            return None
        
        if self.use_depth is not None:
            matching = [e for e in evals if e.get('depth') == self.use_depth]
            if not matching:
                return None
            eval_entry = matching[0]
        else:
            eval_entry = max(evals, key=lambda e: e.get('depth', 0))
        
        pvs = eval_entry.get('pvs', [])
        if not pvs:
            return None
        
        first_pv = pvs[0]
        cp = first_pv.get('cp')
        
        if cp is None:
            mate = first_pv.get('mate')
            if mate is not None:
                cp = self.max_cp if mate > 0 else -self.max_cp
            else:
                return None
        
        return cp
    
    def __len__(self):
        return len(self.valid_offsets)
    
    def __getitem__(self, idx):
        """
        Load and parse position on-demand from file.
        Only called when this specific position is needed for training.
        """
        offset = self.valid_offsets[idx]
        
        # Read the line at this offset
        with open(self.json_file, 'rb') as f:
            f.seek(offset)
            line = f.readline().decode('utf-8').strip()
        
        # Parse JSON
        obj = json.loads(line)
        fen = obj['fen']
        evals = obj['evals']
        
        # Extract CP
        cp = self._extract_cp_quick(evals)
        
        # Convert to features
        try:
            features = fen_to_features(fen)
        except:
            # Fallback to zeros if FEN is invalid
            features = torch.zeros(768)
        
        # Normalize centipawn score
        cp_clamped = max(-self.max_cp, min(self.max_cp, cp))
        target = torch.tensor([cp_clamped / self.max_cp], dtype=torch.float32)
        
        return features, target


class LichessEvalDataset(Dataset):
    """
    Dataset for Lichess positions with evaluations.
    
    Expected JSON format:
    {
        "fen": "...",
        "evals": [
            {
                "pvs": [{"cp": 311, "line": "..."}, ...],
                "depth": 36,
                ...
            },
            ...
        ]
    }
    """
    
    def __init__(self, json_file, max_cp=1000, use_depth=None, max_lines=None):
        """
        Args:
            json_file: Path to JSON lines file (one JSON object per line)
            max_cp: Clamp centipawn values to [-max_cp, max_cp]
            use_depth: If specified, only use evals at this depth (None = use first eval)
            max_lines: Maximum number of lines to read (None = read all)
        """
        self.data = []
        self.max_cp = max_cp
        self.use_depth = use_depth
        self.max_lines = max_lines
        
        print(f"Loading data from {json_file}...")
        if max_lines:
            print(f"  (limiting to {max_lines:,} lines)")
        self._load_data(json_file)
        print(f"Loaded {len(self.data)} positions")
    
    def _load_data(self, json_file):
        """Load and parse the JSON lines file"""
        with open(json_file, 'r') as f:
            for line_num, line in enumerate(f, 1):
                # Check if we've reached max_lines
                if self.max_lines and line_num > self.max_lines:
                    print(f"  Reached max_lines limit ({self.max_lines:,})")
                    break
                
                # Print progress every 10k lines
                if line_num % 10000 == 0:
                    positions_loaded = len(self.data)
                    if self.max_lines:
                        remaining = self.max_lines - line_num
                        print(f"  Progress: {line_num:,}/{self.max_lines:,} lines ({positions_loaded:,} positions loaded, {remaining:,} lines remaining)")
                    else:
                        print(f"  Progress: {line_num:,} lines processed ({positions_loaded:,} positions loaded)")
                
                try:
                    obj = json.loads(line.strip())
                    fen = obj.get('fen')
                    evals = obj.get('evals', [])
                    
                    if not fen or not evals:
                        continue
                    
                    # Extract centipawn evaluation
                    cp = self._extract_cp(evals)
                    if cp is None:
                        continue
                    
                    # Validate FEN
                    try:
                        chess.Board(fen)
                    except:
                        continue
                    
                    self.data.append((fen, cp))
                    
                except json.JSONDecodeError:
                    print(f"Warning: Could not parse line {line_num}")
                    continue
                except Exception as e:
                    print(f"Warning: Error processing line {line_num}: {e}")
                    continue
    
    def _extract_cp(self, evals):
        """
        Extract centipawn evaluation from evals list.
        
        Strategy:
        1. If use_depth is specified, find eval at that depth
        2. Otherwise, use the eval with highest depth
        3. Take the first PV's cp value
        """
        if not evals:
            return None
        
        # Find the right eval entry
        if self.use_depth is not None:
            # Find eval with matching depth
            matching = [e for e in evals if e.get('depth') == self.use_depth]
            if not matching:
                return None
            eval_entry = matching[0]
        else:
            # Use the eval with highest depth
            eval_entry = max(evals, key=lambda e: e.get('depth', 0))
        
        # Extract cp from first PV
        pvs = eval_entry.get('pvs', [])
        if not pvs:
            return None
        
        first_pv = pvs[0]
        cp = first_pv.get('cp')
        
        # Handle mate scores (skip them for now or clamp to max)
        if cp is None:
            mate = first_pv.get('mate')
            if mate is not None:
                # Mate in N moves -> very high score
                cp = self.max_cp if mate > 0 else -self.max_cp
            else:
                return None
        
        return cp
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            features: torch.Tensor of shape [768]
            target: torch.Tensor of shape [1] - normalized cp score
        """
        fen, cp = self.data[idx]
        
        # Convert FEN to features
        features = fen_to_features(fen)
        
        # Normalize centipawn score to [-1, 1]
        cp_clamped = max(-self.max_cp, min(self.max_cp, cp))
        target = torch.tensor([cp_clamped / self.max_cp], dtype=torch.float32)
        
        return features, target


def create_dataloader(json_file, batch_size=256, shuffle=True, max_cp=1000, train_split=0.9, max_lines=None, streaming=False):
    """
    Create train and validation dataloaders.
    
    Args:
        json_file: Path to data file
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        max_cp: Clamp threshold for centipawns
        train_split: Fraction of data to use for training
        max_lines: Maximum number of lines to read from file (None = read all)
        streaming: If True, use streaming mode (low memory, slower). If False, load all into memory (fast, high memory)
    
    Returns:
        train_loader, val_loader
    """
    # Choose dataset class based on streaming mode
    if streaming:
        print("Using STREAMING mode (low memory, on-demand loading)")
        dataset = LichessEvalDatasetStreaming(json_file, max_cp=max_cp, max_lines=max_lines)
    else:
        print("Using IN-MEMORY mode (high memory, fast loading)")
        dataset = LichessEvalDataset(json_file, max_cp=max_cp, max_lines=max_lines)
    
    # Split into train/val
    train_size = int(len(dataset) * train_split)
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,  # Set to 0 for simplicity; increase if you have CPU cores
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Train set: {len(train_dataset)} positions")
    print(f"Val set: {len(val_dataset)} positions")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test the dataset loader
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python dataset.py <json_file>")
        print("\nCreating synthetic test data...")
        
        # Create a small test file
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
        
        with open('test_data.jsonl', 'w') as f:
            for item in test_data:
                f.write(json.dumps(item) + '\n')
        
        dataset = LichessEvalDataset('test_data.jsonl')
        print(f"\nLoaded {len(dataset)} positions")
        
        # Test first item
        features, target = dataset[0]
        print(f"Features shape: {features.shape}")
        print(f"Target shape: {target.shape}")
        print(f"Target value: {target.item():.4f} (normalized)")
        print(f"Active features: {features.sum().item():.0f}")
    else:
        # Test with real file
        dataset = LichessEvalDataset(sys.argv[1])
        print(f"\nDataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            features, target = dataset[0]
            print(f"Sample features shape: {features.shape}")
            print(f"Sample target: {target.item():.4f}")

