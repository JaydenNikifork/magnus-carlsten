#!/usr/bin/env python3
"""
Script to partition GM_games_dataset.csv into files with 250k entries each using pandas.
"""

import pandas as pd
from pathlib import Path

def partition_csv_pandas(input_file, rows_per_file=250000):
    """
    Load a CSV into pandas and partition it into multiple files.
    
    Args:
        input_file: Path to the input CSV file
        rows_per_file: Number of rows per output file (default: 250000)
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        print(f"Error: File {input_file} not found!")
        return
    
    output_dir = input_path.parent / "partitioned"
    output_dir.mkdir(exist_ok=True)
    
    base_name = input_path.stem
    extension = input_path.suffix
    
    print(f"Loading {input_file} into pandas...")
    df = pd.read_csv(input_path)
    
    total_rows = len(df)
    num_files = (total_rows + rows_per_file - 1) // rows_per_file
    
    print(f"Total rows: {total_rows:,}")
    print(f"Rows per file: {rows_per_file:,}")
    print(f"Number of output files: {num_files}")
    print(f"Output directory: {output_dir}")
    print()
    
    for i in range(num_files):
        start_idx = i * rows_per_file
        end_idx = min((i + 1) * rows_per_file, total_rows)
        
        chunk = df.iloc[start_idx:end_idx]
        
        output_filename = output_dir / f"{base_name}_part{i+1:03d}{extension}"
        chunk.to_csv(output_filename, index=False)
        
        rows_in_chunk = len(chunk)
        print(f"  Saved part {i+1:03d}: {rows_in_chunk:,} rows -> {output_filename.name}")
    
    print()
    print(f"✓ Partitioning complete!")
    print(f"  Total rows processed: {total_rows:,}")
    print(f"  Number of output files: {num_files}")
    print(f"  Output location: {output_dir}")

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    input_file = script_dir / "GM_games_dataset.csv"
    
    partition_csv_pandas(input_file, rows_per_file=250000)







