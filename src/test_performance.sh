#!/bin/bash

echo "Testing incremental NNUE performance..."
echo

echo "Running 3 searches at depth 2..."
for i in {1..3}; do
    echo "Search $i:"
    (echo "SEARCH rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"; echo "QUIT") | \
        ./build/chess_engine_onnx model.onnx model_config.txt 2>&1 | \
        grep -E "(Time:|Nodes/sec:|Evaluations:)"
    echo
done
