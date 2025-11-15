#!/bin/bash

set -e

echo "Building C++ Chess Engine..."

cd "$(dirname "$0")"

make clean 2>/dev/null || true
make

echo ""
echo "Build complete! Engine binary: src/build/chess_engine"

