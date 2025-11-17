#!/bin/bash
lldb -o "run" -o "bt" -o "quit" -- ./build/chess_engine_onnx model.onnx model_config.txt < test_nnue.txt 2>&1
