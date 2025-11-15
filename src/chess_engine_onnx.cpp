#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <sstream>
#include <fstream>
#include <chrono>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "chess.hpp"

const int MAX_DEPTH = 4;
const int INF = 1000000;
const int MATE_SCORE = 100000;

class NNUEEvaluator {
private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::string input_name_str;   // Store as string
    std::string output_name_str;  // Store as string
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    int max_cp;
    bool loaded;
    int eval_count;
    
public:
    NNUEEvaluator() : loaded(false), max_cp(1000), eval_count(0) {
        env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ChessEngine");
    }
    
    bool loadModel(const std::string& model_path, const std::string& config_path) {
        try {
            std::cerr << "Loading ONNX model from: " << model_path << std::endl;
            
            // Session options
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            // Load model
            session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);
            
            // Get input/output names
            Ort::AllocatorWithDefaultOptions allocator;
            
            // Input name - store in string first
            auto input_name_alloc = session->GetInputNameAllocated(0, allocator);
            input_name_str = std::string(input_name_alloc.get());
            input_names.push_back(input_name_str.c_str());
            
            // Output name - store in string first
            auto output_name_alloc = session->GetOutputNameAllocated(0, allocator);
            output_name_str = std::string(output_name_alloc.get());
            output_names.push_back(output_name_str.c_str());
            
            // Load config
            std::ifstream config_file(config_path);
            if (config_file.is_open()) {
                config_file >> max_cp;
                config_file.close();
            }
            
            std::cerr << "✓ ONNX Runtime model loaded successfully!" << std::endl;
            std::cerr << "  Max centipawns: ±" << max_cp << std::endl;
            std::cerr << "  Input: " << input_names[0] << std::endl;
            std::cerr << "  Output: " << output_names[0] << std::endl;
            std::cerr << std::endl;
            
            loaded = true;
            return true;
        } catch (const Ort::Exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            loaded = false;
            return false;
        }
    }
    
    void boardToFeatures(const chess::Board& board, float* features) {
        std::fill(features, features + 768, 0.0f);
        
        for (int square = 0; square < 64; square++) {
            auto piece = board.at(chess::Square(square));
            
            if (piece == chess::Piece::NONE) continue;
            
            int piece_type = static_cast<int>(piece.type()) - 1;
            bool is_white = (piece.color() == chess::Color::WHITE);
            
            int color_offset = is_white ? 0 : 6;
            int feature_idx = (piece_type + color_offset) * 64 + square;
            
            features[feature_idx] = 1.0f;
        }
    }
    
    int evaluate(const chess::Board& board) {
        if (!loaded) {
            std::cerr << "Model not loaded!" << std::endl;
            return 0;
        }
        
        eval_count++;
        
        try {
            // Prepare input
            float input_data[768];
            boardToFeatures(board, input_data);
            
            // Create input tensor
            std::vector<int64_t> input_shape = {1, 768};
            size_t input_tensor_size = 768;
            
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, input_data, input_tensor_size, 
                input_shape.data(), input_shape.size()
            );
            
            // Run inference
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names.data(), &input_tensor, 1,
                output_names.data(), 1
            );
            
            // Get output
            float* output_data = output_tensors.front().GetTensorMutableData<float>();
            float normalized = output_data[0];
            int score = static_cast<int>(normalized * max_cp);
            
            return score;
        } catch (const Ort::Exception& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
            return 0;
        }
    }
    
    bool isLoaded() const { return loaded; }
    int getEvalCount() const { return eval_count; }
    void resetEvalCount() { eval_count = 0; }
};

class ChessEngine {
private:
    NNUEEvaluator evaluator;
    
    int alphaBeta(chess::Board& board, int depth, int alpha, int beta, bool maximizing, int& nodes_evaluated) {
        auto game_over = board.isGameOver();
        
        if (depth == 0 || game_over.first != chess::GameResultReason::NONE) {
            nodes_evaluated++;
            
            if (game_over.first != chess::GameResultReason::NONE) {
                auto result = game_over.second;
                if (result == chess::GameResult::WIN) {
                    return board.sideToMove() == chess::Color::WHITE ? MATE_SCORE : -MATE_SCORE;
                } else if (result == chess::GameResult::LOSE) {
                    return board.sideToMove() == chess::Color::WHITE ? -MATE_SCORE : MATE_SCORE;
                } else {
                    return 0;
                }
            }
            
            int score = evaluator.evaluate(board);
            return board.sideToMove() == chess::Color::WHITE ? score : -score;
        }
        
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);
        
        if (moves.empty()) {
            nodes_evaluated++;
            return 0;
        }
        
        if (maximizing) {
            int maxEval = -INF;
            for (const auto& move : moves) {
                board.makeMove(move);
                int eval = alphaBeta(board, depth - 1, alpha, beta, false, nodes_evaluated);
                board.unmakeMove(move);
                
                maxEval = std::max(maxEval, eval);
                alpha = std::max(alpha, eval);
                if (beta <= alpha) {
                    break;
                }
            }
            return maxEval;
        } else {
            int minEval = INF;
            for (const auto& move : moves) {
                board.makeMove(move);
                int eval = alphaBeta(board, depth - 1, alpha, beta, true, nodes_evaluated);
                board.unmakeMove(move);
                
                minEval = std::min(minEval, eval);
                beta = std::min(beta, eval);
                if (beta <= alpha) {
                    break;
                }
            }
            return minEval;
        }
    }

public:
    ChessEngine() {}
    
    bool initialize(const std::string& model_path, const std::string& config_path) {
        return evaluator.loadModel(model_path, config_path);
    }

    std::pair<std::string, int> findBestMove(const std::string& fen, int max_depth = MAX_DEPTH) {
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::cerr << "=== Starting search ===" << std::endl;
        std::cerr << "Position: " << fen.substr(0, 60) << "..." << std::endl;
        std::cerr << "Depth: " << max_depth << std::endl;
        
        chess::Board board(fen);
        
        chess::Movelist moves;
        chess::movegen::legalmoves(moves, board);
        
        if (moves.empty()) {
            std::cerr << "No legal moves available!" << std::endl;
            return {"", 0};
        }
        
        std::cerr << "Legal moves: " << moves.size() << std::endl;
        
        std::string best_move_uci;
        int best_score = -INF;
        int move_count = 0;
        int total_nodes = 0;
        
        evaluator.resetEvalCount();
        
        for (const auto& move : moves) {
            board.makeMove(move);
            int nodes_for_move = 0;
            int score = alphaBeta(board, max_depth - 1, -INF, INF, false, nodes_for_move);
            board.unmakeMove(move);
            
            move_count++;
            total_nodes += nodes_for_move;
            
            std::string move_uci = chess::uci::moveToUci(move);
            
            std::cerr << "  Move " << move_count << "/" << moves.size() 
                      << ": " << move_uci << " -> Score: " << score
                      << " (nodes: " << nodes_for_move << ")";
            
            if (score > best_score) {
                best_score = score;
                best_move_uci = move_uci;
                std::cerr << " (NEW BEST!)";
            }
            std::cerr << std::endl;
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cerr << "=== Search complete ===" << std::endl;
        std::cerr << "Best move: " << best_move_uci << std::endl;
        std::cerr << "Best score: " << best_score << " centipawns" << std::endl;
        std::cerr << "Total nodes evaluated: " << total_nodes << std::endl;
        std::cerr << "NNUE evaluations: " << evaluator.getEvalCount() << std::endl;
        std::cerr << "Time: " << duration.count() << "ms" << std::endl;
        std::cerr << "Nodes/sec: " << (total_nodes * 1000 / std::max(1LL, duration.count())) << std::endl;
        std::cerr << std::endl;
        
        return {best_move_uci, best_score};
    }
};

int main(int argc, char* argv[]) {
    ChessEngine engine;
    
    std::string model_path = "model.onnx";
    std::string config_path = "model_config.txt";
    
    if (argc > 1) {
        model_path = argv[1];
    }
    if (argc > 2) {
        config_path = argv[2];
    }
    
    std::cerr << "=========================================" << std::endl;
    std::cerr << "Chess Engine with ONNX Runtime" << std::endl;
    std::cerr << "High Performance, Easy Setup" << std::endl;
    std::cerr << "=========================================" << std::endl;
    std::cerr << std::endl;
    
    if (!engine.initialize(model_path, config_path)) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return 1;
    }
    
    std::cerr << "Engine ready (ONNX Runtime - optimized)" << std::endl;
    std::cout << "READY" << std::endl;
    std::cout.flush();
    
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;
        
        std::istringstream iss(line);
        std::string command;
        iss >> command;
        
        if (command == "SEARCH") {
            std::string fen;
            std::getline(iss, fen);
            if (!fen.empty() && fen[0] == ' ') {
                fen = fen.substr(1);
            }
            
            int depth = MAX_DEPTH;
            
            auto [best_move, score] = engine.findBestMove(fen, depth);
            std::cout << "BESTMOVE " << best_move << " " << score << std::endl;
            std::cout.flush();
            
        } else if (command == "QUIT") {
            break;
        }
    }
    
    return 0;
}

