#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>
#include <chrono>
#include "chess.hpp"
#include "nnue_evaluator.hpp"

const int MAX_DEPTH = 2;
const int INF = 1000000;
const int MATE_SCORE = 100000;

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
            
            int score = evaluator.evaluate();
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
                evaluator.move(move, board);
                board.makeMove(move);
                int eval = alphaBeta(board, depth - 1, alpha, beta, false, nodes_evaluated);
                board.unmakeMove(move);
                evaluator.undo();
                
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
                evaluator.move(move, board);
                board.makeMove(move);
                int eval = alphaBeta(board, depth - 1, alpha, beta, true, nodes_evaluated);
                board.unmakeMove(move);
                evaluator.undo();
                
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
        evaluator.reset(board);
        
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
            evaluator.move(move, board);
            board.makeMove(move);
            int nodes_for_move = 0;
            int score = alphaBeta(board, max_depth - 1, -INF, INF, false, nodes_for_move);
            board.unmakeMove(move);
            evaluator.undo();
            
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
        std::cerr << "Evaluations: " << evaluator.getEvalCount() << std::endl;
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
    
    std::cerr << "Shutting down gracefully..." << std::endl;
    
    return 0;
}

