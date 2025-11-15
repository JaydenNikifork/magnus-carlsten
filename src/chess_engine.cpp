#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <limits>
#include <sstream>
#include <map>

const int MAX_DEPTH = 4;
const int INF = 1000000;
const int MATE_SCORE = 100000;

struct Move {
    std::string uci;
    int score;
};

struct Position {
    std::string fen;
    std::vector<std::string> legal_moves;
    bool is_terminal;
    int terminal_score;
};

class ChessEngine {
private:
    int evaluatePosition(const std::string& fen) {
        std::cout << "EVAL " << fen << std::endl;
        std::cout.flush();
        
        std::string response;
        std::getline(std::cin, response);
        
        try {
            return std::stoi(response);
        } catch (...) {
            std::cerr << "Error parsing eval response: " << response << std::endl;
            return 0;
        }
    }

    Position getPosition(const std::string& fen) {
        std::cout << "POSITION " << fen << std::endl;
        std::cout.flush();
        
        Position pos;
        pos.fen = fen;
        
        std::string response;
        std::getline(std::cin, response);
        
        std::istringstream iss(response);
        std::string status;
        iss >> status;
        
        if (status == "TERMINAL") {
            pos.is_terminal = true;
            std::string result;
            iss >> result;
            if (result == "1-0") pos.terminal_score = MATE_SCORE;
            else if (result == "0-1") pos.terminal_score = -MATE_SCORE;
            else pos.terminal_score = 0;
        } else if (status == "NORMAL") {
            pos.is_terminal = false;
            pos.terminal_score = 0;
            std::string move;
            while (iss >> move) {
                pos.legal_moves.push_back(move);
            }
        }
        
        return pos;
    }

    std::string makeMove(const std::string& fen, const std::string& move) {
        std::cout << "MAKEMOVE " << fen << " " << move << std::endl;
        std::cout.flush();
        
        std::string new_fen;
        std::getline(std::cin, new_fen);
        return new_fen;
    }

    int alphaBeta(const std::string& fen, int depth, int alpha, int beta, bool maximizing, int& nodes_evaluated) {
        Position pos = getPosition(fen);
        
        if (depth == 0 || pos.is_terminal) {
            nodes_evaluated++;
            if (pos.is_terminal) {
                return pos.terminal_score;
            }
            return evaluatePosition(fen);
        }
        
        if (pos.legal_moves.empty()) {
            nodes_evaluated++;
            return pos.terminal_score;
        }
        
        if (maximizing) {
            int maxEval = -INF;
            for (const auto& move : pos.legal_moves) {
                std::string new_fen = makeMove(fen, move);
                int eval = alphaBeta(new_fen, depth - 1, alpha, beta, false, nodes_evaluated);
                maxEval = std::max(maxEval, eval);
                alpha = std::max(alpha, eval);
                if (beta <= alpha) {
                    break;
                }
            }
            return maxEval;
        } else {
            int minEval = INF;
            for (const auto& move : pos.legal_moves) {
                std::string new_fen = makeMove(fen, move);
                int eval = alphaBeta(new_fen, depth - 1, alpha, beta, true, nodes_evaluated);
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

    Move findBestMove(const std::string& fen, int max_depth = MAX_DEPTH) {
        std::cerr << "=== Starting search ===" << std::endl;
        std::cerr << "Position: " << fen.substr(0, 60) << "..." << std::endl;
        std::cerr << "Depth: " << max_depth << std::endl;
        
        Position pos = getPosition(fen);
        
        if (pos.legal_moves.empty()) {
            std::cerr << "No legal moves available!" << std::endl;
            return {"", 0};
        }
        
        std::cerr << "Legal moves: " << pos.legal_moves.size() << std::endl;
        
        Move best_move = {pos.legal_moves[0], -INF};
        int move_count = 0;
        int total_nodes = 0;
        
        for (const auto& move : pos.legal_moves) {
            std::string new_fen = makeMove(fen, move);
            int nodes_for_move = 0;
            int score = alphaBeta(new_fen, max_depth - 1, -INF, INF, false, nodes_for_move);
            move_count++;
            total_nodes += nodes_for_move;
            
            std::cerr << "  Move " << move_count << "/" << pos.legal_moves.size() 
                      << ": " << move << " -> Score: " << score
                      << " (nodes: " << nodes_for_move << ")";
            
            if (score > best_move.score) {
                best_move = {move, score};
                std::cerr << " (NEW BEST!)";
            }
            std::cerr << std::endl;
        }
        
        std::cerr << "=== Search complete ===" << std::endl;
        std::cerr << "Best move: " << best_move.uci << std::endl;
        std::cerr << "Best score: " << best_move.score << " centipawns" << std::endl;
        std::cerr << "Total nodes evaluated: " << total_nodes << std::endl;
        std::cerr << std::endl;
        
        return best_move;
    }
};

int main() {
    ChessEngine engine;
    
    std::cerr << "C++ Engine ready" << std::endl;
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
            
            Move best = engine.findBestMove(fen, depth);
            std::cout << "BESTMOVE " << best.uci << " " << best.score << std::endl;
            std::cout.flush();
        } else if (command == "QUIT") {
            break;
        }
    }
    
    return 0;
}

