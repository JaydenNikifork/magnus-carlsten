#include "nnue_evaluator.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstring>

NNUEEvaluator::NNUEEvaluator() : loaded(false), max_cp(1000), eval_count(0), h1_size(0), h2_size(0) {
    env = std::make_unique<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ChessEngine");
    std::memset(current_features, 0, sizeof(current_features));
}

bool NNUEEvaluator::loadLayer1Weights(const std::string& weights_path, const std::string& bias_path) {
    std::ifstream weights_file(weights_path, std::ios::binary);
    std::ifstream bias_file(bias_path, std::ios::binary);
    
    if (!weights_file.is_open() || !bias_file.is_open()) {
        std::cerr << "Failed to open layer1 weight files" << std::endl;
        return false;
    }
    
    layer1_weights.resize(768 * h1_size);
    layer1_bias.resize(h1_size);
    layer1_accumulator.resize(h1_size);
    
    weights_file.read(reinterpret_cast<char*>(layer1_weights.data()), 768 * h1_size * sizeof(float));
    bias_file.read(reinterpret_cast<char*>(layer1_bias.data()), h1_size * sizeof(float));
    
    weights_file.close();
    bias_file.close();
    
    std::cerr << "✓ Layer 1 weights loaded for incremental updates" << std::endl;
    std::cerr << "  Shape: [768, " << h1_size << "]" << std::endl;
    
    return true;
}

bool NNUEEvaluator::loadModel(const std::string& model_path, const std::string& config_path) {
    try {
        std::cerr << "Loading ONNX model from: " << model_path << std::endl;
        
        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
        
        std::ifstream config_file(config_path);
        if (config_file.is_open()) {
            config_file >> max_cp;
            config_file >> h1_size;
            config_file >> h2_size;
            config_file.close();
        } else {
            std::cerr << "Warning: Could not read config file, using defaults" << std::endl;
            h1_size = 512;
            h2_size = 64;
        }
        
        std::string model_dir = "";
        size_t last_slash = model_path.find_last_of("/\\");
        if (last_slash != std::string::npos) {
            model_dir = model_path.substr(0, last_slash + 1);
        }
        
        std::string partial_model_path = model_dir + "model_partial.onnx";
        
        try {
            Ort::SessionOptions partial_options;
            partial_options.SetIntraOpNumThreads(1);
            partial_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            std::cerr << "Attempting to load partial model (CPU mode, GPU can be enabled via CMake)..." << std::endl;
            
            partial_session = std::make_unique<Ort::Session>(*env, partial_model_path.c_str(), partial_options);
            
            Ort::AllocatorWithDefaultOptions allocator;
            auto partial_input_name_alloc = partial_session->GetInputNameAllocated(0, allocator);
            partial_input_name_str = std::string(partial_input_name_alloc.get());
            partial_input_name_alloc.reset();
            
            auto partial_output_name_alloc = partial_session->GetOutputNameAllocated(0, allocator);
            partial_output_name_str = std::string(partial_output_name_alloc.get());
            partial_output_name_alloc.reset();
            
            partial_input_names.clear();
            partial_input_names.push_back(partial_input_name_str.c_str());
            partial_output_names.clear();
            partial_output_names.push_back(partial_output_name_str.c_str());
            
            std::cerr << "✓ Partial ONNX model loaded successfully!" << std::endl;
            std::cerr << "  Architecture: " << h1_size << " → " << h2_size << " → 1" << std::endl;
            std::cerr << "  Input: \"" << partial_input_name_str << "\" (length: " << partial_input_name_str.length() << ")" << std::endl;
            std::cerr << "  Output: \"" << partial_output_name_str << "\" (length: " << partial_output_name_str.length() << ")" << std::endl;
        } catch (const Ort::Exception& e) {
            std::cerr << "⚠ Could not load partial model: " << e.what() << std::endl;
            std::cerr << "  Falling back to full model" << std::endl;
            partial_session.reset();
        }
        
        session = std::make_unique<Ort::Session>(*env, model_path.c_str(), session_options);
        
        Ort::AllocatorWithDefaultOptions allocator2;
        
        auto input_name_alloc = session->GetInputNameAllocated(0, allocator2);
        input_name_str = std::string(input_name_alloc.get());
        input_names.clear();
        input_names.push_back(input_name_str.c_str());
        
        auto output_name_alloc = session->GetOutputNameAllocated(0, allocator2);
        output_name_str = std::string(output_name_alloc.get());
        output_names.clear();
        output_names.push_back(output_name_str.c_str());
        
        std::cerr << "✓ Full ONNX Runtime model loaded (fallback)" << std::endl;
        std::cerr << "  Architecture: 768 → " << h1_size << " → " << h2_size << " → 1" << std::endl;
        std::cerr << "  Max centipawns: ±" << max_cp << std::endl;
        std::cerr << std::endl;
        
        std::string weights_path = model_dir + "layer1_weights.bin";
        std::string bias_path = model_dir + "layer1_bias.bin";
        
        if (!loadLayer1Weights(weights_path, bias_path)) {
            std::cerr << "Warning: Could not load layer1 weights, incremental updates disabled" << std::endl;
        }
        
        loaded = true;
        return true;
    } catch (const Ort::Exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        loaded = false;
        return false;
    }
}

void NNUEEvaluator::boardToFeatures(const chess::Board& board, float* features) {
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

int NNUEEvaluator::getFeatureIndex(chess::PieceType type, chess::Color color, int square) {
    int piece_type = static_cast<int>(type) - 1;
    int color_offset = (color == chess::Color::WHITE) ? 0 : 6;
    return (piece_type + color_offset) * 64 + square;
}

void NNUEEvaluator::reset(const chess::Board& board) {
    boardToFeatures(board, current_features);
    pending_diffs.clear();
    while (!history_stack.empty()) {
        history_stack.pop();
    }
    
    if (layer1_weights.empty()) return;
    
    std::fill(layer1_accumulator.begin(), layer1_accumulator.end(), 0.0f);
    
    for (int i = 0; i < h1_size; i++) {
        layer1_accumulator[i] = layer1_bias[i];
    }
    
    for (int feature_idx = 0; feature_idx < 768; feature_idx++) {
        if (current_features[feature_idx] > 0.0f) {
            for (int neuron = 0; neuron < h1_size; neuron++) {
                layer1_accumulator[neuron] += layer1_weights[feature_idx * h1_size + neuron] * current_features[feature_idx];
            }
        }
    }
}

void NNUEEvaluator::move(const chess::Move& move, const chess::Board& board_before) {
    std::vector<FeatureDiff> move_diffs;
    
    int from_square = static_cast<int>(move.from().index());
    int to_square = static_cast<int>(move.to().index());
    auto piece_moved = board_before.at(move.from());
    auto piece_captured = board_before.at(move.to());
    
    auto move_type = move.typeOf();
    
    if (move_type == chess::Move::NORMAL) {
        int from_idx = getFeatureIndex(piece_moved.type(), piece_moved.color(), from_square);
        move_diffs.push_back({from_idx, current_features[from_idx], 0.0f});
        
        int to_idx = getFeatureIndex(piece_moved.type(), piece_moved.color(), to_square);
        move_diffs.push_back({to_idx, current_features[to_idx], 1.0f});
        
        if (piece_captured != chess::Piece::NONE) {
            int captured_idx = getFeatureIndex(piece_captured.type(), piece_captured.color(), to_square);
            move_diffs.push_back({captured_idx, current_features[captured_idx], 0.0f});
        }
    }
    else if (move_type == chess::Move::PROMOTION) {
        auto promoted_type = move.promotionType();
        int from_idx = getFeatureIndex(piece_moved.type(), piece_moved.color(), from_square);
        int prom_idx = getFeatureIndex(promoted_type, piece_moved.color(), to_square);
        
        move_diffs.push_back({from_idx, current_features[from_idx], 0.0f});
        move_diffs.push_back({prom_idx, current_features[prom_idx], 1.0f});
        
        if (piece_captured != chess::Piece::NONE) {
            int captured_idx = getFeatureIndex(piece_captured.type(), piece_captured.color(), to_square);
            move_diffs.push_back({captured_idx, current_features[captured_idx], 0.0f});
        }
    }
    else if (move_type == chess::Move::CASTLING) {
        chess::Color color = piece_moved.color();
        bool king_side = to_square > from_square;
        
        auto rook_from = move.to();
        auto rook_to = chess::Square::castling_rook_square(king_side, color);
        auto king_to = chess::Square::castling_king_square(king_side, color);
        
        int king_from_idx = getFeatureIndex(chess::PieceType::KING, color, from_square);
        int king_to_idx = getFeatureIndex(chess::PieceType::KING, color, king_to.index());
        int rook_from_idx = getFeatureIndex(chess::PieceType::ROOK, color, rook_from.index());
        int rook_to_idx = getFeatureIndex(chess::PieceType::ROOK, color, rook_to.index());
        
        move_diffs.push_back({king_from_idx, current_features[king_from_idx], 0.0f});
        move_diffs.push_back({king_to_idx, current_features[king_to_idx], 1.0f});
        move_diffs.push_back({rook_from_idx, current_features[rook_from_idx], 0.0f});
        move_diffs.push_back({rook_to_idx, current_features[rook_to_idx], 1.0f});
    }
    else if (move_type == chess::Move::ENPASSANT) {
        int from_idx = getFeatureIndex(piece_moved.type(), piece_moved.color(), from_square);
        int to_idx = getFeatureIndex(piece_moved.type(), piece_moved.color(), to_square);
        
        int ep_square = (piece_moved.color() == chess::Color::WHITE) 
                       ? to_square - 8 
                       : to_square + 8;
        int ep_idx = getFeatureIndex(chess::PieceType::PAWN, 
                                     piece_moved.color() == chess::Color::WHITE ? chess::Color::BLACK : chess::Color::WHITE,
                                     ep_square);
        
        move_diffs.push_back({from_idx, current_features[from_idx], 0.0f});
        move_diffs.push_back({to_idx, current_features[to_idx], 1.0f});
        move_diffs.push_back({ep_idx, current_features[ep_idx], 0.0f});
    }
    
    history_stack.push(move_diffs);
    
    for (const auto& diff : move_diffs) {
        pending_diffs.push_back(diff);
        
        if (!layer1_weights.empty()) {
            int feature_idx = diff.index;
            float delta = diff.new_value - diff.old_value;
            
            if (delta != 0.0f) {
                for (int neuron = 0; neuron < h1_size; neuron++) {
                    layer1_accumulator[neuron] += layer1_weights[feature_idx * h1_size + neuron] * delta;
                }
            }
        }
    }
}

void NNUEEvaluator::undo() {
    if (history_stack.empty()) return;
    
    auto last_diffs = history_stack.top();
    history_stack.pop();
    
    for (const auto& diff : last_diffs) {
        pending_diffs.push_back({diff.index, diff.new_value, diff.old_value});
        
        if (!layer1_weights.empty()) {
            int feature_idx = diff.index;
            float delta = diff.old_value - diff.new_value;
            
            if (delta != 0.0f) {
                for (int neuron = 0; neuron < h1_size; neuron++) {
                    layer1_accumulator[neuron] += layer1_weights[feature_idx * h1_size + neuron] * delta;
                }
            }
        }
    }
}

int NNUEEvaluator::evaluate() {
    if (!loaded) {
        std::cerr << "Model not loaded!" << std::endl;
        return 0;
    }
    
    eval_count++;
    
    for (const auto& diff : pending_diffs) {
        current_features[diff.index] = diff.new_value;
    }
    pending_diffs.clear();
    
    if (layer1_weights.empty() || !partial_session) {
        try {
            std::vector<int64_t> input_shape = {1, 768};
            size_t input_tensor_size = 768;
            
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, current_features, input_tensor_size, 
                input_shape.data(), input_shape.size()
            );
            
            auto output_tensors = session->Run(
                Ort::RunOptions{nullptr},
                input_names.data(), &input_tensor, 1,
                output_names.data(), 1
            );
            
            float* output_data = output_tensors.front().GetTensorMutableData<float>();
            float normalized = output_data[0];
            int score = static_cast<int>(normalized * max_cp);
            
            return score;
        } catch (const Ort::Exception& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
            return 0;
        }
    }
    
    std::vector<float> layer1_output(h1_size);
    for (int i = 0; i < h1_size; i++) {
        layer1_output[i] = std::max(0.0f, layer1_accumulator[i]);
    }
    
    try {
        if (partial_input_name_str.empty() || partial_output_name_str.empty()) {
            std::cerr << "Error: Partial model input/output names not initialized" << std::endl;
            return 0;
        }
        
        std::vector<int64_t> input_shape = {1, static_cast<int64_t>(h1_size)};
        
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, layer1_output.data(), h1_size,
            input_shape.data(), input_shape.size()
        );
        
        std::vector<const char*> input_names_vec = {partial_input_name_str.c_str()};
        std::vector<const char*> output_names_vec = {partial_output_name_str.c_str()};
        
        auto output_tensors = partial_session->Run(
            Ort::RunOptions{nullptr},
            input_names_vec.data(), &input_tensor, 1,
            output_names_vec.data(), 1
        );
        
        float* output_data = output_tensors.front().GetTensorMutableData<float>();
        float normalized = output_data[0];
        int score = static_cast<int>(normalized * max_cp);
        
        return score;
    } catch (const Ort::Exception& e) {
        std::cerr << "Partial inference error: " << e.what() << std::endl;
        std::cerr << "  Error code: " << e.GetOrtErrorCode() << std::endl;
        std::cerr << "  Input name stored: \"" << partial_input_name_str << "\"" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Standard exception in partial inference: " << e.what() << std::endl;
        return 0;
    } catch (...) {
        std::cerr << "Unknown exception in partial inference" << std::endl;
        return 0;
    }
}

bool NNUEEvaluator::isLoaded() const {
    return loaded;
}

int NNUEEvaluator::getEvalCount() const {
    return eval_count;
}

void NNUEEvaluator::resetEvalCount() {
    eval_count = 0;
}


