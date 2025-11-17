#ifndef NNUE_EVALUATOR_HPP
#define NNUE_EVALUATOR_HPP

#include <memory>
#include <string>
#include <vector>
#include <stack>
#include <onnxruntime_cxx_api.h>
#include "chess.hpp"

struct FeatureDiff {
    int index;
    float old_value;
    float new_value;
};

class NNUEEvaluator {
private:
    std::unique_ptr<Ort::Env> env;
    std::unique_ptr<Ort::Session> session;
    std::unique_ptr<Ort::Session> partial_session;
    std::string input_name_str;
    std::string output_name_str;
    std::string partial_input_name_str;
    std::string partial_output_name_str;
    std::vector<const char*> input_names;
    std::vector<const char*> output_names;
    std::vector<const char*> partial_input_names;
    std::vector<const char*> partial_output_names;
    int max_cp;
    bool loaded;
    int eval_count;
    
    float current_features[768];
    std::vector<FeatureDiff> pending_diffs;
    std::stack<std::vector<FeatureDiff>> history_stack;
    
    int h1_size;
    int h2_size;
    std::vector<float> layer1_weights;
    std::vector<float> layer1_bias;
    std::vector<float> layer1_accumulator;
    
    void boardToFeatures(const chess::Board& board, float* features);
    int getFeatureIndex(chess::PieceType type, chess::Color color, int square);
    bool loadLayer1Weights(const std::string& weights_path, const std::string& bias_path);
    
public:
    NNUEEvaluator();
    ~NNUEEvaluator() = default;
    
    bool loadModel(const std::string& model_path, const std::string& config_path);
    void reset(const chess::Board& board);
    void move(const chess::Move& move, const chess::Board& board_before);
    void undo();
    int evaluate();
    bool isLoaded() const;
    int getEvalCount() const;
    void resetEvalCount();
};

#endif


