#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include <variant> // C++ heade
#include <stdexcept>
#include <memory>
#include <vector>
#include <cmath>
// use std as namespace
using namespace std;



class Matrix {
public:
    size_t rows, cols;
    std::vector<std::vector<float>> data;
    // Constructor, destructor, copy constructor, and other utility methods
    Matrix(std::vector<std::vector<float>> data)
        : data(data), rows(data.size()), cols(data.empty() ? 0 : data[0].size()) {}

    Matrix(vector<float> data) : data(vector<vector<float>> {data}), rows(1), cols(data.size()) {
        // cout << "data" << endl;
        // cout << 
        // std::vector<std::vector<float>> data_ = this->transpose().data;
        // this->data = data_;
        // this->rows = data_.size();
        // this->cols = data_.empty() ? 0 : data_[0].size();
    }

    Matrix(const Matrix& other) : data(other.data), rows(other.rows), cols(other.cols) {}
    

    // Constructor that takes number of rows and columns
    Matrix(size_t r, size_t c)
        : rows(r), cols(c), data(r, std::vector<float>(c, 0)) {}

    // Destructor
    ~Matrix() {}

    // Methods for matrix operations (addition, multiplication, etc.)
    // Matrix dot(const Matrix& other) const;
    Matrix add(const Matrix& other) const{
        if (rows != other.rows || cols != other.cols) {
            cout << "rows" << rows << endl;
            cout << "other.rows" << other.rows << endl;
            cout << "cols" << cols << endl;
            cout << "other.cols" << other.cols << endl;
            throw std::invalid_argument("Matrix dimensions do not match for addition.");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] + other.data[i][j];
            }
        }
        return result;
    };

    void print() const {
        for (size_t i = 0; i < rows; ++i) {
            std::cout << "[";
            for (size_t j = 0; j < cols; ++j) {
                std::cout << data[i][j] << ", ";
            }
            std::cout << "]\n";
        }
    }
    Matrix subtract(const Matrix& other) const{
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for subtraction.");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                result.data[i][j] = data[i][j] - other.data[i][j];
            }
        }
        return result;
    };

    // Matrix transpose() const;
    Matrix dot_transpose(const Matrix& other) const{
        if (cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        // cout << "weight" << endl;
        // cout << other.rows << " " << other.cols << endl;


        Matrix result(rows, other.rows);
        for (size_t i = 0; i < rows; ++i) { // m
            for (size_t j = 0; j < other.rows; ++j) { // p
                for (size_t k = 0; k < cols; ++k) { // n
                    result.data[i][j] += data[i][k] * other.data[j][k];
                }
            }
        }
        // cout << "result" << endl;
        // cout << result.rows << " " << result.cols << endl;
        return result;
    };


    Matrix transpose() const {
    // Create a new matrix with flipped dimensions
        Matrix transposed(cols, rows);
        // transposed.print();
        // this->print();

        // Transpose the matrix
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                // Assign the transposed values
                transposed.data[j][i] = data[i][j];
            }
        }
        return transposed;
    }

    Matrix in_place_mul(const Matrix& other) const {
        if (rows != other.rows || cols != other.cols) {
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        Matrix result(rows, cols);
        for (size_t i = 0; i < rows; ++i) { // m
            for (size_t j = 0; j < cols; ++j) { // p
                result.data[i][j] = data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    Matrix dot(const Matrix& other) const {
        if (cols != other.rows) {
            cout << "rows" << rows << endl;
            cout << "cols" << cols << endl;
            cout << "other.rows" << other.rows << endl;
            cout << "other.cols" << other.cols << endl;
            throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
        }

        // self is m x n, other is n x p

        // take transpose of other matrix
        // cout << "dot"  << endl; 
        Matrix other_transpose = other.transpose();  // p x n

        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; ++i) { // m
            for (size_t j = 0; j < other.cols; ++j) { // p
                for (size_t k = 0; k < cols; ++k) { // n
                    result.data[i][j] += data[i][k] * other_transpose.data[j][k];
                }
            }
        }
        return result;
    }


};

Matrix softmax(const Matrix& x) {
    Matrix result = x;

    for (size_t i = 0; i < x.rows; ++i) {
        float rowSum = 0;
        // Exponentiate and sum
        for (size_t j = 0; j < x.cols; ++j) {
            result.data[i][j] = std::exp(x.data[i][j]);
            rowSum += result.data[i][j];
        }
        // Normalize
        for (size_t j = 0; j < x.cols; ++j) {
            result.data[i][j] /= rowSum + 1e-4f; // Epsilon added for numerical stability
        }
    }

    return result;
}

class LayerNorm {
public:
    LayerNorm(size_t featureSize) : weights(featureSize, 1), bias(featureSize, 1) {
        // Initialize weights and bias. Typically, weights are initialized to 1 and bias to 0.
        for (size_t i = 0; i < featureSize; ++i) {
            weights.data[i][0] = 1.0;
            bias.data[i][0] = 0.0;
        }
    }

    LayerNorm(const Matrix& weights, const Matrix& bias) : weights(weights.data), bias(bias.data) { }

    Matrix forward(const Matrix& x) const {
        if (x.cols != weights.cols || x.rows != weights.rows) {
            cout << "x.cols" << x.cols << endl;
            cout << "weights.rows" << weights.rows << endl;
            cout << "x.rows" << x.rows << endl;
            cout << "weights.cols" << weights.cols << endl;
            throw std::invalid_argument("Feature size mismatch in LayerNorm.");
        }

        Matrix normalized = normalize(x);
        Matrix scaled = normalized.in_place_mul(weights);
        return addBias(scaled, bias);
    }

    Matrix weights, bias;

    Matrix normalize(const Matrix& x) const {
        Matrix result(x.rows, x.cols);

        for (size_t i = 0; i < x.rows; ++i) {
            float mean = 0, variance = 0;
            x.print();

            // Calculate mean
            for (size_t j = 0; j < x.cols; ++j) {
                mean += x.data[i][j];
            }
            cout << "mean" << mean << " " << x.cols<< endl;
            mean /= x.cols;

            // Calculate variance
            for (size_t j = 0; j < x.cols; ++j) {
                variance += std::pow(x.data[i][j] - mean, 2);
            }
            variance /= x.cols;

            // Normalize
            for (size_t j = 0; j < x.cols; ++j) {
                result.data[i][j] = (x.data[i][j] - mean) / std::sqrt(variance + 1e-2f); // Epsilon added for numerical stability
            }
        }

        return result;
    }

    Matrix addBias(const Matrix& x, const Matrix& b) const {
        Matrix result = x;

        for (size_t i = 0; i < x.rows; ++i) {
            for (size_t j = 0; j < x.cols; ++j) {
                result.data[i][j] += b.data[0][j];
            }
        }

        return result;
    }
};

// class Linear {
// public:
//     Linear(size_t inputSize, size_t outputSize) : weights(inputSize, outputSize), bias(outputSize, 1) {
//         // Initialize weights and bias. Here we just use placeholders for initialization.
//         // In practice, you should initialize them properly (e.g., using a random initializer).
//     }

//     Matrix forward(const Matrix& x) const {
//         if (x.cols != weights.rows) {
//             throw std::invalid_argument("Input size mismatch in Linear layer.");
//         }

//         Matrix result = x.dot(weights);
//         return addBias(result, bias);
//     }

//     Matrix weights, bias;

//     Matrix addBias(const Matrix& x, const Matrix& b) const {
//         Matrix result = x;

//         for (size_t i = 0; i < x.rows; ++i) {
//             for (size_t j = 0; j < x.cols; ++j) {
//                 result.data[i][j] += b.data[j][0];
//             }
//         }

//         return result;
//     }
// };

class FeedForward {
public:
    Matrix weight1, weight2, bias1, bias2;
    LayerNorm ln1, ln2;
    FeedForward(size_t inputSize, size_t hiddenSize, size_t outputSize) :
        weight1(inputSize, hiddenSize), weight2(hiddenSize, outputSize),
        bias1(hiddenSize, 1), bias2(outputSize, 1),
        ln1(hiddenSize), ln2(outputSize) {
        // Initialize weights and biases. Here we just use placeholders for initialization.
        // In practice, you should initialize them properly (e.g., using a random initializer).
    }

    FeedForward(const Matrix& weight1, const Matrix& bias1, const Matrix& weight2, const Matrix& bias2,
                const LayerNorm& ln1, const LayerNorm& ln2) :
        weight1(weight1.data), weight2(weight2.data), bias1(bias1.data), bias2(bias2.data),
        ln1(ln1.weights, ln1.bias), ln2(ln2.weights,ln2.bias ) { }

    Matrix forward(const Matrix& x) const {
        if (x.cols != weight1.cols) {
            cout << "x.cols" << x.cols << endl;
            cout << "weight1.rows" << weight1.rows << endl;
            cout << "x.rows" << x.rows << endl;
            cout << "weight1.cols" << weight1.cols << endl;
            throw std::invalid_argument("Input size mismatch in FeedForward.");
        }
        // Apply first layer transformation
        // cout << x.rows << " " << x.cols << endl;
        Matrix hidden = x.dot_transpose(weight1);
        // cout << hidden.rows << " " << hidden.cols << endl;
        hidden = addBias(hidden, bias1);
        // cout << hidden.rows << " " << hidden.cols << endl;
        hidden = ln1.forward(hidden);
        // cout << hidden.rows << " " << hidden.cols << endl;
        hidden = relu(hidden);
        // cout << hidden.rows << " " << hidden.cols << endl;


        // Apply second layer transformation

        Matrix output = hidden.dot_transpose(weight2);
        // cout << output.rows << " " << output.cols << endl;
        output = addBias(output, bias2);
        // cout << output.rows << " " << output.cols << endl;
        output = ln2.forward(output);
        // cout << output.rows << " " << output.cols << endl;


        return output;
    }


    Matrix addBias(const Matrix& x, const Matrix& b) const {
        Matrix result = x;
        for (size_t i = 0; i < x.rows; ++i) {
            for (size_t j = 0; j < x.cols; ++j) {
                result.data[i][j] += b.data[0][j];
            }
        }
        return result;
    }

    Matrix relu(const Matrix& x) const {
        Matrix result = x;
        for (size_t i = 0; i < x.rows; ++i) {
            for (size_t j = 0; j < x.cols; ++j) {
                result.data[i][j] = std::max(0.0f, x.data[i][j]);
            }
        }
        return result;
    }
};

class SelfAttention {
public:
    Matrix queryWeights, keyWeights, valueWeights, queryBias, keyBias, valueBias;
    // size_t numHeads;
    SelfAttention(size_t modelSize) :
        queryWeights(modelSize, modelSize), 
        queryBias(modelSize, 1),
        keyWeights(modelSize, modelSize),
        keyBias(modelSize, 1),
        valueWeights(modelSize, modelSize),
        valueBias(modelSize, 1)
         {
        // Initialize weights. Here we just use placeholders for initialization.
        // In practice, these should be initialized properly.
    }

    SelfAttention(const Matrix& queryWeights, const Matrix& queryBias ,const Matrix& keyWeights, const Matrix& keyBias ,const Matrix& valueWeights, const Matrix& valueBias) :
        queryWeights(queryWeights), queryBias(queryBias) ,keyWeights(keyWeights), keyBias(keyBias) ,valueWeights(valueWeights), valueBias(valueBias) { }

    Matrix forward(const Matrix& x) const {
        if (x.cols != queryWeights.rows) {
            throw std::invalid_argument("Input size mismatch in SelfAttention.");
        }
        // Matrix x = i.transpose();
        // Calculate query, key, value matrices
        Matrix queries = x.dot(queryWeights);
        queries = queries.add(queryBias);
        Matrix keys = x.dot(keyWeights);
        keys = keys.add(keyBias);
        Matrix values = x.dot(valueWeights);
        values = values.add(valueBias);

        // Split into heads and perform scaled dot-product attention for each head
        // size_t headSize = queries.cols / numHeads;
        // Matrix attentionResults(x.rows, x.cols);
        // for (size_t head = 0; head < numHeads; ++head) {
        //     Matrix headQueries = splitHeads(queries, headSize);
        //     Matrix headKeys = splitHeads(keys, headSize);
        //     Matrix headValues = splitHeads(values, headSize);

        // Matrix attentionResults(scaledDotProductAttention(queries, keys, values));
        // }

        // Final linear transformation can be added here if needed

        return scaledDotProductAttention(queries, keys, values);
    }


    // Matrix splitHeads(const Matrix& x, size_t headSize) const {
    //     // Implement the logic to split the matrix x into multiple heads
    //     // This typically involves reshaping the matrix
    // }

Matrix scaledDotProductAttention(const Matrix& queries, const Matrix& keys, const Matrix& values) const {
    cout << "cols" << keys.cols << endl;
    float scalingFactor = std::sqrt(static_cast<float>(keys.cols));

    // Step 1: Dot product between queries and keys
    queries.print();
    keys.transpose().print();
    // cout << "queries" << endl;
    Matrix dotProducts = queries.transpose().dot(keys);

    dotProducts.print();

    // Step 2: Scale the dot products
    for (size_t i = 0; i < dotProducts.rows; ++i) {
        for (size_t j = 0; j < dotProducts.cols; ++j) {
            dotProducts.data[i][j] /= scalingFactor;
            cout << "-" << " ";
        }
    }

    // Step 3: Apply softmax to dotProducts
    Matrix softmaxResults = softmax(dotProducts);
    softmaxResults.print();

    cout << "softmaxResults" << endl;
    values.transpose().print();

    // Step 4: Dot product with values
    return softmaxResults.dot(values.transpose()).transpose();
}

};



class TransformerBlock {
public:
    SelfAttention attention;
    LayerNorm norm1, norm2;
    FeedForward feedForward;
    TransformerBlock(size_t modelSize, size_t numHeads, size_t feedForwardSize)
        : attention(modelSize),
          norm1(modelSize), norm2(modelSize),
          feedForward(modelSize, feedForwardSize, modelSize) {}

    TransformerBlock(const SelfAttention& attention, const LayerNorm& norm1, const LayerNorm& norm2, const FeedForward& feedForward)
        : attention(attention.queryWeights,attention.queryBias ,attention.keyWeights, attention.keyBias, attention.valueWeights, attention.valueBias),
          norm1(norm1.weights, norm1.bias), norm2(norm2.weights, norm2.bias),
          feedForward(feedForward.weight1, feedForward.bias1, feedForward.weight2,  feedForward.bias2, feedForward.ln1, feedForward.ln2) { }
    

    Matrix forward(const Matrix& x) const {
        if (x.cols != attention.queryWeights.rows) {
            throw std::invalid_argument("Input size mismatch in TransformerBlock.");
        }
        // Step 1: Self-Attention
        Matrix attentionOutput = attention.forward(x);
        cout << "attentionOutput" << endl;
        attentionOutput.print();
        // Step 2: Add & Norm (Residual connection + LayerNorm)
        Matrix addNorm1Output = norm1.forward(x.add( attentionOutput));

        // Step 3: Feed-Forward
        Matrix feedForwardOutput = feedForward.forward(addNorm1Output);

        // Step 4: Add & Norm (Another residual connection + LayerNorm)
        Matrix addNorm2Output = norm2.forward(addNorm1Output.add( feedForwardOutput));

        return addNorm2Output;
    }

private:

    Matrix add(const Matrix& x1, const Matrix& x2) const {
        // Assuming a simple element-wise addition
        Matrix result = x1;
        for (size_t i = 0; i < x1.rows; ++i) {
            for (size_t j = 0; j < x1.cols; ++j) {
                result.data[i][j] += x2.data[i][j];
            }
        }
        return result;
    }
};



class Transformer {
public:
    std::vector<TransformerBlock> blocks;
    Transformer(size_t numLayers, size_t modelSize, size_t numHeads, size_t feedForwardSize) {
        // Initialize the specified number of Transformer blocks
        for (size_t i = 0; i < numLayers; ++i) {
            blocks.emplace_back(modelSize, numHeads, feedForwardSize);
        }
    }

    Transformer(const std::vector<TransformerBlock>& blocks) : blocks(blocks) { }

    Matrix forward(const Matrix& x) const {
        if (blocks.empty()) {
            throw std::invalid_argument("No Transformer blocks found.");
        }
        if (x.cols != blocks[0].attention.queryWeights.rows) {
            cout << "x.cols" << x.cols << endl;
            cout << "blocks[0].attention.queryWeights.rows" << blocks[0].attention.queryWeights.rows << endl;
            throw std::invalid_argument("Input size mismatch in Transformer.");
        }
        Matrix output = x;

        // Pass the input through each Transformer block in sequence
        for (const auto& block : blocks) {
            output = block.forward(output);
        }

        return output;
    }

};



float tanh_tr(float x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}
class ChessBot {
public:
    Transformer transformer;
    FeedForward inputff;
    FeedForward valueHead;
    FeedForward policyHead;
    size_t inputSize;

    // ChessBot(size_t numLayers, size_t modelSize, size_t numHeads, size_t feedForwardSize, size_t inputSize, size_t numMoves) {
    //     transformer = Transformer(numLayers, modelSize, numHeads, feedForwardSize);
    //     inputff = FeedForward(inputSize, feedForwardSize, modelSize);
    //     valueHead = FeedForward(modelSize, 1, 1);
    //     policyHead = FeedForward(modelSize, numMoves, 1);
    //     inputSize = inputSize;
    // }

    ChessBot(const Transformer& transformer, const FeedForward& inputff, const FeedForward& valueHead, const FeedForward& policyHead, size_t inputSize) :
        transformer(transformer.blocks), inputff(inputff.weight1, inputff.bias1, inputff.weight2,  inputff.bias2, inputff.ln1, inputff.ln2),
        valueHead(valueHead.weight1, valueHead.bias1, valueHead.weight2,  valueHead.bias2, valueHead.ln1, valueHead.ln2),
        policyHead(policyHead.weight1,  policyHead.bias1, policyHead.weight2, policyHead.bias2, policyHead.ln1, policyHead.ln2),
        inputSize(inputSize) { }


    vector<float> forward(Matrix x, bool returnPolicy) {
        // x is the multi-channel grid input of shape [batch, channels, height, width]

        // Flatten the board representation to fit the transformer input
        // x = x.view(x.size(0), -1)  # Flatten to [batch, channels * height * width]
        
        x = inputff.forward(x);

        // Apply Transformer layers
        x = transformer.forward(x);

        if (returnPolicy) {
            vector<float> policy = policyHead.forward(x).data[0];
            policy = softmax(Matrix(policy)).transpose().data[0];
            return policy;
        } else {
            float value = tanh_tr(valueHead.forward(x).data[0][0]);
            return vector<float> {value};
        }

    
    }

};


Matrix  transformer_layers_0_attention_query_weight   (
{{-0.1, -0.1, 0.2, 0.0, -0.0, -0.2, -0.0, 0.2, 0.2, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.1, 0.1, -0.1, -0.1, 0.0, 0.0, -0.1, -0.2, 0.0, -0.1},
{0.2, 0.2, 0.1, -0.1, 0.1, -0.2, -0.0, -0.1, -0.2, -0.0, 0.2, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.0, -0.0, -0.1, 0.1},
{-0.2, 0.0, -0.0, -0.1, 0.0, 0.1, -0.2, -0.1, -0.1, 0.0, 0.1, -0.1, -0.1, 0.1, 0.1, 0.0, -0.2, -0.1, 0.0, -0.1, -0.1, 0.1, 0.0, -0.0, 0.1},
{-0.2, -0.2, 0.0, 0.1, 0.2, -0.1, 0.0, -0.2, 0.0, -0.0, -0.1, -0.1, 0.1, -0.1, -0.2, -0.0, -0.2, -0.0, -0.1, -0.0, 0.0, -0.0, -0.2, -0.1, 0.2},
{0.1, 0.2, 0.1, -0.1, -0.0, 0.2, -0.2, 0.1, -0.1, -0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, 0.1, 0.1, -0.1, -0.2, 0.1, 0.0, 0.2, -0.0, 0.2},
{-0.2, 0.1, 0.1, -0.1, 0.1, -0.0, -0.1, -0.1, 0.2, -0.0, 0.1, -0.2, -0.2, -0.0, -0.1, 0.1, 0.1, -0.1, -0.2, -0.0, 0.1, -0.1, 0.2, -0.1, -0.0},
{-0.1, 0.1, 0.0, 0.0, -0.2, -0.2, -0.0, 0.1, 0.1, -0.1, 0.0, 0.1, -0.0, -0.2, -0.1, -0.2, 0.1, 0.2, 0.1, -0.2, -0.0, 0.0, 0.1, 0.1, 0.2},
{0.2, 0.1, -0.0, 0.1, 0.2, -0.1, 0.1, -0.1, 0.1, 0.0, -0.1, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.2, -0.2, -0.2, -0.2, -0.1, 0.2, 0.0},
{0.1, 0.0, -0.0, 0.1, 0.1, -0.2, -0.1, 0.1, 0.0, 0.1, 0.1, -0.1, -0.2, 0.1, -0.0, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.1, -0.2},
{0.1, -0.1, -0.1, -0.1, -0.0, -0.2, -0.1, -0.1, -0.2, -0.1, -0.0, -0.1, -0.0, 0.1, 0.0, 0.1, -0.0, -0.0, -0.1, 0.0, 0.1, 0.2, -0.1, 0.0, -0.1},
{-0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.2, -0.1, -0.1, 0.0, -0.1, 0.1, -0.1, 0.1, -0.2, -0.1, -0.2, 0.2, 0.2, -0.0, 0.1, 0.0, -0.2, -0.1, -0.0},
{-0.2, -0.0, 0.2, 0.1, -0.2, 0.0, -0.2, -0.2, -0.0, 0.2, -0.1, -0.2, -0.0, -0.2, 0.1, -0.0, 0.2, -0.2, -0.1, -0.1, -0.1, -0.1, 0.2, 0.1, -0.1},
{-0.1, 0.1, 0.0, 0.0, 0.0, -0.1, 0.1, 0.1, 0.2, -0.1, -0.2, 0.2, -0.1, -0.0, 0.1, 0.2, -0.1, 0.2, 0.0, 0.1, 0.1, -0.1, -0.0, -0.1, 0.0},
{-0.1, -0.0, 0.1, 0.2, 0.1, 0.0, 0.1, -0.0, 0.2, -0.2, -0.1, -0.0, -0.1, -0.2, 0.2, 0.0, 0.2, 0.2, -0.1, -0.1, -0.1, 0.0, -0.1, -0.0, -0.1},
{0.1, -0.2, -0.2, 0.1, -0.0, -0.1, -0.2, 0.2, -0.0, -0.2, -0.1, 0.1, -0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, -0.1, 0.1, -0.1, -0.1, -0.2, -0.1},
{-0.1, 0.1, -0.1, -0.1, -0.2, 0.0, -0.0, 0.1, 0.0, 0.1, 0.2, -0.2, 0.0, -0.2, 0.1, -0.1, 0.1, 0.1, -0.0, 0.2, 0.1, -0.1, 0.2, -0.1, 0.1},
{0.1, -0.1, 0.0, 0.2, -0.1, 0.1, -0.2, 0.1, 0.2, -0.1, -0.1, 0.1, 0.2, 0.1, -0.1, -0.2, 0.1, -0.1, 0.0, 0.0, -0.0, -0.1, 0.1, 0.0, 0.1},
{-0.1, 0.0, -0.2, -0.2, 0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.2, -0.1, -0.2, -0.0, -0.1, 0.1, 0.1, 0.1, 0.0, -0.2, 0.1, -0.1, -0.1, -0.2, -0.0},
{-0.0, 0.1, -0.0, -0.2, -0.1, 0.1, 0.1, 0.0, 0.0, -0.2, -0.2, 0.0, -0.0, -0.2, -0.1, -0.1, -0.1, 0.2, 0.1, 0.1, -0.0, 0.1, 0.0, -0.1, -0.0},
{-0.1, 0.0, -0.1, 0.2, 0.1, -0.2, -0.2, -0.2, -0.0, 0.1, 0.1, 0.1, -0.0, 0.1, -0.2, -0.2, 0.2, 0.2, -0.2, -0.0, 0.2, 0.2, 0.1, -0.1, 0.2},
{0.2, -0.1, 0.1, 0.1, 0.1, -0.1, -0.2, 0.0, 0.1, -0.0, -0.2, -0.1, 0.2, -0.0, 0.1, 0.1, -0.2, 0.2, 0.1, -0.1, 0.0, 0.1, -0.1, -0.1, 0.1},
{-0.1, 0.0, 0.2, -0.1, 0.2, -0.2, -0.2, 0.2, -0.0, 0.2, -0.1, 0.0, -0.1, -0.1, 0.2, -0.2, -0.1, 0.1, 0.1, 0.0, -0.2, -0.1, -0.2, 0.1, -0.1},
{0.2, -0.2, 0.1, 0.0, -0.1, -0.0, 0.1, -0.0, -0.1, -0.1, 0.2, 0.0, -0.1, 0.1, 0.1, -0.1, 0.1, 0.0, -0.1, 0.0, -0.2, 0.0, -0.2, -0.0, 0.1},
{0.0, 0.0, 0.1, 0.2, 0.1, 0.1, 0.1, -0.1, 0.1, -0.0, -0.2, 0.1, 0.2, 0.1, 0.0, -0.0, 0.1, -0.2, -0.1, -0.1, 0.1, 0.1, 0.0, 0.0, -0.1},
{-0.0, 0.1, -0.2, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0, -0.1, -0.1, -0.0, -0.0, -0.0, -0.1, 0.1, -0.1, 0.0, 0.1, 0.2, 0.2, -0.2, 0.0, 0.2, -0.0},
}
);
Matrix  transformer_layers_0_attention_query_bias   (
{0.1, 0.1, 0.2, -0.1, 0.1, 0.1, -0.0, -0.1, -0.2, 0.1, 0.0, -0.1, 0.0, -0.0, 0.2, 0.1, -0.2, 0.0, -0.2, 0.2, 0.0, -0.0, -0.2, -0.0, 0.2}
);
Matrix  transformer_layers_0_attention_key_weight   (
{{0.2, 0.1, 0.1, 0.1, 0.1, 0.0, -0.1, -0.0, 0.2, 0.1, 0.2, -0.0, 0.2, -0.1, 0.1, -0.2, -0.2, 0.1, -0.0, -0.0, 0.2, -0.1, -0.0, -0.1, 0.1},
{-0.0, -0.1, -0.2, 0.1, 0.1, -0.0, -0.1, -0.0, 0.2, 0.2, 0.1, 0.1, 0.2, 0.1, -0.1, -0.1, 0.2, -0.1, -0.1, 0.1, 0.0, 0.2, -0.1, 0.1, -0.1},
{0.1, 0.1, 0.0, -0.2, -0.1, -0.1, 0.1, 0.0, 0.2, -0.1, -0.2, -0.1, -0.0, -0.0, 0.2, 0.1, 0.2, -0.2, -0.0, 0.1, 0.1, -0.1, -0.1, -0.1, 0.2},
{0.1, 0.1, 0.1, 0.1, 0.2, -0.1, -0.1, -0.1, 0.1, -0.2, -0.1, -0.1, 0.0, -0.2, -0.0, 0.1, -0.1, -0.1, 0.0, 0.1, 0.1, -0.2, -0.1, 0.0, -0.2},
{-0.1, -0.1, 0.0, -0.1, -0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, -0.1, -0.1, 0.2, 0.1, 0.1, 0.1, -0.2, -0.2, 0.0, 0.1, 0.1, -0.1, -0.1, -0.1},
{-0.1, -0.0, 0.2, -0.1, 0.1, 0.2, -0.0, -0.1, -0.2, 0.0, 0.1, -0.1, 0.2, 0.1, -0.0, 0.0, 0.0, -0.0, 0.2, -0.1, 0.1, 0.2, 0.1, -0.1, 0.0},
{-0.1, 0.2, 0.1, 0.1, -0.1, -0.1, 0.1, 0.0, -0.0, 0.1, 0.0, 0.1, -0.2, -0.0, -0.1, 0.1, -0.1, 0.0, -0.2, 0.1, -0.1, 0.2, -0.0, 0.1, -0.2},
{0.2, 0.0, 0.1, 0.0, -0.2, -0.1, 0.1, -0.0, 0.0, -0.2, 0.2, 0.2, 0.1, 0.1, -0.2, 0.0, 0.0, 0.2, -0.2, 0.1, 0.1, 0.1, -0.2, 0.1, -0.1},
{0.0, -0.1, -0.1, -0.1, 0.0, 0.2, 0.1, 0.1, 0.0, 0.2, -0.2, 0.1, -0.0, 0.1, 0.0, 0.1, -0.0, 0.0, 0.1, 0.1, -0.2, 0.1, -0.1, 0.0, 0.1},
{0.1, 0.0, 0.2, -0.2, 0.1, 0.0, -0.0, -0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.2, -0.1, -0.2, -0.1, -0.1, 0.2, -0.1, 0.0, 0.2, 0.0, -0.0},
{-0.1, 0.1, -0.2, -0.2, 0.1, 0.0, 0.2, -0.2, 0.1, -0.1, -0.2, 0.1, 0.2, 0.0, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.2, 0.0, 0.1, 0.1, -0.0},
{0.2, 0.1, 0.0, 0.1, -0.1, -0.0, 0.1, -0.2, -0.1, -0.1, 0.1, -0.0, -0.2, -0.1, -0.2, 0.1, 0.1, -0.0, -0.2, 0.1, 0.1, -0.1, 0.2, -0.2, -0.1},
{0.1, -0.1, -0.0, -0.0, -0.0, -0.0, -0.1, -0.0, -0.2, -0.0, -0.2, 0.1, 0.2, -0.1, 0.2, 0.2, -0.0, 0.1, -0.0, 0.0, -0.1, -0.2, 0.0, 0.0, 0.2},
{-0.2, 0.1, 0.1, -0.2, 0.2, -0.0, 0.1, 0.1, 0.0, 0.2, 0.1, 0.1, -0.1, -0.2, -0.1, 0.2, -0.0, 0.2, 0.1, -0.1, -0.0, -0.1, 0.1, 0.1, -0.1},
{-0.2, 0.1, 0.0, 0.0, -0.1, 0.2, -0.2, -0.2, 0.1, -0.2, 0.2, -0.1, 0.0, 0.2, 0.1, -0.1, -0.2, 0.0, -0.0, -0.2, -0.0, -0.1, 0.1, -0.1, 0.1},
{0.1, 0.2, -0.2, 0.2, -0.1, -0.0, -0.1, -0.1, 0.2, -0.1, -0.2, -0.2, 0.1, 0.0, 0.0, -0.2, 0.0, -0.2, -0.1, 0.1, -0.1, -0.2, 0.1, -0.0, 0.2},
{-0.1, 0.2, 0.1, -0.1, 0.1, -0.0, 0.0, -0.1, 0.1, -0.2, -0.0, -0.1, -0.0, -0.0, 0.1, 0.2, -0.1, -0.0, 0.2, -0.1, 0.0, 0.1, 0.1, -0.1, -0.2},
{0.2, 0.2, -0.1, 0.1, 0.1, 0.1, -0.1, -0.1, 0.2, 0.1, -0.2, 0.0, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0, 0.1, -0.0, 0.1, -0.2, -0.1, 0.1, -0.2},
{0.1, -0.2, -0.0, -0.1, 0.1, -0.1, 0.2, -0.1, 0.1, -0.0, 0.2, 0.2, 0.0, 0.1, -0.2, 0.1, 0.2, -0.1, -0.2, -0.1, -0.2, 0.0, -0.2, -0.1, -0.1},
{-0.1, 0.0, 0.1, -0.2, -0.2, -0.2, -0.1, 0.0, 0.0, 0.2, 0.1, 0.2, 0.0, 0.1, 0.1, 0.1, -0.2, -0.1, -0.1, 0.1, -0.1, 0.2, 0.0, 0.2, 0.1},
{-0.2, -0.0, -0.1, -0.0, 0.1, -0.2, 0.2, 0.0, -0.2, 0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.1, -0.2, 0.1, 0.0, 0.1, -0.1, 0.2, 0.1, 0.1, -0.2},
{0.1, 0.2, -0.0, -0.1, 0.1, -0.1, -0.2, -0.2, 0.0, 0.2, 0.0, 0.2, -0.1, 0.2, -0.0, 0.1, 0.1, -0.1, -0.0, 0.0, -0.1, 0.1, 0.1, -0.1, -0.1},
{0.1, -0.1, 0.0, -0.1, -0.1, 0.2, -0.2, -0.2, 0.2, 0.1, 0.1, -0.2, 0.1, -0.1, 0.1, 0.1, -0.0, -0.2, 0.1, -0.1, 0.2, -0.0, -0.1, -0.0, -0.1},
{-0.1, -0.1, 0.1, -0.0, -0.1, -0.1, 0.1, -0.0, -0.1, 0.1, 0.2, 0.2, 0.2, -0.1, 0.1, -0.1, -0.2, -0.1, -0.0, -0.1, 0.1, -0.1, 0.2, -0.2, -0.0},
{-0.1, -0.1, -0.1, -0.2, 0.1, -0.1, 0.1, -0.2, 0.1, -0.1, 0.2, 0.1, -0.2, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.0, -0.0, -0.1, -0.2, -0.0, -0.1},
}
);
Matrix  transformer_layers_0_attention_key_bias   (
{-0.1, 0.2, 0.1, 0.0, -0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.2, -0.0, 0.1, 0.1, -0.0, 0.1, 0.1, 0.2, -0.1, -0.1, 0.0, -0.1, 0.0, -0.1, -0.0}
);
Matrix  transformer_layers_0_attention_value_weight   (
{{0.1, -0.0, -0.1, -0.1, -0.0, -0.1, -0.2, 0.0, 0.2, 0.0, 0.2, -0.0, 0.2, -0.1, 0.1, 0.1, -0.1, -0.1, -0.2, -0.1, -0.1, -0.0, 0.2, -0.0, 0.1},
{-0.2, -0.1, 0.2, -0.0, -0.1, -0.1, -0.2, -0.1, -0.2, 0.0, -0.2, 0.1, 0.1, -0.1, -0.1, -0.0, -0.1, -0.1, 0.1, 0.0, -0.2, 0.1, 0.1, -0.2, 0.1},
{-0.0, -0.2, 0.0, -0.0, -0.0, -0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, -0.1, 0.0, 0.2, -0.1, 0.1, -0.1, 0.2, 0.0, -0.1, 0.1, 0.2, 0.1, -0.1},
{0.1, -0.1, -0.2, 0.1, -0.1, -0.1, 0.2, -0.1, 0.1, -0.0, -0.2, -0.1, -0.1, 0.0, 0.1, -0.0, 0.1, 0.1, 0.0, -0.0, 0.1, 0.1, 0.1, -0.1, 0.2},
{0.2, 0.1, 0.1, 0.2, -0.1, 0.0, -0.2, 0.2, -0.0, 0.1, 0.0, -0.1, 0.1, -0.1, -0.0, -0.2, 0.2, 0.2, -0.0, -0.0, 0.0, -0.1, 0.0, -0.2, -0.0},
{0.2, -0.2, -0.1, -0.2, -0.2, 0.2, 0.0, -0.2, 0.1, -0.1, -0.1, -0.2, 0.1, -0.2, 0.1, -0.2, -0.0, 0.2, -0.2, 0.0, 0.2, 0.1, 0.1, -0.2, -0.1},
{0.2, -0.1, -0.2, 0.1, -0.0, 0.1, 0.1, 0.1, -0.2, -0.0, -0.1, 0.1, 0.2, 0.2, -0.1, -0.1, -0.2, -0.0, -0.0, -0.2, -0.0, -0.1, -0.1, -0.1, 0.1},
{-0.2, 0.1, 0.0, -0.2, 0.1, 0.1, 0.1, 0.2, -0.2, 0.1, -0.0, 0.1, 0.2, 0.1, 0.2, -0.1, -0.0, 0.1, -0.0, 0.0, 0.0, 0.1, 0.0, -0.1, -0.2},
{0.0, 0.0, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.2, -0.1, 0.0, 0.2, -0.1, -0.0, 0.0, -0.1, -0.1},
{0.1, -0.1, -0.2, -0.0, 0.1, -0.0, -0.0, 0.0, -0.1, 0.1, 0.2, -0.2, 0.1, 0.1, 0.0, 0.0, 0.0, -0.0, -0.0, 0.1, -0.2, 0.2, -0.1, -0.1, 0.1},
{0.1, -0.2, -0.1, -0.2, -0.0, -0.2, 0.1, 0.2, 0.1, -0.2, 0.0, -0.0, -0.1, -0.0, 0.0, 0.0, -0.0, -0.2, -0.1, 0.0, -0.1, -0.1, 0.1, 0.2, 0.2},
{-0.0, -0.1, 0.1, 0.1, 0.1, -0.0, -0.1, -0.1, -0.0, 0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, -0.1, 0.2, 0.1, -0.0, -0.1, 0.2, -0.0, 0.0},
{-0.1, 0.1, -0.1, -0.1, 0.2, 0.0, -0.2, -0.2, 0.0, 0.2, 0.1, 0.2, -0.0, 0.1, 0.1, 0.2, -0.2, -0.1, 0.0, 0.1, -0.0, 0.1, -0.1, 0.1, 0.1},
{-0.1, -0.1, -0.1, 0.2, -0.2, 0.1, -0.1, -0.2, -0.0, 0.0, -0.2, -0.1, -0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1, -0.1, 0.1, -0.2, -0.0, -0.2},
{0.1, 0.1, -0.0, 0.1, -0.0, 0.1, 0.2, -0.1, -0.1, 0.1, 0.1, -0.2, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.0, 0.2, 0.0, 0.1},
{-0.0, 0.1, 0.1, 0.0, 0.1, 0.1, -0.0, -0.1, 0.0, 0.2, 0.0, -0.2, 0.0, 0.2, 0.1, -0.2, 0.2, 0.0, 0.0, 0.0, -0.2, -0.0, -0.1, 0.0, -0.1},
{0.1, -0.0, -0.1, 0.1, -0.2, 0.0, 0.1, -0.2, 0.2, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.2, -0.1, 0.0, 0.0, 0.0, -0.2, 0.2, -0.1, 0.0},
{0.0, -0.1, -0.1, 0.1, -0.0, -0.2, -0.1, 0.2, 0.0, 0.1, 0.1, -0.0, 0.1, -0.0, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1, -0.1, -0.1, 0.1, 0.0, -0.2},
{-0.0, 0.2, 0.1, -0.1, 0.2, 0.2, -0.1, 0.0, 0.1, 0.0, -0.1, -0.2, -0.0, 0.0, 0.1, -0.2, -0.2, -0.0, 0.2, 0.2, 0.0, -0.1, -0.1, 0.1, -0.0},
{-0.1, -0.0, 0.1, -0.1, -0.1, 0.2, 0.1, -0.1, -0.1, -0.1, 0.2, 0.1, -0.1, -0.1, 0.1, -0.2, -0.2, -0.1, 0.1, -0.1, -0.1, 0.2, -0.2, -0.1, -0.1},
{0.1, -0.2, -0.2, -0.0, -0.1, 0.0, 0.2, 0.2, -0.0, -0.2, 0.1, -0.2, -0.1, 0.1, -0.1, 0.2, -0.1, 0.1, -0.1, 0.0, -0.1, -0.1, 0.1, -0.0, 0.0},
{-0.0, -0.0, 0.1, -0.1, 0.0, 0.0, -0.1, -0.1, -0.1, 0.2, 0.1, -0.1, -0.2, 0.2, -0.2, 0.0, 0.1, -0.1, -0.1, 0.1, -0.0, -0.1, 0.2, 0.2, -0.0},
{-0.0, 0.1, -0.1, -0.0, 0.0, 0.1, 0.0, -0.1, 0.0, -0.1, 0.0, -0.2, -0.0, 0.2, -0.1, 0.0, 0.0, 0.1, 0.1, -0.0, 0.2, -0.2, -0.2, 0.1, 0.0},
{0.2, -0.2, -0.1, -0.0, -0.1, 0.0, 0.1, -0.1, -0.1, -0.2, -0.2, -0.1, -0.1, -0.2, 0.2, -0.2, 0.1, -0.1, 0.2, -0.0, -0.1, -0.1, -0.2, 0.2, -0.2},
{-0.1, 0.2, -0.1, -0.1, 0.0, 0.1, 0.1, -0.1, -0.0, 0.1, -0.0, 0.2, -0.1, 0.0, -0.1, 0.2, -0.2, -0.2, 0.2, 0.0, 0.1, -0.1, 0.1, -0.1, -0.2},
}
);
Matrix  transformer_layers_0_attention_value_bias   (
{0.1, 0.1, -0.1, 0.1, 0.0, 0.2, -0.1, 0.0, -0.1, 0.0, -0.1, 0.1, -0.1, -0.2, -0.1, 0.1, 0.1, 0.1, -0.0, 0.1, 0.2, 0.2, 0.1, -0.1, 0.0}
);
Matrix  transformer_layers_0_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_0_norm1_layer_norm_bias   (
{-0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0}
);
Matrix  transformer_layers_0_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_0_norm2_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_0_feed_forward_linear1_weight   (
{{0.1, -0.2, -0.1, 0.2, -0.2, 0.0, 0.1, -0.0, 0.1, 0.2, -0.2, 0.1, -0.2, -0.2, 0.1, 0.0, 0.1, 0.1, -0.0, -0.1, 0.1, 0.2, -0.1, 0.0, 0.0},
{0.1, -0.0, -0.1, -0.2, -0.1, -0.0, 0.0, -0.2, 0.1, 0.1, 0.2, 0.1, -0.1, -0.1, 0.2, 0.1, -0.0, 0.1, 0.1, 0.0, -0.1, -0.1, -0.0, -0.1, 0.1},
{0.0, -0.2, -0.0, 0.1, -0.0, -0.0, 0.0, 0.0, 0.0, -0.1, -0.1, -0.0, 0.0, -0.1, 0.2, -0.1, 0.1, -0.0, 0.0, -0.1, 0.1, 0.2, 0.0, -0.2, 0.1},
{0.2, 0.1, -0.1, 0.0, 0.2, 0.2, 0.1, -0.1, 0.1, -0.0, 0.2, 0.0, -0.1, -0.1, 0.2, 0.2, -0.1, 0.1, -0.0, 0.0, -0.1, -0.0, 0.2, 0.2, 0.2},
{0.1, 0.2, 0.0, -0.1, 0.1, -0.2, -0.1, -0.2, -0.2, 0.1, -0.1, 0.0, -0.0, 0.1, -0.1, 0.0, 0.2, 0.1, -0.2, -0.1, 0.2, 0.1, 0.1, 0.0, -0.2},
{-0.2, -0.2, -0.2, -0.2, -0.2, 0.1, 0.0, -0.2, 0.1, 0.2, 0.1, -0.0, -0.1, 0.1, -0.0, 0.2, -0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.2, 0.0},
{-0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.0, -0.0, -0.0, 0.2, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.1, -0.1, -0.1, 0.2, -0.2},
{0.1, 0.1, -0.2, 0.0, 0.1, -0.2, -0.0, 0.1, -0.1, 0.1, -0.1, -0.0, -0.2, 0.2, 0.2, -0.0, -0.1, 0.2, 0.1, 0.1, 0.2, 0.1, -0.0, 0.0, 0.1},
{0.2, -0.1, -0.0, 0.2, -0.0, -0.1, 0.1, 0.1, 0.0, -0.1, -0.0, -0.1, 0.0, -0.1, 0.1, 0.2, -0.1, -0.1, -0.1, -0.2, -0.1, -0.1, 0.0, 0.1, -0.2},
{-0.0, 0.2, -0.0, 0.1, 0.1, -0.0, 0.1, -0.1, -0.0, 0.1, 0.1, 0.1, -0.1, -0.1, -0.2, -0.1, -0.2, -0.1, -0.1, 0.0, -0.2, -0.0, -0.1, 0.1, -0.0},
{0.2, -0.1, 0.2, -0.1, -0.1, -0.1, -0.0, -0.2, 0.1, 0.1, -0.0, 0.2, -0.2, 0.0, -0.1, 0.1, -0.2, -0.0, 0.0, 0.1, 0.0, 0.1, -0.2, 0.0, 0.1},
{-0.2, -0.1, -0.1, -0.1, 0.0, 0.0, -0.0, -0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.0, 0.1, 0.1, 0.1, 0.0, -0.1, 0.1, -0.1, -0.1, 0.1, -0.0, -0.0},
{-0.0, 0.2, -0.2, -0.2, 0.1, -0.2, 0.1, -0.1, -0.2, -0.1, -0.1, 0.1, -0.2, -0.2, -0.1, 0.1, 0.2, 0.2, -0.0, 0.1, -0.0, -0.1, 0.0, 0.1, -0.1},
{0.2, 0.0, -0.2, 0.1, -0.2, 0.1, -0.0, -0.1, -0.1, 0.1, 0.0, 0.2, -0.1, -0.2, -0.1, -0.1, -0.2, 0.2, 0.0, 0.2, -0.0, 0.2, -0.1, 0.2, 0.2},
{-0.1, 0.2, 0.2, -0.0, -0.1, -0.1, -0.0, -0.2, 0.1, 0.2, 0.1, 0.1, -0.1, -0.1, 0.2, 0.1, -0.0, 0.1, -0.0, -0.1, 0.1, -0.1, -0.0, 0.1, 0.1},
}
);
Matrix  transformer_layers_0_feed_forward_linear1_bias   (
{0.2, 0.1, -0.0, -0.2, 0.1, -0.2, -0.1, 0.2, -0.0, 0.1, 0.0, 0.2, -0.1, -0.2, -0.2}
);
Matrix  transformer_layers_0_feed_forward_linear2_weight   (
{{0.1, -0.1, 0.2, -0.1, 0.1, 0.0, 0.2, 0.0, -0.1, 0.0, 0.0, -0.1, 0.1, 0.2, -0.2},
{0.1, 0.1, -0.1, 0.3, -0.1, 0.1, 0.1, -0.1, 0.1, -0.2, 0.3, -0.2, 0.2, 0.2, 0.1},
{-0.2, -0.1, -0.1, -0.2, -0.1, 0.1, -0.2, 0.1, -0.0, 0.1, -0.2, 0.1, -0.1, -0.0, 0.0},
{0.2, -0.2, 0.0, -0.2, -0.2, -0.2, 0.2, -0.3, -0.1, 0.1, 0.2, -0.2, 0.2, 0.1, 0.1},
{0.1, 0.2, 0.0, -0.1, 0.0, 0.1, -0.1, -0.1, 0.2, -0.2, -0.0, -0.1, -0.1, 0.1, -0.1},
{-0.1, -0.0, -0.1, 0.2, 0.2, -0.2, -0.1, 0.2, -0.2, 0.1, -0.1, 0.1, 0.1, 0.1, -0.0},
{0.2, 0.1, 0.1, -0.2, -0.1, 0.1, -0.1, -0.1, 0.1, -0.2, 0.0, -0.1, 0.1, -0.2, -0.1},
{0.1, 0.1, -0.1, -0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, -0.1, -0.2, -0.0, -0.1},
{-0.1, -0.1, 0.2, 0.2, 0.2, 0.0, -0.2, 0.1, 0.2, 0.1, -0.1, -0.0, 0.2, -0.1, -0.2},
{-0.1, -0.2, -0.0, 0.2, -0.1, -0.1, -0.1, 0.1, -0.1, -0.1, -0.1, 0.2, -0.1, -0.2, 0.3},
{0.1, 0.1, 0.1, -0.0, -0.0, -0.1, 0.2, 0.1, -0.1, -0.1, 0.2, 0.2, 0.2, 0.2, -0.1},
{-0.0, -0.1, 0.3, 0.1, -0.0, -0.0, 0.2, -0.1, -0.0, 0.0, 0.0, 0.1, 0.1, 0.1, -0.0},
{-0.0, 0.0, 0.2, 0.2, -0.2, -0.1, 0.0, -0.2, 0.2, 0.0, 0.0, -0.1, 0.2, 0.2, -0.2},
{0.0, 0.2, -0.2, 0.0, 0.1, 0.2, -0.2, -0.2, -0.0, 0.0, -0.1, 0.1, -0.1, 0.1, 0.1},
{-0.2, -0.1, -0.2, 0.0, -0.1, -0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.2, -0.2, 0.2, -0.1},
{-0.2, 0.0, 0.2, -0.1, 0.0, -0.2, -0.2, -0.2, 0.2, 0.2, 0.0, -0.1, 0.1, -0.2, -0.1},
{-0.2, 0.1, 0.1, -0.1, -0.0, -0.1, -0.0, 0.0, 0.2, 0.2, -0.2, 0.0, 0.1, -0.0, -0.0},
{0.3, -0.1, 0.0, 0.2, 0.0, -0.2, 0.0, -0.2, 0.3, -0.2, 0.0, 0.2, -0.2, -0.1, 0.2},
{0.1, -0.1, -0.2, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.2, 0.2, 0.1, -0.2, -0.2, 0.2},
{0.3, 0.3, -0.1, 0.2, 0.2, -0.1, -0.1, 0.0, -0.2, 0.2, -0.0, 0.1, 0.1, 0.0, 0.1},
{-0.0, -0.0, -0.0, -0.1, -0.1, -0.2, 0.0, 0.1, -0.0, -0.2, 0.2, 0.0, 0.0, 0.2, -0.0},
{0.0, 0.0, 0.1, -0.1, 0.2, -0.3, 0.2, -0.1, 0.2, -0.2, 0.2, -0.1, -0.2, 0.1, -0.2},
{-0.0, 0.2, -0.2, -0.0, 0.2, -0.2, -0.0, 0.0, -0.0, 0.1, 0.1, -0.1, 0.0, -0.2, -0.2},
{0.0, 0.0, 0.2, -0.1, 0.1, -0.1, 0.0, 0.0, -0.1, -0.2, -0.2, 0.0, 0.0, -0.2, 0.2},
{-0.2, -0.2, 0.1, -0.2, -0.1, -0.2, 0.1, -0.2, 0.0, -0.2, -0.0, 0.2, 0.0, -0.0, 0.2},
}
);
Matrix  transformer_layers_0_feed_forward_linear2_bias   (
{0.1, -0.2, -0.1, -0.1, 0.0, -0.1, -0.0, -0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2, -0.2, -0.1, 0.2, -0.0, 0.1, -0.0, -0.1, 0.2, 0.2, -0.1, 0.0}
);
Matrix  transformer_layers_0_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_0_feed_forward_ln1_layer_norm_bias   (
{0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0}
);
Matrix  transformer_layers_0_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_0_feed_forward_ln2_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_1_attention_query_weight   (
{{-0.1, 0.2, 0.2, 0.1, 0.0, -0.1, 0.1, 0.0, 0.2, -0.1, -0.1, -0.2, 0.1, 0.0, 0.1, 0.0, -0.0, 0.0, -0.0, -0.1, 0.0, 0.1, -0.0, -0.1, 0.1},
{0.2, 0.1, -0.0, 0.2, -0.2, -0.0, -0.2, 0.1, 0.2, -0.0, 0.0, 0.2, 0.1, -0.0, -0.2, 0.2, -0.2, 0.1, -0.1, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1},
{-0.1, 0.2, -0.0, -0.1, 0.1, 0.2, -0.1, -0.2, 0.1, 0.1, -0.1, -0.1, 0.1, 0.0, 0.2, -0.2, 0.1, 0.2, -0.1, 0.1, -0.1, 0.1, -0.1, 0.0, 0.1},
{-0.1, 0.0, -0.1, 0.1, -0.0, -0.1, 0.1, 0.0, 0.0, 0.1, -0.1, 0.1, 0.2, -0.0, -0.2, -0.1, -0.2, 0.1, -0.0, -0.1, -0.1, 0.1, -0.1, 0.2, -0.0},
{-0.0, -0.1, -0.2, -0.1, 0.0, 0.1, 0.1, -0.2, 0.1, -0.1, 0.1, -0.2, -0.1, -0.0, -0.0, 0.2, 0.1, -0.1, 0.2, 0.1, 0.1, -0.0, -0.1, 0.0, -0.0},
{0.1, -0.0, 0.1, -0.0, -0.2, -0.1, 0.0, -0.1, 0.1, -0.1, 0.1, 0.0, -0.1, 0.2, 0.0, 0.2, 0.1, 0.2, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1},
{0.0, -0.1, 0.0, 0.0, 0.2, -0.1, -0.1, -0.0, -0.1, 0.1, 0.0, -0.1, 0.1, -0.2, 0.0, -0.0, 0.1, -0.2, -0.1, -0.0, 0.1, 0.0, -0.0, -0.1, 0.2},
{0.1, 0.1, -0.0, 0.2, 0.1, -0.1, 0.1, 0.2, 0.1, -0.1, 0.2, -0.1, -0.1, 0.1, 0.2, 0.1, -0.2, 0.1, 0.1, -0.0, 0.1, 0.1, -0.2, 0.1, 0.2},
{0.1, -0.0, -0.1, -0.2, 0.1, -0.1, 0.0, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.0, -0.0, 0.1, 0.1, 0.1, -0.1, 0.1, -0.1, -0.0, 0.1, 0.1},
{0.1, 0.1, 0.1, -0.0, 0.1, -0.2, -0.1, 0.2, -0.2, 0.1, 0.0, -0.1, -0.2, 0.0, -0.0, -0.2, 0.2, -0.2, 0.2, 0.1, 0.1, -0.2, 0.2, -0.1, -0.0},
{-0.1, 0.1, -0.2, 0.1, 0.2, -0.0, 0.2, -0.1, 0.1, -0.0, 0.2, 0.2, 0.2, -0.2, -0.2, 0.1, 0.2, -0.0, 0.2, 0.1, 0.2, 0.2, 0.2, -0.2, 0.0},
{-0.1, -0.0, 0.0, 0.1, -0.0, -0.1, -0.1, 0.1, -0.2, -0.2, 0.0, -0.0, 0.1, 0.1, -0.1, 0.1, -0.1, -0.0, -0.0, -0.1, -0.2, -0.1, -0.0, 0.1, -0.2},
{-0.2, 0.2, -0.0, 0.0, 0.0, -0.1, -0.0, -0.1, -0.2, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.2, -0.2, 0.2, 0.2, -0.1, 0.1, -0.2, 0.1, -0.1, -0.1},
{0.1, -0.2, -0.1, -0.2, -0.1, -0.2, 0.0, -0.1, 0.2, -0.1, 0.2, 0.0, -0.1, -0.1, 0.1, 0.1, -0.1, -0.0, 0.1, -0.2, 0.1, -0.0, 0.0, -0.0, 0.0},
{-0.1, -0.2, -0.0, -0.0, -0.2, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 0.2, -0.1, 0.0, 0.0, 0.1, -0.1, -0.0, -0.1, 0.1, 0.1, -0.2, 0.0, 0.1},
{0.2, 0.0, 0.2, 0.1, 0.2, 0.2, 0.1, -0.0, 0.2, 0.0, -0.1, -0.1, 0.0, 0.1, -0.1, -0.1, -0.2, -0.0, 0.0, 0.1, -0.2, 0.1, 0.1, 0.1, -0.1},
{-0.0, 0.1, 0.2, 0.1, 0.0, 0.2, 0.0, -0.1, -0.2, 0.1, 0.2, 0.0, 0.2, 0.1, -0.1, -0.1, -0.0, -0.0, 0.0, 0.1, -0.0, -0.1, 0.0, -0.1, 0.2},
{0.0, 0.0, -0.1, -0.0, 0.0, -0.1, -0.0, 0.2, 0.2, -0.1, -0.1, -0.1, -0.0, 0.0, 0.0, 0.2, -0.2, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.0},
{0.1, -0.2, -0.0, 0.1, -0.2, -0.2, 0.1, -0.2, -0.0, -0.0, 0.1, -0.0, -0.0, 0.2, 0.2, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1, 0.2, 0.1, 0.1},
{-0.1, 0.2, 0.2, 0.2, -0.2, 0.0, -0.0, 0.1, -0.0, -0.1, 0.0, -0.1, -0.0, 0.1, -0.2, -0.1, 0.1, -0.1, -0.1, -0.0, -0.1, 0.1, 0.1, 0.2, 0.1},
{-0.2, -0.1, 0.2, -0.0, 0.1, -0.1, 0.1, -0.1, 0.0, -0.1, 0.1, -0.1, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.2, -0.1, -0.2, 0.0},
{-0.1, 0.2, -0.1, 0.2, -0.1, 0.0, -0.0, -0.2, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.2, 0.0, -0.0, 0.1, -0.2, -0.0, -0.0, -0.1, 0.0},
{0.2, 0.1, 0.1, -0.1, 0.1, -0.1, -0.2, 0.1, -0.0, -0.0, 0.0, -0.1, -0.1, 0.2, -0.1, -0.0, -0.1, -0.1, -0.1, -0.1, -0.0, 0.0, -0.1, -0.2, 0.1},
{-0.0, -0.0, 0.1, 0.1, -0.1, -0.1, 0.2, 0.0, -0.2, -0.2, 0.1, 0.1, -0.1, -0.2, -0.1, 0.1, -0.0, 0.1, 0.2, 0.0, -0.1, -0.2, 0.2, -0.1, -0.1},
{-0.2, -0.0, -0.0, 0.1, 0.2, -0.0, 0.1, -0.1, -0.1, 0.0, 0.2, -0.0, 0.0, 0.1, 0.0, -0.1, -0.2, 0.0, 0.1, 0.2, 0.2, 0.1, 0.1, -0.2, 0.1},
}
);
Matrix  transformer_layers_1_attention_query_bias   (
{0.2, -0.1, 0.1, -0.0, -0.1, -0.1, -0.1, 0.2, 0.1, -0.1, 0.1, -0.1, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.0, 0.1, -0.1, 0.1, 0.0}
);
Matrix  transformer_layers_1_attention_key_weight   (
{{-0.1, -0.0, -0.1, -0.1, 0.1, 0.2, 0.1, 0.0, 0.1, 0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2, 0.2, -0.1, 0.0, 0.1, 0.1, -0.1, -0.2, -0.1, 0.1},
{0.1, 0.1, -0.1, 0.2, 0.1, -0.0, 0.1, 0.0, 0.1, 0.2, -0.1, 0.2, -0.0, -0.2, -0.1, -0.1, -0.0, 0.1, -0.2, 0.0, -0.0, 0.2, 0.1, 0.0, 0.2},
{-0.2, -0.1, -0.2, 0.1, -0.2, -0.0, -0.0, 0.2, 0.0, 0.2, 0.2, 0.0, 0.1, 0.2, -0.0, 0.2, 0.1, -0.1, 0.2, -0.1, 0.2, -0.2, 0.1, 0.1, 0.1},
{0.1, 0.1, 0.1, -0.2, -0.2, -0.1, 0.1, 0.2, 0.0, 0.1, 0.2, 0.2, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, -0.0, -0.1},
{0.1, 0.0, 0.1, 0.0, -0.0, 0.1, -0.0, -0.2, 0.2, 0.1, -0.1, -0.1, -0.1, 0.1, -0.2, -0.2, -0.2, 0.1, -0.0, 0.1, 0.1, 0.2, 0.0, -0.1, -0.1},
{0.0, 0.1, 0.2, 0.1, -0.2, 0.0, -0.1, -0.1, -0.1, 0.2, -0.2, -0.2, -0.2, -0.1, -0.0, -0.1, 0.1, -0.0, -0.1, -0.0, -0.0, -0.1, 0.1, 0.1, 0.1},
{-0.1, -0.1, -0.0, -0.0, -0.1, -0.0, 0.2, -0.1, 0.1, 0.2, -0.1, -0.1, -0.1, -0.2, 0.1, 0.1, 0.0, 0.2, 0.0, -0.2, -0.1, -0.0, -0.1, 0.0, 0.0},
{0.1, -0.1, 0.2, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.2, 0.0, 0.1, 0.0, 0.2, -0.1, 0.1, 0.1, 0.2, 0.2, -0.2, 0.1, -0.0, 0.1, -0.0},
{-0.1, 0.2, -0.2, 0.1, -0.2, 0.1, -0.1, -0.0, -0.1, -0.1, -0.0, -0.2, -0.1, 0.0, -0.0, -0.0, 0.0, 0.1, 0.1, 0.2, 0.2, -0.0, 0.1, 0.0, -0.1},
{0.0, -0.0, -0.1, -0.1, -0.2, 0.0, -0.2, -0.0, -0.1, 0.0, -0.1, -0.1, 0.1, 0.0, 0.0, -0.1, 0.2, 0.2, 0.1, 0.1, -0.2, 0.2, 0.0, 0.2, -0.2},
{0.2, -0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.2, -0.0, -0.1, -0.0, 0.0, 0.0, 0.0, 0.0, -0.1, -0.1, -0.2, -0.1, 0.0, 0.2, 0.2, 0.2, -0.1, -0.1},
{-0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, -0.1, -0.2, -0.2, 0.0, 0.0, -0.1, -0.2, -0.1, 0.1, -0.0, -0.2, -0.1, 0.1, 0.0, 0.1},
{-0.2, 0.1, -0.1, -0.0, -0.2, 0.0, -0.1, -0.0, -0.1, -0.1, -0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, 0.1, 0.1, -0.1, -0.2, 0.0, -0.0, -0.1, -0.1},
{0.1, 0.2, 0.1, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.2, 0.2, -0.0, 0.1, 0.1, 0.1, 0.1, -0.2, -0.0, -0.1, 0.1, -0.1, -0.2, 0.0, 0.1, -0.1},
{0.2, -0.2, 0.2, -0.0, 0.1, -0.1, -0.0, -0.0, 0.1, 0.1, 0.0, 0.1, 0.0, 0.2, -0.2, 0.0, -0.1, 0.2, -0.1, -0.1, 0.0, 0.0, -0.0, -0.1, 0.0},
{-0.1, -0.0, 0.2, 0.2, -0.1, -0.1, 0.1, 0.2, 0.2, -0.1, 0.2, 0.0, -0.1, -0.1, 0.0, 0.0, -0.2, 0.0, -0.1, -0.1, -0.1, 0.1, 0.0, -0.1, 0.2},
{-0.2, 0.1, -0.0, 0.0, 0.1, 0.0, -0.1, -0.2, -0.2, -0.0, 0.1, -0.0, -0.1, -0.1, 0.1, 0.1, -0.0, 0.1, -0.0, -0.1, 0.2, -0.1, -0.2, 0.1, 0.0},
{0.0, 0.1, 0.1, 0.0, -0.1, 0.0, -0.1, -0.1, -0.2, 0.1, -0.1, 0.2, -0.2, -0.0, -0.1, 0.1, 0.0, -0.0, -0.0, 0.1, -0.2, 0.1, -0.1, 0.1, 0.1},
{-0.2, 0.1, -0.1, 0.0, -0.1, 0.1, -0.2, 0.1, -0.0, -0.1, -0.0, 0.0, 0.1, -0.2, -0.0, -0.0, -0.1, -0.2, 0.0, 0.1, -0.1, 0.1, -0.1, 0.0, -0.2},
{-0.0, 0.2, -0.0, 0.2, 0.2, -0.0, 0.2, 0.0, 0.1, -0.1, -0.0, 0.0, -0.1, 0.1, -0.1, -0.1, -0.1, -0.0, -0.1, -0.1, -0.0, -0.1, 0.1, -0.1, -0.1},
{0.1, -0.1, -0.0, 0.1, 0.0, -0.1, -0.0, 0.1, 0.1, 0.2, 0.1, 0.1, -0.2, 0.1, -0.2, 0.1, -0.2, 0.2, -0.2, -0.1, -0.0, 0.0, 0.1, -0.0, 0.2},
{-0.0, -0.0, 0.0, -0.1, 0.0, 0.1, 0.0, 0.0, 0.2, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1, 0.1, -0.1, -0.2, 0.0, -0.1, 0.1, 0.1, 0.1, 0.2, 0.0},
{-0.2, -0.1, 0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.1, -0.0, 0.0, -0.1, -0.1, -0.1, 0.1, 0.0, -0.2, 0.1, 0.1, 0.0, 0.2, -0.1, -0.2, -0.1, 0.1},
{0.2, -0.1, 0.0, -0.0, -0.1, 0.1, -0.2, 0.1, 0.1, -0.2, -0.0, 0.2, 0.2, 0.1, -0.1, 0.1, -0.2, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.0},
{0.0, -0.1, -0.1, 0.2, -0.2, -0.1, 0.0, -0.0, -0.1, -0.0, 0.0, -0.1, 0.2, -0.1, -0.2, -0.1, 0.1, -0.1, 0.2, 0.1, -0.1, -0.1, -0.1, 0.2, 0.1},
}
);
Matrix  transformer_layers_1_attention_key_bias   (
{-0.0, 0.2, 0.1, 0.0, 0.1, 0.0, -0.1, 0.2, 0.2, -0.0, 0.0, -0.2, 0.0, -0.0, -0.1, -0.0, 0.1, -0.2, -0.1, 0.0, -0.0, -0.0, -0.2, 0.0, -0.1}
);
Matrix  transformer_layers_1_attention_value_weight   (
{{0.1, -0.1, 0.0, 0.1, 0.1, -0.2, 0.1, -0.1, 0.1, 0.1, -0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, -0.2, -0.2, 0.1, -0.2, 0.1, 0.1},
{0.2, 0.1, -0.1, -0.1, -0.2, 0.1, -0.0, -0.1, 0.1, 0.1, -0.2, -0.1, -0.0, 0.1, -0.0, 0.1, -0.1, -0.0, -0.0, -0.1, -0.0, -0.1, 0.1, 0.1, -0.0},
{-0.1, -0.0, -0.2, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.2, -0.1, -0.2, -0.2, -0.1, -0.1, -0.1, -0.1, 0.1, 0.2, 0.1, -0.1, -0.1, 0.1, -0.2},
{0.1, -0.2, -0.2, 0.1, -0.1, 0.0, -0.1, -0.2, 0.1, 0.1, -0.1, 0.2, -0.1, -0.2, 0.1, -0.2, 0.0, 0.2, 0.0, -0.1, -0.1, 0.1, 0.1, 0.2, -0.0},
{-0.0, -0.2, -0.2, 0.2, -0.0, 0.0, -0.0, -0.1, -0.1, -0.1, 0.1, -0.0, -0.1, -0.2, -0.0, 0.0, -0.2, 0.2, -0.0, 0.2, -0.1, 0.1, 0.1, 0.2, -0.2},
{0.0, -0.2, -0.1, 0.1, 0.1, -0.2, -0.1, -0.1, 0.0, 0.1, 0.1, -0.1, -0.1, 0.2, 0.1, 0.1, 0.1, -0.2, -0.0, -0.1, 0.2, 0.1, 0.0, 0.1, 0.1},
{0.1, -0.1, -0.2, -0.2, -0.1, 0.1, 0.1, 0.1, -0.2, 0.1, -0.1, 0.0, 0.2, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.0, 0.1, -0.2, -0.1, -0.1, 0.2},
{0.1, 0.2, 0.0, -0.1, -0.0, 0.2, 0.1, 0.0, -0.1, -0.2, 0.1, -0.1, 0.2, -0.0, 0.1, -0.2, -0.0, -0.1, -0.1, -0.2, 0.2, -0.2, -0.1, -0.1, -0.2},
{-0.2, 0.1, -0.1, -0.2, -0.2, -0.1, -0.0, -0.0, -0.0, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1, -0.1, -0.2, 0.1, 0.1, -0.1, -0.0},
{-0.2, 0.0, 0.0, -0.1, 0.0, 0.2, 0.0, -0.2, -0.1, 0.0, 0.1, -0.2, 0.1, -0.1, -0.1, -0.2, -0.2, 0.2, -0.1, -0.2, 0.0, 0.1, 0.2, 0.1, -0.2},
{-0.2, 0.1, -0.0, 0.2, 0.2, -0.1, 0.1, -0.1, -0.1, 0.2, 0.2, -0.2, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.2, -0.1, -0.2, 0.1, 0.2, -0.0},
{0.2, -0.1, 0.2, 0.2, 0.1, -0.0, -0.1, 0.0, 0.1, -0.1, 0.1, 0.1, 0.1, -0.2, -0.2, 0.0, -0.1, 0.1, 0.1, -0.2, 0.1, 0.1, 0.1, -0.1, -0.1},
{-0.0, 0.2, -0.1, -0.1, 0.1, -0.1, -0.1, -0.1, 0.1, -0.0, -0.0, -0.1, -0.1, 0.1, 0.1, 0.0, -0.1, -0.2, -0.1, -0.1, -0.0, 0.0, -0.2, -0.1, 0.2},
{-0.1, -0.1, -0.1, 0.1, -0.1, -0.2, -0.2, -0.0, 0.2, 0.2, -0.1, 0.0, -0.0, 0.1, 0.1, 0.1, 0.2, 0.1, -0.0, -0.2, -0.2, -0.1, -0.1, 0.1, -0.0},
{0.1, 0.1, 0.0, -0.1, 0.1, 0.0, -0.1, -0.0, 0.0, 0.1, -0.0, -0.0, -0.2, -0.1, 0.0, 0.0, 0.0, 0.1, -0.0, 0.2, 0.0, -0.1, -0.1, -0.1, 0.1},
{0.1, 0.0, 0.2, -0.1, -0.0, 0.0, -0.0, 0.1, 0.1, 0.0, 0.1, -0.2, 0.2, -0.2, -0.1, 0.2, -0.2, -0.1, -0.1, 0.0, -0.2, 0.1, 0.1, -0.0, -0.1},
{0.1, -0.2, -0.1, -0.2, -0.0, -0.1, 0.1, 0.2, 0.2, 0.2, 0.0, -0.0, -0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, -0.1, -0.0, 0.0, 0.0, 0.1, 0.2},
{0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.0, -0.1, -0.2, -0.0, -0.1, 0.0, 0.2, -0.1, -0.1, -0.1, 0.2, -0.0, -0.2, -0.2, -0.2, -0.2, 0.0, 0.1, -0.2},
{-0.1, -0.0, 0.0, 0.2, -0.0, -0.1, -0.2, 0.1, -0.2, 0.1, -0.1, -0.1, 0.1, -0.1, -0.0, 0.0, -0.0, 0.1, -0.1, 0.0, 0.0, -0.2, 0.1, -0.1, 0.1},
{0.1, 0.2, -0.1, -0.2, -0.0, 0.2, -0.2, -0.2, 0.1, 0.1, -0.0, -0.1, -0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.2, 0.1, 0.1, 0.1, -0.2, -0.2, 0.1},
{0.0, 0.2, 0.2, -0.1, -0.1, -0.1, 0.1, -0.2, -0.2, 0.1, -0.0, 0.1, 0.1, -0.0, 0.2, -0.2, -0.2, 0.2, 0.2, -0.0, -0.1, 0.1, -0.1, -0.2, -0.2},
{-0.1, -0.0, -0.1, 0.1, -0.0, 0.0, -0.1, 0.0, -0.1, -0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.2, 0.1, 0.1, 0.1, -0.0, 0.0, 0.1, 0.1, -0.0, 0.1},
{0.2, 0.0, -0.1, -0.2, -0.1, -0.1, 0.1, 0.1, -0.1, -0.0, 0.1, -0.0, -0.0, -0.1, 0.1, 0.0, 0.2, 0.0, 0.0, 0.2, -0.2, -0.0, 0.0, 0.2, -0.0},
{-0.0, 0.2, -0.0, 0.2, -0.2, -0.1, 0.1, 0.1, -0.2, -0.1, 0.2, 0.1, 0.1, 0.1, -0.2, -0.0, -0.2, -0.1, 0.0, 0.1, -0.0, 0.2, 0.1, 0.2, 0.2},
{0.0, -0.1, 0.2, 0.0, 0.1, -0.0, 0.1, -0.1, 0.1, 0.1, -0.2, 0.1, -0.2, 0.1, -0.1, 0.1, -0.1, 0.2, -0.2, -0.1, 0.1, -0.1, -0.1, 0.2, 0.1},
}
);
Matrix  transformer_layers_1_attention_value_bias   (
{-0.0, 0.0, -0.1, 0.1, 0.2, -0.2, 0.1, -0.1, -0.1, 0.1, -0.0, -0.2, 0.1, -0.2, 0.2, 0.2, -0.2, 0.0, 0.1, 0.2, 0.1, 0.1, -0.2, 0.0, 0.0}
);
Matrix  transformer_layers_1_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_1_norm1_layer_norm_bias   (
{-0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_1_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_1_norm2_layer_norm_bias   (
{-0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_1_feed_forward_linear1_weight   (
{{0.2, -0.2, 0.1, 0.0, 0.2, -0.0, 0.1, -0.0, -0.0, 0.1, 0.1, 0.0, -0.1, 0.2, -0.1, -0.2, -0.0, 0.1, -0.1, -0.0, 0.1, 0.0, 0.2, 0.2, -0.0},
{-0.0, 0.0, -0.1, -0.0, -0.1, -0.1, -0.2, 0.1, 0.1, -0.0, 0.2, -0.1, 0.1, 0.1, 0.0, -0.0, -0.0, -0.0, -0.1, 0.1, -0.1, -0.2, 0.1, 0.2, -0.1},
{0.1, -0.1, -0.1, -0.0, 0.1, -0.1, 0.1, -0.1, -0.2, 0.2, 0.0, 0.2, 0.1, -0.0, -0.0, 0.2, 0.0, 0.1, -0.2, 0.1, -0.2, -0.1, -0.0, 0.2, 0.0},
{-0.0, -0.2, -0.1, 0.0, -0.0, 0.0, 0.0, -0.1, 0.1, -0.2, 0.1, -0.2, -0.2, 0.1, -0.0, -0.0, 0.1, -0.1, -0.1, -0.2, -0.0, 0.0, 0.0, -0.0, 0.1},
{0.1, 0.0, 0.2, 0.2, -0.0, -0.2, -0.1, 0.0, -0.1, -0.0, 0.1, 0.1, -0.2, 0.2, 0.1, 0.2, 0.1, -0.1, -0.0, -0.1, 0.1, -0.1, -0.0, 0.1, -0.2},
{-0.0, -0.2, 0.1, 0.2, -0.2, 0.2, 0.0, -0.1, -0.1, 0.2, 0.1, 0.1, -0.2, -0.1, 0.1, 0.0, 0.1, -0.2, 0.2, 0.2, 0.1, -0.1, 0.1, -0.1, 0.0},
{-0.2, -0.2, 0.0, -0.1, -0.1, -0.1, 0.1, 0.0, -0.1, 0.2, -0.1, 0.0, -0.2, -0.1, -0.2, 0.1, 0.0, 0.2, 0.1, -0.1, 0.1, -0.2, -0.0, 0.1, 0.0},
{0.1, 0.1, -0.0, 0.1, -0.2, -0.0, -0.2, -0.0, 0.2, 0.0, 0.2, 0.1, 0.1, -0.2, 0.0, -0.0, -0.2, 0.2, -0.1, -0.0, -0.0, 0.1, -0.2, 0.1, -0.1},
{0.2, 0.2, -0.2, 0.2, -0.1, 0.1, -0.1, 0.0, -0.1, -0.0, 0.1, -0.1, 0.0, 0.2, 0.2, -0.1, -0.2, 0.0, 0.1, 0.2, 0.0, -0.0, -0.2, 0.0, -0.0},
{0.1, 0.1, 0.1, 0.2, 0.2, -0.0, 0.1, -0.0, 0.2, 0.2, -0.1, 0.2, -0.0, -0.1, -0.2, -0.0, 0.0, 0.1, 0.1, 0.2, 0.1, -0.2, 0.0, -0.1, -0.0},
{0.1, -0.0, 0.1, -0.1, -0.1, -0.1, -0.0, 0.1, 0.1, -0.1, 0.2, -0.0, -0.1, 0.1, 0.2, -0.1, 0.2, 0.0, 0.2, 0.1, 0.1, 0.1, 0.2, 0.0, 0.2},
{-0.0, 0.0, -0.1, -0.0, -0.1, -0.1, 0.1, -0.2, -0.2, 0.1, -0.2, -0.0, -0.2, -0.2, -0.0, -0.1, 0.2, -0.1, 0.0, -0.0, 0.0, -0.2, -0.1, 0.2, 0.0},
{0.1, -0.1, 0.0, 0.1, -0.1, 0.2, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.1, 0.0, 0.2, 0.1, 0.1, 0.2, -0.1, 0.1, 0.1, 0.2, -0.2, -0.1, 0.0},
{-0.1, 0.0, 0.2, -0.1, -0.2, 0.2, -0.2, -0.1, 0.1, -0.1, 0.0, 0.1, 0.2, 0.0, -0.2, 0.1, 0.0, -0.2, 0.1, 0.0, 0.0, -0.0, 0.2, 0.0, -0.0},
{0.1, -0.2, -0.0, -0.2, -0.1, 0.1, -0.1, -0.0, 0.1, 0.2, 0.1, 0.1, 0.1, -0.0, 0.2, 0.1, -0.2, -0.2, 0.2, -0.1, 0.2, -0.1, 0.1, -0.2, 0.0},
}
);
Matrix  transformer_layers_1_feed_forward_linear1_bias   (
{0.2, -0.2, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.2, 0.1, -0.1, -0.2, -0.2, -0.0}
);
Matrix  transformer_layers_1_feed_forward_linear2_weight   (
{{0.1, -0.2, 0.2, -0.2, 0.1, 0.2, -0.1, 0.2, -0.1, 0.2, 0.1, 0.0, 0.1, 0.2, -0.1},
{-0.2, -0.1, 0.2, -0.2, 0.2, 0.0, 0.1, -0.1, -0.2, -0.2, 0.1, -0.2, 0.0, 0.2, 0.0},
{0.2, -0.1, -0.1, 0.2, 0.1, 0.2, 0.2, -0.1, 0.0, -0.2, -0.1, -0.0, 0.0, 0.2, -0.2},
{0.2, 0.1, 0.1, -0.2, -0.3, 0.1, 0.0, 0.2, 0.2, 0.0, -0.0, 0.1, -0.0, 0.1, 0.0},
{-0.1, -0.1, -0.1, 0.1, -0.2, 0.1, -0.2, 0.0, -0.2, 0.1, -0.3, -0.2, 0.1, 0.2, -0.2},
{0.0, -0.1, 0.1, 0.1, 0.1, 0.2, -0.1, 0.2, -0.2, 0.2, 0.2, -0.2, -0.1, 0.2, -0.3},
{-0.2, 0.3, 0.1, 0.1, 0.0, -0.1, -0.2, 0.2, 0.2, 0.1, -0.2, -0.1, 0.1, -0.0, -0.0},
{0.2, -0.2, 0.1, -0.2, -0.1, 0.2, 0.3, 0.2, 0.1, -0.0, 0.1, 0.2, -0.0, 0.2, -0.3},
{0.2, -0.2, 0.1, 0.2, -0.1, 0.2, 0.1, -0.1, -0.1, 0.2, 0.1, 0.2, 0.2, -0.1, 0.1},
{-0.2, 0.1, 0.2, 0.0, 0.2, 0.1, 0.1, 0.0, 0.3, 0.1, 0.1, 0.1, -0.2, -0.1, -0.2},
{-0.2, -0.2, 0.1, -0.0, -0.0, -0.2, -0.0, -0.3, -0.2, 0.3, -0.0, 0.1, -0.0, 0.2, -0.2},
{-0.0, -0.1, 0.0, -0.2, -0.1, 0.0, 0.0, -0.1, -0.2, 0.1, 0.1, -0.1, 0.1, -0.1, -0.1},
{0.2, -0.3, 0.2, -0.0, 0.0, -0.1, 0.2, -0.3, 0.1, 0.2, 0.0, -0.2, -0.2, -0.2, 0.2},
{0.1, -0.2, 0.1, -0.0, -0.3, -0.3, -0.1, -0.1, 0.1, 0.1, 0.2, 0.3, 0.0, -0.1, -0.2},
{-0.1, -0.2, 0.1, -0.0, 0.1, 0.0, -0.2, 0.1, 0.1, -0.2, -0.2, 0.1, 0.0, 0.0, -0.0},
{-0.2, -0.1, -0.1, 0.1, 0.0, 0.2, 0.1, 0.2, -0.0, 0.0, -0.0, -0.2, -0.0, -0.1, -0.0},
{0.2, -0.2, -0.2, -0.0, -0.2, 0.1, -0.0, -0.2, 0.2, 0.2, -0.2, 0.0, 0.1, -0.1, 0.0},
{0.1, -0.0, 0.0, 0.1, -0.1, -0.1, -0.2, 0.1, -0.1, -0.2, -0.0, 0.0, 0.2, 0.2, -0.0},
{0.0, -0.2, -0.1, -0.0, -0.1, -0.0, -0.2, -0.0, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, -0.1},
{0.2, -0.2, -0.1, -0.1, -0.2, 0.2, -0.1, 0.1, -0.0, 0.2, -0.0, -0.0, -0.2, -0.1, -0.2},
{-0.0, -0.2, 0.1, 0.2, 0.2, -0.0, 0.2, 0.2, 0.2, -0.1, -0.2, -0.1, -0.2, -0.1, 0.1},
{0.2, 0.2, -0.1, 0.0, -0.2, -0.1, -0.0, 0.0, -0.1, 0.1, 0.2, 0.0, 0.0, -0.1, -0.3},
{0.1, 0.0, -0.2, 0.2, 0.1, -0.2, 0.2, -0.2, 0.3, -0.1, -0.1, -0.1, 0.2, 0.1, -0.0},
{0.2, -0.2, -0.0, -0.0, 0.0, 0.1, -0.1, 0.2, 0.3, 0.3, 0.0, 0.1, -0.0, -0.1, -0.1},
{0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.2, -0.1, 0.1, -0.1, -0.2, 0.1, 0.0, -0.0, -0.3},
}
);
Matrix  transformer_layers_1_feed_forward_linear2_bias   (
{0.2, 0.2, -0.2, -0.0, -0.1, 0.1, 0.1, -0.1, -0.2, 0.2, 0.0, -0.0, 0.1, -0.1, -0.2, 0.1, 0.2, -0.2, -0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.0}
);
Matrix  transformer_layers_1_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_1_feed_forward_ln1_layer_norm_bias   (
{0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_1_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_1_feed_forward_ln2_layer_norm_bias   (
{-0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_2_attention_query_weight   (
{{-0.1, 0.0, 0.0, 0.0, 0.0, 0.2, 0.1, 0.2, -0.1, -0.1, -0.1, 0.2, -0.0, 0.1, 0.2, 0.0, -0.1, 0.1, 0.0, 0.1, -0.0, 0.0, 0.1, 0.0, 0.0},
{0.0, 0.1, -0.1, 0.1, -0.0, -0.2, -0.0, -0.2, -0.0, 0.1, 0.2, -0.0, -0.0, -0.0, -0.1, 0.2, -0.0, -0.0, 0.0, 0.2, 0.2, -0.1, -0.2, -0.2, 0.1},
{0.1, 0.1, 0.0, -0.2, -0.2, -0.0, -0.1, -0.0, 0.1, 0.1, -0.0, -0.1, 0.1, 0.1, -0.0, -0.2, -0.2, -0.1, -0.0, 0.2, -0.0, -0.1, 0.1, -0.0, 0.1},
{-0.1, 0.1, 0.1, -0.2, -0.2, -0.1, 0.0, 0.1, -0.0, -0.1, -0.2, -0.2, 0.1, 0.1, -0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.2, 0.2, -0.1, -0.0, 0.1},
{0.1, -0.0, -0.1, 0.1, -0.2, 0.0, -0.1, -0.1, 0.0, 0.1, 0.0, -0.1, 0.2, -0.2, -0.2, -0.2, 0.1, 0.2, -0.1, -0.1, -0.2, -0.0, 0.2, -0.2, 0.2},
{0.1, 0.0, 0.1, 0.1, 0.2, -0.2, 0.0, 0.1, -0.1, 0.1, -0.1, 0.0, 0.0, -0.1, -0.2, -0.0, -0.2, -0.1, 0.1, -0.1, 0.1, 0.2, -0.1, -0.0, -0.1},
{-0.1, -0.1, -0.2, -0.1, -0.1, -0.0, -0.0, 0.1, -0.0, -0.1, -0.2, -0.2, 0.1, -0.1, 0.0, 0.0, 0.1, 0.1, -0.2, -0.0, -0.2, 0.1, 0.2, 0.1, -0.0},
{0.0, -0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.2, 0.1, 0.0, -0.1, 0.1, 0.2, 0.1, -0.2, -0.1, 0.1, 0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.0, -0.1},
{0.2, -0.1, -0.1, 0.1, 0.1, -0.2, -0.1, 0.1, 0.1, -0.2, 0.1, -0.1, 0.2, 0.0, 0.2, 0.1, 0.1, -0.1, -0.1, -0.2, 0.1, 0.2, -0.1, -0.1, -0.0},
{0.1, 0.1, -0.2, -0.1, 0.1, -0.2, -0.0, -0.2, -0.0, 0.0, 0.2, 0.2, 0.1, 0.0, 0.1, 0.1, 0.0, 0.2, 0.0, 0.2, 0.1, -0.2, 0.1, -0.1, 0.1},
{0.2, -0.1, -0.1, 0.1, 0.0, 0.0, 0.1, 0.1, -0.0, -0.1, 0.0, 0.0, -0.1, -0.1, 0.2, -0.1, 0.0, -0.1, 0.0, -0.2, -0.0, -0.2, -0.1, -0.2, -0.0},
{-0.1, -0.1, 0.0, 0.2, 0.2, -0.2, 0.1, -0.1, -0.1, -0.0, 0.0, 0.0, 0.0, 0.0, 0.1, -0.1, -0.1, 0.2, -0.0, -0.0, -0.1, 0.1, -0.2, 0.1, 0.0},
{-0.1, -0.2, -0.0, -0.2, 0.2, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.2, 0.2, 0.1, -0.0, -0.0, -0.1, 0.1, -0.1, 0.2, 0.0, -0.0, 0.0, 0.2},
{0.1, 0.0, 0.0, 0.0, -0.1, -0.0, -0.1, -0.0, 0.1, -0.0, -0.1, -0.1, 0.0, 0.1, -0.2, 0.0, 0.1, 0.2, -0.1, 0.2, 0.2, 0.1, -0.1, 0.1, -0.1},
{-0.1, -0.2, 0.1, 0.1, 0.0, 0.2, 0.2, -0.1, -0.1, -0.0, -0.1, -0.2, 0.1, 0.1, 0.1, -0.0, 0.2, -0.1, 0.0, 0.2, -0.2, -0.2, 0.2, 0.1, 0.1},
{-0.2, -0.1, -0.2, -0.1, 0.1, -0.2, -0.2, -0.0, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.1, -0.2, -0.0, -0.1, 0.1, -0.1, -0.2, 0.1, -0.2, 0.2},
{0.1, 0.0, -0.2, -0.1, -0.1, -0.1, 0.2, 0.1, -0.2, -0.1, 0.2, 0.1, -0.1, -0.1, 0.0, 0.1, 0.0, -0.1, -0.1, -0.2, -0.1, -0.0, -0.1, 0.1, 0.2},
{-0.1, -0.0, 0.1, -0.0, 0.1, -0.1, -0.0, 0.1, -0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.2, 0.1, 0.2, -0.0, -0.1, 0.1, 0.1, -0.0, 0.2},
{0.2, -0.2, 0.1, -0.1, -0.2, -0.1, 0.0, -0.1, -0.1, 0.1, 0.0, -0.1, -0.1, 0.1, -0.1, 0.1, -0.0, -0.2, 0.1, -0.2, 0.1, 0.1, 0.0, -0.1, 0.2},
{-0.1, 0.0, -0.1, -0.1, 0.2, 0.0, 0.1, -0.1, 0.0, 0.2, 0.0, -0.1, -0.1, -0.0, 0.1, -0.1, 0.2, 0.1, -0.1, 0.1, -0.1, -0.1, -0.1, 0.2, -0.1},
{0.0, 0.2, 0.1, -0.2, 0.1, 0.2, 0.2, 0.1, -0.2, -0.0, 0.1, -0.2, 0.0, -0.0, -0.0, -0.0, 0.2, 0.1, -0.2, -0.1, 0.1, -0.2, 0.2, -0.1, 0.2},
{-0.1, 0.0, -0.2, 0.1, -0.0, -0.1, -0.1, -0.1, 0.2, -0.1, 0.0, -0.0, -0.0, 0.2, 0.1, 0.0, -0.0, 0.1, 0.0, -0.2, -0.0, 0.1, -0.1, 0.0, 0.2},
{-0.0, 0.0, 0.0, 0.1, 0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.2, -0.2, -0.2, -0.2, -0.1, -0.2, -0.1, -0.1, 0.1, -0.1, -0.0, 0.1, -0.0, 0.2, -0.2},
{-0.2, 0.1, -0.2, 0.1, 0.1, 0.1, 0.2, 0.1, -0.1, 0.1, -0.0, 0.1, 0.2, 0.1, 0.2, -0.1, 0.0, 0.0, 0.0, -0.0, 0.2, 0.0, 0.1, 0.0, -0.1},
{-0.0, -0.1, 0.2, 0.0, 0.1, -0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.1, 0.2, 0.0, -0.1, -0.1, 0.1, 0.1, 0.1, -0.2, 0.2, 0.2, -0.0, 0.1, 0.1},
}
);
Matrix  transformer_layers_2_attention_query_bias   (
{0.1, -0.1, -0.2, 0.1, -0.1, 0.1, 0.1, 0.2, -0.2, -0.2, 0.1, 0.1, 0.2, -0.1, 0.1, -0.2, 0.2, 0.1, 0.1, -0.1, -0.1, 0.2, -0.1, -0.2, 0.0}
);
Matrix  transformer_layers_2_attention_key_weight   (
{{-0.2, 0.0, -0.2, 0.2, 0.1, 0.1, -0.0, 0.2, -0.1, -0.1, 0.1, 0.2, -0.0, -0.1, -0.2, 0.0, -0.2, 0.0, -0.0, -0.2, -0.2, 0.1, -0.2, -0.2, 0.2},
{-0.1, -0.2, 0.1, 0.2, -0.0, -0.0, -0.2, -0.0, 0.1, -0.0, -0.1, 0.1, -0.2, 0.1, -0.1, -0.1, 0.1, -0.0, -0.0, -0.1, 0.2, 0.1, 0.0, -0.0, 0.0},
{-0.2, 0.1, 0.1, 0.2, 0.2, -0.0, 0.0, 0.1, 0.2, -0.0, 0.1, -0.1, 0.1, -0.1, -0.1, 0.2, -0.2, -0.2, -0.1, 0.1, -0.1, -0.0, 0.0, 0.0, -0.2},
{-0.1, 0.2, 0.0, -0.2, 0.1, -0.1, -0.0, -0.0, 0.1, 0.1, 0.1, -0.2, -0.0, 0.1, 0.1, -0.1, 0.1, 0.0, 0.1, 0.2, -0.1, -0.2, 0.2, 0.1, 0.1},
{0.1, 0.0, -0.0, 0.1, -0.0, 0.1, -0.2, 0.1, 0.0, -0.0, 0.1, 0.0, -0.2, -0.1, -0.2, 0.1, -0.0, 0.1, -0.1, 0.2, -0.1, -0.1, 0.2, 0.1, -0.0},
{0.1, 0.1, 0.2, -0.1, 0.0, -0.1, -0.1, -0.0, 0.1, 0.1, -0.1, -0.0, 0.1, -0.0, 0.2, -0.1, 0.0, -0.1, 0.1, 0.0, 0.2, 0.2, -0.1, -0.1, -0.0},
{0.1, -0.1, -0.0, -0.1, -0.1, -0.1, -0.0, 0.1, -0.1, -0.1, 0.2, 0.0, -0.2, -0.1, -0.2, -0.0, 0.0, -0.0, 0.1, -0.2, 0.0, -0.1, 0.1, -0.2, -0.1},
{-0.1, 0.2, 0.0, 0.0, -0.2, 0.0, 0.0, -0.1, -0.2, 0.0, -0.1, 0.1, 0.0, 0.1, -0.1, -0.2, -0.0, -0.1, -0.1, 0.2, 0.1, -0.0, 0.2, -0.0, -0.1},
{-0.1, 0.1, 0.1, -0.2, -0.2, -0.0, -0.1, -0.0, 0.1, -0.0, 0.1, -0.1, -0.0, -0.2, -0.1, 0.2, 0.0, 0.1, -0.1, 0.1, -0.1, -0.2, -0.2, 0.0, -0.2},
{0.0, 0.2, -0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.0, -0.2, -0.1, 0.1, 0.0, -0.1, -0.0, 0.1, -0.0, 0.1, -0.2, -0.2, 0.2, 0.1, 0.0},
{0.2, 0.1, -0.0, -0.1, -0.0, 0.1, -0.1, 0.2, 0.2, -0.1, 0.1, 0.1, 0.2, 0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.0, -0.2, 0.1, -0.0, -0.0, -0.0},
{0.1, -0.2, -0.1, 0.0, -0.1, -0.2, 0.0, 0.1, -0.0, -0.1, 0.1, 0.1, -0.2, 0.2, 0.1, 0.0, 0.1, -0.1, -0.0, -0.0, -0.0, 0.1, 0.0, 0.1, -0.1},
{-0.2, -0.1, -0.2, -0.1, -0.0, 0.1, 0.0, 0.0, -0.0, 0.0, -0.1, -0.0, -0.1, 0.2, -0.1, -0.1, 0.0, -0.2, 0.1, 0.2, 0.1, -0.0, 0.1, 0.0, -0.1},
{0.1, -0.1, 0.1, -0.2, -0.1, -0.0, -0.1, 0.0, -0.2, -0.1, 0.0, -0.2, -0.0, -0.1, -0.1, 0.0, 0.1, 0.1, 0.1, 0.2, -0.0, 0.0, -0.1, -0.1, 0.1},
{-0.0, 0.1, 0.0, -0.1, -0.1, 0.2, 0.1, 0.0, 0.1, -0.0, -0.1, 0.2, -0.1, -0.2, 0.1, -0.0, -0.2, 0.0, 0.2, -0.1, 0.1, -0.1, 0.1, -0.0, -0.1},
{0.1, -0.0, 0.0, -0.2, -0.1, 0.2, 0.1, -0.1, 0.1, 0.2, 0.0, 0.1, -0.1, 0.2, -0.1, -0.1, 0.1, -0.0, 0.1, -0.1, -0.1, -0.2, 0.0, 0.2, 0.1},
{-0.2, 0.2, -0.2, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, -0.1, -0.0, -0.1, -0.0, -0.0, 0.1, -0.0, 0.1, 0.1, 0.1, 0.2, 0.0, -0.0, 0.1, 0.2, -0.1},
{0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, -0.2, 0.2, 0.0, 0.0, -0.1, -0.1, 0.1, -0.0, -0.0, 0.1, -0.0, -0.1, -0.0, 0.1, -0.0},
{-0.2, 0.0, -0.1, 0.0, 0.1, -0.2, -0.2, -0.0, 0.0, -0.0, 0.1, 0.1, -0.2, 0.1, 0.0, 0.1, -0.1, -0.0, 0.1, -0.1, -0.1, -0.0, -0.0, -0.0, -0.0},
{0.1, 0.1, -0.1, 0.0, 0.0, 0.1, -0.1, 0.0, -0.1, -0.0, 0.2, -0.2, 0.1, -0.1, -0.0, 0.1, 0.1, 0.0, 0.2, -0.1, 0.0, -0.1, -0.2, 0.0, -0.1},
{-0.2, 0.2, -0.1, -0.1, 0.0, -0.1, -0.1, 0.0, 0.1, -0.1, -0.1, 0.1, -0.2, 0.0, 0.2, 0.0, 0.1, 0.1, -0.0, -0.0, 0.1, 0.0, 0.1, -0.2, 0.1},
{-0.2, -0.0, -0.0, -0.2, -0.1, 0.1, -0.1, -0.0, 0.2, -0.0, 0.0, -0.0, 0.0, 0.2, 0.1, 0.0, 0.2, -0.0, 0.1, -0.1, 0.2, 0.1, 0.1, 0.2, -0.2},
{0.1, -0.2, -0.0, -0.2, -0.0, 0.1, -0.1, 0.1, -0.1, 0.0, 0.2, 0.0, -0.1, 0.2, -0.0, -0.1, 0.1, -0.2, 0.0, -0.1, 0.1, -0.0, 0.1, 0.1, -0.2},
{-0.1, 0.2, -0.1, -0.1, -0.2, -0.1, 0.2, -0.2, 0.0, -0.1, -0.1, 0.1, -0.1, -0.1, -0.1, 0.0, -0.2, -0.1, 0.1, 0.2, -0.1, 0.0, 0.2, -0.1, -0.1},
{0.1, -0.0, -0.1, 0.1, 0.0, 0.0, 0.1, -0.1, -0.2, -0.1, 0.2, -0.1, -0.0, -0.2, -0.1, -0.1, -0.1, 0.0, -0.1, 0.1, -0.1, 0.1, -0.1, -0.1, -0.2},
}
);
Matrix  transformer_layers_2_attention_key_bias   (
{-0.1, 0.1, 0.0, -0.0, 0.0, -0.1, 0.1, -0.1, 0.0, 0.2, 0.1, 0.1, -0.2, 0.2, -0.0, 0.0, 0.0, -0.1, -0.1, 0.1, 0.0, 0.2, -0.1, -0.0, -0.0}
);
Matrix  transformer_layers_2_attention_value_weight   (
{{0.1, -0.0, -0.0, -0.1, -0.1, 0.2, 0.1, 0.2, 0.1, -0.0, -0.0, 0.1, -0.1, 0.2, -0.1, 0.0, 0.1, -0.0, 0.1, -0.2, 0.0, -0.1, -0.2, 0.1, -0.0},
{0.1, -0.1, -0.0, 0.1, 0.1, -0.1, 0.2, -0.2, 0.2, -0.1, -0.1, 0.1, 0.2, 0.0, 0.1, 0.2, -0.0, -0.2, -0.0, 0.0, 0.1, -0.0, -0.0, -0.2, 0.1},
{-0.1, 0.1, -0.2, 0.1, 0.0, -0.1, -0.2, -0.1, -0.1, 0.0, -0.1, 0.1, 0.2, -0.0, 0.0, -0.1, -0.1, 0.2, -0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0.1},
{-0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.2, 0.2, -0.1, -0.1, 0.2, -0.2, 0.1, -0.1, -0.2, -0.0, -0.2, 0.1, 0.2, 0.0, 0.1, -0.1, 0.1, -0.2, 0.1},
{-0.0, 0.2, -0.1, -0.1, -0.0, 0.1, -0.0, -0.1, -0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, 0.1, -0.1, -0.1, -0.2, -0.1, -0.1, 0.2, -0.0, 0.0, 0.1},
{0.2, -0.1, 0.0, -0.1, -0.1, 0.0, 0.1, -0.1, -0.2, -0.1, -0.1, 0.2, 0.2, 0.1, -0.0, -0.1, -0.0, 0.1, -0.0, 0.0, -0.2, 0.1, -0.1, -0.2, -0.0},
{0.1, 0.1, -0.2, -0.1, 0.0, 0.1, 0.2, 0.0, -0.1, 0.2, 0.2, -0.1, 0.1, 0.2, -0.1, 0.2, 0.0, 0.0, -0.2, 0.1, -0.1, 0.1, -0.1, 0.0, -0.0},
{-0.0, 0.2, -0.0, 0.1, -0.2, -0.1, 0.0, -0.2, 0.2, -0.0, 0.1, -0.0, 0.1, -0.1, -0.1, -0.1, -0.2, -0.2, 0.1, 0.0, 0.1, -0.1, 0.1, -0.1, -0.1},
{0.2, -0.2, 0.1, 0.1, -0.1, 0.1, 0.1, 0.2, -0.2, 0.1, -0.2, -0.2, -0.1, -0.1, 0.2, -0.2, -0.1, 0.1, -0.1, 0.0, 0.1, -0.2, 0.1, -0.0, -0.1},
{0.1, -0.2, -0.1, 0.1, -0.1, 0.0, -0.1, 0.0, -0.2, 0.0, 0.2, -0.2, -0.1, 0.1, 0.0, 0.2, 0.1, -0.2, 0.2, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1},
{-0.1, 0.0, -0.1, -0.2, -0.0, -0.1, 0.2, -0.1, -0.0, 0.0, -0.1, -0.0, 0.0, -0.1, -0.0, 0.0, 0.1, -0.0, -0.0, -0.1, 0.0, 0.0, 0.2, -0.1, 0.0},
{-0.0, -0.0, -0.2, -0.1, -0.1, 0.1, -0.1, -0.0, 0.2, 0.1, -0.1, -0.1, -0.1, 0.1, 0.2, -0.2, 0.1, 0.1, 0.2, -0.0, -0.1, -0.0, -0.2, -0.1, -0.1},
{0.2, -0.0, 0.1, 0.0, -0.2, -0.1, 0.0, 0.2, -0.1, -0.1, 0.0, 0.0, -0.2, 0.2, 0.1, 0.1, -0.0, -0.2, -0.0, 0.1, -0.0, 0.1, 0.1, 0.1, -0.1},
{-0.1, 0.1, 0.2, -0.0, 0.2, -0.2, 0.1, 0.1, -0.1, 0.2, 0.0, -0.1, -0.1, 0.1, 0.1, -0.2, 0.0, -0.1, -0.0, -0.1, 0.1, 0.1, 0.1, -0.1, 0.2},
{0.1, 0.1, 0.2, -0.0, 0.0, -0.1, -0.1, 0.1, -0.1, 0.2, -0.2, -0.1, -0.2, -0.0, 0.2, -0.1, -0.1, -0.2, 0.2, 0.0, -0.1, 0.2, 0.1, -0.0, 0.1},
{0.1, -0.2, 0.2, 0.2, 0.1, -0.1, 0.1, -0.1, 0.0, -0.2, -0.0, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0, -0.1, 0.2, -0.2, -0.2, 0.1, 0.0, 0.1, 0.1},
{-0.0, 0.0, 0.1, 0.1, -0.1, -0.2, -0.1, 0.1, -0.2, 0.0, 0.1, -0.1, 0.1, 0.2, -0.1, 0.1, -0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.2, -0.2, 0.2},
{-0.1, -0.1, 0.0, 0.1, 0.2, 0.0, -0.0, 0.0, -0.1, -0.2, -0.1, 0.1, 0.0, 0.2, -0.0, 0.1, -0.1, -0.1, 0.1, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2},
{0.1, 0.0, 0.0, 0.2, 0.1, -0.1, -0.0, -0.1, -0.0, 0.0, -0.2, 0.1, 0.1, -0.1, 0.1, 0.1, -0.2, -0.1, 0.0, 0.1, -0.1, 0.1, 0.0, -0.1, -0.0},
{-0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.2, 0.0, -0.1, 0.2, -0.1, -0.1, -0.1, 0.1, -0.0, -0.0, 0.2, -0.2, -0.0, -0.2, -0.1, 0.2},
{0.1, 0.1, 0.2, -0.2, 0.2, -0.1, 0.2, 0.1, 0.1, 0.1, -0.1, -0.0, 0.2, 0.1, -0.0, -0.1, -0.2, 0.0, -0.0, 0.1, -0.1, 0.0, 0.1, 0.1, -0.1},
{0.0, 0.1, -0.0, 0.2, 0.2, -0.1, -0.1, 0.2, -0.2, 0.0, 0.2, -0.0, 0.1, 0.0, 0.2, -0.1, 0.0, 0.1, -0.1, 0.1, 0.1, 0.2, -0.1, 0.1, 0.2},
{-0.1, -0.2, 0.0, 0.2, 0.0, 0.1, -0.1, 0.1, -0.0, 0.1, 0.2, 0.1, -0.1, 0.0, 0.0, 0.0, -0.1, -0.2, -0.2, -0.0, -0.1, -0.1, 0.0, 0.2, 0.1},
{0.0, 0.1, 0.1, -0.1, -0.2, -0.0, -0.2, 0.1, -0.0, -0.2, -0.2, 0.0, 0.1, -0.1, -0.2, -0.1, -0.1, 0.2, 0.1, 0.1, -0.1, 0.0, 0.1, 0.0, 0.1},
{0.0, 0.2, 0.1, 0.0, 0.1, 0.0, 0.0, -0.1, 0.1, -0.1, 0.2, 0.2, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, -0.1, -0.0, -0.1, 0.2, 0.2, -0.0, 0.0},
}
);
Matrix  transformer_layers_2_attention_value_bias   (
{-0.2, 0.0, -0.0, 0.1, -0.0, -0.2, -0.2, 0.2, -0.1, 0.1, -0.1, 0.0, 0.1, 0.1, -0.1, -0.1, -0.0, -0.0, -0.1, -0.0, 0.1, 0.2, 0.1, 0.0, -0.1}
);
Matrix  transformer_layers_2_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_2_norm1_layer_norm_bias   (
{-0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_2_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_2_norm2_layer_norm_bias   (
{-0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_2_feed_forward_linear1_weight   (
{{0.1, -0.2, 0.0, -0.0, -0.2, 0.0, -0.2, -0.1, 0.2, -0.1, 0.1, -0.1, 0.1, -0.0, -0.0, -0.2, -0.0, 0.2, 0.2, -0.0, 0.0, 0.1, 0.0, -0.0, -0.0},
{0.1, -0.1, 0.1, 0.2, -0.0, 0.1, 0.1, -0.1, 0.0, -0.1, 0.1, 0.1, 0.2, 0.0, 0.1, -0.1, 0.1, 0.1, -0.2, 0.0, -0.1, -0.1, -0.2, -0.2, -0.1},
{0.2, 0.2, 0.0, -0.1, -0.1, -0.1, -0.1, 0.1, -0.1, -0.0, -0.2, -0.0, 0.1, -0.1, 0.2, -0.1, 0.2, -0.1, -0.1, 0.2, -0.2, -0.2, 0.1, -0.1, -0.1},
{0.0, -0.2, 0.1, -0.1, -0.1, -0.1, 0.2, 0.2, -0.0, 0.2, 0.0, -0.2, 0.0, -0.1, 0.1, 0.2, 0.1, 0.0, 0.0, 0.1, -0.2, -0.2, -0.0, 0.2, 0.1},
{-0.0, -0.1, 0.1, -0.1, 0.0, -0.1, -0.0, 0.0, -0.1, 0.2, 0.1, 0.1, 0.1, -0.1, -0.2, 0.0, -0.1, 0.1, -0.1, -0.1, 0.1, 0.0, -0.1, -0.2, 0.2},
{0.1, -0.1, -0.2, 0.2, 0.1, -0.0, 0.1, -0.2, -0.2, 0.1, -0.1, 0.1, -0.1, 0.2, 0.1, -0.2, 0.2, 0.1, -0.1, -0.1, -0.1, 0.2, 0.1, 0.1, -0.2},
{-0.1, -0.2, -0.1, -0.1, -0.1, 0.0, -0.2, 0.0, 0.1, -0.2, 0.2, -0.2, 0.1, -0.1, 0.1, 0.0, -0.1, -0.0, -0.0, -0.2, -0.2, -0.0, -0.0, 0.0, 0.1},
{0.0, 0.1, 0.1, -0.0, 0.1, 0.1, -0.0, 0.2, 0.1, 0.1, -0.2, 0.1, -0.1, 0.1, -0.1, 0.2, -0.1, 0.1, 0.2, 0.1, 0.1, 0.2, -0.1, -0.2, -0.0},
{-0.1, -0.1, -0.0, -0.2, 0.1, -0.0, -0.1, 0.0, 0.2, 0.0, 0.0, 0.1, 0.2, -0.1, 0.2, 0.1, -0.0, 0.2, -0.1, 0.1, 0.1, 0.2, 0.0, 0.2, 0.2},
{0.1, -0.1, 0.1, 0.1, -0.0, 0.2, 0.1, 0.1, -0.0, 0.2, 0.0, -0.1, -0.1, 0.2, -0.1, -0.1, -0.0, -0.1, 0.0, -0.1, -0.1, -0.0, -0.1, 0.2, -0.0},
{-0.2, -0.2, 0.1, 0.2, 0.1, -0.1, 0.0, -0.0, 0.0, 0.0, 0.1, -0.0, 0.2, 0.1, 0.0, 0.2, -0.2, -0.2, 0.2, -0.2, 0.0, 0.2, 0.1, 0.1, -0.1},
{0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.0, -0.2, -0.2, -0.1, 0.1, 0.1, 0.1, 0.1, -0.2, -0.1, 0.1, -0.0, 0.0, -0.0, 0.1, 0.0, -0.1, 0.1, -0.1},
{0.0, 0.1, -0.0, -0.1, 0.1, -0.1, 0.0, -0.1, 0.0, 0.0, -0.1, -0.0, -0.1, 0.2, 0.2, -0.0, 0.1, 0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.0, 0.2},
{-0.0, -0.1, -0.0, -0.1, -0.1, -0.1, 0.1, -0.0, -0.1, -0.0, 0.1, 0.2, 0.1, -0.1, -0.0, 0.2, 0.1, -0.2, 0.0, -0.2, -0.1, -0.1, 0.1, -0.1, 0.1},
{-0.1, 0.1, -0.1, -0.2, -0.2, -0.0, 0.2, -0.1, 0.2, 0.1, 0.1, -0.0, -0.0, 0.0, -0.1, 0.0, -0.2, 0.2, -0.0, 0.1, 0.2, 0.0, -0.1, 0.2, -0.2},
}
);
Matrix  transformer_layers_2_feed_forward_linear1_bias   (
{-0.2, 0.1, -0.1, 0.1, 0.1, -0.2, 0.1, -0.0, -0.1, -0.2, 0.0, -0.1, 0.1, -0.1, 0.2}
);
Matrix  transformer_layers_2_feed_forward_linear2_weight   (
{{-0.3, -0.2, 0.1, -0.2, 0.1, 0.1, -0.2, -0.2, -0.2, 0.2, 0.2, 0.2, 0.2, -0.1, 0.1},
{0.2, -0.2, -0.2, -0.2, -0.2, 0.2, 0.1, 0.1, -0.1, 0.2, 0.0, 0.0, -0.2, 0.0, 0.1},
{-0.2, -0.0, 0.1, 0.3, -0.1, 0.2, 0.2, 0.1, 0.2, 0.0, -0.1, 0.0, 0.2, -0.2, 0.2},
{-0.1, 0.0, 0.2, -0.2, -0.1, 0.1, -0.1, -0.2, -0.2, -0.0, -0.1, 0.1, 0.0, -0.2, -0.1},
{-0.2, -0.1, -0.1, 0.0, -0.1, -0.1, -0.0, -0.1, -0.1, -0.0, -0.1, 0.1, 0.0, 0.1, 0.1},
{-0.0, -0.2, 0.1, -0.2, -0.1, -0.1, 0.2, -0.1, -0.2, 0.1, 0.1, -0.1, -0.2, 0.0, 0.1},
{-0.2, 0.1, 0.1, 0.1, -0.2, -0.1, 0.1, 0.3, -0.2, 0.1, -0.1, 0.2, -0.1, 0.2, 0.1},
{0.0, 0.1, 0.0, 0.0, -0.2, -0.1, 0.0, 0.1, -0.3, 0.2, -0.1, 0.2, -0.1, -0.1, 0.1},
{0.2, -0.1, -0.1, -0.2, -0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.1, 0.2, -0.2, 0.0, 0.2},
{-0.1, -0.2, -0.0, 0.1, -0.0, 0.3, -0.1, 0.2, 0.2, 0.1, -0.1, 0.1, -0.0, -0.1, 0.1},
{-0.2, 0.2, 0.2, -0.1, 0.1, 0.0, -0.1, 0.0, -0.1, -0.2, 0.0, -0.2, -0.2, -0.2, -0.2},
{-0.2, -0.2, 0.2, 0.1, 0.3, -0.1, 0.0, 0.2, 0.0, -0.2, 0.1, 0.2, 0.2, -0.2, 0.1},
{-0.0, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, -0.0, 0.2, -0.1, 0.1, 0.0, -0.1, -0.1},
{0.0, 0.2, 0.2, 0.1, -0.0, 0.1, -0.1, 0.2, 0.1, 0.1, 0.0, -0.1, 0.1, 0.1, 0.0},
{-0.1, 0.2, -0.2, 0.2, -0.0, 0.1, 0.2, -0.2, 0.2, 0.2, 0.1, -0.2, -0.2, 0.2, 0.2},
{0.2, -0.0, -0.1, 0.1, -0.1, -0.0, 0.2, 0.1, -0.1, -0.1, 0.2, 0.2, -0.1, 0.1, 0.1},
{0.1, -0.2, -0.1, 0.1, 0.1, 0.0, -0.2, 0.2, -0.0, -0.2, 0.2, 0.0, 0.0, 0.1, -0.0},
{-0.0, -0.0, -0.2, 0.1, -0.1, -0.0, -0.2, -0.1, 0.3, 0.2, -0.0, -0.0, 0.1, -0.2, 0.1},
{-0.2, 0.2, -0.2, -0.1, 0.1, -0.0, -0.1, 0.2, 0.1, -0.0, 0.2, -0.2, -0.2, 0.0, 0.1},
{-0.1, -0.1, 0.1, -0.0, 0.0, -0.0, 0.2, 0.0, 0.1, -0.2, -0.2, 0.1, -0.0, 0.2, -0.2},
{0.2, -0.1, -0.2, 0.2, 0.2, -0.1, 0.1, -0.2, -0.1, 0.2, 0.0, 0.0, 0.2, -0.0, 0.1},
{0.0, 0.1, 0.2, -0.1, 0.1, -0.1, -0.2, -0.2, 0.2, 0.0, -0.0, -0.1, 0.1, -0.3, 0.0},
{0.2, -0.1, -0.2, -0.1, 0.1, 0.2, 0.0, -0.2, -0.1, -0.0, 0.1, -0.1, 0.2, 0.2, 0.0},
{-0.1, 0.2, -0.1, -0.2, -0.2, 0.1, 0.1, 0.0, -0.0, 0.1, -0.3, 0.1, 0.3, -0.0, -0.1},
{0.1, 0.1, 0.0, 0.3, -0.1, -0.0, -0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.2, 0.0, -0.2},
}
);
Matrix  transformer_layers_2_feed_forward_linear2_bias   (
{0.0, -0.2, -0.1, 0.2, -0.0, -0.2, -0.2, -0.1, -0.2, -0.1, -0.0, 0.1, 0.2, -0.0, 0.1, -0.3, 0.2, 0.2, -0.1, 0.2, 0.0, -0.1, -0.3, -0.0, -0.0}
);
Matrix  transformer_layers_2_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_2_feed_forward_ln1_layer_norm_bias   (
{0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_2_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_2_feed_forward_ln2_layer_norm_bias   (
{-0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_3_attention_query_weight   (
{{0.1, 0.0, 0.2, 0.1, -0.1, -0.0, -0.0, -0.1, -0.1, -0.0, -0.1, -0.2, -0.0, -0.0, -0.1, -0.0, -0.2, 0.1, -0.0, -0.1, 0.1, -0.0, -0.0, -0.1, 0.2},
{-0.0, 0.0, -0.1, 0.2, 0.1, -0.2, 0.0, 0.0, -0.1, 0.0, 0.2, -0.1, 0.2, -0.1, 0.1, -0.1, 0.1, -0.0, -0.1, 0.1, -0.1, 0.1, 0.0, -0.0, -0.2},
{-0.1, 0.0, 0.1, -0.0, -0.1, 0.2, -0.1, 0.2, 0.1, 0.0, 0.0, 0.1, -0.1, -0.2, -0.2, -0.0, 0.1, 0.1, 0.0, -0.2, -0.2, -0.1, 0.2, 0.0, 0.2},
{0.1, -0.2, -0.2, 0.2, -0.2, -0.2, 0.2, -0.1, 0.2, 0.1, 0.2, -0.1, -0.0, 0.2, 0.0, 0.1, 0.2, 0.2, 0.2, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1},
{-0.0, 0.1, 0.0, 0.1, -0.1, -0.1, -0.1, -0.2, -0.0, -0.1, 0.2, 0.2, -0.1, -0.0, 0.2, -0.0, -0.1, -0.0, -0.1, 0.0, 0.0, -0.1, 0.1, -0.1, -0.1},
{0.1, -0.1, -0.1, 0.2, 0.0, 0.0, 0.1, 0.0, -0.0, 0.1, -0.1, 0.1, -0.1, 0.2, 0.2, -0.1, -0.2, -0.0, 0.1, -0.0, 0.0, 0.1, 0.1, 0.1, -0.0},
{0.1, 0.1, -0.2, 0.1, -0.0, 0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.0, -0.2, -0.1, -0.1, -0.1, 0.1, -0.1, -0.1, -0.2, -0.1, -0.0, 0.1, 0.1},
{-0.1, -0.2, 0.1, -0.1, 0.2, -0.1, -0.1, -0.2, -0.2, 0.1, -0.0, -0.1, -0.1, -0.1, 0.2, -0.2, -0.1, 0.2, 0.1, -0.1, 0.0, -0.1, -0.2, 0.1, -0.1},
{-0.2, 0.0, 0.0, 0.2, -0.1, -0.0, -0.1, -0.1, 0.0, -0.0, 0.2, 0.1, -0.2, 0.1, 0.1, 0.1, -0.1, 0.1, 0.0, 0.2, 0.1, 0.1, -0.0, -0.1, 0.0},
{0.2, -0.0, -0.1, 0.0, -0.1, -0.0, 0.1, -0.2, 0.1, -0.1, 0.2, -0.1, 0.1, -0.0, 0.1, -0.1, 0.2, -0.0, -0.2, 0.0, -0.1, 0.0, -0.2, -0.0, -0.2},
{-0.0, -0.2, -0.1, 0.0, -0.1, -0.2, 0.2, -0.0, 0.2, -0.0, -0.2, -0.0, -0.1, 0.1, -0.0, 0.1, 0.0, -0.2, 0.1, -0.1, -0.0, 0.1, -0.1, 0.2, 0.1},
{0.1, -0.1, -0.1, 0.2, 0.1, 0.0, 0.1, 0.1, -0.0, 0.1, -0.1, 0.1, -0.1, 0.2, -0.2, -0.2, -0.2, 0.1, 0.2, -0.1, -0.1, -0.1, -0.0, -0.2, 0.1},
{-0.2, 0.0, -0.2, 0.2, -0.1, -0.1, -0.0, 0.1, -0.0, 0.1, -0.0, -0.1, 0.1, 0.1, -0.0, 0.1, -0.1, 0.0, -0.2, -0.0, 0.2, 0.2, 0.2, 0.1, -0.1},
{-0.0, -0.1, 0.2, 0.1, 0.0, 0.2, -0.1, 0.2, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.0, 0.1, -0.1, 0.1, -0.2, 0.1, 0.1, -0.2, -0.2, -0.2, 0.1},
{0.1, 0.1, -0.2, -0.1, 0.1, -0.1, -0.1, -0.2, -0.1, -0.0, -0.1, 0.2, -0.1, 0.2, -0.2, -0.0, -0.0, 0.2, 0.1, 0.1, -0.1, -0.0, -0.1, 0.1, 0.2},
{-0.0, 0.1, -0.1, 0.2, 0.2, 0.1, -0.2, 0.1, -0.0, -0.2, 0.1, 0.1, 0.1, -0.2, 0.2, 0.0, -0.0, 0.1, 0.1, 0.0, -0.1, -0.2, -0.1, -0.1, 0.2},
{-0.1, 0.1, 0.0, -0.0, 0.0, -0.2, -0.1, 0.2, 0.1, 0.1, -0.1, -0.2, -0.0, 0.0, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.1, -0.2, -0.2, -0.2, 0.2},
{0.1, -0.1, 0.1, 0.1, 0.1, -0.1, -0.0, -0.0, 0.0, -0.1, 0.2, -0.0, -0.0, -0.1, -0.2, -0.1, -0.1, -0.1, 0.1, 0.2, -0.1, -0.2, -0.1, -0.1, 0.1},
{-0.1, -0.1, 0.2, -0.1, -0.2, -0.0, 0.0, -0.2, -0.2, -0.1, 0.2, 0.0, -0.1, 0.2, 0.0, -0.0, 0.1, -0.1, -0.1, -0.1, -0.2, -0.1, 0.2, 0.0, 0.0},
{-0.1, -0.2, 0.1, -0.1, -0.2, 0.1, 0.2, 0.2, -0.0, 0.1, -0.1, 0.1, -0.1, 0.0, 0.0, -0.2, 0.2, -0.2, -0.0, -0.0, 0.2, 0.1, -0.2, 0.0, 0.0},
{0.1, 0.2, 0.2, 0.1, 0.0, 0.1, -0.2, 0.2, -0.2, 0.2, -0.1, -0.1, -0.0, 0.1, 0.2, -0.2, 0.1, 0.2, -0.1, -0.0, -0.2, 0.1, -0.1, -0.0, -0.0},
{0.0, 0.2, 0.2, -0.1, -0.1, -0.0, 0.0, -0.0, 0.1, 0.1, -0.0, 0.0, 0.1, -0.0, 0.0, -0.1, 0.0, 0.1, -0.1, 0.0, 0.0, -0.2, -0.0, 0.1, -0.2},
{0.1, -0.2, 0.1, 0.1, 0.2, -0.1, -0.1, -0.1, -0.0, 0.0, -0.2, -0.1, 0.2, 0.1, -0.0, 0.1, 0.0, 0.1, -0.2, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1},
{0.1, 0.1, 0.0, -0.0, 0.1, 0.2, -0.1, -0.2, 0.1, -0.0, 0.0, -0.1, 0.2, 0.2, -0.2, -0.2, -0.1, 0.1, 0.1, 0.0, 0.2, -0.0, 0.0, 0.1, 0.0},
{-0.2, -0.2, 0.0, -0.0, 0.1, -0.1, -0.0, -0.2, 0.1, -0.0, 0.1, 0.1, 0.1, -0.2, 0.2, 0.2, -0.2, 0.0, -0.0, -0.1, 0.1, 0.2, -0.1, -0.1, -0.0},
}
);
Matrix  transformer_layers_3_attention_query_bias   (
{-0.2, -0.0, -0.0, -0.2, -0.1, 0.1, -0.1, 0.0, 0.0, 0.0, -0.1, -0.2, 0.0, -0.0, 0.0, 0.1, 0.2, 0.1, 0.2, 0.1, -0.2, -0.2, 0.1, 0.1, -0.1}
);
Matrix  transformer_layers_3_attention_key_weight   (
{{-0.1, 0.1, -0.0, -0.1, 0.2, -0.1, 0.2, 0.0, -0.2, 0.1, 0.1, 0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.1, -0.2, -0.2, -0.1, 0.1, 0.1, -0.1, 0.0},
{0.1, -0.2, -0.1, 0.0, -0.2, -0.0, -0.1, 0.0, 0.0, -0.0, 0.0, -0.1, -0.0, 0.2, 0.2, 0.0, 0.1, 0.1, -0.1, 0.2, -0.0, -0.2, -0.1, -0.2, -0.0},
{-0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.2, 0.1, 0.0, -0.2, 0.1, 0.1, 0.2, -0.2, -0.2, 0.2, 0.0, 0.1, -0.1, 0.1, 0.2, 0.0},
{0.0, 0.1, 0.0, -0.0, 0.2, -0.1, -0.2, 0.1, 0.1, 0.2, 0.0, -0.1, 0.1, 0.0, 0.0, 0.0, 0.1, -0.0, -0.1, -0.2, -0.1, -0.0, 0.2, 0.2, -0.1},
{0.1, -0.2, 0.1, 0.0, -0.1, 0.1, -0.2, 0.0, 0.0, -0.1, -0.2, 0.1, -0.1, -0.1, -0.2, -0.1, -0.0, 0.1, 0.2, -0.2, -0.0, -0.1, 0.1, 0.2, 0.2},
{0.2, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.0, 0.2, -0.1, 0.1, 0.0, 0.1, -0.1, -0.1, -0.1, -0.0, 0.0, -0.2, 0.1, -0.1, -0.2, -0.2, -0.1, 0.1},
{0.0, 0.1, 0.2, -0.1, 0.0, -0.1, -0.2, 0.1, -0.2, 0.1, 0.0, 0.1, 0.1, -0.1, -0.1, -0.1, 0.1, -0.2, -0.1, -0.0, 0.0, -0.0, -0.1, -0.1, -0.1},
{-0.2, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.2, -0.2, -0.2, 0.2, -0.1, -0.2, -0.1, 0.1, 0.0, -0.0, 0.1, -0.1, -0.2, -0.1, 0.2, -0.1},
{0.0, 0.1, 0.1, 0.1, -0.2, -0.2, -0.2, -0.1, 0.1, -0.0, -0.1, 0.2, 0.1, -0.1, -0.2, -0.0, 0.2, -0.1, -0.1, 0.2, 0.2, 0.1, 0.2, -0.1, 0.1},
{-0.1, -0.1, 0.1, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.0, 0.2, -0.0, -0.0, 0.1, -0.0, 0.2, -0.1, -0.1, 0.1, -0.1, 0.1, 0.0, 0.1, -0.1, -0.1},
{-0.0, -0.2, -0.0, -0.1, 0.2, 0.1, -0.1, -0.1, -0.0, -0.1, -0.2, 0.2, -0.1, 0.1, 0.0, -0.1, 0.0, 0.0, -0.0, -0.1, 0.2, -0.1, 0.2, -0.1, -0.0},
{0.0, 0.2, 0.1, -0.0, 0.2, -0.2, 0.2, -0.0, -0.2, -0.0, 0.1, -0.1, -0.0, -0.0, -0.2, -0.1, 0.1, 0.0, -0.0, -0.0, -0.1, 0.2, -0.1, -0.1, 0.2},
{-0.1, -0.1, 0.1, -0.0, 0.1, -0.2, 0.2, 0.1, -0.1, 0.2, 0.1, -0.0, -0.1, 0.0, -0.1, -0.1, -0.0, 0.2, 0.1, -0.1, 0.1, 0.1, -0.0, -0.1, -0.0},
{-0.1, -0.2, -0.1, -0.1, 0.0, 0.0, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.2, 0.1, 0.1, -0.0, -0.1, -0.0, 0.1, 0.1, -0.1, 0.0, 0.1, -0.0, 0.1},
{-0.1, 0.0, 0.1, -0.0, 0.1, -0.0, 0.1, -0.0, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1, -0.1, -0.0, 0.1, -0.1, -0.2, -0.2, 0.0, 0.1},
{0.2, 0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1, -0.0, -0.1, -0.1, 0.2, -0.1, -0.2, -0.0, -0.2, 0.0, 0.1, -0.0, -0.1, -0.2, 0.0, -0.1, -0.1},
{0.1, 0.0, -0.1, -0.0, -0.1, -0.1, 0.2, -0.0, -0.2, 0.2, 0.1, 0.1, -0.1, 0.1, 0.2, 0.2, -0.2, 0.2, 0.0, 0.1, -0.2, -0.2, -0.2, -0.2, 0.2},
{0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.2, -0.2, -0.1, -0.1, 0.2, 0.1, -0.1, 0.1, -0.1, -0.2, -0.1, -0.2, -0.2, -0.1, -0.1, 0.2, 0.0, -0.0},
{-0.1, -0.0, 0.0, 0.0, 0.1, -0.1, 0.0, -0.0, -0.2, 0.0, -0.0, -0.2, -0.2, 0.2, 0.2, -0.0, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.0, 0.0, -0.0},
{-0.1, -0.0, -0.1, 0.1, 0.2, 0.1, -0.2, -0.2, 0.0, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.2, -0.2, 0.1, -0.0, -0.1, 0.2},
{-0.0, 0.2, 0.2, 0.0, 0.2, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.2, 0.1, 0.0, 0.1, 0.0, -0.0, -0.1, 0.1, -0.0, -0.0, 0.0, 0.1, 0.1, -0.1},
{-0.1, 0.2, -0.0, 0.1, -0.0, 0.1, -0.0, 0.1, -0.1, -0.2, 0.1, -0.1, -0.1, -0.0, -0.1, -0.0, -0.1, 0.0, -0.2, 0.0, -0.0, 0.1, 0.1, 0.0, -0.1},
{-0.0, 0.0, -0.2, -0.2, -0.1, -0.1, -0.0, 0.1, 0.1, 0.0, 0.2, 0.0, 0.0, -0.0, 0.1, 0.2, -0.1, 0.2, -0.0, 0.2, 0.2, 0.2, -0.1, 0.1, -0.2},
{0.1, 0.1, -0.0, 0.2, 0.1, -0.1, 0.1, 0.1, -0.0, 0.1, 0.1, 0.2, 0.0, 0.1, 0.0, 0.2, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.2, 0.2, -0.2},
{0.0, -0.0, 0.1, 0.1, 0.2, -0.1, -0.0, -0.0, -0.2, -0.1, -0.1, -0.0, 0.1, 0.2, -0.2, -0.0, -0.1, -0.1, -0.0, 0.1, 0.2, -0.1, 0.1, -0.0, -0.1},
}
);
Matrix  transformer_layers_3_attention_key_bias   (
{-0.1, -0.0, 0.0, -0.2, -0.0, 0.1, 0.2, 0.0, 0.0, -0.1, -0.1, -0.2, -0.2, 0.2, 0.2, -0.0, 0.0, -0.2, -0.1, -0.0, 0.0, 0.0, 0.1, -0.2, -0.1}
);
Matrix  transformer_layers_3_attention_value_weight   (
{{0.1, -0.1, 0.0, 0.2, -0.1, 0.2, 0.1, 0.1, 0.0, 0.0, 0.0, -0.1, 0.1, -0.0, -0.0, -0.1, -0.0, 0.2, 0.0, -0.1, -0.1, -0.0, -0.1, 0.1, -0.1},
{0.2, -0.1, 0.2, 0.1, -0.0, 0.1, 0.1, 0.0, -0.0, 0.1, -0.1, -0.1, 0.0, 0.2, -0.2, 0.0, -0.1, -0.1, -0.0, 0.2, -0.2, 0.1, -0.1, 0.1, 0.1},
{-0.0, 0.1, -0.2, 0.1, -0.0, -0.1, -0.0, 0.1, 0.0, 0.0, -0.1, 0.2, -0.1, 0.1, 0.0, 0.1, -0.1, -0.1, -0.0, 0.1, 0.1, 0.2, -0.0, -0.1, 0.2},
{0.1, 0.1, -0.1, 0.0, -0.2, 0.1, -0.1, -0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, 0.2, 0.2, 0.2, 0.1, -0.1, -0.0, -0.0, -0.2, 0.1, -0.1, 0.1},
{-0.1, -0.2, 0.1, -0.1, -0.1, 0.1, 0.0, 0.1, 0.1, -0.1, 0.1, -0.1, 0.2, 0.1, -0.2, -0.0, -0.1, -0.1, -0.0, -0.0, -0.2, 0.1, -0.0, 0.1, 0.0},
{-0.1, 0.0, -0.1, -0.1, -0.0, -0.1, 0.1, 0.0, 0.1, 0.2, -0.1, -0.1, 0.0, 0.1, 0.1, 0.1, -0.0, 0.1, 0.1, 0.1, -0.1, -0.0, 0.1, 0.1, 0.2},
{-0.1, -0.2, 0.0, 0.1, -0.1, -0.0, 0.1, -0.2, 0.0, -0.1, -0.1, 0.1, 0.2, 0.0, 0.0, -0.1, 0.0, -0.0, 0.0, -0.1, -0.0, -0.2, 0.0, -0.2, -0.2},
{-0.1, 0.2, -0.1, -0.1, 0.2, -0.2, -0.1, 0.2, -0.1, -0.1, 0.2, 0.0, 0.1, 0.2, 0.1, -0.1, 0.1, -0.1, 0.1, 0.0, -0.1, -0.1, 0.1, -0.2, -0.0},
{0.1, -0.0, 0.1, -0.0, 0.0, 0.1, 0.2, 0.1, 0.2, 0.2, 0.2, -0.2, 0.0, -0.0, -0.0, -0.2, 0.2, 0.0, -0.0, 0.2, -0.0, -0.1, 0.1, 0.0, -0.1},
{-0.1, -0.2, -0.1, -0.1, -0.0, -0.1, -0.2, 0.1, -0.1, 0.2, 0.0, 0.1, -0.0, -0.2, -0.1, -0.1, -0.0, 0.2, -0.0, -0.1, -0.1, -0.0, 0.2, -0.0, 0.2},
{0.0, 0.0, -0.0, -0.0, -0.1, 0.2, -0.0, 0.2, -0.1, -0.1, 0.1, 0.0, -0.2, -0.0, -0.1, 0.0, 0.2, 0.0, -0.1, 0.1, 0.0, -0.1, -0.0, -0.1, -0.2},
{-0.1, 0.2, -0.1, 0.0, -0.1, -0.1, 0.1, -0.1, 0.0, -0.1, 0.0, -0.1, 0.1, 0.1, -0.1, 0.1, 0.2, 0.2, -0.0, -0.1, -0.0, 0.2, -0.1, -0.1, 0.2},
{0.2, -0.1, 0.1, 0.1, 0.2, 0.0, -0.0, -0.1, 0.1, 0.0, -0.1, 0.0, -0.0, -0.0, -0.0, -0.1, -0.0, 0.1, -0.1, 0.1, -0.1, -0.0, 0.0, -0.1, 0.0},
{-0.1, 0.0, -0.0, 0.0, -0.1, 0.1, -0.1, 0.2, -0.1, -0.1, -0.1, 0.0, 0.2, 0.1, -0.1, 0.0, -0.0, -0.2, 0.0, 0.0, 0.0, 0.2, 0.0, -0.2, -0.2},
{0.0, 0.1, 0.1, -0.0, 0.0, -0.1, 0.0, 0.0, -0.0, -0.0, 0.1, 0.2, 0.0, 0.2, -0.2, -0.1, -0.0, -0.1, -0.2, -0.1, 0.0, -0.1, -0.1, 0.1, -0.0},
{0.1, -0.0, 0.1, -0.2, -0.0, -0.1, -0.2, -0.1, -0.0, 0.2, -0.2, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, -0.0, -0.2, 0.1, 0.0, 0.1, -0.2},
{-0.1, -0.2, -0.2, 0.1, 0.0, 0.2, 0.1, -0.1, -0.0, 0.1, 0.0, -0.1, 0.1, 0.2, -0.2, 0.2, -0.1, 0.1, 0.0, 0.2, 0.2, 0.2, 0.1, -0.0, 0.1},
{0.1, 0.2, 0.0, -0.0, 0.0, 0.2, 0.1, 0.2, 0.1, -0.1, -0.1, -0.2, 0.2, -0.1, -0.0, -0.1, 0.2, 0.1, -0.1, -0.2, 0.2, 0.1, 0.1, -0.2, 0.1},
{-0.2, 0.1, -0.1, 0.1, 0.1, -0.0, -0.1, 0.1, 0.0, -0.1, -0.1, -0.0, -0.0, 0.2, 0.2, -0.2, -0.2, -0.2, -0.0, 0.2, 0.2, -0.0, -0.1, -0.1, -0.0},
{-0.2, 0.0, -0.2, -0.2, -0.0, -0.2, -0.0, 0.2, -0.0, 0.1, 0.1, 0.1, 0.0, -0.2, 0.0, 0.2, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.0, -0.2, 0.1},
{-0.0, 0.1, 0.1, -0.2, -0.2, 0.1, 0.0, -0.2, -0.0, -0.1, -0.1, 0.0, 0.0, 0.1, 0.1, -0.2, -0.1, 0.1, 0.2, 0.1, 0.0, -0.2, -0.2, 0.1, 0.1},
{-0.1, -0.0, -0.1, -0.1, -0.2, 0.1, -0.0, 0.1, -0.0, 0.1, -0.1, -0.1, 0.0, 0.0, -0.0, -0.1, -0.2, 0.0, -0.1, -0.0, 0.0, 0.2, 0.2, -0.1, 0.1},
{-0.1, -0.0, 0.1, 0.0, -0.0, 0.1, 0.1, -0.0, -0.2, 0.0, -0.0, -0.1, -0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0, 0.2, -0.0, -0.1, 0.1, -0.1, 0.1},
{-0.2, -0.1, 0.1, -0.0, 0.2, -0.1, 0.1, 0.0, 0.1, 0.1, 0.0, -0.2, -0.0, -0.0, 0.2, 0.2, 0.1, 0.2, -0.2, 0.2, 0.1, -0.1, -0.0, 0.1, -0.1},
{0.1, -0.1, 0.2, -0.1, 0.1, -0.2, -0.1, -0.1, 0.1, -0.2, -0.2, -0.0, -0.1, -0.1, -0.2, 0.1, -0.1, -0.2, -0.2, -0.1, 0.1, -0.2, 0.2, 0.2, 0.2},
}
);
Matrix  transformer_layers_3_attention_value_bias   (
{-0.1, -0.0, 0.0, -0.1, -0.1, 0.1, -0.2, 0.2, -0.0, 0.1, 0.0, 0.1, -0.1, -0.0, -0.2, 0.1, -0.0, -0.2, 0.1, -0.1, -0.1, -0.1, 0.2, -0.0, 0.1}
);
Matrix  transformer_layers_3_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_3_norm1_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_3_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_3_norm2_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0}
);
Matrix  transformer_layers_3_feed_forward_linear1_weight   (
{{-0.0, 0.2, 0.2, -0.1, 0.1, -0.2, 0.1, 0.0, 0.1, -0.1, -0.1, -0.0, 0.1, -0.2, -0.0, 0.1, -0.1, 0.0, 0.0, -0.0, 0.1, -0.1, -0.1, -0.0, 0.2},
{-0.2, 0.2, 0.1, -0.1, -0.2, 0.1, 0.1, -0.1, -0.1, -0.1, 0.2, 0.0, -0.1, 0.1, -0.2, -0.1, -0.2, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, -0.0},
{-0.1, 0.1, 0.0, 0.1, 0.1, 0.2, -0.1, -0.1, -0.0, -0.0, 0.1, -0.0, -0.1, -0.1, 0.1, 0.2, -0.0, -0.1, 0.1, 0.1, -0.0, 0.2, -0.1, 0.2, 0.2},
{0.2, -0.1, -0.1, 0.0, -0.0, 0.0, -0.0, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.0, -0.1, -0.2, -0.1, -0.2, 0.1, 0.1},
{0.1, -0.0, 0.2, 0.0, 0.0, -0.0, 0.2, 0.0, 0.2, -0.0, -0.1, 0.1, 0.0, 0.1, -0.2, -0.1, 0.1, -0.0, -0.0, -0.0, -0.2, 0.2, -0.1, 0.0, -0.0},
{0.1, 0.1, -0.1, -0.0, 0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.0, 0.0, 0.1, 0.0, -0.2, 0.1, -0.0, -0.1, 0.1, 0.1, -0.2, 0.0, 0.1, -0.1, 0.2},
{0.0, 0.1, -0.1, 0.0, 0.0, 0.1, 0.0, -0.1, 0.1, 0.2, -0.1, -0.1, -0.1, 0.1, -0.2, 0.0, -0.1, 0.0, 0.0, -0.1, -0.0, -0.2, 0.1, -0.1, -0.2},
{-0.0, -0.1, -0.1, 0.2, 0.2, 0.2, 0.1, -0.1, 0.0, -0.0, 0.1, -0.0, -0.2, -0.1, 0.0, 0.2, 0.1, 0.2, -0.1, 0.0, 0.1, -0.1, -0.0, 0.0, 0.1},
{0.1, 0.1, -0.2, -0.0, -0.0, -0.0, -0.1, -0.2, -0.1, -0.0, -0.1, -0.1, -0.0, 0.0, -0.1, 0.2, 0.0, 0.1, 0.1, -0.1, 0.2, -0.1, -0.2, 0.0, 0.1},
{-0.1, -0.2, 0.1, -0.1, 0.1, 0.1, 0.0, -0.2, 0.2, 0.1, 0.0, 0.1, 0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, -0.1, 0.1},
{0.1, -0.1, 0.1, 0.2, -0.0, 0.0, -0.0, 0.0, 0.0, 0.1, -0.1, -0.2, -0.1, -0.1, 0.0, -0.1, -0.1, 0.0, 0.0, 0.2, 0.2, -0.1, 0.1, 0.1, -0.0},
{0.2, 0.1, 0.0, 0.2, -0.0, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, 0.0, -0.0, 0.1, -0.2, -0.1, -0.0, 0.2, 0.2, -0.1, -0.1, 0.2, -0.1, 0.1, -0.1},
{-0.1, -0.1, -0.1, -0.1, 0.2, -0.2, -0.0, 0.2, -0.1, 0.2, 0.1, -0.1, 0.1, -0.1, -0.2, 0.1, -0.2, 0.2, -0.1, 0.1, -0.1, -0.2, 0.1, -0.1, -0.1},
{-0.0, -0.0, 0.1, -0.2, 0.0, -0.2, 0.1, 0.1, 0.0, -0.0, -0.2, 0.1, 0.1, -0.1, 0.1, 0.1, 0.0, 0.1, 0.2, 0.1, -0.2, 0.2, 0.1, -0.0, 0.1},
{0.1, 0.2, -0.2, -0.0, -0.1, 0.1, -0.0, 0.1, -0.0, 0.2, 0.1, -0.1, -0.0, 0.1, 0.1, -0.1, 0.2, -0.2, 0.2, 0.2, -0.1, -0.1, -0.1, 0.2, -0.1},
}
);
Matrix  transformer_layers_3_feed_forward_linear1_bias   (
{-0.1, 0.0, 0.0, -0.1, -0.0, 0.1, -0.2, 0.1, -0.2, -0.2, 0.2, -0.2, -0.1, -0.0, 0.0}
);
Matrix  transformer_layers_3_feed_forward_linear2_weight   (
{{-0.2, -0.2, 0.2, -0.1, 0.2, 0.3, 0.0, 0.1, -0.2, -0.0, 0.1, 0.2, -0.1, 0.2, -0.1},
{0.0, -0.1, 0.0, -0.1, 0.0, -0.2, -0.2, 0.0, 0.0, 0.2, 0.2, -0.1, 0.0, -0.1, 0.1},
{0.0, -0.2, 0.2, -0.2, -0.1, -0.1, 0.1, -0.2, -0.3, -0.2, 0.0, 0.2, -0.0, 0.1, -0.0},
{0.2, 0.0, -0.0, -0.1, 0.0, -0.2, -0.1, -0.2, -0.2, 0.1, 0.1, -0.2, 0.0, 0.1, -0.1},
{-0.2, 0.2, -0.0, 0.1, 0.2, 0.2, -0.1, 0.0, -0.1, 0.0, -0.2, -0.2, 0.2, 0.1, -0.2},
{0.0, -0.1, 0.1, 0.3, 0.3, -0.1, -0.1, -0.1, 0.2, 0.1, 0.1, -0.2, -0.2, 0.0, -0.1},
{0.1, -0.0, 0.2, 0.1, -0.2, -0.1, 0.1, 0.0, 0.1, -0.2, 0.2, -0.0, -0.1, -0.2, -0.2},
{0.2, 0.1, -0.2, -0.1, 0.1, 0.1, 0.0, -0.0, -0.0, 0.2, -0.1, -0.1, 0.2, -0.0, -0.2},
{0.1, 0.1, 0.0, 0.2, -0.2, 0.1, 0.2, 0.0, 0.2, -0.1, 0.1, -0.1, 0.0, 0.1, -0.2},
{0.2, 0.2, -0.2, 0.1, -0.2, 0.2, -0.2, -0.2, 0.2, -0.1, 0.1, 0.2, -0.2, 0.2, 0.2},
{-0.1, -0.2, 0.0, 0.1, 0.1, 0.1, -0.0, -0.2, 0.1, -0.1, 0.2, 0.0, -0.2, -0.3, -0.2},
{-0.1, -0.0, -0.2, -0.2, -0.2, -0.1, 0.1, 0.1, 0.2, -0.2, -0.2, -0.2, 0.0, 0.1, -0.0},
{-0.2, 0.1, 0.2, -0.2, 0.0, -0.1, 0.1, 0.2, 0.1, -0.2, 0.1, 0.0, -0.2, -0.1, -0.2},
{0.2, -0.2, -0.1, 0.1, 0.2, -0.1, -0.2, 0.1, 0.1, 0.1, -0.0, 0.0, -0.1, -0.0, -0.0},
{0.2, 0.2, -0.1, 0.2, -0.2, -0.2, -0.0, 0.1, 0.1, -0.2, -0.1, -0.0, 0.2, 0.1, 0.2},
{0.0, 0.1, -0.2, 0.2, 0.2, 0.2, 0.1, 0.0, 0.2, -0.1, 0.1, 0.2, -0.2, -0.1, -0.2},
{0.2, -0.2, 0.2, 0.1, -0.1, -0.2, 0.1, -0.1, 0.0, -0.1, -0.1, -0.2, 0.0, -0.1, -0.1},
{0.1, -0.0, 0.0, 0.2, -0.2, 0.2, -0.2, 0.1, -0.1, 0.1, 0.2, -0.2, 0.2, -0.2, 0.0},
{0.1, -0.2, 0.3, -0.2, 0.1, 0.1, 0.2, -0.2, -0.1, -0.0, 0.1, -0.0, -0.1, 0.2, -0.2},
{0.2, -0.1, -0.2, 0.2, -0.2, 0.1, 0.0, 0.0, 0.1, -0.0, 0.2, -0.0, -0.0, -0.3, -0.1},
{-0.1, -0.1, 0.1, 0.2, -0.1, 0.2, 0.1, 0.2, 0.2, -0.1, -0.2, -0.1, -0.1, -0.2, 0.0},
{-0.1, 0.2, -0.1, -0.0, 0.1, -0.1, 0.2, 0.0, -0.0, -0.0, -0.2, -0.2, 0.1, -0.2, -0.0},
{-0.2, 0.2, 0.2, -0.2, -0.0, 0.2, 0.1, -0.0, 0.1, 0.0, -0.2, 0.1, -0.1, 0.3, 0.0},
{-0.2, -0.1, 0.2, -0.1, -0.2, 0.1, 0.1, -0.2, 0.2, -0.2, 0.2, -0.2, -0.2, -0.3, 0.2},
{0.1, 0.2, 0.1, -0.0, -0.2, -0.1, -0.2, 0.0, -0.2, -0.2, -0.2, -0.1, 0.0, -0.0, 0.2},
}
);
Matrix  transformer_layers_3_feed_forward_linear2_bias   (
{0.2, 0.1, 0.2, 0.1, 0.2, -0.1, -0.2, -0.2, -0.3, 0.2, -0.0, 0.1, 0.2, 0.2, -0.1, 0.0, -0.2, 0.1, 0.1, 0.3, -0.1, 0.1, -0.1, -0.2, 0.2}
);
Matrix  transformer_layers_3_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_3_feed_forward_ln1_layer_norm_bias   (
{-0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0}
);
Matrix  transformer_layers_3_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_3_feed_forward_ln2_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0}
);
Matrix  transformer_layers_4_attention_query_weight   (
{{0.1, 0.1, -0.2, -0.2, 0.0, 0.1, -0.1, -0.1, -0.1, 0.0, -0.1, 0.1, 0.1, 0.2, 0.1, -0.1, 0.1, -0.1, 0.1, -0.0, 0.2, 0.1, 0.2, 0.1, 0.0},
{-0.1, 0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, -0.0, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, -0.0, -0.1, -0.0, -0.2, -0.1, -0.2, -0.1, 0.1, -0.1},
{-0.2, 0.1, -0.1, -0.2, -0.0, -0.1, -0.0, -0.1, -0.1, -0.2, 0.2, 0.2, -0.2, -0.2, 0.1, 0.2, -0.1, 0.1, -0.0, -0.2, -0.1, 0.1, -0.1, -0.1, 0.0},
{0.2, 0.1, -0.0, -0.0, 0.2, 0.2, -0.2, 0.0, -0.1, 0.1, -0.0, -0.0, 0.2, -0.0, -0.1, 0.1, 0.1, 0.1, 0.1, -0.0, 0.1, -0.0, 0.1, -0.1, -0.2},
{-0.1, -0.2, -0.1, 0.2, -0.2, -0.1, -0.1, 0.2, 0.0, -0.1, 0.1, 0.0, 0.1, 0.0, -0.2, -0.0, -0.0, -0.1, -0.0, -0.0, 0.1, 0.0, 0.1, 0.1, 0.2},
{-0.0, -0.2, 0.2, 0.2, 0.1, -0.1, -0.1, 0.1, 0.0, 0.2, 0.0, 0.1, 0.1, 0.0, -0.0, -0.1, -0.0, 0.1, -0.1, -0.1, -0.1, 0.0, -0.0, -0.1, 0.2},
{0.1, -0.1, -0.2, 0.1, 0.2, 0.1, 0.1, -0.2, -0.2, 0.2, 0.1, 0.1, -0.1, -0.1, -0.0, -0.0, -0.2, 0.0, -0.1, 0.1, -0.1, -0.0, -0.1, 0.0, 0.1},
{0.0, -0.2, -0.2, 0.2, 0.2, -0.0, 0.1, -0.0, -0.1, 0.0, -0.1, 0.1, -0.2, -0.0, 0.1, -0.1, 0.0, 0.1, -0.0, -0.0, -0.1, 0.0, -0.1, -0.1, 0.1},
{0.2, 0.1, -0.1, -0.0, -0.1, -0.1, 0.1, -0.1, 0.0, 0.1, -0.1, 0.2, 0.2, -0.1, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.0, 0.2, 0.1, 0.2, 0.0},
{0.1, -0.2, -0.1, 0.1, 0.0, -0.1, -0.0, 0.1, -0.2, -0.0, 0.2, -0.2, 0.1, -0.1, 0.0, -0.2, 0.1, -0.2, -0.1, 0.1, -0.1, -0.1, 0.1, 0.0, 0.1},
{0.0, 0.1, 0.1, -0.1, -0.1, -0.1, -0.0, -0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.0, -0.1, -0.1, -0.1, 0.2, 0.2, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1},
{-0.1, -0.0, -0.1, 0.1, -0.0, -0.0, 0.0, -0.2, 0.1, -0.1, -0.0, 0.0, 0.1, -0.2, -0.1, -0.0, 0.1, 0.1, -0.0, 0.0, -0.1, 0.1, -0.2, -0.2, -0.0},
{0.1, 0.0, -0.1, -0.1, -0.1, -0.1, 0.2, -0.0, 0.1, 0.0, 0.0, 0.1, 0.2, -0.1, 0.1, 0.1, -0.2, 0.0, -0.0, -0.1, 0.0, -0.0, -0.1, 0.0, 0.2},
{-0.2, -0.1, 0.1, 0.0, -0.0, 0.2, 0.0, 0.0, 0.0, -0.2, -0.2, 0.2, -0.2, -0.0, 0.1, 0.1, 0.2, 0.1, -0.2, 0.1, -0.1, -0.0, 0.2, -0.1, 0.0},
{-0.0, 0.2, -0.1, 0.1, 0.2, -0.1, 0.1, 0.2, 0.0, -0.0, -0.0, 0.1, -0.1, 0.1, 0.1, 0.0, -0.1, -0.1, 0.1, 0.1, -0.0, -0.0, -0.1, -0.0, 0.1},
{-0.2, -0.2, 0.1, 0.2, -0.1, -0.1, 0.1, -0.1, -0.1, 0.2, 0.0, -0.0, 0.1, 0.1, 0.1, 0.1, -0.0, 0.1, -0.1, 0.1, 0.1, -0.2, -0.1, -0.1, -0.1},
{-0.1, -0.2, 0.2, 0.0, -0.1, -0.1, -0.2, -0.1, -0.1, 0.2, 0.1, 0.0, -0.1, 0.1, 0.0, -0.0, -0.1, 0.2, -0.1, -0.1, 0.1, -0.1, -0.1, 0.0, -0.2},
{0.2, 0.2, 0.1, 0.2, 0.1, -0.1, 0.2, -0.0, -0.1, 0.1, -0.1, 0.1, 0.1, -0.0, -0.1, -0.0, 0.1, 0.2, -0.1, 0.0, 0.1, 0.0, -0.2, -0.2, 0.1},
{0.1, 0.1, -0.2, 0.0, -0.1, -0.1, 0.1, -0.1, -0.1, -0.2, -0.1, -0.2, 0.0, -0.2, -0.1, 0.2, -0.0, -0.1, -0.2, 0.1, -0.2, 0.2, -0.0, 0.2, -0.2},
{-0.1, -0.1, 0.0, -0.2, 0.1, -0.1, 0.0, -0.0, 0.2, -0.2, -0.1, -0.2, 0.1, -0.0, 0.1, 0.1, -0.1, 0.0, 0.2, -0.1, -0.1, 0.2, 0.1, 0.0, 0.0},
{-0.1, 0.1, -0.0, 0.0, -0.2, -0.1, 0.2, 0.2, -0.0, 0.2, 0.0, -0.1, -0.0, 0.2, -0.1, -0.1, 0.0, -0.1, 0.0, -0.2, -0.0, -0.1, -0.1, -0.2, 0.1},
{-0.1, -0.0, 0.0, 0.2, -0.0, -0.2, -0.0, -0.2, -0.2, 0.1, 0.1, -0.2, -0.2, -0.1, -0.1, 0.0, -0.1, -0.0, -0.0, -0.1, -0.1, -0.0, 0.1, -0.1, 0.1},
{0.2, -0.0, -0.2, 0.1, -0.2, 0.2, -0.2, -0.0, -0.2, 0.0, -0.0, 0.1, 0.0, 0.2, -0.1, -0.2, 0.1, -0.2, -0.1, -0.1, -0.0, 0.2, -0.0, 0.1, -0.2},
{0.0, -0.2, -0.1, -0.1, -0.2, 0.1, -0.1, -0.0, -0.1, 0.1, -0.0, -0.0, -0.0, -0.0, -0.2, -0.2, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, -0.1, -0.1, 0.2},
{-0.2, 0.1, -0.2, -0.1, 0.2, -0.1, -0.1, -0.1, -0.1, 0.2, -0.1, -0.0, 0.1, 0.2, 0.1, 0.0, -0.2, -0.1, -0.1, -0.1, 0.1, 0.2, -0.1, -0.0, 0.0},
}
);
Matrix  transformer_layers_4_attention_query_bias   (
{0.1, 0.0, 0.2, -0.1, 0.0, -0.0, 0.1, 0.1, -0.0, -0.1, -0.0, 0.2, 0.1, -0.1, 0.0, -0.2, 0.1, -0.0, -0.1, 0.1, 0.0, 0.2, 0.0, 0.0, 0.1}
);
Matrix  transformer_layers_4_attention_key_weight   (
{{0.1, 0.2, -0.1, 0.0, 0.1, 0.2, 0.0, -0.2, 0.1, -0.2, 0.1, 0.1, 0.2, -0.2, -0.0, -0.1, 0.2, -0.1, -0.1, 0.1, 0.2, 0.0, -0.2, 0.1, -0.0},
{0.1, 0.1, -0.2, 0.0, 0.1, -0.0, -0.1, 0.1, -0.1, 0.1, 0.1, -0.2, -0.2, 0.1, -0.1, 0.0, 0.1, 0.1, 0.2, -0.1, -0.1, 0.1, -0.1, 0.0, -0.1},
{-0.1, 0.1, -0.1, -0.2, -0.0, 0.1, 0.1, 0.1, 0.2, -0.1, -0.0, -0.1, -0.1, -0.2, -0.2, 0.1, -0.0, 0.2, -0.2, -0.1, -0.1, 0.1, -0.1, 0.0, 0.2},
{-0.1, -0.1, -0.1, 0.1, -0.1, -0.0, -0.2, 0.1, 0.0, -0.1, -0.1, -0.0, -0.0, 0.1, 0.1, -0.2, 0.0, -0.1, 0.1, -0.1, -0.1, -0.1, -0.0, 0.2, -0.1},
{0.1, 0.2, 0.0, 0.0, 0.2, -0.2, 0.1, -0.1, -0.2, -0.0, 0.1, -0.2, 0.0, 0.1, 0.2, 0.1, 0.1, -0.0, -0.0, -0.0, -0.2, 0.1, 0.2, 0.1, -0.2},
{0.1, 0.1, 0.1, -0.0, -0.1, 0.1, 0.0, -0.1, -0.0, 0.1, 0.1, -0.1, 0.2, 0.1, -0.1, -0.2, 0.0, -0.0, 0.1, -0.2, -0.0, 0.1, 0.0, -0.0, 0.0},
{0.1, 0.2, 0.0, -0.1, -0.1, -0.2, -0.2, -0.1, 0.1, -0.0, -0.1, -0.1, 0.1, -0.1, 0.2, 0.2, 0.0, -0.0, -0.2, -0.1, 0.0, 0.2, 0.0, 0.1, 0.1},
{-0.0, -0.1, 0.0, -0.1, -0.1, 0.1, -0.1, -0.0, -0.2, -0.2, -0.1, 0.0, 0.1, -0.0, 0.0, -0.0, -0.1, 0.0, -0.0, 0.1, -0.0, 0.0, 0.1, -0.1, 0.1},
{-0.0, 0.1, 0.0, 0.1, -0.1, -0.1, -0.2, -0.2, 0.2, 0.2, -0.0, 0.1, 0.2, 0.0, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.0, -0.1},
{-0.1, -0.1, -0.1, -0.0, -0.2, -0.2, -0.0, 0.1, -0.2, 0.1, -0.0, 0.1, -0.0, -0.1, 0.1, -0.1, 0.1, 0.0, 0.1, -0.0, -0.1, 0.2, -0.2, -0.1, -0.0},
{-0.1, -0.0, 0.1, -0.2, 0.1, -0.2, -0.2, 0.1, 0.1, 0.1, 0.2, -0.2, -0.0, 0.0, 0.1, 0.1, 0.1, -0.2, 0.0, 0.2, -0.1, 0.0, 0.1, 0.2, 0.1},
{-0.2, 0.1, 0.2, 0.0, -0.1, -0.1, -0.1, 0.1, -0.2, -0.0, 0.2, -0.1, -0.0, -0.2, 0.1, -0.1, -0.2, -0.1, 0.0, -0.2, 0.2, -0.0, 0.0, -0.1, -0.1},
{0.2, 0.2, -0.1, -0.2, 0.1, 0.1, 0.1, -0.0, -0.2, 0.1, -0.1, -0.1, -0.2, 0.2, -0.0, -0.2, -0.1, 0.0, 0.2, 0.2, 0.1, -0.0, -0.2, 0.2, -0.1},
{0.0, -0.1, 0.2, 0.1, -0.2, -0.1, -0.1, -0.1, -0.2, 0.1, -0.2, 0.2, 0.1, 0.1, 0.0, -0.0, 0.1, -0.1, 0.1, 0.1, 0.0, -0.1, -0.0, 0.1, -0.1},
{0.1, -0.0, -0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0, -0.1, -0.1, 0.2, 0.2, 0.0, -0.1, 0.2, 0.1, -0.1, 0.0, -0.2, 0.0, 0.1, -0.1, 0.0},
{0.0, 0.0, -0.1, -0.0, -0.2, -0.1, 0.0, -0.1, -0.1, 0.1, 0.1, 0.0, 0.1, 0.1, -0.1, 0.0, 0.1, 0.2, 0.1, 0.1, -0.2, -0.0, 0.1, -0.1, -0.0},
{0.1, -0.1, 0.0, -0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2, 0.1, 0.2, -0.1, 0.2, 0.0, 0.1, -0.0, -0.0, -0.1, 0.1, -0.2, 0.1, -0.1, -0.0, -0.1},
{0.2, 0.1, 0.0, 0.0, 0.0, 0.2, -0.0, 0.1, -0.0, -0.0, 0.2, 0.2, -0.2, 0.2, -0.1, 0.0, -0.0, 0.0, -0.0, 0.2, 0.1, -0.0, 0.0, 0.1, -0.0},
{0.1, 0.2, -0.2, 0.1, -0.1, -0.1, 0.0, -0.2, 0.1, 0.1, 0.1, 0.0, -0.2, 0.0, -0.2, 0.2, 0.1, -0.2, -0.2, 0.0, -0.0, 0.1, 0.0, 0.2, 0.0},
{-0.2, -0.2, 0.0, -0.2, -0.0, 0.1, -0.2, 0.2, -0.1, 0.1, -0.1, -0.1, -0.2, 0.0, -0.0, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, 0.2, -0.1, 0.1},
{-0.1, -0.1, -0.1, 0.1, -0.0, -0.0, -0.2, 0.1, 0.0, 0.2, 0.1, -0.1, 0.2, 0.2, -0.2, -0.2, 0.0, -0.2, -0.2, 0.1, 0.0, -0.2, -0.0, -0.0, -0.2},
{0.0, -0.0, -0.2, -0.1, -0.2, -0.1, -0.1, 0.0, -0.2, 0.2, -0.1, 0.1, -0.1, 0.2, 0.1, 0.1, -0.1, 0.2, 0.2, -0.0, 0.1, -0.1, -0.0, -0.0, 0.2},
{-0.1, 0.2, 0.0, 0.1, -0.0, -0.2, -0.1, 0.1, -0.0, 0.0, -0.1, -0.1, 0.2, 0.1, -0.2, -0.1, 0.2, 0.1, 0.2, 0.1, 0.1, -0.0, -0.1, -0.1, -0.0},
{0.2, -0.0, 0.1, 0.1, 0.0, -0.0, -0.0, 0.1, 0.0, -0.2, 0.0, 0.2, 0.2, 0.2, 0.0, -0.1, 0.0, 0.0, 0.1, 0.2, 0.1, -0.1, -0.1, -0.2, -0.1},
{0.0, -0.1, -0.2, -0.2, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.2, -0.1, 0.1, -0.0, 0.0, 0.1, -0.2, 0.1, 0.2, -0.2, 0.1, -0.0},
}
);
Matrix  transformer_layers_4_attention_key_bias   (
{-0.0, -0.0, -0.0, -0.2, 0.1, -0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.0, -0.0, 0.2, 0.0, 0.0, 0.2, 0.0, -0.2, 0.0, 0.2, 0.0, -0.1, -0.0}
);
Matrix  transformer_layers_4_attention_value_weight   (
{{-0.2, -0.2, 0.1, 0.2, 0.2, -0.0, 0.2, 0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.2, 0.2, -0.1, 0.1, 0.1, 0.0, 0.0, 0.2, 0.0},
{-0.2, 0.1, 0.2, -0.2, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.0, 0.1, -0.1, 0.0, 0.1, 0.2, -0.0, 0.1, 0.1, -0.2, 0.1, -0.2, 0.1, 0.2},
{-0.1, -0.2, 0.0, 0.2, 0.1, -0.1, -0.1, -0.2, 0.0, -0.1, -0.0, -0.0, 0.0, 0.1, 0.1, 0.2, 0.0, -0.1, 0.1, -0.0, -0.1, 0.1, 0.1, 0.0, 0.1},
{0.1, 0.2, -0.2, 0.0, -0.2, 0.2, 0.0, 0.1, 0.1, -0.1, 0.1, -0.1, -0.1, 0.0, 0.1, -0.0, 0.1, -0.1, 0.2, -0.1, 0.1, -0.1, -0.1, -0.0, -0.0},
{0.1, 0.0, -0.1, 0.1, -0.2, -0.2, -0.0, 0.0, -0.1, -0.2, 0.0, 0.1, -0.1, 0.1, -0.1, 0.2, 0.1, -0.2, -0.2, -0.1, -0.1, -0.2, 0.0, 0.0, 0.1},
{-0.1, -0.1, -0.1, 0.2, -0.1, -0.2, 0.1, 0.0, -0.1, 0.2, 0.1, 0.0, 0.0, -0.1, 0.1, 0.1, 0.2, 0.1, -0.1, -0.1, 0.2, 0.0, 0.0, -0.2, 0.1},
{0.2, 0.0, -0.1, -0.1, -0.0, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.0, -0.1, 0.1, 0.0, -0.1, -0.1, 0.0, 0.1, -0.0, 0.0, 0.0, 0.0},
{0.1, 0.2, 0.2, -0.1, 0.2, -0.1, 0.1, -0.1, 0.1, -0.2, 0.2, 0.0, 0.0, -0.1, -0.1, -0.0, 0.2, 0.0, 0.1, -0.2, -0.0, 0.1, -0.2, -0.1, -0.0},
{-0.0, -0.0, 0.1, 0.2, -0.2, -0.0, -0.1, -0.1, -0.2, -0.0, -0.2, 0.1, 0.0, -0.0, -0.2, 0.1, 0.1, 0.1, -0.0, -0.0, 0.1, -0.1, -0.1, 0.1, 0.2},
{-0.2, -0.2, 0.0, 0.0, -0.1, 0.1, -0.2, 0.1, -0.1, -0.0, -0.1, 0.0, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.2, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1},
{-0.0, -0.1, -0.2, -0.0, 0.2, -0.0, 0.2, -0.2, 0.0, 0.0, 0.1, -0.1, -0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.1, -0.0, 0.0, -0.1, -0.1, 0.2, -0.2},
{-0.1, -0.1, -0.0, -0.1, -0.0, 0.0, 0.1, -0.0, 0.0, -0.1, -0.1, -0.1, -0.2, -0.2, -0.0, -0.1, -0.2, 0.1, 0.1, -0.0, 0.2, -0.1, 0.0, -0.1, 0.1},
{-0.0, 0.0, -0.1, -0.1, 0.1, 0.1, -0.2, -0.0, -0.2, -0.1, -0.1, -0.1, 0.1, 0.0, 0.2, -0.1, -0.1, -0.2, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1},
{0.0, 0.1, 0.1, -0.1, -0.1, -0.0, -0.1, 0.1, 0.0, -0.1, -0.2, -0.1, -0.1, -0.2, 0.1, 0.0, 0.1, -0.1, -0.2, -0.2, -0.1, 0.1, 0.1, -0.0, 0.1},
{0.2, 0.1, -0.0, 0.1, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.0, -0.1, 0.2, -0.0, -0.2, 0.1, 0.1, 0.1, -0.0, -0.1, 0.1, 0.1, 0.1, 0.1},
{0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.0, 0.2, 0.1, -0.0, -0.1, 0.1, -0.1, -0.1, 0.2, 0.0, -0.2, -0.2, 0.0, -0.2, -0.1, 0.1, 0.1, 0.1, -0.1},
{0.0, -0.1, -0.2, -0.0, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.1, -0.0, 0.0, 0.0, 0.2, -0.1, 0.2, 0.1, -0.0, -0.2, -0.0, 0.2, -0.0, -0.1},
{-0.0, 0.1, 0.2, -0.0, -0.2, 0.1, 0.1, -0.2, -0.1, -0.1, 0.1, -0.1, -0.0, -0.1, -0.2, -0.2, -0.0, -0.1, -0.2, 0.1, 0.2, -0.1, 0.1, -0.2, -0.0},
{0.2, -0.0, -0.1, -0.0, -0.1, -0.0, 0.2, -0.0, 0.0, -0.2, -0.1, -0.1, -0.1, -0.0, 0.1, 0.1, 0.0, 0.1, 0.1, -0.2, -0.2, -0.1, 0.2, 0.1, -0.2},
{-0.1, 0.2, -0.1, 0.2, 0.2, -0.0, -0.1, -0.1, 0.1, 0.2, -0.0, 0.2, 0.2, -0.1, 0.1, -0.0, 0.0, -0.0, -0.0, 0.2, 0.1, 0.1, -0.0, 0.0, 0.2},
{-0.0, -0.1, 0.1, -0.1, 0.2, 0.1, 0.2, -0.1, -0.0, -0.2, -0.1, -0.0, 0.1, 0.0, -0.2, -0.0, 0.1, -0.2, -0.1, -0.2, 0.2, -0.0, 0.2, 0.1, 0.0},
{-0.0, -0.0, -0.2, 0.0, -0.1, -0.1, -0.0, 0.0, 0.1, 0.0, -0.2, -0.1, -0.1, 0.1, 0.1, -0.2, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.2, -0.1, 0.1},
{0.0, 0.2, 0.1, -0.2, 0.1, 0.1, 0.1, 0.0, -0.1, 0.1, 0.0, 0.1, -0.0, -0.0, 0.1, 0.2, -0.2, -0.2, 0.1, -0.0, 0.0, 0.1, 0.1, -0.2, -0.2},
{0.1, 0.1, 0.2, 0.1, -0.1, 0.1, 0.2, -0.1, -0.1, 0.1, 0.1, -0.0, -0.1, 0.0, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1, 0.1, -0.0, 0.0, -0.1, -0.0},
{0.0, 0.1, -0.1, -0.1, -0.1, -0.2, 0.1, 0.1, 0.1, 0.2, -0.1, 0.0, 0.1, -0.1, 0.2, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.0, -0.2, -0.1, 0.0},
}
);
Matrix  transformer_layers_4_attention_value_bias   (
{0.1, -0.0, 0.2, 0.1, -0.1, 0.2, -0.1, 0.1, -0.0, -0.2, 0.0, 0.1, -0.0, -0.0, -0.1, -0.1, -0.1, -0.2, -0.1, -0.1, 0.2, -0.0, 0.1, -0.1, 0.1}
);
Matrix  transformer_layers_4_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_4_norm1_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0}
);
Matrix  transformer_layers_4_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_4_norm2_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_4_feed_forward_linear1_weight   (
{{-0.2, -0.1, 0.2, -0.1, -0.2, 0.2, -0.0, 0.2, -0.2, -0.0, 0.1, -0.2, -0.1, -0.1, 0.0, 0.1, 0.1, -0.0, 0.1, -0.2, 0.2, 0.2, -0.0, 0.1, -0.0},
{0.2, 0.2, -0.1, 0.1, -0.1, 0.1, -0.2, 0.0, 0.2, -0.0, -0.0, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.0, -0.2, 0.1, 0.1, 0.1, -0.0, -0.1, -0.2},
{-0.2, -0.1, -0.1, -0.0, -0.1, -0.1, -0.0, -0.0, 0.1, 0.2, -0.1, 0.1, 0.1, -0.1, 0.0, 0.1, -0.0, 0.1, -0.2, -0.2, 0.1, -0.1, 0.0, -0.1, 0.1},
{0.1, -0.2, 0.0, -0.1, -0.0, -0.0, -0.1, 0.0, -0.1, 0.0, -0.1, -0.0, 0.1, -0.2, -0.1, 0.1, -0.2, -0.1, 0.0, -0.1, 0.0, 0.1, -0.1, 0.2, -0.1},
{0.2, -0.1, 0.0, 0.1, -0.0, 0.2, -0.1, 0.0, 0.1, -0.1, -0.2, -0.1, 0.2, 0.1, 0.1, -0.2, -0.1, -0.2, 0.1, -0.1, -0.1, 0.2, 0.2, 0.1, 0.1},
{-0.1, -0.1, -0.0, -0.0, -0.1, 0.1, -0.1, 0.2, 0.2, 0.2, 0.2, -0.2, 0.1, 0.0, 0.1, -0.1, 0.0, 0.0, -0.1, 0.2, 0.1, 0.0, -0.2, 0.0, 0.2},
{0.2, -0.2, 0.0, 0.2, -0.1, 0.0, -0.1, -0.2, -0.0, -0.0, 0.2, 0.0, 0.2, 0.1, 0.1, 0.1, -0.1, -0.1, 0.1, -0.0, -0.0, -0.2, 0.0, -0.0, 0.0},
{0.2, 0.1, 0.1, 0.0, 0.0, 0.2, 0.0, 0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, 0.1, -0.0, -0.1, 0.1, 0.1, 0.1, -0.1, 0.0, 0.0, 0.0, -0.1},
{-0.1, 0.1, -0.0, 0.0, -0.1, 0.0, 0.1, -0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.2, -0.0, 0.2, 0.1, 0.0, 0.1, 0.1, -0.1, 0.1, -0.2, -0.1, -0.2},
{0.0, 0.2, -0.1, 0.2, -0.2, -0.1, 0.1, -0.1, -0.1, -0.2, -0.2, 0.1, -0.1, 0.1, 0.1, 0.2, -0.1, 0.0, 0.1, -0.2, -0.1, 0.1, 0.1, -0.0, 0.0},
{-0.2, -0.1, 0.1, -0.2, -0.1, 0.2, -0.1, 0.0, -0.0, -0.0, 0.1, -0.1, -0.1, -0.1, 0.1, 0.0, 0.1, 0.0, -0.1, 0.1, -0.1, -0.1, -0.0, -0.0, -0.2},
{0.1, 0.2, -0.1, 0.2, -0.0, -0.1, -0.0, 0.0, -0.2, 0.1, 0.2, 0.0, -0.2, 0.2, 0.1, -0.2, -0.1, -0.1, 0.1, -0.2, -0.2, 0.2, -0.0, -0.0, -0.1},
{-0.2, 0.0, 0.1, -0.1, 0.1, 0.0, 0.0, 0.0, -0.2, -0.2, -0.0, -0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.0, 0.0, 0.2, 0.1, 0.1, -0.0, 0.2},
{0.1, 0.1, 0.1, 0.1, -0.2, 0.1, -0.2, 0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.2, 0.1, -0.2, 0.1, -0.0, -0.2, 0.0, -0.2, -0.0, -0.1, 0.1},
{-0.1, 0.2, -0.1, 0.1, -0.2, -0.0, -0.2, 0.2, 0.0, -0.1, -0.0, -0.2, 0.2, 0.1, -0.1, 0.1, -0.1, -0.1, 0.2, -0.1, -0.2, 0.1, -0.2, -0.2, -0.1},
}
);
Matrix  transformer_layers_4_feed_forward_linear1_bias   (
{-0.0, 0.1, 0.1, 0.1, 0.0, 0.0, -0.1, -0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1}
);
Matrix  transformer_layers_4_feed_forward_linear2_weight   (
{{-0.1, -0.1, 0.0, -0.0, 0.1, -0.2, -0.2, -0.2, 0.1, 0.2, -0.1, 0.1, -0.1, -0.0, -0.1},
{-0.2, 0.0, -0.2, -0.1, 0.1, 0.3, -0.2, 0.1, -0.1, -0.2, 0.1, -0.2, 0.1, -0.1, 0.1},
{0.0, -0.2, 0.1, -0.0, -0.0, 0.2, -0.1, 0.1, 0.0, -0.2, 0.2, 0.2, -0.2, 0.0, -0.2},
{0.1, -0.1, -0.1, -0.0, 0.2, -0.1, 0.2, 0.1, -0.1, -0.2, -0.2, -0.1, 0.2, 0.1, -0.1},
{-0.1, 0.2, 0.0, 0.2, -0.1, -0.2, 0.1, 0.0, 0.1, 0.1, -0.1, 0.1, -0.0, -0.2, -0.2},
{-0.1, -0.1, 0.1, 0.2, -0.1, -0.1, -0.1, -0.1, 0.1, -0.2, -0.0, -0.1, 0.1, 0.2, 0.2},
{0.1, 0.2, 0.1, -0.2, -0.1, 0.1, -0.1, -0.1, 0.1, -0.2, 0.2, 0.1, 0.1, 0.2, -0.2},
{-0.2, 0.2, 0.3, -0.1, 0.1, 0.1, 0.1, -0.2, 0.1, -0.3, 0.2, -0.3, -0.0, -0.1, 0.3},
{0.2, 0.1, 0.1, 0.0, 0.1, -0.1, 0.0, -0.2, 0.3, 0.0, 0.2, -0.0, -0.1, 0.2, -0.2},
{0.1, -0.1, -0.1, 0.2, 0.1, -0.2, 0.2, -0.2, 0.1, 0.1, 0.1, 0.1, -0.1, -0.0, 0.2},
{-0.0, -0.0, -0.3, 0.0, -0.3, 0.0, 0.0, 0.1, -0.2, -0.1, 0.1, -0.0, 0.0, -0.2, 0.0},
{0.3, 0.0, -0.2, -0.1, -0.2, -0.2, -0.2, 0.1, 0.1, -0.2, -0.0, -0.2, -0.2, 0.1, -0.2},
{-0.0, -0.2, -0.2, 0.2, -0.2, -0.1, -0.1, 0.1, 0.3, -0.1, -0.0, 0.1, -0.0, 0.1, 0.2},
{0.2, -0.0, -0.1, 0.2, 0.1, -0.0, 0.0, 0.2, -0.0, -0.2, 0.2, -0.1, 0.1, 0.0, -0.0},
{0.3, 0.2, 0.2, -0.2, 0.1, 0.2, -0.1, -0.0, 0.2, -0.1, 0.1, 0.0, 0.1, -0.2, 0.1},
{0.2, 0.1, 0.2, 0.2, 0.1, -0.2, -0.2, 0.1, 0.0, 0.0, -0.1, -0.0, 0.0, -0.1, 0.1},
{0.1, -0.0, -0.1, -0.0, -0.2, 0.2, 0.0, 0.2, -0.2, 0.2, -0.2, -0.0, -0.2, -0.2, 0.1},
{-0.1, 0.0, -0.2, -0.0, 0.1, -0.2, -0.2, 0.2, -0.2, 0.1, -0.0, -0.2, -0.1, -0.2, -0.2},
{-0.1, 0.0, -0.0, -0.1, -0.0, 0.2, -0.2, 0.2, 0.0, 0.2, 0.2, -0.1, -0.1, -0.3, -0.0},
{0.3, -0.2, 0.2, -0.2, 0.1, 0.1, 0.1, -0.2, 0.1, -0.0, -0.0, 0.2, -0.0, -0.2, -0.2},
{0.1, -0.2, 0.0, -0.2, -0.0, 0.2, -0.1, 0.1, -0.2, 0.0, 0.0, 0.0, 0.2, 0.2, -0.2},
{-0.2, 0.2, -0.2, 0.0, 0.2, -0.0, -0.1, -0.2, -0.1, -0.1, 0.1, -0.1, -0.0, 0.1, -0.1},
{-0.1, -0.1, -0.0, -0.2, 0.2, -0.2, 0.1, -0.2, 0.2, 0.1, -0.2, 0.2, -0.2, 0.2, 0.2},
{0.2, 0.1, 0.2, -0.0, -0.1, -0.1, 0.0, -0.2, 0.1, -0.1, 0.1, -0.1, 0.2, -0.2, -0.1},
{0.0, -0.1, -0.1, -0.0, 0.1, 0.0, 0.0, 0.1, 0.2, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2},
}
);
Matrix  transformer_layers_4_feed_forward_linear2_bias   (
{-0.2, 0.2, -0.1, -0.2, 0.2, 0.0, -0.0, 0.2, 0.1, 0.1, -0.0, -0.2, -0.2, -0.0, 0.1, -0.2, 0.1, -0.2, 0.0, 0.0, 0.0, -0.1, 0.1, 0.2, -0.0}
);
Matrix  transformer_layers_4_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_4_feed_forward_ln1_layer_norm_bias   (
{0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0}
);
Matrix  transformer_layers_4_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_4_feed_forward_ln2_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_5_attention_query_weight   (
{{-0.0, 0.2, 0.1, 0.1, -0.0, 0.2, 0.1, -0.2, 0.1, 0.1, 0.2, 0.2, -0.1, -0.1, 0.1, 0.2, 0.0, -0.0, 0.0, -0.1, -0.1, 0.1, 0.1, -0.0, -0.1},
{0.1, 0.0, -0.1, 0.1, -0.2, 0.2, 0.0, -0.2, 0.1, -0.0, -0.0, -0.2, -0.2, -0.1, 0.1, 0.1, -0.2, -0.0, -0.2, 0.0, -0.2, 0.2, -0.0, -0.0, -0.0},
{0.2, -0.1, 0.0, -0.1, 0.1, -0.1, 0.2, 0.1, 0.2, -0.1, -0.0, -0.1, 0.2, 0.1, -0.2, -0.0, 0.2, -0.1, -0.1, -0.2, 0.1, -0.1, 0.2, 0.1, 0.1},
{-0.2, -0.1, -0.2, 0.2, 0.2, 0.2, 0.2, -0.0, -0.1, -0.1, -0.2, -0.0, 0.0, -0.2, -0.1, 0.2, -0.2, 0.2, 0.1, 0.0, -0.1, 0.0, -0.1, -0.2, 0.1},
{-0.0, -0.0, -0.0, 0.1, -0.0, 0.2, 0.1, -0.1, 0.1, 0.0, 0.1, 0.0, -0.0, 0.0, -0.1, 0.2, -0.0, 0.0, 0.1, -0.2, -0.1, -0.1, -0.1, 0.1, -0.2},
{0.2, -0.1, 0.1, 0.2, -0.0, 0.1, 0.1, 0.0, -0.2, 0.1, 0.1, 0.1, 0.2, -0.0, 0.2, 0.1, 0.1, -0.1, 0.1, 0.2, 0.2, 0.1, 0.2, -0.0, -0.1},
{0.1, -0.2, -0.1, -0.1, 0.0, -0.1, 0.1, 0.0, 0.0, 0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.2, -0.1, -0.2, -0.1, -0.1, -0.0, -0.1, 0.1, -0.2, 0.1},
{0.2, -0.1, -0.0, -0.0, -0.1, -0.1, -0.1, -0.2, 0.2, -0.1, -0.2, -0.1, -0.0, -0.1, 0.2, 0.0, 0.2, 0.2, -0.1, -0.0, -0.2, 0.1, -0.0, -0.0, -0.0},
{0.1, 0.0, 0.1, 0.1, 0.1, -0.1, 0.1, -0.0, 0.1, 0.1, -0.1, -0.0, 0.2, -0.2, 0.2, 0.2, -0.2, -0.1, 0.0, -0.2, 0.1, -0.2, 0.2, 0.2, 0.0},
{-0.1, 0.1, -0.2, -0.2, -0.1, 0.1, -0.1, 0.1, -0.2, 0.1, -0.0, -0.2, -0.1, 0.0, 0.2, -0.1, -0.1, 0.2, 0.2, 0.1, -0.1, -0.1, 0.1, -0.0, 0.2},
{0.1, 0.2, -0.0, 0.1, -0.1, 0.1, -0.2, 0.1, -0.0, 0.1, 0.1, -0.1, 0.2, -0.2, -0.2, 0.0, 0.1, -0.2, -0.1, -0.1, 0.1, -0.2, 0.1, 0.1, -0.2},
{-0.0, 0.0, 0.1, 0.1, -0.2, -0.2, 0.1, 0.2, 0.1, 0.1, -0.2, -0.1, 0.1, 0.2, -0.1, -0.1, 0.2, -0.0, 0.2, -0.2, 0.1, -0.1, 0.1, 0.0, -0.1},
{-0.1, -0.2, -0.1, 0.1, 0.0, 0.0, 0.1, -0.2, 0.2, 0.1, -0.1, 0.0, -0.2, 0.0, 0.2, 0.1, 0.0, -0.2, -0.1, -0.1, -0.2, 0.1, 0.1, 0.1, -0.1},
{0.2, -0.2, 0.1, -0.1, -0.1, 0.1, 0.1, -0.0, -0.2, -0.1, 0.1, -0.1, -0.1, 0.0, -0.2, 0.1, 0.1, -0.1, -0.1, 0.0, 0.0, -0.1, -0.1, 0.2, 0.2},
{-0.1, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, -0.2, 0.0, -0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.2, 0.1, -0.2, 0.0, 0.2, -0.1, 0.1, -0.0, -0.1, -0.1},
{0.1, -0.2, 0.1, 0.1, 0.1, -0.0, -0.0, 0.1, 0.1, 0.1, -0.1, 0.2, 0.1, -0.1, -0.0, 0.1, 0.0, 0.1, -0.0, 0.2, 0.0, 0.1, -0.1, 0.2, 0.0},
{0.1, -0.1, 0.0, 0.0, -0.0, 0.2, -0.1, 0.1, -0.1, -0.0, 0.1, 0.1, -0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.1, 0.1, -0.2, 0.1, -0.0, -0.2, -0.0},
{-0.1, -0.2, -0.1, -0.0, 0.1, -0.1, 0.1, -0.1, -0.1, 0.2, -0.2, 0.2, -0.1, 0.1, 0.1, -0.2, 0.1, 0.2, 0.1, -0.1, 0.1, 0.2, 0.1, -0.0, -0.1},
{0.0, 0.2, 0.1, -0.2, -0.1, -0.1, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1, 0.2, 0.1, 0.2, -0.2, -0.1, 0.0, -0.0, -0.1, -0.0, -0.2},
{-0.2, 0.2, 0.0, -0.1, -0.1, 0.1, -0.2, -0.0, 0.1, -0.1, -0.2, -0.2, -0.1, -0.1, -0.1, 0.1, 0.0, 0.0, 0.1, 0.1, -0.0, -0.2, -0.1, 0.1, 0.2},
{0.1, 0.2, 0.1, 0.0, -0.1, 0.1, 0.2, 0.0, 0.0, 0.1, -0.1, 0.2, 0.2, 0.1, 0.1, 0.2, 0.0, -0.2, 0.2, -0.2, -0.0, 0.1, -0.1, 0.0, 0.2},
{0.1, 0.1, -0.1, -0.1, 0.1, -0.0, -0.0, 0.2, -0.2, 0.0, -0.1, -0.1, -0.0, 0.0, -0.0, -0.2, -0.1, -0.1, 0.1, -0.2, -0.1, 0.2, -0.1, -0.1, 0.0},
{0.2, 0.1, 0.1, -0.1, 0.0, -0.0, -0.2, -0.1, 0.1, -0.1, 0.1, -0.0, -0.1, -0.1, 0.1, 0.0, 0.2, -0.1, -0.1, 0.1, 0.1, 0.1, -0.1, -0.1, 0.2},
{0.1, -0.1, 0.1, -0.0, -0.0, 0.1, -0.1, -0.0, -0.2, -0.1, 0.2, -0.0, 0.0, 0.1, 0.0, -0.1, 0.2, -0.0, 0.1, -0.2, 0.2, -0.1, -0.1, 0.2, 0.2},
{-0.2, -0.1, 0.2, 0.1, -0.1, -0.2, -0.1, 0.1, -0.1, -0.1, 0.2, 0.2, 0.2, 0.1, -0.0, 0.0, -0.1, 0.0, 0.2, 0.1, 0.1, 0.1, -0.1, 0.2, 0.1},
}
);
Matrix  transformer_layers_5_attention_query_bias   (
{-0.0, 0.1, -0.1, 0.1, -0.1, 0.0, -0.1, -0.1, -0.1, 0.2, 0.1, 0.0, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.0, -0.2, 0.1, -0.0, 0.1, -0.1, 0.1}
);
Matrix  transformer_layers_5_attention_key_weight   (
{{0.1, 0.2, -0.2, -0.0, 0.2, -0.1, 0.1, 0.1, 0.2, -0.2, 0.1, 0.1, -0.0, 0.1, 0.1, 0.2, -0.1, 0.0, 0.0, 0.2, 0.0, -0.2, -0.1, -0.1, 0.1},
{-0.1, -0.0, -0.2, -0.1, -0.1, -0.1, 0.0, -0.0, -0.1, 0.1, 0.1, 0.1, -0.2, 0.0, -0.1, 0.0, 0.1, -0.0, -0.1, -0.1, -0.2, -0.2, -0.0, 0.1, 0.2},
{-0.1, -0.1, -0.1, 0.1, 0.2, 0.1, -0.2, -0.1, -0.2, -0.1, 0.2, 0.1, 0.0, 0.0, -0.0, 0.2, -0.1, 0.1, -0.0, -0.1, 0.1, -0.1, 0.0, 0.1, -0.2},
{-0.1, -0.1, 0.2, 0.2, -0.1, 0.1, -0.1, 0.1, 0.2, 0.2, 0.1, 0.0, -0.1, 0.1, 0.2, -0.2, 0.1, 0.1, -0.1, -0.2, -0.1, -0.2, -0.0, -0.0, 0.1},
{0.1, 0.1, 0.2, -0.2, -0.2, -0.0, -0.0, 0.1, -0.1, 0.0, 0.0, -0.2, 0.1, -0.1, 0.2, -0.2, -0.1, 0.0, -0.0, -0.1, -0.1, -0.0, -0.0, 0.2, -0.1},
{-0.1, 0.1, -0.2, -0.1, -0.0, -0.1, 0.1, 0.0, -0.0, -0.1, 0.1, 0.1, -0.2, -0.1, -0.2, 0.1, -0.1, 0.0, -0.0, -0.1, 0.2, 0.1, -0.0, 0.2, 0.0},
{-0.1, -0.1, 0.1, 0.1, -0.1, 0.2, -0.1, 0.1, 0.1, -0.0, -0.2, -0.0, 0.0, -0.1, -0.0, 0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, 0.1, -0.1},
{-0.1, -0.1, 0.0, -0.0, 0.1, -0.0, 0.2, -0.1, 0.1, 0.1, -0.2, 0.2, -0.1, 0.1, -0.1, 0.2, -0.1, 0.2, 0.1, 0.2, 0.1, 0.2, -0.1, 0.2, 0.1},
{-0.0, 0.0, 0.1, -0.1, 0.2, -0.1, 0.1, -0.2, 0.1, -0.0, -0.0, 0.1, -0.0, 0.0, -0.0, -0.1, 0.0, -0.2, -0.2, 0.1, -0.2, -0.1, 0.1, -0.1, -0.1},
{-0.2, -0.0, 0.0, -0.1, 0.2, 0.1, 0.0, 0.1, -0.0, 0.0, -0.1, -0.1, 0.1, 0.2, -0.0, 0.2, -0.1, -0.1, -0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.1},
{-0.2, 0.1, 0.1, 0.1, -0.1, -0.0, 0.2, 0.1, -0.1, -0.1, 0.1, -0.0, 0.2, -0.0, 0.2, 0.2, 0.1, -0.1, -0.0, 0.1, 0.2, -0.2, -0.1, 0.1, -0.2},
{0.1, 0.2, -0.2, -0.2, 0.2, 0.0, 0.1, -0.1, -0.2, -0.2, 0.2, -0.1, -0.1, 0.0, -0.1, 0.1, -0.1, 0.2, -0.1, 0.1, 0.1, 0.2, -0.0, 0.1, -0.1},
{-0.2, 0.0, 0.0, -0.1, -0.1, -0.1, 0.2, -0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.0, 0.0, 0.1, -0.0, -0.1, -0.1, -0.2, -0.2, -0.2, -0.0, -0.2, 0.2},
{0.1, -0.0, -0.1, 0.2, -0.2, 0.0, -0.2, 0.0, -0.2, 0.1, 0.2, 0.0, -0.0, -0.0, 0.1, -0.1, 0.0, -0.1, -0.1, -0.0, 0.1, 0.1, -0.2, -0.1, -0.1},
{-0.1, 0.0, -0.1, -0.2, -0.0, -0.1, -0.0, 0.1, -0.0, 0.1, -0.0, -0.1, -0.0, 0.1, 0.2, -0.1, 0.1, -0.0, 0.0, -0.1, -0.1, 0.2, 0.1, -0.1, 0.1},
{-0.1, -0.0, -0.2, -0.2, -0.1, 0.1, -0.2, 0.1, 0.1, 0.1, -0.1, -0.0, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, -0.1, -0.0, -0.1, -0.1, 0.1, -0.1, -0.0},
{0.0, 0.1, -0.1, -0.0, -0.0, -0.0, 0.0, -0.1, -0.1, -0.1, 0.1, 0.1, -0.0, 0.0, 0.0, -0.2, -0.1, -0.2, 0.0, -0.0, 0.2, 0.2, 0.2, 0.0, -0.2},
{-0.2, -0.2, 0.0, -0.1, -0.1, -0.0, 0.2, -0.0, 0.1, 0.0, -0.1, 0.2, 0.0, 0.1, 0.2, 0.1, -0.0, -0.1, -0.1, 0.1, 0.0, -0.0, -0.1, -0.1, 0.1},
{0.2, -0.0, -0.0, 0.0, -0.1, -0.1, -0.1, -0.2, -0.1, -0.2, 0.0, -0.2, 0.0, -0.1, 0.1, 0.2, 0.1, 0.0, -0.1, -0.0, 0.2, 0.1, -0.0, 0.1, -0.1},
{-0.0, -0.1, -0.0, 0.1, 0.0, 0.1, -0.0, 0.1, 0.1, -0.1, -0.0, -0.1, 0.0, 0.2, -0.1, -0.0, -0.1, -0.1, 0.1, -0.0, 0.1, -0.2, 0.1, -0.0, -0.2},
{0.1, -0.2, 0.1, 0.1, 0.2, -0.1, 0.2, -0.1, 0.1, -0.2, 0.1, 0.1, -0.0, -0.2, -0.1, -0.0, 0.1, -0.2, 0.0, -0.2, 0.2, 0.2, -0.0, 0.1, 0.1},
{0.1, 0.1, 0.2, -0.1, 0.2, 0.2, 0.2, -0.2, 0.2, 0.1, -0.0, 0.1, 0.1, 0.1, 0.2, -0.2, 0.1, -0.1, -0.1, 0.0, -0.1, -0.0, -0.0, 0.2, 0.2},
{0.1, 0.0, -0.1, -0.0, -0.1, 0.1, 0.2, 0.2, 0.2, 0.1, -0.1, -0.2, -0.0, -0.2, -0.1, -0.1, 0.2, -0.1, -0.1, 0.0, -0.1, 0.1, -0.0, 0.2, 0.0},
{0.1, 0.1, 0.0, 0.0, -0.2, 0.0, 0.1, -0.0, -0.1, 0.2, 0.0, 0.1, -0.2, 0.1, 0.1, 0.1, 0.1, -0.0, 0.1, -0.0, -0.1, -0.1, -0.0, 0.0, 0.1},
{-0.0, -0.2, -0.1, 0.1, 0.1, -0.0, -0.1, 0.0, 0.2, -0.1, -0.1, 0.1, 0.1, -0.0, 0.1, 0.2, -0.1, -0.0, -0.1, 0.2, 0.0, 0.1, -0.1, 0.1, 0.0},
}
);
Matrix  transformer_layers_5_attention_key_bias   (
{-0.1, -0.0, -0.1, 0.0, 0.0, 0.2, 0.1, 0.1, -0.1, -0.0, 0.0, 0.1, -0.0, -0.1, -0.2, -0.1, -0.2, 0.0, -0.2, 0.1, -0.0, -0.0, 0.1, -0.2, 0.2}
);
Matrix  transformer_layers_5_attention_value_weight   (
{{-0.1, -0.1, -0.1, 0.2, -0.0, -0.2, 0.2, -0.1, -0.1, 0.2, -0.0, -0.1, 0.0, 0.1, -0.2, -0.0, 0.2, -0.2, -0.1, 0.1, -0.2, -0.2, -0.1, 0.2, 0.0},
{0.1, 0.1, 0.2, -0.1, -0.2, 0.1, -0.2, 0.2, 0.0, 0.2, 0.1, -0.2, 0.2, -0.1, 0.1, -0.1, 0.1, 0.0, 0.1, 0.0, 0.2, -0.0, 0.0, 0.0, 0.2},
{0.0, -0.2, -0.0, -0.1, 0.1, 0.2, -0.1, -0.2, 0.1, 0.1, 0.1, -0.1, 0.0, -0.1, -0.0, 0.0, -0.2, -0.1, -0.1, -0.1, 0.1, 0.2, 0.2, 0.1, 0.2},
{0.1, 0.0, -0.1, 0.0, -0.1, 0.0, -0.0, -0.0, 0.2, -0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, -0.0, 0.0, -0.2, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0},
{-0.1, -0.2, -0.0, -0.0, -0.1, -0.0, -0.2, -0.2, -0.1, -0.2, 0.2, 0.0, -0.1, -0.0, -0.1, 0.1, -0.1, 0.1, 0.1, -0.2, 0.1, 0.1, 0.0, -0.1, -0.1},
{-0.2, -0.1, -0.1, -0.1, -0.2, -0.2, -0.0, 0.1, 0.0, -0.2, 0.2, 0.1, -0.0, 0.0, -0.2, -0.0, -0.1, -0.2, -0.1, 0.1, 0.1, -0.2, -0.1, 0.1, -0.1},
{0.1, -0.0, -0.0, 0.0, 0.1, -0.0, 0.0, 0.1, 0.1, 0.1, -0.2, -0.0, 0.1, -0.1, -0.2, 0.2, 0.1, 0.1, 0.1, -0.1, -0.1, 0.0, -0.1, 0.0, 0.0},
{-0.1, -0.0, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.2, -0.1, -0.0, -0.1, 0.1, -0.2, -0.1, 0.1, 0.1, -0.1, 0.1, 0.2, -0.1, 0.1},
{-0.2, 0.1, -0.1, 0.1, 0.2, 0.0, 0.2, 0.0, -0.2, -0.1, 0.1, 0.2, -0.2, -0.0, -0.1, -0.0, 0.2, -0.0, -0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.0},
{0.1, 0.2, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.1, 0.0, 0.2, 0.0, 0.0, 0.2, -0.1, 0.2, -0.0, 0.2, 0.2, 0.1, -0.1, 0.2, 0.1, 0.1, -0.2},
{-0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.2, 0.1, -0.2, 0.1, 0.1, 0.2, -0.1, 0.0, 0.1, -0.2, 0.1, -0.1, -0.1, 0.1, 0.2, 0.1, -0.2, -0.2},
{0.1, -0.1, 0.0, -0.0, -0.0, -0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, -0.0, 0.1, 0.0, 0.0, 0.1, -0.2, -0.2, -0.0, 0.2, 0.0, 0.0, -0.1, 0.1},
{0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.0, -0.1, 0.0, -0.2, -0.1, -0.1, 0.1, 0.2, -0.1, -0.1, -0.1, -0.1, 0.1, -0.2, 0.1, 0.1, 0.1, -0.2, -0.2},
{0.0, 0.2, 0.2, 0.1, -0.1, -0.1, 0.1, -0.0, 0.1, -0.0, -0.1, -0.0, -0.0, -0.1, -0.0, -0.0, -0.0, -0.0, -0.1, 0.1, -0.2, -0.1, -0.1, 0.0, 0.2},
{-0.2, 0.0, -0.1, -0.1, -0.2, 0.0, 0.1, -0.0, 0.1, 0.1, -0.2, 0.0, 0.1, 0.1, -0.1, -0.0, 0.1, -0.2, -0.1, 0.1, 0.2, -0.1, 0.0, -0.0, 0.1},
{-0.2, 0.0, -0.0, -0.1, -0.1, 0.2, -0.2, 0.2, -0.2, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, 0.0, -0.2, -0.0, 0.2, -0.0, -0.1, 0.0, 0.1, 0.1},
{-0.0, -0.2, -0.2, 0.1, 0.0, -0.0, 0.1, -0.2, -0.1, 0.1, -0.2, 0.1, 0.0, 0.2, -0.1, -0.2, 0.2, -0.1, -0.1, 0.2, 0.1, 0.1, -0.0, 0.2, 0.2},
{0.2, -0.2, -0.1, 0.2, 0.1, 0.1, 0.1, 0.2, -0.1, -0.1, -0.0, -0.0, -0.0, -0.2, -0.0, -0.2, -0.1, -0.1, 0.1, -0.2, 0.2, -0.1, 0.1, -0.1, -0.2},
{0.1, -0.1, 0.0, 0.1, 0.0, -0.1, -0.0, -0.2, -0.0, 0.1, 0.2, 0.0, 0.0, -0.2, -0.1, -0.1, -0.1, -0.2, 0.2, 0.2, -0.1, 0.1, -0.1, 0.0, 0.1},
{0.0, -0.1, -0.2, 0.0, -0.0, 0.1, 0.0, 0.2, 0.1, -0.1, -0.2, 0.0, 0.1, -0.2, 0.0, 0.2, -0.0, 0.0, -0.1, -0.0, 0.1, -0.1, -0.1, 0.1, -0.1},
{-0.0, 0.2, 0.1, 0.1, -0.2, 0.0, 0.0, -0.1, 0.1, 0.1, 0.1, -0.1, -0.1, 0.0, -0.1, 0.2, 0.1, -0.0, -0.1, 0.1, -0.2, -0.1, -0.2, -0.2, 0.0},
{-0.1, 0.1, -0.1, 0.1, -0.1, -0.2, 0.1, 0.1, 0.2, -0.1, 0.2, 0.1, 0.2, 0.1, 0.0, 0.2, 0.0, -0.1, -0.2, 0.1, -0.1, -0.1, 0.1, 0.2, 0.1},
{-0.1, -0.1, 0.2, 0.0, -0.1, -0.2, 0.1, -0.1, 0.1, 0.2, -0.0, -0.2, 0.1, -0.0, -0.1, 0.0, 0.1, 0.1, -0.2, -0.2, -0.1, 0.0, -0.1, -0.0, 0.0},
{-0.1, -0.2, 0.2, 0.1, -0.2, -0.0, -0.0, 0.0, 0.1, 0.1, 0.1, -0.2, -0.2, -0.1, 0.2, -0.1, 0.1, -0.0, 0.1, -0.1, 0.0, 0.2, 0.1, 0.2, 0.1},
{0.2, -0.1, -0.0, -0.0, 0.1, 0.1, -0.2, -0.1, -0.0, 0.0, 0.0, 0.2, -0.0, 0.1, -0.1, -0.1, 0.1, 0.2, 0.0, 0.1, 0.1, -0.1, 0.1, 0.0, -0.1},
}
);
Matrix  transformer_layers_5_attention_value_bias   (
{0.0, 0.0, 0.0, -0.1, -0.1, 0.0, 0.1, 0.1, -0.0, -0.1, 0.2, 0.2, -0.0, -0.2, 0.0, -0.1, -0.1, 0.1, -0.1, -0.1, 0.0, -0.1, -0.1, 0.1, -0.1}
);
Matrix  transformer_layers_5_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_5_norm1_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_5_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_5_norm2_layer_norm_bias   (
{-0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_5_feed_forward_linear1_weight   (
{{0.1, 0.0, 0.2, -0.2, -0.1, 0.1, -0.1, -0.2, 0.0, -0.0, -0.2, -0.0, 0.0, -0.0, -0.2, -0.1, 0.0, 0.1, -0.1, -0.2, -0.1, -0.0, 0.1, 0.2, -0.2},
{0.1, 0.0, 0.2, 0.1, -0.0, -0.1, 0.0, 0.1, -0.2, 0.0, -0.1, 0.2, 0.2, -0.0, 0.2, 0.1, -0.1, 0.2, 0.1, 0.1, -0.1, -0.0, 0.0, -0.1, -0.1},
{0.0, -0.0, -0.1, -0.2, -0.0, -0.1, 0.1, -0.2, 0.0, -0.0, 0.2, -0.2, -0.1, -0.2, 0.1, -0.0, 0.1, -0.1, -0.0, -0.1, 0.1, -0.0, -0.2, 0.2, -0.2},
{-0.0, 0.1, 0.1, 0.0, -0.1, -0.1, -0.0, -0.1, -0.1, -0.1, -0.1, 0.2, 0.0, -0.1, 0.1, 0.1, -0.1, -0.2, -0.2, -0.0, 0.2, -0.0, -0.0, 0.1, -0.2},
{0.0, -0.1, 0.0, -0.0, -0.0, -0.2, -0.1, -0.1, -0.0, -0.0, -0.2, 0.0, 0.0, 0.1, 0.2, 0.1, 0.1, 0.2, -0.1, -0.1, 0.2, 0.1, -0.1, -0.2, -0.2},
{-0.1, -0.1, -0.0, 0.1, 0.0, 0.1, -0.1, -0.1, 0.2, -0.0, 0.2, -0.0, -0.1, 0.1, 0.0, 0.2, 0.1, -0.0, 0.1, -0.1, -0.1, -0.2, -0.1, 0.1, 0.1},
{-0.0, 0.1, -0.2, 0.1, -0.2, 0.1, 0.1, -0.1, 0.0, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, -0.0, 0.1, -0.2, -0.1, 0.1, -0.1, -0.2, 0.0, 0.1, -0.1},
{-0.1, 0.2, -0.2, 0.1, 0.0, 0.2, -0.0, 0.1, -0.1, -0.0, -0.1, 0.0, 0.2, 0.1, -0.1, 0.2, 0.1, -0.2, -0.1, 0.1, -0.2, 0.1, 0.2, -0.0, -0.1},
{0.0, -0.0, -0.2, 0.2, 0.0, -0.2, 0.0, -0.2, -0.1, 0.2, -0.1, -0.1, -0.1, 0.2, -0.1, -0.0, 0.1, -0.0, 0.0, 0.2, -0.1, -0.1, -0.1, -0.1, 0.2},
{0.0, 0.1, 0.2, 0.0, 0.1, -0.1, 0.2, -0.2, 0.2, 0.0, 0.2, -0.1, 0.1, 0.0, 0.2, -0.1, -0.1, -0.1, -0.0, -0.1, -0.2, -0.1, 0.1, 0.2, -0.0},
{-0.0, -0.0, -0.2, -0.0, -0.2, -0.1, 0.2, 0.1, -0.2, -0.2, -0.0, -0.2, -0.2, -0.1, -0.0, 0.0, 0.2, -0.1, 0.1, -0.2, -0.1, 0.0, 0.2, 0.0, 0.0},
{-0.0, -0.0, -0.0, 0.2, -0.2, 0.1, 0.1, -0.2, -0.2, 0.2, -0.1, 0.0, -0.0, 0.1, -0.2, -0.0, -0.1, 0.2, 0.2, 0.2, -0.2, 0.1, 0.1, 0.1, -0.1},
{-0.1, -0.1, 0.1, 0.1, 0.1, 0.0, -0.1, -0.1, -0.0, -0.1, -0.0, -0.1, -0.1, -0.2, -0.0, -0.0, -0.1, 0.0, -0.1, 0.1, 0.0, 0.2, 0.1, 0.1, -0.2},
{-0.2, 0.2, -0.1, 0.2, 0.2, 0.1, 0.1, 0.1, -0.1, 0.0, 0.1, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.1, -0.0, -0.2, 0.0, -0.1, -0.0, 0.2, -0.1},
{-0.1, 0.1, -0.2, -0.1, 0.0, -0.1, -0.1, -0.1, -0.0, -0.0, 0.1, -0.1, -0.2, -0.2, -0.1, -0.1, -0.2, 0.0, -0.1, 0.1, 0.2, -0.2, 0.2, 0.1, -0.1},
}
);
Matrix  transformer_layers_5_feed_forward_linear1_bias   (
{0.1, 0.1, 0.0, 0.1, -0.2, -0.1, -0.2, -0.0, -0.1, -0.2, 0.2, -0.2, -0.2, -0.1, -0.2}
);
Matrix  transformer_layers_5_feed_forward_linear2_weight   (
{{-0.1, -0.2, -0.2, 0.2, 0.3, 0.2, -0.2, 0.0, 0.2, 0.2, -0.1, -0.1, 0.1, -0.2, -0.1},
{-0.1, -0.2, -0.2, 0.0, 0.1, 0.1, -0.1, 0.1, 0.2, 0.1, -0.0, -0.2, 0.2, -0.1, -0.1},
{-0.0, -0.1, -0.0, -0.0, 0.0, -0.2, 0.1, 0.1, -0.2, -0.1, -0.0, -0.2, -0.1, -0.2, -0.1},
{-0.1, 0.1, 0.0, 0.2, 0.1, -0.1, -0.1, -0.1, -0.0, -0.2, 0.2, -0.0, -0.1, 0.1, -0.2},
{0.2, -0.1, -0.1, 0.2, -0.1, 0.2, -0.0, -0.0, 0.1, -0.2, -0.0, -0.1, -0.1, -0.2, 0.1},
{-0.0, -0.2, -0.2, 0.1, -0.3, -0.0, 0.0, 0.2, 0.1, 0.1, -0.1, -0.2, -0.0, 0.2, -0.1},
{-0.2, -0.1, -0.2, -0.1, -0.1, 0.1, -0.0, -0.2, -0.1, 0.2, -0.1, -0.2, 0.0, 0.2, 0.0},
{-0.1, -0.2, 0.2, 0.0, -0.2, -0.1, -0.0, -0.2, -0.2, -0.2, -0.0, 0.0, -0.0, -0.1, -0.1},
{-0.1, -0.3, 0.0, 0.1, 0.2, 0.0, -0.1, -0.2, -0.2, 0.1, 0.2, -0.2, -0.1, -0.2, 0.1},
{0.2, -0.2, 0.1, -0.2, -0.2, 0.2, -0.3, 0.2, -0.1, 0.1, 0.2, 0.1, 0.0, 0.2, 0.1},
{0.1, -0.0, -0.3, -0.1, -0.1, 0.1, -0.2, 0.2, 0.0, 0.0, 0.1, -0.2, 0.1, 0.0, 0.2},
{0.2, 0.1, 0.1, -0.1, -0.1, -0.3, -0.2, 0.1, -0.2, 0.2, -0.0, -0.2, -0.1, -0.1, 0.1},
{0.2, 0.3, 0.0, -0.0, 0.2, 0.2, 0.2, -0.1, -0.1, 0.2, -0.2, 0.2, -0.1, -0.1, 0.1},
{0.0, 0.1, 0.1, 0.1, -0.2, 0.2, -0.2, 0.2, -0.1, 0.2, -0.1, -0.2, 0.2, -0.1, -0.2},
{0.0, -0.0, -0.2, 0.2, 0.0, 0.1, 0.2, -0.1, 0.1, -0.2, 0.2, 0.1, -0.0, -0.1, -0.2},
{-0.3, 0.2, 0.2, 0.1, -0.2, -0.1, 0.2, -0.1, 0.2, 0.1, 0.1, -0.2, 0.2, 0.1, -0.1},
{0.0, 0.0, 0.1, 0.2, -0.0, -0.2, 0.1, 0.2, -0.1, -0.2, 0.1, -0.1, -0.0, -0.1, -0.0},
{0.1, -0.2, -0.0, 0.1, -0.0, -0.0, -0.1, 0.1, 0.2, -0.1, 0.2, -0.1, -0.0, -0.2, -0.0},
{-0.1, 0.2, -0.2, 0.1, -0.1, 0.1, -0.1, 0.0, -0.0, -0.2, 0.1, -0.2, 0.2, 0.2, 0.0},
{0.2, -0.0, -0.0, -0.2, 0.1, -0.2, -0.2, 0.0, -0.2, -0.1, -0.0, 0.1, -0.1, -0.2, -0.1},
{0.2, -0.1, -0.2, -0.0, -0.1, -0.1, 0.1, -0.0, 0.2, 0.0, -0.2, -0.1, -0.1, -0.2, -0.1},
{0.2, -0.1, 0.2, 0.1, 0.0, 0.2, 0.1, 0.0, 0.1, -0.2, 0.1, 0.1, -0.1, 0.0, 0.2},
{0.2, -0.2, -0.1, 0.1, -0.0, -0.1, -0.1, -0.3, -0.0, -0.2, 0.2, -0.1, 0.0, -0.0, -0.1},
{-0.1, 0.2, -0.2, -0.2, 0.2, 0.1, 0.0, 0.1, 0.1, 0.1, -0.2, 0.2, -0.1, 0.2, -0.0},
{0.2, -0.2, 0.2, 0.2, 0.0, -0.2, 0.1, -0.2, 0.2, -0.0, 0.2, 0.2, 0.1, 0.1, 0.2},
}
);
Matrix  transformer_layers_5_feed_forward_linear2_bias   (
{-0.2, -0.1, 0.0, 0.2, -0.2, 0.2, 0.1, -0.2, 0.2, 0.1, 0.0, 0.1, -0.0, -0.1, 0.1, -0.1, -0.2, -0.0, 0.1, -0.2, 0.3, 0.0, 0.2, -0.1, 0.2}
);
Matrix  transformer_layers_5_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_5_feed_forward_ln1_layer_norm_bias   (
{0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_5_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_5_feed_forward_ln2_layer_norm_bias   (
{-0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_6_attention_query_weight   (
{{-0.1, 0.1, 0.1, -0.2, -0.0, -0.0, 0.0, -0.1, -0.1, 0.2, 0.0, -0.1, 0.1, -0.2, -0.2, -0.1, 0.1, -0.1, -0.1, 0.1, -0.0, 0.1, 0.1, -0.2, 0.2},
{-0.1, -0.0, -0.1, -0.0, -0.1, -0.0, -0.1, -0.1, -0.1, 0.1, -0.2, 0.0, 0.2, 0.0, -0.2, -0.1, -0.1, 0.0, 0.1, -0.1, -0.0, 0.1, -0.2, -0.2, 0.0},
{0.1, -0.1, 0.1, -0.0, -0.2, 0.0, -0.1, -0.2, -0.2, -0.2, -0.2, -0.1, 0.0, 0.1, -0.0, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.2, -0.1, 0.1, 0.1},
{0.2, -0.1, 0.0, 0.1, -0.1, -0.1, 0.1, -0.1, 0.0, -0.0, 0.1, -0.1, 0.2, -0.2, -0.0, -0.1, -0.1, -0.2, -0.2, 0.1, 0.0, -0.1, -0.1, 0.0, 0.0},
{0.1, -0.1, 0.1, -0.2, -0.0, 0.1, -0.1, 0.1, 0.0, -0.0, 0.1, 0.1, -0.1, 0.1, 0.0, -0.1, 0.2, -0.1, 0.2, 0.0, -0.0, 0.0, 0.1, -0.1, -0.2},
{-0.1, -0.0, -0.1, -0.1, -0.1, -0.2, -0.1, 0.0, -0.0, 0.1, -0.0, -0.1, -0.2, -0.1, 0.1, 0.2, -0.2, -0.2, 0.2, -0.2, 0.2, -0.2, -0.1, -0.2, 0.2},
{0.0, -0.0, -0.0, -0.2, 0.2, 0.2, -0.2, 0.1, -0.2, -0.1, 0.1, 0.1, 0.0, 0.1, -0.1, 0.0, -0.1, 0.1, -0.2, -0.0, -0.1, -0.1, 0.1, -0.1, 0.0},
{0.0, -0.2, -0.0, 0.1, 0.1, -0.0, -0.0, -0.1, -0.0, -0.2, -0.1, -0.1, -0.0, -0.1, 0.1, -0.1, -0.2, -0.1, 0.2, -0.0, -0.1, 0.0, 0.2, -0.2, 0.2},
{-0.0, 0.2, -0.1, -0.1, 0.1, -0.2, -0.2, 0.1, -0.0, -0.1, 0.0, -0.0, 0.0, -0.0, -0.1, -0.0, -0.1, -0.2, -0.2, 0.0, 0.0, -0.1, 0.0, 0.1, -0.2},
{-0.1, -0.1, -0.0, -0.0, 0.1, -0.0, -0.0, -0.1, -0.0, 0.2, -0.0, 0.0, -0.0, 0.0, 0.1, 0.0, 0.2, -0.1, -0.1, -0.0, -0.0, -0.0, -0.1, 0.2, 0.1},
{-0.0, 0.1, 0.0, 0.1, -0.2, 0.0, 0.0, 0.1, 0.0, -0.1, 0.1, 0.2, -0.1, -0.0, -0.1, -0.2, 0.1, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1},
{0.1, -0.1, 0.2, 0.2, -0.1, 0.0, 0.1, 0.1, -0.2, -0.0, -0.1, 0.1, 0.1, -0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, -0.0, 0.1, -0.0, -0.2, 0.1},
{0.2, 0.2, 0.1, -0.0, 0.1, 0.1, 0.1, -0.0, -0.1, -0.1, -0.0, 0.1, 0.1, 0.2, 0.0, -0.1, 0.1, -0.0, 0.1, 0.2, 0.1, -0.1, -0.1, 0.1, -0.1},
{-0.0, -0.1, -0.1, -0.1, -0.2, 0.0, 0.0, -0.1, 0.0, 0.0, -0.2, -0.2, -0.2, -0.1, -0.1, 0.1, -0.0, -0.1, -0.2, 0.0, 0.0, -0.2, 0.2, 0.2, 0.2},
{0.1, -0.0, -0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.0, -0.0, 0.0, 0.1, -0.0, -0.2, 0.1, -0.1, -0.2, 0.0, -0.0, -0.0, 0.0, 0.1},
{0.0, 0.1, -0.1, 0.1, -0.2, -0.1, 0.0, 0.1, -0.0, 0.0, 0.0, -0.1, -0.0, 0.1, 0.1, 0.2, -0.1, -0.1, -0.1, 0.2, 0.2, 0.1, -0.1, 0.1, -0.0},
{-0.1, -0.1, 0.1, 0.2, 0.2, -0.1, -0.2, 0.1, 0.1, -0.0, 0.2, 0.1, 0.0, -0.0, -0.1, 0.1, -0.2, 0.1, 0.1, 0.1, -0.1, -0.0, 0.2, -0.1, -0.0},
{0.1, -0.0, 0.2, -0.1, 0.0, -0.2, -0.1, -0.1, -0.2, 0.2, -0.1, -0.1, 0.2, -0.2, 0.2, -0.1, 0.0, -0.1, 0.1, 0.2, 0.0, 0.0, -0.0, -0.0, 0.0},
{0.1, -0.1, 0.1, -0.2, 0.2, 0.2, -0.2, -0.2, -0.1, 0.1, 0.2, 0.2, -0.1, 0.1, 0.1, -0.1, 0.2, -0.2, -0.2, -0.2, 0.1, -0.1, 0.0, -0.1, 0.0},
{-0.2, 0.2, -0.0, 0.0, -0.2, -0.1, 0.0, -0.2, 0.1, -0.0, -0.2, 0.1, 0.1, 0.1, 0.1, -0.1, -0.2, -0.0, 0.1, 0.1, -0.1, -0.0, 0.2, 0.1, 0.1},
{-0.1, -0.1, 0.2, -0.0, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1, 0.1, 0.1, 0.2, -0.2, 0.1, 0.0, -0.0, 0.2, -0.1, -0.2, 0.2, -0.2, 0.1, -0.1, -0.2},
{0.1, 0.1, -0.1, 0.1, 0.0, -0.2, -0.2, -0.1, -0.1, -0.1, 0.1, -0.0, -0.0, 0.2, 0.1, -0.1, -0.2, -0.0, -0.2, -0.2, 0.2, 0.2, -0.0, 0.1, 0.0},
{-0.0, -0.2, 0.1, -0.2, -0.0, 0.2, 0.2, 0.2, -0.1, 0.1, 0.1, 0.1, -0.0, -0.1, -0.0, 0.2, -0.1, 0.2, -0.2, -0.2, -0.1, 0.1, -0.1, 0.2, -0.1},
{0.1, -0.1, -0.1, -0.1, 0.0, 0.0, 0.1, -0.2, -0.2, -0.2, -0.1, -0.1, 0.1, -0.2, -0.1, -0.2, 0.0, 0.1, -0.1, -0.0, 0.1, -0.1, 0.1, -0.0, -0.1},
{0.1, -0.0, -0.0, -0.2, -0.0, 0.0, 0.0, -0.2, -0.1, 0.1, -0.0, -0.2, 0.0, -0.2, 0.0, 0.0, 0.0, -0.1, -0.1, 0.0, 0.1, 0.1, 0.1, 0.0, 0.1},
}
);
Matrix  transformer_layers_6_attention_query_bias   (
{0.1, -0.1, -0.1, -0.1, -0.0, 0.1, 0.0, 0.0, 0.2, -0.0, 0.0, -0.1, -0.1, -0.1, 0.0, -0.1, 0.1, -0.1, 0.2, -0.1, -0.0, 0.1, 0.1, 0.2, 0.1}
);
Matrix  transformer_layers_6_attention_key_weight   (
{{-0.0, -0.0, -0.1, -0.1, 0.0, -0.1, -0.1, 0.1, 0.0, -0.2, -0.1, -0.1, -0.0, -0.0, -0.1, 0.2, -0.1, 0.2, -0.1, -0.1, -0.1, 0.1, 0.2, -0.1, 0.1},
{0.1, 0.1, 0.1, -0.0, -0.1, 0.1, -0.1, 0.2, -0.1, 0.2, -0.0, -0.2, -0.2, -0.1, 0.1, 0.2, 0.1, -0.2, -0.1, 0.2, 0.1, -0.1, -0.1, 0.1, 0.0},
{-0.0, -0.1, 0.0, -0.1, 0.1, -0.0, -0.2, 0.2, 0.2, -0.2, 0.1, -0.0, 0.1, -0.0, -0.1, 0.0, -0.2, -0.2, -0.0, 0.2, 0.0, -0.2, -0.0, 0.1, -0.1},
{0.2, -0.1, -0.2, -0.1, 0.2, 0.1, -0.1, -0.2, -0.0, 0.1, -0.1, -0.0, 0.0, -0.0, 0.1, 0.0, -0.1, -0.1, -0.2, -0.0, -0.0, 0.1, 0.1, 0.2, -0.2},
{-0.0, -0.2, 0.2, 0.0, 0.0, -0.2, 0.2, 0.2, 0.1, 0.1, 0.2, -0.1, 0.2, 0.1, -0.1, -0.0, -0.0, -0.0, 0.0, -0.2, 0.0, 0.2, 0.1, 0.1, -0.1},
{-0.0, -0.0, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, -0.0, -0.1, 0.1, 0.0, 0.0, -0.1, -0.2, 0.1, 0.1, -0.2, -0.1, 0.2, 0.2, 0.2, 0.2, -0.2},
{-0.1, 0.2, 0.1, 0.1, -0.0, -0.1, -0.1, -0.1, 0.2, 0.2, 0.1, -0.1, 0.2, -0.1, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0, 0.1, 0.1, -0.0, 0.1},
{-0.1, -0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, -0.2, -0.1, -0.1, 0.2, 0.0, 0.1, -0.2, 0.2, 0.2, -0.0, 0.1, 0.0, -0.2, -0.1, 0.2, 0.0, -0.2},
{0.2, 0.1, -0.1, -0.1, -0.2, -0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 0.1, 0.0, -0.1, 0.1, -0.2, -0.0, 0.1, -0.2, 0.1, 0.1, -0.1, 0.1, -0.1, 0.0},
{0.2, 0.1, 0.0, -0.1, 0.1, -0.1, 0.1, -0.1, -0.1, -0.1, -0.0, -0.2, 0.2, -0.2, -0.1, -0.1, 0.2, 0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.0, 0.1},
{-0.0, -0.0, -0.1, 0.1, -0.1, 0.1, 0.2, -0.1, 0.1, -0.1, 0.1, 0.0, -0.1, -0.2, -0.0, 0.1, 0.1, 0.1, 0.1, -0.0, -0.0, 0.2, -0.1, 0.1, -0.1},
{0.0, -0.1, 0.2, -0.1, 0.1, 0.0, -0.0, -0.0, -0.2, -0.1, -0.1, 0.0, 0.1, -0.1, -0.1, 0.0, -0.1, -0.0, 0.0, -0.1, 0.2, 0.0, -0.0, -0.0, 0.1},
{-0.0, 0.2, -0.1, -0.2, -0.2, -0.1, -0.1, -0.0, -0.1, -0.2, 0.0, 0.0, -0.0, 0.0, -0.0, -0.1, -0.0, -0.1, 0.0, -0.0, 0.2, 0.1, 0.1, -0.2, 0.0},
{0.0, -0.0, 0.2, 0.2, -0.1, -0.1, 0.2, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, 0.1, -0.0, 0.2, -0.1, -0.0, 0.0, 0.0, 0.2, -0.1},
{-0.1, 0.1, -0.1, 0.1, 0.1, 0.2, -0.2, 0.2, 0.1, 0.0, -0.1, -0.0, -0.1, -0.2, 0.2, 0.2, -0.1, -0.2, -0.1, -0.1, 0.1, -0.2, -0.1, 0.1, 0.1},
{0.0, 0.2, -0.0, 0.2, -0.1, 0.1, -0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.2, 0.1, -0.1, -0.0, -0.1, 0.2, -0.0, -0.2, 0.1, -0.2, 0.1, -0.1, -0.2},
{-0.1, 0.2, -0.2, -0.0, 0.1, 0.2, 0.1, -0.1, -0.2, 0.2, -0.1, 0.1, 0.1, -0.1, 0.0, 0.0, 0.1, 0.2, -0.1, 0.2, -0.2, 0.2, 0.2, 0.1, 0.2},
{-0.1, -0.0, -0.1, -0.1, 0.0, -0.2, -0.1, 0.0, -0.1, 0.0, 0.2, 0.1, 0.0, 0.0, -0.2, 0.2, -0.2, -0.1, -0.1, 0.0, 0.0, -0.1, -0.0, -0.0, 0.0},
{0.2, 0.0, -0.1, 0.1, -0.1, 0.1, 0.2, -0.0, -0.0, -0.2, 0.1, -0.1, -0.2, 0.1, -0.1, 0.2, -0.1, 0.2, -0.0, 0.2, -0.2, -0.0, -0.1, 0.1, 0.0},
{0.1, 0.0, 0.1, 0.1, -0.0, 0.1, -0.1, 0.1, -0.1, 0.2, -0.1, 0.0, -0.2, 0.0, 0.1, -0.1, -0.2, 0.0, 0.1, -0.1, -0.0, -0.2, 0.2, 0.0, 0.0},
{-0.0, 0.1, -0.0, 0.1, 0.0, 0.2, 0.0, -0.0, 0.1, -0.2, -0.0, 0.1, -0.2, 0.1, 0.1, 0.1, 0.1, -0.2, -0.1, -0.1, 0.2, 0.1, -0.1, -0.1, 0.1},
{-0.1, 0.2, -0.2, -0.1, -0.1, 0.1, 0.1, -0.2, -0.1, -0.0, -0.2, -0.1, 0.1, -0.1, -0.0, 0.1, 0.0, -0.1, 0.0, -0.0, 0.1, 0.1, 0.2, -0.0, 0.1},
{-0.2, 0.1, -0.1, -0.0, 0.1, -0.1, -0.1, 0.0, -0.0, -0.1, 0.2, -0.1, 0.0, -0.0, -0.1, -0.1, -0.0, 0.2, -0.1, -0.0, -0.1, -0.0, -0.1, 0.0, 0.2},
{0.2, 0.1, -0.2, 0.0, 0.2, 0.0, 0.0, 0.0, 0.1, 0.0, 0.1, -0.0, 0.1, -0.0, -0.1, -0.0, -0.0, 0.2, 0.2, -0.0, -0.1, -0.0, 0.2, -0.0, 0.2},
{-0.2, 0.2, 0.0, 0.1, 0.1, -0.0, -0.1, -0.0, -0.2, 0.1, -0.1, 0.0, 0.1, -0.2, 0.2, 0.2, 0.0, 0.0, 0.1, 0.2, -0.0, 0.1, 0.2, -0.1, 0.1},
}
);
Matrix  transformer_layers_6_attention_key_bias   (
{0.1, -0.1, 0.2, -0.1, 0.2, 0.2, 0.0, -0.1, -0.1, 0.0, -0.1, 0.0, 0.1, -0.1, 0.0, -0.1, -0.1, 0.1, -0.2, -0.0, -0.1, -0.0, 0.2, -0.0, -0.2}
);
Matrix  transformer_layers_6_attention_value_weight   (
{{-0.0, 0.1, 0.2, -0.1, 0.2, 0.2, 0.2, -0.1, -0.1, -0.1, -0.1, 0.0, 0.1, -0.1, 0.2, 0.2, -0.1, -0.1, 0.2, -0.2, -0.1, -0.2, 0.1, -0.0, 0.2},
{0.0, -0.2, 0.0, 0.1, 0.1, 0.0, -0.1, 0.2, -0.2, 0.1, 0.2, -0.1, -0.1, -0.2, -0.1, 0.2, 0.1, -0.2, 0.1, -0.0, 0.1, -0.0, -0.0, -0.2, -0.1},
{0.0, 0.2, -0.2, -0.1, -0.1, -0.1, -0.2, -0.1, -0.2, -0.1, -0.1, -0.2, 0.1, -0.0, 0.1, -0.2, -0.1, 0.1, 0.1, -0.1, -0.1, 0.2, 0.0, 0.0, -0.1},
{0.0, 0.2, -0.1, -0.0, -0.1, 0.2, -0.1, 0.1, -0.0, 0.1, 0.0, 0.1, -0.0, -0.1, -0.1, -0.1, -0.2, -0.1, 0.1, 0.1, -0.2, -0.2, -0.2, -0.1, -0.0},
{0.0, -0.1, 0.0, -0.0, -0.1, -0.2, 0.1, -0.2, -0.0, -0.1, -0.1, 0.1, -0.1, 0.1, 0.2, 0.0, 0.0, 0.1, 0.0, -0.2, 0.2, 0.2, -0.1, 0.1, -0.1},
{0.0, 0.1, 0.1, -0.2, 0.1, -0.2, 0.1, -0.1, 0.1, 0.2, 0.1, -0.0, 0.0, 0.1, -0.1, 0.1, 0.2, 0.2, 0.2, -0.1, -0.1, -0.0, -0.1, 0.1, 0.0},
{0.2, -0.1, -0.0, 0.1, -0.1, 0.0, 0.0, 0.2, -0.1, -0.1, -0.2, 0.0, -0.0, 0.2, -0.0, -0.1, 0.2, -0.1, 0.2, 0.1, -0.1, 0.2, -0.1, -0.1, -0.1},
{-0.1, 0.1, -0.0, -0.1, -0.0, 0.1, 0.2, -0.0, -0.1, 0.2, 0.1, 0.0, -0.2, -0.0, 0.1, 0.2, -0.0, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.0, -0.1},
{-0.1, 0.1, -0.1, 0.2, 0.1, 0.0, 0.2, 0.1, 0.1, -0.1, 0.1, -0.0, -0.1, -0.0, 0.0, -0.0, 0.1, -0.1, -0.1, 0.0, 0.2, 0.0, -0.1, 0.1, -0.1},
{-0.1, 0.0, -0.0, -0.1, -0.2, -0.1, -0.1, 0.2, 0.1, -0.1, 0.1, 0.1, -0.0, -0.2, 0.1, 0.1, -0.1, 0.2, 0.0, -0.0, 0.2, 0.1, -0.0, 0.1, 0.1},
{0.1, 0.2, -0.0, -0.0, 0.2, -0.2, -0.0, -0.0, -0.1, -0.1, -0.1, -0.1, -0.2, -0.1, 0.0, 0.1, -0.2, 0.0, -0.2, -0.1, -0.2, 0.2, -0.1, 0.2, 0.2},
{-0.1, -0.1, -0.2, 0.2, -0.1, 0.2, 0.2, -0.1, 0.0, -0.1, -0.1, 0.1, 0.0, -0.0, 0.1, -0.1, -0.0, -0.2, -0.1, -0.1, -0.0, -0.1, 0.1, -0.1, -0.2},
{-0.2, -0.1, 0.1, -0.1, 0.1, -0.1, 0.0, -0.2, 0.1, 0.2, -0.1, 0.1, 0.1, -0.1, 0.0, 0.2, -0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, -0.1, -0.2},
{-0.1, -0.1, -0.2, 0.2, -0.0, 0.0, 0.0, -0.0, -0.2, -0.1, -0.2, 0.1, 0.2, 0.2, 0.0, -0.0, -0.1, -0.1, 0.0, -0.1, -0.0, 0.2, 0.1, 0.1, 0.1},
{0.2, -0.0, 0.2, 0.1, -0.0, 0.0, -0.1, 0.0, 0.1, 0.0, -0.0, 0.1, -0.1, -0.0, 0.0, -0.1, 0.0, 0.1, -0.0, 0.1, 0.2, 0.2, -0.2, -0.1, -0.1},
{-0.2, -0.0, -0.1, -0.1, -0.1, -0.0, -0.1, 0.1, -0.1, -0.2, -0.1, 0.2, -0.2, -0.1, 0.1, 0.1, -0.0, -0.1, -0.1, -0.1, -0.2, 0.0, 0.1, -0.0, 0.2},
{-0.0, -0.2, 0.2, -0.1, -0.1, 0.0, 0.2, -0.2, -0.1, -0.0, 0.2, 0.1, -0.1, -0.0, -0.1, -0.1, -0.1, 0.0, -0.0, 0.0, 0.0, -0.1, 0.2, 0.1, -0.2},
{0.1, -0.1, -0.2, -0.0, -0.2, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.0, 0.1, -0.2, -0.2, -0.1, -0.1, 0.1, 0.0, 0.0, -0.1, -0.1, -0.1, -0.1, 0.0},
{-0.1, 0.2, 0.1, 0.2, 0.1, -0.0, -0.1, -0.2, -0.2, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, -0.0, -0.0, 0.1, 0.1, -0.1, 0.2},
{0.0, -0.2, 0.1, -0.2, 0.0, -0.2, 0.2, 0.0, -0.1, 0.2, 0.0, 0.0, 0.2, 0.0, 0.1, 0.2, -0.2, 0.0, -0.1, 0.0, -0.1, 0.1, 0.0, 0.1, -0.2},
{0.0, 0.1, 0.1, -0.0, -0.2, 0.1, 0.1, 0.1, 0.0, 0.1, 0.0, -0.1, 0.1, -0.1, 0.0, -0.1, 0.1, 0.2, -0.2, 0.1, 0.1, 0.1, -0.0, 0.0, -0.1},
{0.1, -0.0, -0.2, -0.2, -0.2, 0.2, 0.2, 0.1, -0.1, -0.2, -0.0, -0.1, 0.0, 0.1, 0.1, -0.0, -0.0, 0.0, 0.0, -0.1, 0.2, -0.1, -0.2, 0.1, -0.1},
{-0.1, 0.1, 0.0, -0.2, 0.0, -0.0, -0.0, 0.0, -0.1, -0.2, -0.1, 0.1, -0.0, -0.2, 0.1, -0.2, 0.2, -0.1, 0.1, -0.1, -0.1, 0.2, 0.1, -0.2, 0.0},
{-0.1, -0.2, -0.1, -0.1, -0.1, 0.0, -0.0, 0.1, 0.2, 0.1, -0.2, -0.1, 0.1, 0.2, -0.2, 0.1, 0.2, 0.1, -0.2, 0.0, -0.2, -0.1, 0.1, -0.1, -0.2},
{-0.2, -0.2, -0.0, 0.1, 0.2, 0.2, 0.0, -0.1, -0.1, -0.1, -0.2, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1},
}
);
Matrix  transformer_layers_6_attention_value_bias   (
{-0.2, -0.2, 0.1, 0.2, 0.1, 0.2, -0.1, 0.1, 0.0, 0.2, -0.1, -0.0, -0.1, -0.1, -0.2, -0.1, 0.2, -0.0, -0.0, 0.1, -0.1, -0.1, 0.1, -0.2, 0.1}
);
Matrix  transformer_layers_6_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_6_norm1_layer_norm_bias   (
{-0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_6_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_6_norm2_layer_norm_bias   (
{0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_6_feed_forward_linear1_weight   (
{{0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, -0.2, -0.1, -0.2, 0.2, 0.1, -0.1, 0.1, -0.1, 0.0, -0.1, -0.1, 0.1, 0.0, -0.1},
{0.2, -0.1, -0.1, 0.2, -0.1, 0.1, 0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.1, 0.0, -0.0, -0.2, -0.1, -0.1, 0.1, -0.0, 0.2, -0.1, 0.1, 0.2, 0.1},
{-0.0, -0.1, 0.1, -0.1, 0.1, -0.1, -0.1, -0.1, 0.0, -0.1, -0.0, -0.2, 0.1, 0.2, -0.1, 0.0, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.2, -0.1, 0.1},
{0.2, -0.1, 0.1, -0.1, 0.1, -0.1, -0.0, -0.1, -0.2, 0.1, 0.2, 0.1, -0.1, -0.1, -0.2, 0.2, 0.2, -0.2, 0.1, 0.2, -0.0, -0.0, 0.1, -0.0, -0.0},
{-0.0, 0.2, -0.1, 0.1, -0.1, -0.2, -0.0, -0.0, -0.2, -0.1, 0.2, 0.0, 0.2, 0.1, 0.2, 0.1, -0.1, -0.1, 0.0, 0.1, 0.1, -0.2, 0.1, -0.0, -0.0},
{0.1, -0.0, 0.1, 0.0, 0.1, 0.0, 0.1, -0.2, 0.1, -0.1, 0.1, -0.1, -0.1, -0.1, 0.1, 0.2, 0.0, -0.1, -0.0, -0.0, -0.1, 0.1, 0.1, 0.2, -0.1},
{-0.1, 0.1, 0.1, 0.2, -0.2, -0.0, 0.0, 0.1, 0.2, -0.1, 0.1, -0.2, 0.2, 0.0, 0.2, -0.1, 0.1, -0.0, 0.1, -0.1, 0.1, 0.2, -0.1, -0.0, 0.0},
{0.1, -0.0, -0.2, -0.2, 0.2, -0.0, -0.2, -0.2, -0.2, 0.1, 0.1, -0.0, 0.0, 0.2, 0.1, 0.1, -0.0, 0.1, -0.0, -0.1, -0.2, 0.1, 0.1, -0.2, -0.0},
{-0.1, 0.2, 0.1, -0.0, -0.1, -0.1, 0.0, -0.1, -0.2, -0.1, -0.1, -0.2, -0.1, 0.1, -0.2, -0.2, 0.2, 0.1, -0.2, -0.1, 0.1, 0.1, -0.1, -0.1, -0.2},
{-0.1, 0.0, -0.1, -0.1, -0.1, 0.2, -0.1, 0.2, 0.1, 0.0, 0.2, 0.0, -0.0, 0.1, -0.1, -0.2, 0.1, -0.1, -0.0, -0.0, 0.1, 0.2, -0.0, -0.1, -0.1},
{-0.1, 0.0, -0.1, -0.2, 0.2, 0.1, 0.0, -0.0, -0.2, -0.1, 0.0, -0.1, -0.2, 0.0, -0.0, 0.1, 0.0, 0.1, 0.2, -0.1, 0.2, 0.2, -0.1, 0.1, 0.2},
{0.1, -0.1, 0.1, 0.1, 0.1, 0.0, 0.0, -0.1, -0.0, -0.0, 0.1, -0.0, -0.1, -0.1, -0.2, -0.0, -0.1, -0.0, 0.1, 0.0, -0.1, -0.2, -0.1, -0.1, 0.1},
{0.0, 0.2, -0.2, -0.2, 0.1, -0.2, 0.1, 0.1, -0.1, -0.2, -0.2, -0.1, 0.1, -0.2, 0.1, -0.1, -0.1, -0.1, 0.1, -0.1, -0.2, -0.0, 0.2, -0.1, 0.0},
{-0.0, -0.1, 0.0, 0.2, 0.0, -0.1, -0.0, -0.0, 0.2, -0.1, -0.0, 0.0, -0.0, 0.1, 0.1, 0.0, 0.0, 0.2, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1},
{-0.0, 0.1, 0.1, 0.2, 0.1, 0.1, -0.0, -0.0, -0.2, -0.1, 0.0, -0.1, 0.0, -0.1, 0.2, 0.0, 0.1, 0.1, 0.1, 0.2, -0.2, 0.1, 0.1, 0.1, 0.0},
}
);
Matrix  transformer_layers_6_feed_forward_linear1_bias   (
{0.1, -0.2, -0.1, -0.0, 0.2, 0.1, -0.1, 0.0, -0.1, 0.0, 0.1, -0.0, 0.2, -0.1, 0.0}
);
Matrix  transformer_layers_6_feed_forward_linear2_weight   (
{{-0.1, 0.2, -0.2, 0.1, 0.1, 0.3, 0.1, -0.2, 0.1, -0.2, -0.2, 0.1, 0.2, 0.1, -0.2},
{0.1, -0.1, -0.0, 0.1, 0.0, -0.2, -0.1, -0.0, 0.2, -0.1, 0.1, 0.1, -0.1, -0.1, 0.0},
{-0.1, -0.0, -0.1, 0.1, 0.0, -0.3, 0.1, 0.0, 0.1, 0.2, -0.0, 0.0, -0.2, 0.1, 0.1},
{-0.0, -0.2, -0.1, 0.1, 0.2, -0.2, 0.1, 0.1, -0.0, -0.2, -0.0, 0.1, 0.2, -0.2, -0.2},
{0.2, -0.1, 0.0, -0.2, 0.1, 0.0, 0.2, -0.1, -0.1, 0.1, -0.2, -0.0, -0.1, 0.2, 0.1},
{-0.2, -0.2, 0.1, 0.0, 0.1, 0.2, 0.1, -0.1, -0.2, -0.2, -0.0, 0.1, 0.1, 0.0, -0.1},
{-0.1, 0.2, 0.2, -0.1, -0.0, -0.0, 0.1, 0.0, -0.2, -0.1, -0.2, -0.0, -0.2, -0.1, -0.2},
{0.2, 0.2, 0.1, -0.0, -0.1, -0.3, -0.1, -0.0, -0.0, -0.1, -0.2, 0.1, 0.2, 0.1, -0.2},
{-0.1, -0.3, -0.1, 0.0, 0.0, 0.0, 0.2, 0.0, -0.1, 0.1, -0.2, 0.0, 0.2, 0.2, -0.2},
{-0.1, 0.0, -0.2, -0.1, -0.1, -0.0, 0.2, 0.1, 0.1, -0.2, 0.2, 0.2, 0.2, -0.0, -0.2},
{0.2, 0.1, -0.1, -0.1, -0.0, -0.1, 0.2, 0.1, -0.2, -0.1, 0.2, -0.1, -0.0, 0.2, 0.0},
{-0.1, -0.1, -0.2, 0.1, -0.1, -0.2, 0.1, 0.2, 0.2, 0.3, -0.3, 0.0, 0.1, 0.0, -0.1},
{-0.1, -0.1, 0.1, -0.2, 0.2, -0.0, -0.1, -0.3, 0.1, -0.1, 0.2, -0.2, 0.2, -0.1, -0.1},
{0.2, -0.1, -0.2, 0.1, 0.1, -0.1, 0.2, -0.1, -0.2, 0.1, 0.0, -0.1, 0.0, -0.1, -0.0},
{0.1, -0.1, -0.2, 0.1, 0.0, 0.1, -0.0, -0.0, -0.1, -0.2, 0.2, 0.1, 0.1, -0.2, -0.0},
{-0.1, 0.1, 0.2, 0.1, -0.2, -0.0, 0.2, 0.0, 0.1, 0.1, 0.0, -0.0, 0.2, 0.1, 0.2},
{-0.1, 0.0, -0.2, -0.1, -0.1, 0.1, -0.0, -0.0, -0.1, -0.2, -0.0, -0.1, -0.1, 0.1, 0.2},
{-0.2, 0.0, 0.1, 0.1, 0.2, 0.1, -0.1, 0.2, -0.2, 0.0, 0.1, -0.1, -0.3, -0.2, 0.2},
{-0.2, -0.2, -0.1, -0.1, -0.1, 0.2, 0.2, -0.1, -0.2, -0.1, 0.1, 0.2, 0.2, -0.1, -0.1},
{0.2, -0.0, 0.0, -0.0, 0.1, -0.2, -0.2, -0.1, 0.2, 0.2, -0.2, 0.1, 0.2, -0.2, -0.2},
{0.2, -0.0, 0.1, -0.2, 0.1, 0.1, -0.1, -0.2, 0.0, 0.1, -0.1, 0.0, 0.2, 0.2, 0.2},
{0.2, -0.2, 0.2, 0.1, -0.2, -0.2, 0.1, 0.1, 0.1, -0.2, -0.2, 0.2, -0.3, -0.2, -0.2},
{-0.2, 0.1, -0.1, -0.2, -0.0, 0.0, -0.1, -0.2, 0.0, -0.0, -0.2, 0.1, 0.2, -0.2, 0.2},
{-0.1, -0.2, -0.1, 0.1, 0.1, 0.2, 0.1, -0.2, -0.1, -0.2, -0.3, 0.1, 0.2, 0.1, -0.1},
{-0.1, -0.0, -0.0, -0.2, 0.2, 0.1, 0.0, -0.2, 0.2, 0.0, -0.1, -0.2, -0.0, 0.1, -0.2},
}
);
Matrix  transformer_layers_6_feed_forward_linear2_bias   (
{0.1, 0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.2, 0.1, 0.0, 0.2, 0.0, 0.3, 0.1, -0.0, 0.1, 0.3, 0.1, 0.2, 0.2, 0.0, -0.1, 0.1, 0.1, -0.2}
);
Matrix  transformer_layers_6_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_6_feed_forward_ln1_layer_norm_bias   (
{-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_6_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_6_feed_forward_ln2_layer_norm_bias   (
{0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_7_attention_query_weight   (
{{-0.0, 0.1, -0.1, 0.2, -0.0, 0.0, 0.1, 0.2, 0.0, -0.2, 0.1, -0.1, 0.2, 0.1, 0.0, -0.0, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.0, -0.1},
{0.0, -0.1, 0.1, 0.1, 0.1, 0.0, 0.1, -0.2, -0.2, -0.1, -0.1, 0.1, -0.1, 0.0, -0.2, -0.0, -0.1, 0.1, -0.1, -0.2, 0.1, 0.1, -0.2, 0.2, 0.0},
{0.2, -0.1, -0.1, -0.1, 0.2, -0.0, 0.1, -0.1, -0.0, 0.1, -0.1, 0.2, 0.1, 0.0, -0.1, 0.1, 0.1, -0.2, -0.1, 0.1, -0.1, -0.2, -0.2, -0.0, -0.0},
{0.1, -0.0, -0.0, 0.1, 0.1, -0.1, 0.1, -0.2, -0.0, 0.1, 0.2, 0.2, -0.2, -0.1, 0.1, -0.2, 0.1, -0.2, -0.1, 0.1, -0.0, -0.1, 0.0, 0.0, -0.0},
{-0.1, -0.1, 0.0, -0.1, -0.1, 0.0, 0.1, -0.0, 0.0, -0.0, 0.1, 0.2, -0.1, -0.1, -0.1, -0.0, 0.1, 0.1, -0.0, 0.1, 0.0, 0.0, -0.1, -0.2, -0.2},
{0.1, 0.1, -0.1, -0.1, -0.2, 0.0, 0.1, -0.0, 0.0, -0.1, -0.2, -0.1, 0.1, 0.0, 0.2, -0.1, 0.1, -0.1, -0.0, 0.1, -0.1, -0.1, 0.2, 0.0, -0.1},
{-0.2, -0.0, 0.1, -0.2, 0.0, 0.0, 0.0, -0.2, 0.1, -0.0, 0.0, -0.0, -0.0, 0.2, -0.0, -0.1, -0.2, -0.0, -0.1, -0.1, 0.2, 0.1, 0.1, -0.1, -0.1},
{0.0, -0.1, 0.1, -0.0, -0.1, -0.2, 0.2, 0.1, 0.1, -0.2, -0.2, 0.0, -0.0, -0.2, 0.1, -0.1, 0.0, -0.1, 0.0, 0.2, 0.1, 0.1, 0.2, -0.2, -0.1},
{-0.0, -0.1, 0.0, -0.1, 0.1, -0.1, 0.2, -0.2, 0.0, -0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.2, 0.1, -0.2, 0.2, -0.0, -0.0, 0.0, 0.2, 0.1, 0.1},
{0.2, -0.1, -0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.2, 0.0, 0.1, 0.2, 0.1, -0.0, -0.1, 0.0, -0.1, 0.2, 0.0, 0.1, -0.0, 0.0, -0.2, 0.2, -0.1},
{0.2, -0.1, -0.1, -0.2, 0.1, 0.0, -0.0, 0.2, 0.2, 0.1, 0.0, 0.1, 0.1, -0.2, 0.1, -0.1, -0.1, 0.1, -0.0, -0.0, -0.0, -0.1, 0.1, 0.0, -0.2},
{0.0, -0.2, -0.2, 0.1, 0.2, -0.2, -0.1, 0.2, 0.2, 0.0, -0.2, 0.0, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, -0.2, 0.1, -0.1, 0.0, -0.1, -0.1, 0.1},
{-0.2, 0.1, -0.1, -0.1, -0.1, 0.0, -0.1, -0.1, -0.1, -0.2, -0.2, 0.1, 0.1, 0.0, -0.0, -0.0, 0.1, -0.2, -0.0, -0.1, -0.2, -0.0, 0.2, -0.0, 0.1},
{0.1, 0.0, -0.2, -0.1, -0.2, -0.2, -0.1, 0.2, -0.0, -0.1, -0.0, -0.2, 0.2, 0.0, -0.1, 0.2, 0.1, 0.2, -0.2, -0.2, -0.1, 0.1, -0.0, 0.1, 0.1},
{-0.2, 0.0, 0.2, 0.0, 0.0, -0.1, -0.0, -0.1, 0.1, 0.1, -0.1, 0.0, 0.0, 0.1, -0.1, -0.1, -0.1, -0.0, -0.1, 0.1, 0.2, -0.1, 0.0, -0.1, 0.0},
{0.0, -0.1, 0.2, -0.2, 0.2, -0.1, -0.2, -0.1, 0.1, 0.2, 0.0, -0.0, -0.1, -0.0, -0.2, -0.1, 0.2, 0.1, -0.1, 0.2, -0.1, -0.0, -0.1, 0.1, -0.0},
{0.1, -0.2, 0.1, 0.0, -0.0, -0.2, 0.2, -0.1, -0.0, 0.1, -0.0, -0.2, -0.0, -0.2, -0.2, -0.1, -0.1, -0.0, 0.0, 0.1, 0.2, 0.0, 0.1, -0.1, -0.0},
{0.1, -0.1, -0.1, -0.1, 0.1, -0.1, 0.0, -0.1, 0.0, -0.2, -0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, -0.0, -0.0, -0.1, -0.2},
{-0.0, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.2, -0.2, -0.1, -0.2, 0.2, 0.2, -0.2, -0.1, -0.1, 0.1, -0.0, 0.2, -0.1, 0.0, -0.1, -0.1, -0.0, -0.2},
{-0.2, 0.2, -0.1, -0.1, 0.1, -0.1, -0.0, 0.1, 0.0, -0.0, -0.0, 0.1, 0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.2, -0.1, 0.1, 0.0, -0.0, 0.1, -0.1},
{0.1, -0.2, 0.2, -0.1, -0.2, 0.1, -0.1, 0.1, -0.0, 0.1, 0.1, -0.2, -0.2, 0.1, -0.2, -0.1, -0.2, -0.0, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.0},
{-0.1, -0.0, -0.0, 0.1, 0.1, -0.1, -0.1, -0.1, -0.2, -0.0, -0.0, -0.1, -0.0, 0.2, -0.0, 0.2, -0.2, 0.2, -0.1, 0.1, -0.1, -0.2, -0.1, -0.1, 0.2},
{0.0, -0.2, -0.0, -0.2, -0.1, 0.1, -0.1, -0.1, -0.0, -0.0, 0.2, -0.0, -0.1, -0.0, 0.2, 0.1, 0.0, 0.2, 0.1, -0.1, -0.1, -0.1, -0.2, -0.2, 0.0},
{0.2, -0.1, 0.0, 0.0, -0.0, 0.2, -0.1, 0.2, -0.1, 0.2, -0.1, -0.0, -0.1, -0.0, 0.0, 0.0, -0.1, 0.0, -0.1, 0.2, 0.1, -0.1, 0.1, -0.1, 0.0},
{-0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, -0.1, -0.1, 0.0, -0.2, 0.2, 0.1, -0.2, 0.0, 0.1, 0.0, -0.2, 0.1, -0.0, -0.2, 0.2, -0.1, 0.2},
}
);
Matrix  transformer_layers_7_attention_query_bias   (
{0.0, -0.2, -0.0, -0.1, 0.0, -0.1, 0.1, -0.1, -0.1, 0.2, -0.1, -0.0, -0.1, 0.2, 0.2, 0.1, 0.2, -0.1, 0.1, -0.2, -0.0, 0.1, 0.2, -0.1, -0.1}
);
Matrix  transformer_layers_7_attention_key_weight   (
{{0.2, 0.1, 0.2, -0.1, 0.0, -0.1, -0.1, -0.1, 0.2, -0.0, 0.0, 0.1, 0.1, -0.1, 0.1, -0.1, -0.1, 0.1, 0.0, -0.2, 0.2, -0.0, -0.2, 0.0, 0.1},
{0.1, -0.0, 0.1, 0.1, 0.0, -0.1, 0.2, 0.1, 0.1, -0.2, -0.0, -0.1, -0.2, 0.2, 0.1, -0.0, 0.1, 0.0, -0.1, 0.0, -0.1, 0.2, 0.0, -0.1, -0.2},
{0.2, -0.1, -0.2, 0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, -0.0, 0.1, 0.0, 0.0, -0.1, 0.2, 0.1, -0.1, 0.2, 0.1, 0.1, 0.1, 0.0, -0.2, 0.0},
{-0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.0, -0.0, 0.1, 0.2, -0.1, 0.1, -0.1, 0.2, -0.2, -0.1, -0.1, -0.1, -0.1, -0.0, 0.0, -0.2, 0.2, -0.1, -0.2},
{-0.2, -0.0, 0.1, 0.1, -0.0, 0.2, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.2, 0.2, -0.0, -0.1, 0.0, 0.1, 0.1, -0.2, 0.2, -0.2, -0.1},
{-0.1, 0.2, 0.1, -0.2, 0.1, -0.1, 0.1, -0.1, -0.2, 0.1, 0.0, 0.0, 0.2, 0.2, -0.0, -0.1, 0.1, -0.2, -0.0, 0.2, -0.0, -0.0, 0.1, -0.1, 0.1},
{0.1, 0.0, -0.0, 0.1, -0.1, -0.0, -0.2, -0.1, -0.1, 0.0, 0.1, 0.2, -0.1, 0.1, -0.1, 0.2, 0.2, -0.2, 0.2, 0.0, 0.1, 0.0, -0.2, 0.1, 0.0},
{-0.2, 0.0, 0.1, 0.0, 0.1, -0.2, -0.1, 0.0, -0.1, 0.1, 0.0, 0.0, 0.2, 0.0, 0.2, 0.2, 0.0, -0.1, 0.1, -0.1, -0.2, -0.2, 0.1, 0.1, -0.1},
{-0.1, -0.2, 0.0, 0.0, -0.1, 0.2, 0.1, -0.1, 0.1, -0.1, 0.2, -0.1, -0.1, -0.1, 0.2, -0.1, 0.1, -0.1, -0.0, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1},
{-0.2, 0.1, 0.2, 0.0, 0.0, 0.2, -0.1, 0.2, 0.2, 0.1, -0.1, 0.2, -0.0, 0.1, 0.1, 0.1, -0.1, -0.0, 0.0, -0.2, 0.1, -0.2, -0.0, 0.1, 0.0},
{-0.1, 0.0, -0.1, -0.2, -0.1, 0.0, -0.0, 0.0, 0.1, 0.2, -0.2, 0.1, 0.2, -0.1, 0.1, -0.1, -0.1, 0.2, -0.0, -0.2, -0.2, 0.1, 0.1, -0.0, -0.1},
{0.0, 0.2, 0.1, 0.0, 0.1, 0.0, -0.1, 0.2, 0.1, 0.0, 0.2, 0.1, -0.2, 0.1, 0.0, 0.0, 0.0, -0.1, 0.1, -0.1, -0.0, 0.0, 0.0, -0.1, -0.1},
{0.1, -0.2, -0.2, 0.1, 0.2, -0.1, -0.0, -0.2, 0.2, -0.2, -0.2, -0.2, -0.2, 0.2, -0.1, 0.1, 0.2, 0.2, 0.0, 0.0, 0.1, -0.1, -0.1, -0.1, 0.0},
{-0.2, 0.0, -0.1, 0.2, 0.1, -0.2, 0.1, -0.1, 0.1, -0.0, 0.0, -0.1, 0.1, 0.1, -0.2, 0.1, -0.1, 0.1, 0.1, -0.0, -0.1, -0.2, 0.2, -0.1, -0.2},
{-0.1, -0.0, -0.1, 0.0, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.2, 0.0, 0.2, 0.0, -0.1, 0.2, 0.1, -0.1, 0.0, -0.1, -0.0, -0.0, -0.0},
{0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.2, 0.2, -0.1, 0.1, -0.1, -0.1, -0.1, 0.1, -0.2, 0.1, -0.1, -0.1, -0.1, 0.2, 0.0, 0.1, -0.2, 0.2, 0.0},
{-0.1, 0.2, -0.0, 0.1, 0.2, -0.0, -0.0, 0.0, -0.1, -0.2, 0.0, -0.1, -0.1, 0.1, 0.2, 0.2, -0.2, -0.1, -0.0, 0.0, 0.1, -0.0, 0.1, 0.2, 0.2},
{0.2, 0.1, -0.1, 0.1, -0.2, 0.0, -0.0, -0.1, 0.1, -0.1, -0.0, 0.1, -0.2, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 0.1, 0.0, -0.0},
{0.2, -0.1, 0.1, -0.1, -0.1, 0.1, -0.1, 0.0, -0.1, 0.2, -0.1, -0.2, 0.0, 0.1, -0.2, -0.1, 0.0, 0.2, 0.0, -0.0, 0.0, -0.2, 0.1, 0.2, 0.0},
{-0.1, 0.1, 0.2, -0.1, -0.0, -0.1, 0.0, -0.1, -0.1, 0.0, 0.1, 0.2, -0.0, 0.0, 0.1, 0.2, 0.2, -0.0, 0.1, 0.1, 0.1, -0.1, 0.1, -0.1, -0.1},
{-0.2, -0.2, 0.2, -0.1, -0.2, 0.0, -0.1, -0.0, -0.0, -0.2, -0.0, 0.0, 0.1, -0.1, -0.1, -0.2, -0.2, 0.0, 0.2, 0.1, -0.1, 0.2, -0.2, -0.1, -0.1},
{0.0, -0.1, 0.2, 0.1, 0.2, 0.2, 0.1, 0.2, 0.1, -0.1, -0.0, -0.0, 0.2, 0.1, -0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.2, 0.1, -0.1, 0.1, 0.1},
{-0.1, -0.0, -0.1, -0.0, 0.0, -0.0, -0.1, 0.1, -0.2, 0.1, 0.2, 0.1, -0.1, -0.2, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.2, 0.1, 0.1, 0.1, 0.1},
{0.2, -0.1, 0.1, 0.0, 0.1, -0.2, 0.0, 0.1, 0.2, -0.2, 0.1, 0.1, 0.1, 0.1, -0.0, 0.2, 0.0, 0.1, 0.2, 0.1, -0.0, 0.0, -0.2, 0.2, 0.1},
{0.0, -0.2, -0.2, -0.2, -0.2, 0.1, -0.2, -0.0, 0.1, -0.2, -0.0, -0.1, -0.2, 0.2, 0.1, 0.1, -0.0, 0.2, -0.2, 0.2, 0.0, 0.2, 0.2, 0.1, 0.0},
}
);
Matrix  transformer_layers_7_attention_key_bias   (
{0.0, -0.1, -0.1, 0.0, -0.2, 0.1, -0.1, 0.1, 0.1, 0.1, -0.2, -0.1, 0.2, 0.1, -0.1, 0.1, 0.1, -0.2, 0.0, -0.1, 0.0, 0.1, -0.1, 0.1, -0.1}
);
Matrix  transformer_layers_7_attention_value_weight   (
{{-0.2, 0.1, 0.1, -0.2, -0.0, -0.0, -0.0, 0.1, 0.0, -0.1, -0.0, 0.1, 0.0, 0.0, 0.1, 0.1, -0.0, 0.1, -0.1, -0.1, 0.2, -0.0, -0.0, 0.2, 0.1},
{0.0, -0.0, 0.1, -0.1, -0.1, 0.0, 0.1, 0.1, -0.1, -0.1, -0.2, -0.1, 0.0, 0.0, 0.2, -0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, -0.1, -0.1},
{-0.0, 0.1, -0.1, -0.1, 0.1, 0.1, 0.0, -0.1, -0.1, -0.2, 0.0, 0.1, -0.1, 0.0, -0.2, 0.2, 0.1, -0.1, 0.1, -0.1, 0.2, 0.0, -0.2, -0.2, 0.1},
{0.2, 0.1, -0.1, -0.0, -0.2, -0.1, 0.2, -0.1, -0.1, -0.1, -0.1, 0.0, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.0, -0.2, 0.2, -0.1, 0.0, -0.2, -0.2},
{0.0, 0.1, 0.2, -0.1, -0.1, -0.2, -0.1, 0.1, -0.1, 0.2, -0.0, -0.1, -0.1, -0.1, -0.0, 0.1, 0.2, -0.1, -0.1, -0.0, 0.1, 0.1, -0.2, -0.2, -0.1},
{0.2, -0.1, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, -0.0, 0.1, -0.2, 0.1, 0.2, 0.1, -0.1, -0.2, 0.2, -0.0, 0.1, 0.0, -0.2, 0.1, 0.0, -0.2, -0.1},
{-0.1, 0.1, 0.1, 0.0, -0.2, -0.0, 0.0, -0.2, -0.0, -0.2, 0.1, 0.2, 0.2, 0.0, -0.2, 0.1, 0.0, 0.1, 0.1, 0.1, -0.2, -0.1, -0.1, -0.2, 0.1},
{0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.1, 0.1, -0.1, 0.0, -0.2, -0.2, -0.1, 0.2, 0.1, -0.1, -0.0, -0.1, 0.1, -0.1, 0.0, -0.1, -0.1, 0.1, 0.2},
{0.2, -0.1, 0.0, 0.2, 0.0, -0.0, -0.1, 0.1, -0.1, 0.0, -0.1, 0.1, 0.0, 0.0, -0.1, 0.0, 0.1, -0.1, 0.1, 0.0, -0.2, -0.0, 0.1, 0.1, 0.1},
{0.1, 0.1, -0.1, 0.1, -0.2, 0.2, -0.1, -0.1, -0.2, 0.1, -0.1, 0.2, -0.0, 0.2, 0.1, -0.1, -0.2, 0.1, -0.2, -0.1, -0.0, -0.1, 0.2, -0.1, 0.1},
{0.1, -0.0, -0.1, 0.1, -0.2, 0.2, -0.2, 0.0, -0.1, 0.0, 0.2, -0.1, 0.0, 0.1, -0.1, 0.0, 0.0, -0.1, 0.2, 0.0, -0.0, 0.2, 0.0, -0.0, 0.0},
{0.2, -0.0, 0.0, -0.1, -0.0, -0.0, -0.2, 0.1, -0.1, 0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.2, 0.1, -0.1, -0.1, 0.1, 0.2, 0.1, 0.2},
{-0.1, -0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.2, 0.2, -0.1, -0.2, -0.0, -0.2, -0.1, 0.2, 0.1, -0.0, 0.2, -0.1, -0.1, 0.2, 0.1, -0.2, -0.1, 0.2},
{-0.2, 0.1, -0.1, -0.0, 0.2, 0.1, -0.0, 0.0, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.0, -0.1, 0.2, -0.1, 0.2, -0.1, -0.2, 0.2, -0.2, -0.1, 0.2},
{0.2, 0.1, 0.2, -0.2, 0.1, 0.0, 0.2, 0.1, -0.1, -0.2, -0.2, -0.0, -0.2, 0.1, -0.2, 0.0, 0.0, 0.0, 0.1, -0.1, -0.1, 0.1, 0.1, -0.2, 0.1},
{-0.0, -0.1, 0.1, 0.1, 0.2, -0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.0, 0.0, 0.0, 0.1, 0.1, -0.2, 0.0, -0.2, -0.1, -0.0, -0.0, 0.1, 0.0},
{-0.1, 0.1, 0.0, -0.0, -0.1, 0.1, -0.2, -0.2, -0.2, -0.2, 0.2, 0.1, 0.1, -0.1, 0.2, 0.2, 0.0, -0.2, 0.2, -0.1, -0.1, 0.2, -0.1, 0.1, 0.1},
{-0.1, -0.2, -0.0, -0.0, -0.2, 0.2, -0.1, -0.1, -0.2, 0.2, -0.0, -0.1, -0.0, -0.1, -0.1, 0.0, -0.2, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.1},
{0.0, -0.1, 0.0, 0.0, 0.0, -0.1, 0.1, -0.0, 0.1, -0.1, 0.1, -0.0, 0.2, 0.0, 0.1, 0.1, -0.2, -0.2, 0.1, -0.1, -0.1, 0.0, -0.0, 0.1, 0.0},
{0.1, -0.2, -0.1, 0.0, -0.2, -0.0, -0.0, -0.1, -0.0, -0.2, 0.1, 0.1, -0.1, 0.1, 0.1, -0.2, 0.1, 0.0, 0.2, -0.1, 0.2, 0.2, -0.1, -0.2, -0.1},
{-0.0, -0.1, -0.1, -0.1, -0.1, -0.2, -0.1, 0.1, -0.2, 0.1, -0.1, 0.1, -0.1, -0.0, 0.2, -0.1, -0.2, -0.1, 0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.0},
{-0.1, 0.1, -0.1, -0.1, -0.0, 0.1, 0.1, -0.2, -0.1, -0.0, 0.1, 0.0, 0.1, 0.1, 0.0, -0.0, -0.1, 0.2, 0.1, 0.0, -0.1, -0.2, -0.1, -0.1, -0.1},
{-0.1, -0.1, 0.0, 0.2, -0.1, -0.1, -0.0, -0.1, 0.1, 0.1, -0.1, 0.1, -0.0, 0.1, 0.1, 0.2, 0.1, 0.1, -0.2, -0.2, -0.0, -0.0, -0.1, -0.2, 0.0},
{0.0, 0.0, 0.2, -0.2, 0.0, 0.2, -0.2, -0.1, 0.0, 0.1, -0.1, -0.1, -0.0, 0.1, 0.1, 0.0, -0.1, -0.0, 0.1, 0.1, 0.2, -0.1, 0.1, -0.2, -0.2},
{0.2, -0.0, 0.1, 0.1, 0.1, -0.2, 0.1, 0.0, 0.2, 0.1, 0.2, 0.1, 0.0, -0.1, 0.2, 0.2, 0.1, -0.1, -0.2, -0.2, 0.2, 0.0, -0.1, 0.2, -0.2},
}
);
Matrix  transformer_layers_7_attention_value_bias   (
{-0.1, -0.1, -0.1, 0.1, -0.1, -0.0, 0.0, -0.2, 0.2, -0.1, -0.1, 0.1, 0.0, -0.2, 0.1, 0.2, 0.0, -0.1, 0.1, 0.0, -0.1, 0.1, -0.1, -0.2, -0.1}
);
Matrix  transformer_layers_7_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_7_norm1_layer_norm_bias   (
{0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_7_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_7_norm2_layer_norm_bias   (
{0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_7_feed_forward_linear1_weight   (
{{0.1, 0.1, -0.0, 0.2, 0.0, -0.1, 0.0, 0.1, -0.0, -0.2, 0.2, -0.1, -0.1, -0.0, 0.1, -0.2, 0.1, -0.1, -0.1, 0.0, -0.0, 0.2, 0.2, -0.2, -0.1},
{-0.2, 0.1, -0.1, -0.0, 0.1, -0.1, 0.2, 0.1, -0.1, -0.2, 0.1, -0.0, 0.1, -0.1, 0.1, 0.2, 0.2, -0.1, 0.2, 0.2, -0.1, 0.1, 0.1, -0.0, -0.2},
{-0.1, -0.0, 0.1, 0.1, -0.0, 0.0, -0.0, 0.2, -0.1, 0.2, 0.1, -0.1, 0.0, 0.1, -0.1, 0.2, -0.1, 0.2, 0.1, 0.2, -0.1, -0.1, -0.1, 0.0, 0.0},
{0.1, 0.2, 0.2, 0.2, 0.1, -0.1, 0.0, -0.0, 0.1, 0.0, -0.1, -0.1, 0.1, -0.1, 0.0, -0.0, 0.0, -0.1, -0.2, 0.1, 0.0, 0.1, 0.2, 0.0, 0.1},
{0.1, -0.1, -0.0, -0.0, 0.1, 0.0, -0.1, 0.1, 0.2, 0.1, -0.1, -0.2, -0.1, 0.1, -0.1, 0.2, -0.2, 0.2, -0.0, 0.2, -0.0, -0.1, -0.0, -0.1, -0.1},
{0.0, -0.2, -0.1, -0.2, -0.1, 0.1, -0.0, -0.0, -0.2, 0.1, 0.0, -0.1, 0.2, 0.2, -0.0, 0.1, 0.1, -0.2, -0.2, -0.2, -0.1, -0.0, -0.2, 0.2, -0.1},
{-0.1, 0.1, -0.2, -0.0, 0.1, 0.0, -0.0, 0.1, 0.1, 0.2, 0.0, -0.0, 0.0, 0.1, -0.1, -0.1, -0.1, -0.1, 0.0, -0.1, -0.2, -0.2, -0.1, -0.2, 0.0},
{-0.1, 0.2, -0.2, 0.2, 0.2, -0.2, -0.2, 0.1, 0.1, -0.2, -0.1, -0.2, -0.1, 0.1, 0.1, -0.0, -0.1, 0.2, 0.2, 0.1, 0.1, -0.1, 0.2, 0.2, -0.0},
{0.1, -0.1, -0.1, 0.2, -0.1, -0.1, 0.0, 0.1, 0.1, -0.0, -0.1, 0.1, -0.1, 0.1, 0.1, 0.2, -0.2, -0.1, 0.0, 0.0, 0.2, 0.0, 0.2, 0.1, -0.1},
{-0.0, -0.0, -0.0, -0.1, 0.0, 0.0, 0.2, 0.0, 0.1, -0.1, 0.0, -0.1, -0.1, -0.2, -0.1, 0.2, 0.2, -0.0, 0.1, 0.0, -0.2, -0.1, -0.1, 0.0, 0.2},
{-0.2, 0.0, -0.1, 0.2, -0.0, -0.2, -0.0, -0.1, 0.0, 0.0, 0.1, -0.2, 0.0, -0.1, -0.2, 0.1, 0.1, -0.1, -0.1, -0.2, 0.0, 0.0, -0.2, -0.0, -0.1},
{0.0, -0.1, -0.0, -0.2, -0.1, -0.0, -0.1, 0.1, -0.1, -0.1, -0.0, -0.0, 0.1, -0.1, 0.1, -0.2, 0.0, -0.1, -0.1, 0.0, -0.1, 0.2, 0.1, 0.1, 0.1},
{0.2, -0.1, 0.1, 0.0, -0.1, -0.1, -0.1, 0.1, -0.1, -0.0, 0.1, 0.1, -0.2, -0.1, 0.1, -0.1, -0.2, 0.2, 0.1, 0.1, -0.2, 0.2, -0.2, 0.2, 0.1},
{-0.0, -0.1, 0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.0, -0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.2, -0.2, 0.0, -0.1, -0.2, -0.0, 0.1, 0.2, -0.0, -0.0},
{-0.1, 0.2, -0.0, 0.0, -0.0, -0.1, 0.1, -0.2, -0.1, -0.1, 0.2, 0.1, -0.1, 0.1, 0.2, -0.1, 0.1, -0.1, -0.1, -0.0, -0.1, 0.2, 0.1, -0.1, 0.2},
}
);
Matrix  transformer_layers_7_feed_forward_linear1_bias   (
{-0.2, 0.1, 0.1, 0.0, -0.1, 0.1, -0.2, -0.1, 0.1, 0.1, -0.0, 0.1, 0.2, -0.0, 0.1}
);
Matrix  transformer_layers_7_feed_forward_linear2_weight   (
{{0.2, 0.2, -0.1, -0.1, -0.1, -0.1, 0.0, 0.1, -0.0, 0.1, -0.2, -0.1, -0.2, -0.0, -0.2},
{0.2, -0.2, -0.0, 0.1, -0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.0, -0.1, -0.1, 0.0, 0.2},
{-0.3, -0.1, -0.0, 0.1, -0.1, -0.0, -0.0, 0.1, -0.2, -0.0, 0.2, 0.2, 0.1, -0.1, -0.2},
{0.1, -0.2, 0.1, 0.1, 0.0, -0.1, 0.1, -0.1, 0.2, -0.1, -0.1, -0.0, -0.0, 0.2, 0.2},
{-0.1, 0.1, -0.0, 0.1, -0.0, -0.2, 0.2, -0.1, -0.0, 0.0, 0.1, 0.1, 0.0, -0.1, -0.2},
{-0.2, -0.1, 0.2, 0.0, 0.1, -0.2, 0.2, 0.2, 0.2, 0.2, 0.1, -0.2, -0.2, 0.0, -0.3},
{0.1, -0.2, -0.1, 0.2, 0.2, -0.1, 0.0, -0.1, 0.2, -0.0, 0.1, -0.1, -0.1, -0.0, 0.1},
{0.2, -0.3, 0.1, 0.0, -0.0, -0.2, 0.2, 0.0, 0.2, 0.1, -0.2, -0.0, -0.1, 0.1, -0.2},
{0.2, 0.1, 0.1, -0.0, -0.1, 0.2, -0.1, -0.2, -0.0, -0.0, -0.2, 0.1, -0.1, -0.2, 0.0},
{0.2, -0.2, -0.2, 0.2, -0.0, 0.1, -0.2, 0.1, -0.1, 0.2, 0.1, 0.1, -0.2, -0.1, 0.1},
{0.0, -0.1, 0.1, 0.2, -0.2, 0.0, -0.2, -0.2, 0.2, 0.2, 0.1, -0.2, 0.1, 0.2, -0.2},
{0.1, -0.0, 0.1, -0.1, -0.3, -0.2, 0.0, 0.1, 0.2, -0.2, -0.1, 0.2, 0.1, 0.1, 0.0},
{0.1, 0.2, 0.1, 0.0, -0.2, -0.1, 0.1, 0.0, 0.1, 0.0, -0.1, -0.2, -0.2, -0.0, 0.0},
{0.1, -0.2, 0.0, 0.1, 0.2, 0.1, 0.2, -0.2, 0.1, -0.1, -0.0, 0.1, 0.1, -0.1, -0.1},
{-0.3, 0.1, 0.0, 0.2, -0.2, -0.0, 0.1, 0.2, 0.1, 0.2, 0.0, -0.2, -0.2, -0.0, -0.3},
{-0.2, 0.2, -0.2, 0.2, -0.1, -0.3, -0.1, -0.0, -0.1, -0.0, -0.2, 0.2, -0.1, 0.1, -0.0},
{-0.2, 0.2, 0.2, -0.1, 0.1, 0.1, 0.3, -0.1, 0.1, 0.1, -0.1, 0.0, -0.1, -0.1, -0.1},
{-0.1, -0.2, 0.2, 0.1, 0.0, -0.3, 0.2, 0.0, 0.2, -0.0, -0.1, 0.2, 0.1, 0.2, 0.2},
{0.1, 0.0, 0.2, 0.1, 0.0, 0.1, -0.0, -0.2, 0.0, 0.2, 0.2, 0.1, -0.1, -0.2, 0.1},
{0.2, 0.1, 0.1, 0.2, 0.0, 0.1, -0.1, -0.1, -0.2, 0.2, -0.0, 0.3, -0.2, 0.2, -0.2},
{-0.1, 0.1, -0.0, 0.2, -0.2, 0.2, 0.1, 0.1, 0.2, 0.0, -0.0, -0.2, 0.0, -0.0, -0.2},
{-0.1, 0.0, -0.1, 0.2, 0.2, -0.2, -0.1, -0.1, -0.2, 0.2, -0.1, 0.2, 0.2, -0.2, 0.0},
{-0.1, -0.2, 0.2, 0.1, -0.2, -0.2, 0.1, -0.1, 0.1, 0.0, -0.1, -0.2, -0.0, 0.0, 0.2},
{-0.1, 0.1, -0.1, 0.2, 0.2, 0.2, -0.1, -0.2, -0.2, 0.1, -0.2, -0.2, -0.1, 0.2, 0.2},
{-0.3, -0.1, 0.0, -0.2, -0.1, 0.2, -0.1, 0.1, 0.0, 0.0, 0.2, 0.1, -0.2, -0.2, 0.2},
}
);
Matrix  transformer_layers_7_feed_forward_linear2_bias   (
{0.2, -0.2, 0.1, 0.2, 0.2, -0.0, 0.1, 0.2, -0.2, 0.2, -0.2, 0.1, 0.3, -0.2, -0.2, 0.1, 0.0, -0.2, -0.0, 0.0, -0.1, -0.1, -0.2, -0.1, -0.2}
);
Matrix  transformer_layers_7_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_7_feed_forward_ln1_layer_norm_bias   (
{0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0}
);
Matrix  transformer_layers_7_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_7_feed_forward_ln2_layer_norm_bias   (
{0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_8_attention_query_weight   (
{{-0.0, -0.1, 0.1, -0.1, 0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.2, -0.1, -0.0, -0.2, -0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.0, 0.1, -0.0, 0.1, -0.1},
{0.0, -0.0, -0.1, -0.2, 0.2, -0.2, 0.0, -0.0, 0.1, -0.1, 0.0, 0.0, -0.1, -0.0, -0.0, 0.1, -0.1, 0.1, -0.1, -0.2, -0.1, 0.1, -0.2, 0.1, -0.1},
{0.2, 0.2, -0.1, -0.1, 0.1, 0.2, -0.1, 0.0, -0.0, 0.1, -0.2, -0.0, -0.1, 0.1, 0.0, -0.2, -0.0, -0.1, 0.1, 0.2, 0.1, 0.0, -0.2, -0.2, -0.0},
{-0.2, 0.2, -0.0, 0.1, 0.0, 0.2, -0.1, 0.2, 0.1, 0.2, 0.1, -0.1, 0.1, -0.1, -0.1, -0.2, 0.1, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0, 0.0, 0.0},
{0.2, 0.1, -0.0, -0.1, -0.2, -0.2, -0.1, -0.1, -0.1, -0.2, -0.1, 0.0, 0.0, -0.1, 0.1, -0.1, -0.1, -0.1, -0.2, -0.0, 0.0, -0.1, 0.1, -0.2, 0.1},
{0.1, -0.1, -0.0, -0.0, 0.0, 0.1, -0.0, -0.0, -0.1, 0.2, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.2, 0.0, 0.0, -0.2, -0.1, 0.0, 0.1, -0.1, 0.2},
{0.1, -0.2, -0.1, 0.0, 0.2, 0.1, -0.0, 0.1, -0.1, 0.0, 0.1, -0.1, -0.1, 0.2, 0.0, -0.0, 0.1, -0.2, -0.1, -0.1, -0.1, -0.0, -0.1, -0.1, -0.0},
{-0.1, 0.1, 0.1, 0.2, -0.1, 0.2, -0.2, -0.2, 0.2, -0.2, 0.2, -0.1, 0.0, -0.2, 0.1, 0.2, 0.2, 0.1, -0.1, -0.0, -0.1, -0.0, 0.1, -0.1, -0.1},
{0.1, 0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.0, -0.0, -0.1, -0.1, -0.0, 0.1, 0.2, -0.1, -0.1, 0.1, 0.1, -0.1, -0.0, -0.1, -0.2, 0.2, 0.1, 0.1},
{-0.2, -0.0, -0.1, -0.1, -0.1, -0.1, -0.0, -0.2, 0.0, -0.1, -0.0, -0.2, 0.2, 0.2, 0.1, -0.0, -0.2, -0.2, -0.1, -0.2, 0.1, -0.1, -0.1, 0.1, 0.0},
{0.1, -0.2, -0.0, -0.0, 0.0, 0.2, -0.0, -0.2, 0.0, 0.1, -0.1, 0.1, 0.2, 0.1, -0.2, -0.2, -0.0, -0.0, -0.2, -0.1, -0.0, 0.1, 0.0, -0.2, 0.1},
{0.2, 0.1, 0.0, 0.0, 0.0, -0.1, -0.2, -0.1, 0.1, -0.2, -0.2, -0.2, 0.0, 0.1, -0.2, -0.1, 0.0, -0.1, -0.1, -0.1, 0.1, -0.1, 0.2, -0.0, 0.2},
{0.1, 0.2, 0.2, 0.1, -0.1, 0.0, 0.1, -0.1, 0.0, -0.1, -0.1, -0.2, 0.1, 0.1, -0.1, -0.2, 0.2, -0.1, 0.1, -0.1, -0.0, -0.0, 0.0, -0.1, -0.0},
{0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.0, -0.1, -0.1, 0.1, 0.0, 0.2, -0.0, 0.1, 0.2, 0.2, 0.1, 0.0, 0.0, -0.1, 0.0, 0.1, -0.0, 0.1, 0.1},
{0.2, -0.0, -0.2, -0.1, 0.1, 0.0, -0.1, 0.1, 0.2, -0.1, -0.1, -0.1, 0.1, -0.1, 0.2, -0.1, -0.1, -0.1, 0.0, 0.1, -0.1, 0.0, -0.1, -0.0, 0.1},
{-0.1, -0.2, 0.2, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, -0.1, 0.1, -0.2, 0.1, -0.1, -0.1, -0.0, 0.2, 0.1, -0.1, -0.1, -0.0, -0.0, -0.2, 0.1, 0.0},
{0.0, 0.1, -0.1, 0.0, -0.0, 0.2, -0.0, 0.1, 0.0, 0.1, 0.0, 0.1, -0.1, 0.1, 0.0, -0.2, -0.0, -0.2, -0.1, -0.1, -0.1, 0.0, 0.1, -0.1, -0.0},
{-0.1, 0.1, 0.0, -0.0, 0.1, 0.2, -0.1, -0.1, 0.1, 0.0, 0.1, -0.2, 0.2, -0.2, -0.1, -0.1, -0.1, -0.2, 0.2, 0.1, -0.0, -0.1, 0.0, -0.1, 0.1},
{0.2, -0.1, -0.1, -0.0, -0.0, 0.1, 0.2, 0.1, 0.1, 0.1, -0.2, -0.1, -0.1, -0.1, -0.2, -0.0, 0.1, -0.1, -0.1, -0.1, 0.1, -0.2, 0.2, -0.0, -0.1},
{0.2, -0.0, -0.0, -0.1, -0.0, -0.1, 0.1, -0.1, 0.2, -0.0, -0.1, 0.2, -0.1, 0.2, 0.0, -0.0, 0.2, 0.2, -0.1, -0.1, 0.0, 0.0, -0.0, -0.0, -0.1},
{-0.1, -0.0, -0.1, 0.1, -0.1, -0.2, 0.2, 0.1, -0.1, -0.0, -0.2, -0.2, -0.2, 0.1, -0.1, 0.0, -0.0, 0.2, 0.1, -0.2, 0.1, -0.1, 0.2, -0.1, 0.2},
{-0.1, 0.2, 0.1, -0.1, 0.2, 0.2, -0.2, -0.1, -0.2, -0.1, 0.2, 0.0, 0.2, 0.1, -0.1, -0.2, -0.1, -0.1, -0.1, 0.1, 0.2, -0.0, 0.1, 0.1, 0.1},
{-0.1, 0.1, 0.1, -0.1, -0.0, -0.0, 0.1, -0.0, -0.0, 0.1, -0.2, 0.1, 0.2, 0.1, 0.0, 0.2, 0.1, -0.1, -0.1, -0.1, 0.2, 0.1, 0.0, -0.0, -0.1},
{-0.0, -0.1, -0.0, -0.2, 0.1, -0.0, -0.1, 0.1, 0.1, -0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.0, 0.0, -0.2, 0.0, -0.2, -0.1, 0.2, -0.1, -0.1, -0.1},
{0.1, 0.1, -0.1, -0.0, -0.0, -0.2, 0.1, 0.0, 0.0, -0.1, -0.2, 0.1, -0.0, -0.1, -0.1, -0.0, -0.1, -0.1, 0.0, -0.0, 0.1, -0.1, 0.1, 0.0, 0.1},
}
);
Matrix  transformer_layers_8_attention_query_bias   (
{0.1, -0.2, 0.0, -0.1, -0.0, 0.1, -0.1, -0.1, 0.2, 0.2, -0.0, -0.0, -0.1, 0.0, 0.1, 0.0, 0.0, 0.2, -0.1, 0.0, 0.2, -0.0, 0.2, 0.2, -0.0}
);
Matrix  transformer_layers_8_attention_key_weight   (
{{-0.0, -0.1, -0.2, -0.1, 0.1, -0.0, -0.0, 0.1, 0.2, -0.2, 0.0, -0.1, 0.1, 0.2, 0.0, -0.1, 0.1, 0.2, -0.2, -0.1, -0.0, -0.1, 0.1, 0.1, 0.1},
{-0.0, -0.2, 0.1, 0.1, -0.2, -0.1, -0.1, -0.0, 0.1, -0.1, 0.1, 0.1, 0.0, -0.0, -0.0, 0.1, 0.1, 0.2, -0.0, -0.1, 0.0, -0.0, -0.2, -0.2, -0.1},
{0.1, -0.1, 0.2, 0.2, 0.1, -0.0, -0.1, 0.2, -0.1, -0.1, -0.2, -0.0, 0.1, -0.0, 0.1, 0.0, -0.1, -0.2, 0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.2},
{-0.2, 0.1, -0.1, -0.0, -0.2, -0.1, 0.2, -0.0, 0.0, 0.0, -0.1, 0.0, -0.1, -0.2, -0.1, 0.1, -0.1, 0.2, -0.2, -0.1, 0.0, 0.1, -0.0, 0.2, 0.2},
{0.1, -0.0, 0.1, -0.0, -0.0, -0.1, -0.0, 0.2, -0.1, -0.2, -0.0, 0.1, 0.1, -0.0, -0.0, -0.1, -0.2, -0.0, -0.1, 0.1, 0.2, 0.1, 0.2, -0.1, 0.0},
{0.0, 0.0, -0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.0, -0.1, 0.1, -0.1, -0.2, -0.1, 0.2, 0.1, -0.1, -0.1, -0.2, -0.2, -0.0, 0.0, -0.1, -0.1, 0.0},
{-0.0, 0.2, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.2, 0.0, 0.1, -0.0, -0.0, -0.1, 0.0, 0.1, -0.2, -0.1, -0.1, 0.1, 0.2, 0.0, 0.1, -0.1},
{0.1, 0.2, 0.1, -0.2, 0.0, -0.1, 0.2, 0.0, -0.2, -0.2, 0.1, -0.2, -0.0, 0.2, 0.0, -0.2, -0.1, -0.1, -0.1, -0.2, 0.0, 0.1, 0.1, -0.1, 0.1},
{0.1, 0.2, -0.1, -0.1, 0.2, 0.0, 0.0, 0.1, 0.0, 0.1, -0.0, -0.0, 0.0, 0.1, 0.0, 0.1, 0.0, -0.1, 0.2, 0.0, 0.2, -0.2, -0.1, 0.0, -0.1},
{-0.1, 0.1, -0.1, 0.1, -0.2, 0.1, 0.2, -0.1, 0.2, 0.1, -0.0, -0.1, -0.1, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.1, 0.0, 0.2, 0.0, 0.1, -0.1},
{0.2, -0.0, 0.0, -0.0, -0.1, 0.2, -0.0, 0.1, 0.1, -0.0, 0.1, -0.1, -0.2, -0.1, -0.2, 0.0, 0.1, -0.2, 0.2, 0.1, -0.2, 0.1, 0.0, 0.0, 0.2},
{0.1, -0.2, -0.1, -0.1, -0.1, 0.2, -0.2, 0.1, 0.0, 0.1, 0.2, 0.1, -0.1, -0.2, 0.1, 0.0, 0.1, -0.1, 0.0, 0.1, -0.0, 0.1, 0.2, 0.0, -0.1},
{0.1, -0.1, 0.0, -0.1, 0.2, -0.1, 0.0, 0.1, 0.1, 0.2, -0.2, -0.1, 0.1, -0.1, 0.1, -0.1, 0.2, 0.1, 0.1, 0.2, -0.1, 0.1, -0.2, -0.1, -0.1},
{-0.1, -0.0, -0.2, 0.1, 0.1, -0.0, 0.1, 0.1, 0.0, -0.1, -0.2, 0.1, 0.1, 0.1, 0.1, -0.2, 0.2, -0.2, 0.2, -0.2, -0.0, -0.2, -0.1, -0.2, -0.2},
{-0.0, 0.0, -0.2, -0.2, 0.0, -0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.0, -0.2, -0.1, 0.0, -0.2, -0.2, 0.2, -0.2, -0.0, -0.1, 0.1},
{-0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.2, 0.1, -0.2, 0.2, -0.1, -0.2, 0.2, 0.1, -0.1, -0.2, -0.0, 0.1, 0.0, -0.1, -0.1, -0.1, -0.0, 0.1, -0.0},
{0.1, -0.2, -0.1, 0.1, -0.1, -0.1, -0.2, -0.1, -0.2, -0.0, 0.0, -0.1, -0.0, -0.1, -0.0, 0.1, -0.1, 0.0, 0.1, -0.1, 0.1, -0.2, -0.1, 0.2, -0.0},
{0.1, -0.2, 0.1, 0.1, 0.1, -0.1, 0.1, -0.2, -0.2, 0.1, 0.0, 0.1, 0.1, -0.1, 0.1, -0.2, 0.2, 0.1, -0.0, 0.1, 0.2, 0.1, 0.2, -0.1, 0.2},
{-0.1, -0.0, -0.1, 0.1, -0.1, 0.1, 0.1, -0.2, -0.1, 0.0, -0.1, -0.0, 0.1, 0.2, -0.1, 0.0, 0.2, 0.0, -0.0, 0.1, -0.1, 0.1, 0.2, -0.0, -0.2},
{-0.1, -0.0, 0.2, 0.0, -0.1, 0.1, 0.1, 0.0, 0.2, -0.1, 0.0, -0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, 0.0, 0.1, 0.2, 0.2, -0.1, 0.0, -0.2},
{0.1, -0.1, 0.1, 0.1, -0.1, 0.2, 0.1, -0.2, 0.2, -0.2, -0.1, -0.1, -0.0, 0.1, -0.1, -0.1, -0.1, 0.0, -0.2, 0.1, 0.1, 0.1, -0.0, -0.1, 0.1},
{0.0, -0.0, -0.1, -0.2, -0.1, -0.1, -0.1, -0.0, -0.0, -0.2, -0.1, -0.2, 0.1, 0.2, -0.0, 0.1, -0.0, -0.1, 0.2, -0.1, 0.2, 0.1, 0.0, -0.2, 0.1},
{0.0, 0.2, -0.1, 0.0, 0.1, 0.2, 0.0, 0.1, -0.2, -0.1, -0.0, -0.0, 0.1, 0.1, 0.1, -0.0, -0.1, 0.0, -0.0, 0.0, 0.2, -0.1, 0.2, 0.1, 0.0},
{0.1, -0.0, -0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.0, -0.1, 0.2, 0.1, -0.2, 0.1, 0.1, 0.1, -0.1, 0.0, -0.0, 0.0, -0.1, -0.0, -0.2, -0.1},
{0.1, -0.1, -0.0, -0.0, 0.1, -0.1, -0.1, 0.2, 0.1, 0.0, 0.1, -0.2, -0.2, 0.1, 0.1, 0.2, -0.2, 0.1, 0.1, -0.2, -0.2, -0.0, 0.0, -0.2, 0.1},
}
);
Matrix  transformer_layers_8_attention_key_bias   (
{-0.1, -0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.2, 0.2, 0.1, -0.0, 0.0, -0.0, 0.2, -0.1, 0.1, 0.2, 0.1, -0.0, -0.1}
);
Matrix  transformer_layers_8_attention_value_weight   (
{{-0.1, -0.2, 0.0, -0.2, -0.1, 0.1, -0.0, 0.1, -0.1, -0.1, 0.0, -0.1, -0.2, -0.1, -0.0, 0.1, -0.0, 0.1, -0.1, -0.1, 0.0, 0.1, 0.1, -0.1, 0.1},
{0.0, 0.1, 0.1, 0.2, 0.1, 0.2, -0.0, 0.1, -0.2, 0.0, 0.1, -0.2, -0.0, -0.1, -0.1, 0.0, 0.1, 0.2, -0.1, 0.2, 0.1, 0.1, -0.2, -0.1, -0.1},
{-0.1, -0.2, 0.1, -0.0, 0.2, -0.0, -0.0, -0.0, 0.1, -0.0, 0.0, -0.0, 0.1, -0.0, 0.2, -0.1, 0.1, -0.2, 0.0, -0.1, 0.0, 0.0, 0.1, -0.2, -0.1},
{0.1, 0.2, 0.1, -0.1, 0.0, 0.0, 0.0, 0.1, -0.1, 0.1, 0.1, -0.1, -0.0, -0.1, -0.1, -0.2, -0.1, 0.0, 0.0, -0.1, -0.1, 0.1, -0.2, 0.1, 0.1},
{0.0, 0.1, -0.2, -0.0, 0.0, -0.1, 0.2, 0.1, -0.1, 0.0, -0.0, 0.1, 0.1, -0.0, 0.0, 0.1, -0.1, 0.0, 0.1, 0.0, -0.2, 0.0, 0.1, 0.0, -0.0},
{-0.2, -0.1, 0.1, -0.1, 0.0, -0.2, 0.0, 0.1, 0.1, 0.1, 0.2, 0.0, -0.1, -0.2, 0.1, -0.1, -0.2, -0.2, -0.1, -0.1, 0.0, 0.2, -0.1, 0.1, -0.1},
{-0.1, 0.1, 0.0, -0.1, 0.0, -0.0, 0.1, 0.0, 0.1, 0.1, 0.0, -0.0, 0.1, -0.1, 0.2, -0.1, 0.1, -0.1, -0.1, 0.2, 0.2, 0.1, -0.1, -0.1, -0.1},
{-0.0, 0.1, 0.1, -0.1, 0.0, 0.1, -0.0, -0.1, 0.0, -0.1, -0.1, 0.1, -0.1, -0.2, -0.2, 0.1, -0.2, 0.1, 0.0, -0.0, -0.0, -0.1, 0.0, 0.0, -0.1},
{-0.2, -0.1, -0.1, 0.0, 0.2, -0.1, 0.1, 0.1, -0.1, 0.2, 0.1, -0.1, 0.1, -0.0, -0.0, 0.2, 0.2, 0.0, 0.1, 0.1, -0.2, -0.0, -0.2, -0.1, -0.1},
{0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.1, -0.1, -0.2, 0.1, 0.0, 0.1, 0.1, -0.0, -0.1, -0.1, -0.1, 0.2, 0.0, 0.1, -0.1, 0.2, 0.0, 0.1, 0.1},
{0.1, -0.1, 0.1, 0.2, -0.0, 0.2, 0.2, -0.1, 0.2, 0.1, -0.2, 0.0, 0.1, -0.1, -0.1, -0.2, -0.2, -0.1, -0.1, -0.2, -0.0, 0.2, -0.0, -0.2, -0.1},
{0.0, 0.1, 0.0, -0.2, -0.0, 0.1, 0.2, 0.2, 0.1, 0.1, 0.0, -0.2, -0.1, 0.2, 0.1, -0.1, -0.2, 0.2, 0.1, -0.2, 0.1, -0.1, 0.0, 0.0, 0.2},
{-0.0, -0.0, 0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.0, -0.2, 0.1, -0.0, 0.2, -0.1, 0.2, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.2, -0.1, -0.1, -0.1},
{-0.2, -0.0, -0.2, -0.1, 0.0, 0.0, 0.1, 0.1, 0.2, -0.1, 0.0, 0.1, 0.1, -0.0, 0.1, -0.1, -0.1, 0.1, -0.2, 0.0, -0.1, 0.1, -0.1, 0.0, -0.1},
{-0.2, -0.2, -0.1, 0.1, 0.2, 0.0, -0.1, 0.0, 0.1, -0.0, 0.2, 0.1, 0.2, -0.1, 0.2, -0.1, 0.0, -0.1, 0.0, 0.1, -0.1, 0.1, 0.2, -0.1, -0.1},
{-0.0, -0.1, 0.1, 0.1, -0.1, -0.0, -0.1, -0.1, 0.1, -0.1, -0.2, 0.2, -0.0, 0.1, -0.0, 0.1, -0.1, 0.2, -0.1, 0.2, 0.1, 0.2, -0.0, -0.0, -0.2},
{-0.0, 0.0, 0.0, -0.1, -0.1, 0.0, 0.0, -0.0, 0.1, 0.1, -0.0, 0.1, -0.2, -0.1, 0.1, 0.1, 0.2, 0.1, -0.1, -0.2, -0.2, 0.1, -0.2, 0.2, -0.1},
{0.2, -0.0, 0.1, -0.1, -0.1, -0.2, 0.1, 0.0, -0.1, -0.1, -0.1, -0.0, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.0, 0.1, 0.2, -0.0, 0.2, -0.1},
{0.2, 0.1, -0.2, 0.1, -0.2, 0.1, 0.2, -0.2, -0.1, 0.1, 0.0, 0.2, -0.0, 0.1, -0.1, -0.1, 0.2, 0.0, 0.1, -0.0, -0.1, -0.1, -0.1, 0.1, -0.0},
{0.2, 0.1, -0.1, -0.1, 0.1, -0.0, -0.2, 0.1, -0.2, -0.0, -0.2, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.0, 0.2, -0.0, -0.0, 0.0, 0.1, 0.1, 0.1},
{-0.1, -0.0, 0.1, 0.1, 0.2, -0.2, -0.2, 0.0, -0.2, -0.0, 0.0, 0.1, 0.1, -0.1, 0.0, -0.1, 0.2, -0.0, -0.1, -0.2, -0.1, -0.2, -0.1, -0.0, 0.2},
{-0.2, -0.0, -0.1, 0.1, 0.0, -0.2, -0.1, -0.2, 0.1, 0.2, 0.2, 0.1, -0.2, -0.1, -0.2, 0.2, 0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.1, -0.0, 0.0},
{0.1, 0.1, 0.0, -0.2, -0.0, -0.0, 0.1, 0.1, 0.1, -0.2, -0.2, 0.0, -0.0, 0.1, -0.2, -0.0, -0.0, -0.1, 0.2, -0.1, 0.0, 0.0, -0.1, 0.1, 0.1},
{-0.1, 0.1, 0.1, 0.1, -0.0, 0.1, 0.1, -0.1, -0.2, 0.0, -0.0, -0.2, 0.2, -0.0, 0.1, -0.0, -0.1, -0.1, -0.1, 0.1, 0.1, 0.2, 0.0, -0.2, 0.1},
{-0.0, -0.1, -0.1, -0.1, 0.1, 0.1, -0.0, -0.1, -0.0, 0.1, -0.1, -0.1, 0.1, -0.0, -0.1, 0.1, -0.1, -0.1, -0.0, -0.2, -0.1, 0.1, 0.2, -0.0, -0.2},
}
);
Matrix  transformer_layers_8_attention_value_bias   (
{0.1, 0.0, 0.2, -0.1, -0.0, -0.1, 0.2, 0.1, -0.1, 0.1, -0.0, -0.1, 0.0, -0.1, 0.1, -0.0, 0.2, 0.0, -0.1, 0.1, 0.1, 0.2, -0.0, -0.1, 0.1}
);
Matrix  transformer_layers_8_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_8_norm1_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_8_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_8_norm2_layer_norm_bias   (
{-0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_8_feed_forward_linear1_weight   (
{{0.2, 0.0, -0.2, -0.2, 0.1, 0.0, -0.1, -0.1, -0.1, -0.2, -0.0, 0.1, -0.1, -0.1, 0.0, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.2, -0.0, 0.0},
{-0.1, 0.0, -0.2, 0.0, 0.2, 0.2, -0.1, -0.1, 0.1, -0.1, 0.2, 0.1, -0.0, 0.2, 0.0, 0.0, 0.2, -0.1, -0.1, -0.2, 0.1, 0.1, 0.0, 0.2, 0.1},
{0.1, 0.0, 0.1, 0.1, 0.2, 0.2, -0.1, -0.1, -0.1, -0.1, -0.0, -0.0, -0.1, 0.2, -0.0, -0.0, -0.0, -0.1, -0.0, 0.0, -0.2, 0.1, 0.2, -0.0, 0.1},
{-0.1, 0.1, -0.2, -0.0, -0.0, -0.0, -0.2, -0.1, -0.1, 0.2, -0.1, 0.1, -0.0, -0.0, 0.1, -0.2, -0.0, 0.1, -0.2, -0.1, 0.0, -0.1, -0.1, -0.0, -0.1},
{-0.1, 0.0, 0.1, -0.0, 0.1, -0.1, 0.1, -0.2, -0.0, -0.1, -0.1, 0.0, 0.1, 0.2, -0.1, -0.0, 0.2, 0.1, -0.0, -0.1, 0.2, -0.1, 0.0, -0.2, -0.0},
{-0.0, 0.1, -0.0, 0.0, 0.1, -0.1, 0.1, -0.1, -0.2, 0.0, 0.1, -0.1, -0.1, 0.1, -0.1, 0.1, 0.2, -0.1, -0.0, 0.1, 0.1, 0.0, -0.2, 0.1, -0.1},
{0.0, -0.1, 0.1, -0.1, 0.0, -0.1, 0.1, -0.1, -0.0, -0.0, 0.1, -0.2, -0.0, 0.1, -0.1, 0.2, 0.0, 0.0, -0.1, 0.1, 0.1, 0.1, 0.0, -0.0, -0.1},
{-0.2, -0.1, 0.2, -0.2, 0.1, 0.1, 0.2, 0.2, 0.1, -0.0, 0.0, -0.1, -0.1, -0.0, -0.0, -0.1, -0.0, -0.1, 0.2, 0.1, 0.0, 0.2, -0.2, 0.1, 0.1},
{0.1, 0.1, 0.2, -0.1, -0.1, -0.2, 0.1, -0.0, 0.1, 0.0, 0.1, -0.0, -0.0, 0.1, 0.2, 0.0, 0.2, -0.2, 0.1, 0.1, 0.1, -0.2, 0.2, -0.2, 0.2},
{-0.2, 0.1, 0.2, 0.2, 0.1, 0.1, -0.2, -0.2, 0.0, -0.2, -0.1, 0.0, 0.2, -0.0, -0.1, 0.0, 0.1, -0.0, 0.2, -0.1, 0.1, -0.2, -0.0, -0.1, -0.1},
{-0.2, -0.1, -0.1, 0.1, 0.0, -0.0, 0.0, -0.2, 0.1, 0.1, -0.0, 0.2, 0.2, -0.1, -0.1, -0.1, -0.0, 0.0, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.2},
{-0.1, 0.2, -0.1, 0.1, 0.0, 0.0, -0.2, -0.1, -0.1, 0.1, -0.1, -0.1, 0.0, 0.1, -0.1, -0.2, -0.0, -0.1, -0.1, -0.0, 0.0, -0.1, 0.0, -0.2, 0.1},
{-0.2, -0.2, -0.1, 0.0, -0.0, 0.2, -0.1, -0.2, -0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2, -0.0, -0.1, -0.1, 0.1, 0.2, 0.2, -0.0, 0.1, 0.1},
{-0.1, 0.1, 0.0, -0.2, 0.1, 0.1, -0.1, 0.1, 0.0, -0.0, 0.1, 0.1, 0.1, -0.0, -0.2, 0.2, -0.1, 0.1, -0.1, -0.1, -0.1, 0.0, 0.1, 0.1, -0.1},
{-0.1, -0.1, -0.1, 0.1, -0.0, -0.1, 0.2, -0.2, 0.1, 0.1, -0.2, 0.0, 0.1, -0.1, 0.1, 0.0, -0.1, 0.1, 0.1, 0.0, -0.2, -0.2, 0.1, -0.2, 0.1},
}
);
Matrix  transformer_layers_8_feed_forward_linear1_bias   (
{-0.0, -0.0, 0.1, -0.0, 0.0, -0.0, 0.0, 0.1, 0.1, 0.2, -0.1, -0.0, 0.2, -0.1, 0.1}
);
Matrix  transformer_layers_8_feed_forward_linear2_weight   (
{{-0.2, -0.2, -0.2, 0.1, 0.1, -0.1, -0.1, 0.1, -0.0, -0.2, -0.1, -0.2, 0.2, -0.0, -0.1},
{-0.1, 0.2, -0.0, 0.1, 0.2, 0.3, 0.1, 0.0, -0.0, 0.2, -0.1, 0.0, -0.1, 0.1, -0.2},
{-0.0, 0.1, -0.1, 0.1, -0.2, 0.0, 0.1, -0.1, 0.1, 0.1, -0.0, 0.1, 0.1, 0.1, -0.0},
{0.0, 0.2, -0.2, -0.1, 0.2, 0.0, -0.1, 0.2, -0.0, -0.1, -0.2, -0.0, 0.1, 0.1, 0.0},
{-0.0, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.1, 0.3, 0.1, 0.1, -0.1, -0.2, 0.1, -0.1},
{-0.1, -0.2, -0.3, -0.2, 0.2, -0.2, 0.1, -0.0, -0.2, -0.2, -0.3, 0.2, 0.1, 0.1, -0.0},
{0.0, 0.0, 0.0, -0.2, -0.2, 0.0, -0.0, -0.0, -0.0, 0.1, -0.2, -0.1, -0.2, -0.2, -0.1},
{-0.0, 0.0, -0.0, -0.1, 0.1, -0.1, 0.2, 0.1, 0.2, -0.2, -0.1, -0.2, -0.2, 0.0, -0.2},
{-0.1, -0.2, 0.1, 0.1, -0.1, 0.1, -0.1, 0.2, -0.1, -0.0, -0.2, -0.0, -0.1, 0.2, -0.2},
{0.1, 0.2, -0.0, -0.2, -0.2, 0.2, -0.1, -0.1, -0.2, 0.2, -0.1, 0.1, -0.2, -0.2, -0.2},
{-0.1, 0.2, 0.2, 0.1, 0.3, -0.2, 0.0, 0.1, -0.1, -0.1, 0.2, -0.1, -0.2, 0.0, -0.1},
{0.2, 0.2, -0.1, 0.0, 0.0, 0.1, -0.1, 0.2, -0.2, 0.1, 0.1, 0.2, 0.1, -0.2, 0.2},
{0.1, -0.1, 0.2, -0.1, 0.0, -0.1, 0.3, -0.1, 0.2, -0.2, 0.0, -0.2, -0.1, -0.2, 0.2},
{0.1, 0.1, -0.0, -0.1, 0.2, 0.1, 0.0, 0.1, -0.2, -0.2, 0.0, -0.2, 0.0, -0.3, 0.1},
{-0.2, -0.1, 0.2, -0.1, 0.1, -0.2, -0.2, -0.1, -0.1, -0.0, 0.1, 0.1, -0.1, -0.2, 0.2},
{-0.0, 0.1, -0.2, 0.1, 0.2, 0.2, -0.0, 0.1, -0.3, -0.1, 0.1, 0.2, 0.1, -0.0, -0.2},
{-0.0, 0.0, -0.2, 0.1, 0.2, -0.1, -0.2, -0.0, 0.2, 0.1, -0.2, 0.1, -0.2, -0.1, -0.2},
{-0.1, -0.2, -0.1, -0.3, -0.1, 0.1, 0.1, 0.1, -0.1, -0.0, 0.2, -0.1, 0.1, -0.0, 0.1},
{-0.2, -0.1, 0.2, -0.0, -0.1, 0.1, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1, -0.1, -0.2},
{0.0, -0.1, 0.1, 0.1, 0.0, -0.2, -0.1, -0.2, -0.2, 0.0, -0.2, 0.1, -0.1, -0.2, 0.1},
{-0.1, -0.1, 0.3, -0.0, -0.1, -0.2, 0.3, -0.1, -0.2, -0.1, 0.2, 0.0, 0.1, 0.1, -0.2},
{0.2, 0.1, -0.1, 0.2, 0.1, -0.2, -0.1, -0.1, -0.2, -0.1, -0.0, -0.1, 0.2, -0.2, -0.0},
{0.1, -0.2, 0.1, 0.1, -0.2, 0.1, 0.2, 0.2, 0.1, 0.0, -0.1, -0.2, 0.2, -0.2, 0.2},
{-0.0, -0.0, -0.0, 0.2, 0.1, 0.1, -0.2, -0.2, 0.2, -0.0, 0.1, -0.1, 0.2, 0.1, 0.2},
{0.0, 0.1, -0.2, 0.2, -0.1, -0.2, 0.1, -0.2, -0.2, -0.2, -0.2, -0.1, -0.1, -0.2, -0.0},
}
);
Matrix  transformer_layers_8_feed_forward_linear2_bias   (
{0.1, -0.2, 0.1, 0.2, -0.1, 0.2, 0.2, 0.2, -0.1, 0.1, 0.2, 0.2, -0.1, 0.1, 0.2, -0.1, 0.1, -0.1, 0.1, -0.1, -0.2, -0.2, 0.2, -0.1, 0.2}
);
Matrix  transformer_layers_8_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_8_feed_forward_ln1_layer_norm_bias   (
{-0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_8_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_8_feed_forward_ln2_layer_norm_bias   (
{-0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_9_attention_query_weight   (
{{-0.1, 0.1, 0.0, -0.2, -0.1, 0.0, -0.0, 0.2, 0.1, -0.0, 0.1, 0.0, -0.1, -0.2, -0.1, -0.1, -0.1, 0.2, -0.0, -0.0, -0.1, -0.0, -0.1, -0.0, -0.1},
{0.1, 0.1, 0.2, -0.0, 0.1, -0.1, -0.0, 0.1, -0.2, -0.2, 0.1, -0.2, -0.1, -0.0, 0.1, 0.0, -0.2, 0.1, 0.1, 0.1, -0.2, -0.1, -0.1, 0.1, 0.1},
{-0.0, -0.2, -0.1, -0.0, -0.2, -0.1, 0.1, -0.0, -0.2, 0.0, 0.1, -0.1, -0.1, -0.0, 0.0, 0.2, 0.1, 0.0, 0.1, -0.1, 0.1, 0.2, -0.1, 0.1, -0.1},
{-0.2, -0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.2, 0.1, 0.1, 0.2, -0.1, -0.0, 0.2, 0.2, 0.1, -0.0, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1, 0.1},
{-0.1, -0.1, -0.0, 0.2, -0.2, 0.2, 0.1, 0.1, 0.0, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.0, 0.0, 0.0, 0.1, -0.2, -0.2, -0.1, 0.0, -0.1, 0.2},
{-0.2, 0.1, 0.0, -0.2, 0.1, -0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.0, -0.2, 0.1, 0.2, 0.0, -0.2, -0.1, -0.2, 0.1, -0.1, -0.1, 0.1, 0.0, -0.0},
{-0.1, -0.2, -0.1, -0.1, 0.0, -0.1, 0.1, 0.1, 0.0, -0.1, 0.1, 0.0, -0.1, 0.0, 0.1, -0.2, -0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.1, -0.1, -0.1},
{-0.1, 0.1, -0.0, -0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.0, -0.1, -0.2, 0.1, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1, -0.1, -0.1, 0.0, -0.1, -0.2},
{-0.2, 0.2, 0.1, 0.2, 0.0, -0.0, 0.1, 0.1, -0.2, 0.2, -0.0, 0.1, 0.1, 0.1, 0.2, -0.2, -0.1, -0.1, 0.1, 0.1, -0.1, -0.2, 0.1, 0.0, 0.2},
{0.1, -0.0, 0.1, 0.2, -0.2, 0.0, 0.1, -0.1, 0.1, 0.1, 0.0, 0.1, -0.2, 0.0, 0.2, 0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.0, 0.1, 0.2},
{0.1, 0.2, -0.0, -0.1, 0.0, 0.1, -0.0, -0.1, 0.1, -0.1, 0.1, -0.2, 0.1, 0.1, -0.1, 0.1, 0.1, -0.0, -0.1, -0.0, 0.2, 0.2, -0.0, -0.2, -0.1},
{-0.1, 0.2, 0.1, -0.2, 0.2, 0.1, 0.1, 0.1, -0.0, 0.0, -0.1, -0.0, -0.1, -0.1, 0.1, -0.2, -0.0, -0.0, -0.1, 0.1, -0.2, -0.1, 0.2, 0.1, -0.0},
{-0.1, -0.1, 0.1, 0.0, -0.1, 0.1, -0.0, -0.0, 0.1, 0.1, 0.1, -0.2, -0.1, -0.2, -0.0, 0.2, -0.2, -0.1, -0.0, 0.1, 0.0, 0.1, 0.2, -0.0, 0.0},
{-0.1, -0.2, -0.0, 0.2, 0.0, -0.1, -0.0, 0.1, -0.2, -0.2, 0.1, 0.1, 0.0, -0.1, -0.0, 0.1, -0.0, 0.1, -0.1, -0.1, -0.0, -0.2, -0.1, -0.1, 0.1},
{0.0, -0.1, 0.1, 0.2, 0.0, 0.1, 0.0, -0.1, -0.0, 0.1, 0.1, -0.2, -0.1, -0.0, 0.2, 0.1, 0.1, 0.2, -0.2, 0.2, 0.1, -0.0, -0.2, -0.2, -0.0},
{-0.1, 0.1, -0.1, 0.1, 0.1, 0.0, 0.1, -0.1, 0.2, 0.2, 0.2, -0.1, 0.1, -0.0, 0.1, -0.1, 0.0, 0.1, 0.2, -0.2, 0.0, -0.0, -0.0, -0.1, -0.0},
{-0.0, 0.2, -0.1, -0.0, 0.0, -0.1, 0.1, 0.2, 0.2, 0.2, -0.1, 0.1, 0.0, 0.1, 0.1, -0.1, -0.1, 0.0, -0.1, 0.1, -0.0, -0.1, -0.2, -0.0, 0.1},
{0.1, -0.0, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, -0.0, -0.2, -0.1, -0.1, 0.1, -0.1, -0.2, 0.2, -0.2, -0.0, -0.2, 0.2, 0.2},
{-0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.0, -0.2, -0.0, -0.2, 0.0, 0.1, -0.2, 0.1, 0.0, -0.1, 0.1, 0.1, -0.2, 0.2, -0.1, 0.0, 0.2, -0.0, 0.1},
{0.1, -0.1, -0.2, 0.2, -0.1, 0.1, 0.1, 0.1, -0.1, 0.1, -0.0, -0.1, -0.1, 0.1, -0.1, -0.0, -0.2, 0.0, -0.1, -0.0, -0.1, 0.1, 0.2, 0.1, 0.1},
{0.1, -0.1, -0.1, -0.0, 0.1, 0.1, 0.2, -0.1, -0.1, 0.1, -0.2, -0.1, 0.1, 0.1, -0.1, 0.0, -0.1, 0.2, 0.1, -0.1, -0.1, 0.1, -0.2, 0.1, -0.0},
{0.1, 0.2, -0.2, -0.1, 0.0, -0.0, 0.1, -0.0, 0.0, -0.1, 0.1, 0.1, 0.1, -0.0, 0.0, 0.1, 0.2, -0.2, -0.1, 0.0, -0.0, 0.1, -0.2, 0.1, -0.2},
{-0.1, -0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.2, -0.0, 0.1, 0.0, -0.1, -0.1, 0.2, -0.1, -0.1, -0.1, -0.2, 0.0, 0.1, 0.0, 0.0, -0.0, -0.1, -0.2},
{0.2, -0.2, -0.1, -0.1, -0.1, -0.0, -0.1, -0.2, -0.2, -0.1, 0.0, -0.2, 0.1, 0.2, -0.2, -0.0, -0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, -0.1, -0.1},
{-0.1, -0.1, 0.2, 0.0, 0.0, -0.2, -0.1, -0.1, -0.1, 0.2, 0.1, 0.1, -0.1, 0.1, -0.1, -0.1, 0.1, 0.2, -0.0, 0.1, 0.1, -0.2, 0.0, -0.1, -0.2},
}
);
Matrix  transformer_layers_9_attention_query_bias   (
{0.1, -0.1, 0.1, -0.2, -0.1, -0.2, -0.2, -0.0, 0.1, -0.2, 0.2, -0.0, 0.1, 0.1, -0.0, -0.0, 0.1, -0.1, 0.1, 0.0, -0.2, 0.2, -0.1, 0.1, -0.2}
);
Matrix  transformer_layers_9_attention_key_weight   (
{{0.0, -0.1, -0.1, 0.0, 0.0, 0.1, -0.0, 0.1, 0.1, 0.2, 0.1, 0.0, 0.2, 0.1, 0.0, 0.1, 0.1, -0.1, 0.1, -0.1, 0.2, 0.2, 0.2, 0.1, -0.2},
{-0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0, 0.1, -0.0, -0.2, 0.0, -0.2, 0.1, -0.0, -0.1, 0.0, 0.1, 0.1, -0.1},
{-0.1, -0.0, -0.2, -0.1, -0.0, 0.1, -0.2, -0.1, 0.1, -0.0, 0.2, 0.0, 0.1, -0.1, 0.0, -0.2, 0.0, -0.2, 0.0, 0.1, 0.1, -0.1, -0.2, -0.1, -0.2},
{0.2, 0.2, 0.2, 0.1, -0.2, 0.1, 0.1, 0.1, 0.2, 0.1, -0.0, -0.2, 0.1, 0.1, -0.1, -0.2, 0.0, -0.1, -0.2, -0.2, 0.1, -0.0, -0.0, -0.0, -0.0},
{-0.1, -0.0, 0.2, 0.2, 0.1, 0.0, -0.1, 0.0, 0.2, -0.1, -0.0, -0.1, -0.0, 0.1, -0.0, -0.0, 0.1, 0.0, -0.1, 0.1, -0.0, -0.1, -0.1, 0.1, -0.0},
{-0.2, -0.1, -0.0, 0.0, 0.1, -0.1, -0.1, -0.1, -0.0, 0.0, 0.1, 0.1, -0.2, 0.2, 0.1, 0.0, 0.1, 0.1, -0.2, -0.1, 0.1, -0.1, -0.2, -0.1, 0.1},
{0.1, -0.1, 0.1, 0.1, 0.0, 0.0, 0.1, -0.2, -0.1, -0.2, -0.1, -0.0, 0.1, 0.2, 0.1, -0.2, -0.0, -0.0, -0.2, -0.2, 0.1, 0.0, 0.1, 0.1, 0.1},
{0.1, 0.1, -0.1, 0.0, -0.0, 0.1, 0.1, -0.0, 0.1, 0.2, 0.2, -0.0, 0.2, 0.2, -0.1, -0.1, 0.1, 0.2, -0.2, 0.2, 0.0, -0.1, 0.2, 0.1, 0.2},
{0.1, -0.0, -0.2, 0.1, 0.1, 0.1, 0.1, -0.1, -0.2, -0.0, -0.1, -0.2, 0.1, -0.1, -0.2, 0.2, -0.1, 0.1, 0.0, -0.1, 0.2, -0.0, -0.2, -0.0, 0.2},
{0.2, 0.2, 0.0, -0.2, -0.0, 0.1, -0.1, -0.2, 0.0, -0.2, -0.1, -0.1, 0.1, -0.1, -0.2, 0.1, 0.0, -0.0, 0.2, -0.0, -0.0, 0.0, 0.0, -0.0, -0.2},
{-0.1, 0.2, 0.1, 0.1, -0.0, -0.0, -0.0, -0.0, 0.1, 0.2, 0.2, 0.0, -0.1, 0.2, 0.2, 0.1, -0.0, 0.1, 0.1, -0.0, 0.0, -0.2, -0.1, -0.1, 0.1},
{0.1, 0.2, 0.1, -0.1, 0.2, -0.1, 0.1, -0.0, 0.1, 0.0, -0.0, -0.2, 0.1, 0.1, 0.2, 0.0, 0.0, 0.0, 0.1, -0.1, -0.0, 0.1, 0.0, -0.2, 0.0},
{-0.1, -0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.0, -0.1, 0.0, 0.0, 0.0, -0.2, -0.2, -0.1, -0.2, 0.1, 0.1, 0.1, 0.1, -0.1, 0.1},
{0.0, -0.0, 0.1, 0.2, -0.2, 0.0, 0.0, -0.2, 0.0, -0.1, -0.0, 0.1, -0.1, -0.2, -0.0, -0.2, 0.2, -0.2, 0.1, -0.1, 0.0, 0.1, -0.2, -0.2, -0.0},
{0.1, -0.1, -0.2, -0.1, -0.2, 0.0, -0.2, 0.1, -0.1, -0.1, 0.2, -0.1, 0.1, 0.0, 0.2, 0.1, 0.1, -0.0, 0.0, 0.1, 0.2, -0.0, 0.1, 0.1, -0.2},
{0.1, -0.0, -0.1, -0.1, -0.2, 0.0, -0.2, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, -0.2, -0.2, -0.0, 0.1, 0.0, -0.1, 0.0, -0.1, -0.2, 0.1, 0.0, -0.2},
{-0.1, -0.0, -0.1, 0.1, -0.0, -0.2, 0.1, 0.1, 0.1, 0.0, -0.1, 0.1, 0.1, 0.0, -0.0, 0.1, -0.1, -0.1, 0.1, -0.0, -0.1, 0.1, -0.1, 0.1, -0.0},
{-0.2, -0.0, -0.0, -0.1, 0.0, 0.2, -0.2, 0.1, 0.1, 0.0, 0.1, -0.0, -0.0, 0.1, -0.1, 0.1, 0.1, 0.0, 0.0, 0.2, 0.1, 0.1, -0.1, -0.0, -0.0},
{0.1, -0.0, 0.1, 0.0, 0.1, -0.1, 0.2, -0.2, 0.0, -0.0, -0.2, -0.1, -0.2, 0.1, 0.2, -0.1, 0.1, 0.0, -0.1, 0.1, -0.2, -0.1, 0.2, 0.2, -0.1},
{-0.0, -0.0, 0.1, -0.2, -0.2, -0.0, -0.0, 0.1, -0.1, -0.2, -0.1, -0.2, -0.1, 0.1, 0.0, 0.0, -0.1, -0.0, 0.2, -0.1, 0.2, 0.0, -0.2, -0.2, 0.1},
{-0.1, 0.2, 0.1, 0.0, 0.0, -0.1, -0.0, -0.1, 0.1, 0.1, -0.1, 0.0, -0.0, -0.1, 0.1, -0.0, 0.0, 0.2, -0.2, 0.1, -0.0, -0.1, -0.2, -0.2, -0.1},
{0.1, -0.1, -0.2, 0.1, 0.1, -0.0, 0.1, -0.0, -0.0, 0.2, 0.2, 0.1, -0.2, -0.1, -0.1, 0.1, -0.2, -0.0, -0.0, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1},
{-0.0, -0.1, -0.2, -0.1, -0.1, -0.2, -0.1, 0.2, -0.2, 0.1, 0.1, 0.0, -0.0, -0.1, -0.1, -0.2, -0.1, -0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.1, 0.1},
{-0.1, -0.1, -0.0, 0.1, -0.2, 0.1, -0.0, 0.2, 0.2, -0.1, 0.1, -0.2, 0.1, 0.2, 0.1, -0.0, -0.1, 0.1, -0.1, -0.1, 0.0, 0.2, 0.1, 0.0, 0.1},
{0.1, 0.0, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, -0.2, -0.1, 0.1, -0.1, 0.1, 0.0, 0.1, 0.0, 0.0, 0.0, 0.2, 0.1, 0.1, -0.2, -0.1},
}
);
Matrix  transformer_layers_9_attention_key_bias   (
{0.0, 0.1, 0.0, -0.1, 0.1, 0.0, -0.2, -0.0, 0.2, -0.2, -0.2, 0.0, -0.2, 0.1, -0.1, -0.2, -0.1, 0.1, -0.1, -0.1, 0.2, 0.1, -0.1, 0.0, 0.2}
);
Matrix  transformer_layers_9_attention_value_weight   (
{{-0.1, -0.1, -0.1, -0.1, 0.2, -0.1, -0.1, -0.1, -0.1, 0.1, 0.0, 0.1, -0.1, -0.0, -0.0, 0.1, -0.1, 0.1, 0.0, -0.1, 0.0, -0.1, 0.0, 0.2, 0.2},
{-0.1, -0.0, 0.1, -0.0, 0.1, 0.1, 0.2, 0.0, -0.1, -0.1, -0.2, -0.1, -0.0, 0.2, -0.2, 0.2, 0.1, 0.1, -0.1, 0.0, -0.1, -0.1, 0.0, -0.1, -0.2},
{0.0, 0.2, 0.0, -0.1, -0.1, -0.1, -0.0, 0.1, -0.1, -0.1, -0.2, 0.2, -0.1, -0.1, -0.1, 0.0, 0.0, 0.1, 0.0, -0.1, 0.0, 0.2, -0.0, -0.1, 0.0},
{0.2, 0.1, 0.1, 0.1, -0.0, 0.1, 0.1, 0.1, -0.2, -0.0, -0.0, -0.2, 0.0, 0.2, 0.1, 0.1, -0.0, 0.2, -0.0, -0.0, 0.1, 0.1, -0.1, -0.2, 0.1},
{0.1, 0.1, -0.2, 0.1, -0.0, -0.2, -0.0, 0.0, -0.2, 0.1, 0.2, 0.2, 0.2, -0.2, -0.1, -0.1, 0.1, 0.2, 0.2, -0.2, 0.0, 0.0, 0.1, 0.1, 0.2},
{-0.2, -0.2, -0.0, 0.1, 0.1, 0.1, 0.0, -0.0, -0.1, -0.0, -0.2, -0.0, -0.1, -0.2, 0.1, -0.1, 0.0, -0.2, 0.1, 0.0, 0.1, 0.1, -0.0, 0.1, 0.1},
{0.2, 0.1, -0.1, -0.0, -0.0, -0.0, 0.1, 0.1, 0.1, 0.2, 0.2, -0.0, 0.1, -0.1, 0.1, -0.2, -0.0, 0.1, 0.1, 0.2, 0.1, -0.1, 0.1, -0.0, -0.1},
{-0.0, -0.1, 0.1, -0.1, -0.0, 0.0, -0.0, -0.1, -0.1, 0.0, 0.0, 0.0, -0.1, -0.2, 0.1, -0.2, 0.2, 0.0, 0.1, 0.1, 0.0, 0.1, 0.0, -0.1, -0.1},
{0.2, 0.0, 0.2, 0.2, 0.1, -0.1, -0.1, -0.0, -0.0, -0.2, 0.2, -0.0, -0.1, -0.1, 0.0, -0.1, -0.1, 0.0, 0.1, 0.2, -0.2, 0.1, 0.2, -0.0, -0.1},
{-0.1, 0.2, 0.2, 0.0, 0.2, -0.1, 0.2, -0.0, -0.2, 0.1, 0.0, -0.0, 0.1, 0.2, -0.1, -0.1, 0.0, 0.1, 0.1, -0.1, 0.0, 0.2, 0.0, -0.1, 0.1},
{0.1, 0.1, 0.1, -0.1, 0.1, -0.1, -0.0, -0.1, -0.1, -0.1, -0.1, 0.1, 0.0, 0.1, 0.1, -0.1, 0.2, 0.1, -0.2, -0.2, -0.1, 0.2, 0.1, 0.2, 0.1},
{0.1, 0.1, -0.2, -0.0, 0.2, 0.2, 0.1, 0.1, -0.1, -0.1, -0.2, 0.1, 0.1, 0.2, -0.1, -0.1, -0.1, -0.1, 0.1, -0.1, 0.2, 0.1, -0.1, -0.2, 0.1},
{-0.2, -0.2, -0.1, -0.1, -0.1, -0.0, 0.1, -0.0, -0.1, 0.2, 0.0, -0.1, -0.1, -0.1, -0.2, 0.1, 0.2, 0.2, -0.0, 0.0, -0.1, -0.1, 0.1, 0.1, -0.1},
{0.2, -0.1, -0.1, 0.2, -0.1, -0.2, 0.2, -0.1, -0.1, 0.0, 0.1, 0.1, 0.1, -0.2, -0.1, -0.1, -0.0, -0.0, -0.1, 0.1, 0.0, 0.2, -0.1, 0.2, -0.1},
{0.1, -0.1, -0.1, -0.0, 0.0, -0.0, 0.1, 0.1, -0.1, 0.0, 0.1, -0.1, -0.0, 0.2, -0.0, -0.2, -0.1, 0.1, 0.0, -0.0, 0.1, 0.1, -0.2, 0.1, 0.2},
{-0.1, -0.0, -0.0, 0.0, 0.2, 0.1, -0.0, 0.0, 0.0, 0.0, -0.2, -0.1, 0.1, 0.2, 0.1, -0.1, 0.0, 0.2, 0.2, -0.0, -0.0, -0.1, -0.1, -0.0, -0.1},
{-0.2, 0.0, 0.0, -0.2, 0.2, -0.2, -0.0, 0.1, 0.1, 0.2, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.2, -0.1, -0.1, 0.1, -0.0, 0.0, -0.1, 0.1, 0.1},
{0.1, -0.2, 0.1, -0.0, 0.0, 0.1, -0.1, -0.1, 0.1, 0.1, -0.2, 0.1, -0.0, -0.1, 0.1, -0.0, 0.1, 0.2, -0.1, 0.1, -0.1, 0.0, 0.0, -0.1, -0.0},
{-0.0, 0.2, 0.1, 0.1, -0.0, -0.1, 0.0, -0.1, -0.1, -0.0, -0.0, 0.1, -0.1, 0.2, 0.0, 0.2, -0.0, 0.1, -0.1, 0.2, -0.2, -0.1, -0.0, 0.1, 0.2},
{0.2, -0.1, -0.1, -0.0, -0.0, -0.1, 0.2, 0.2, 0.2, 0.2, 0.0, -0.1, 0.2, 0.2, 0.2, 0.1, -0.1, -0.1, 0.1, -0.1, 0.0, -0.2, -0.2, 0.1, -0.1},
{-0.2, -0.1, -0.1, -0.2, 0.0, 0.1, 0.0, -0.2, 0.2, 0.1, -0.1, 0.2, -0.0, -0.2, -0.1, 0.0, 0.1, 0.1, 0.2, -0.0, 0.1, -0.2, -0.1, 0.1, 0.1},
{-0.2, 0.1, 0.2, -0.1, -0.1, 0.2, 0.1, 0.2, -0.2, 0.1, -0.1, 0.1, 0.0, -0.2, 0.2, 0.1, -0.1, 0.1, 0.0, -0.2, -0.0, 0.0, 0.2, 0.2, -0.0},
{-0.1, 0.1, -0.1, 0.0, -0.1, 0.2, 0.1, 0.1, 0.1, 0.2, -0.0, 0.1, 0.1, -0.2, 0.2, 0.1, 0.1, 0.2, 0.1, -0.1, 0.0, -0.1, 0.1, 0.0, -0.2},
{-0.1, -0.0, -0.1, 0.1, -0.2, -0.2, -0.1, -0.1, 0.0, -0.2, 0.0, -0.2, 0.2, -0.2, 0.1, -0.2, -0.0, 0.1, 0.2, 0.1, -0.2, -0.1, 0.1, -0.0, -0.2},
{0.2, -0.2, 0.0, 0.1, -0.1, 0.1, 0.2, -0.1, 0.2, -0.2, 0.0, -0.1, 0.0, -0.1, -0.1, 0.1, -0.2, 0.0, 0.2, -0.1, 0.2, -0.0, -0.1, 0.0, -0.2},
}
);
Matrix  transformer_layers_9_attention_value_bias   (
{0.1, 0.0, -0.1, 0.1, 0.2, 0.0, 0.2, 0.0, 0.1, -0.2, -0.1, -0.2, 0.1, -0.0, -0.1, 0.1, 0.0, 0.1, 0.1, 0.0, -0.2, 0.1, 0.1, 0.1, -0.2}
);
Matrix  transformer_layers_9_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_9_norm1_layer_norm_bias   (
{-0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_9_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_9_norm2_layer_norm_bias   (
{-0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_9_feed_forward_linear1_weight   (
{{0.0, 0.1, -0.1, -0.0, -0.0, -0.1, 0.1, 0.1, 0.0, 0.1, 0.2, 0.2, 0.1, 0.2, -0.2, -0.0, 0.0, -0.1, -0.0, -0.0, -0.1, -0.1, 0.1, -0.0, -0.0},
{0.1, -0.0, -0.2, 0.1, 0.1, -0.1, 0.1, 0.1, -0.0, -0.1, 0.1, 0.1, 0.0, 0.2, 0.0, -0.1, 0.2, -0.1, 0.2, 0.2, 0.1, -0.1, 0.0, -0.1, -0.1},
{0.0, 0.1, -0.0, 0.2, -0.1, -0.1, -0.1, -0.0, 0.0, 0.0, -0.2, 0.0, 0.1, -0.1, -0.1, 0.2, 0.0, 0.2, -0.0, 0.1, 0.1, 0.0, 0.0, 0.1, -0.0},
{0.0, 0.2, 0.0, 0.2, -0.0, -0.1, 0.2, 0.2, -0.1, -0.1, 0.1, -0.0, 0.0, -0.0, -0.1, -0.1, -0.2, -0.1, -0.1, 0.2, -0.0, -0.1, -0.0, 0.1, -0.0},
{0.0, -0.0, 0.2, 0.1, 0.1, -0.0, -0.1, 0.2, 0.1, 0.2, 0.0, -0.1, 0.0, 0.2, 0.1, -0.0, 0.1, 0.1, 0.2, 0.2, 0.1, -0.2, -0.2, -0.2, -0.0},
{0.2, -0.2, 0.1, -0.2, 0.2, 0.2, -0.0, -0.0, -0.1, -0.0, -0.1, 0.1, 0.1, -0.2, 0.1, 0.0, -0.0, 0.2, -0.2, 0.2, -0.2, -0.0, -0.1, 0.1, 0.2},
{0.0, -0.1, 0.1, 0.1, 0.0, 0.1, 0.0, 0.1, -0.0, 0.2, -0.1, -0.2, 0.0, -0.1, 0.0, 0.0, -0.2, -0.0, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1},
{-0.0, 0.2, -0.1, -0.1, 0.2, -0.0, -0.2, -0.1, -0.0, -0.2, -0.1, -0.1, 0.1, 0.0, 0.1, 0.2, 0.1, -0.1, 0.1, -0.0, -0.1, 0.2, -0.1, 0.0, -0.1},
{0.1, -0.1, 0.2, -0.0, -0.1, -0.1, -0.0, 0.0, 0.0, -0.2, 0.1, -0.2, -0.0, -0.0, -0.1, -0.2, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1},
{0.2, -0.1, -0.1, -0.1, 0.1, -0.1, -0.2, 0.1, 0.1, -0.2, -0.1, 0.0, -0.0, 0.1, -0.2, -0.2, -0.0, -0.1, 0.1, -0.1, -0.2, -0.1, 0.1, 0.1, -0.2},
{-0.2, 0.0, -0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.2, -0.0, -0.1, 0.1, 0.1, 0.2, -0.1, 0.2, 0.2, 0.1, 0.1, 0.0},
{-0.1, -0.1, 0.0, 0.2, 0.0, -0.1, -0.2, -0.1, 0.1, -0.1, -0.0, 0.2, -0.0, 0.2, -0.0, -0.0, 0.2, -0.0, 0.1, -0.1, -0.1, 0.0, 0.1, 0.0, 0.1},
{0.0, -0.1, -0.0, -0.0, -0.1, 0.0, 0.1, -0.0, 0.1, -0.2, 0.2, 0.0, 0.0, -0.0, -0.2, -0.0, 0.2, -0.0, 0.1, 0.2, -0.0, 0.2, 0.0, -0.1, 0.1},
{0.1, -0.2, 0.0, -0.1, 0.0, 0.0, -0.0, 0.2, -0.1, -0.0, 0.0, -0.1, 0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.0, -0.2, -0.1, 0.1, -0.2},
{0.1, -0.1, 0.1, -0.2, -0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.2, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2},
}
);
Matrix  transformer_layers_9_feed_forward_linear1_bias   (
{0.1, -0.1, 0.2, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.2, 0.0}
);
Matrix  transformer_layers_9_feed_forward_linear2_weight   (
{{-0.2, 0.2, 0.2, -0.1, 0.0, -0.0, -0.0, -0.2, -0.1, -0.2, 0.1, 0.0, -0.3, 0.2, -0.2},
{0.2, -0.2, -0.2, 0.1, 0.0, 0.2, -0.2, -0.2, -0.0, 0.0, -0.0, -0.3, -0.1, -0.1, 0.2},
{-0.1, 0.1, -0.2, 0.2, 0.2, 0.2, -0.0, 0.0, -0.0, 0.0, -0.0, 0.1, -0.1, -0.2, -0.3},
{0.0, 0.0, 0.0, -0.2, -0.1, -0.1, -0.1, 0.1, -0.0, -0.1, -0.1, -0.2, 0.0, -0.1, 0.2},
{-0.0, 0.2, -0.2, -0.1, 0.1, -0.2, -0.2, -0.2, -0.0, -0.2, 0.2, -0.2, 0.2, -0.2, -0.2},
{0.1, -0.2, 0.1, -0.2, 0.1, 0.2, -0.2, 0.2, -0.2, 0.2, 0.2, -0.0, -0.2, -0.1, 0.2},
{-0.0, -0.2, -0.3, 0.2, 0.1, 0.0, 0.2, 0.2, 0.2, 0.0, -0.2, -0.1, 0.0, 0.2, 0.2},
{0.2, -0.1, 0.1, -0.0, -0.1, -0.0, -0.1, -0.0, -0.1, -0.1, 0.2, 0.2, -0.1, 0.1, 0.2},
{0.1, 0.2, -0.1, 0.2, 0.1, -0.1, 0.1, -0.2, -0.2, 0.2, 0.0, -0.1, 0.0, -0.2, 0.0},
{-0.2, -0.2, 0.0, -0.2, 0.2, 0.2, -0.1, -0.0, 0.0, 0.2, -0.1, -0.2, 0.1, -0.2, 0.2},
{0.2, -0.0, 0.0, 0.1, -0.0, 0.2, 0.2, 0.2, -0.0, -0.1, -0.0, 0.2, -0.1, 0.2, -0.1},
{-0.1, -0.2, -0.0, -0.1, -0.2, 0.1, -0.2, -0.2, 0.2, -0.2, -0.1, 0.1, -0.1, 0.0, 0.1},
{0.1, -0.1, 0.0, -0.1, -0.0, -0.2, 0.0, 0.0, 0.2, 0.2, 0.0, 0.1, -0.0, -0.1, -0.2},
{-0.0, -0.1, 0.1, 0.1, -0.2, 0.3, -0.2, -0.0, -0.1, -0.2, 0.2, 0.2, 0.1, -0.2, 0.2},
{0.2, -0.2, 0.1, 0.0, -0.2, 0.1, -0.3, 0.2, -0.1, -0.2, 0.2, -0.2, -0.2, -0.1, 0.1},
{-0.2, -0.0, -0.2, -0.2, -0.2, 0.2, 0.1, -0.2, 0.2, -0.3, -0.0, -0.1, -0.2, 0.0, 0.1},
{-0.2, -0.0, 0.1, 0.2, 0.2, 0.0, 0.0, 0.1, 0.0, 0.0, 0.1, 0.1, -0.1, 0.2, 0.2},
{0.2, -0.3, 0.2, 0.0, -0.0, -0.1, 0.2, -0.0, 0.1, 0.1, 0.2, -0.1, -0.2, -0.2, -0.0},
{0.1, 0.1, -0.2, -0.0, 0.2, 0.0, -0.2, 0.0, -0.2, 0.1, 0.2, 0.2, -0.2, -0.1, 0.0},
{0.1, 0.1, -0.2, 0.1, -0.2, -0.0, 0.1, 0.1, 0.2, -0.1, -0.2, -0.2, -0.1, 0.3, 0.0},
{-0.0, 0.0, -0.0, 0.1, 0.1, -0.0, -0.2, 0.1, -0.0, -0.2, 0.2, 0.2, 0.0, -0.2, 0.2},
{0.1, 0.2, 0.0, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.2, 0.2, -0.1, 0.1},
{-0.1, 0.1, 0.1, 0.2, 0.2, -0.1, -0.2, 0.2, 0.1, 0.2, -0.2, 0.0, 0.1, -0.1, 0.0},
{-0.2, 0.2, -0.1, -0.1, 0.1, 0.2, -0.0, -0.0, -0.1, 0.2, -0.1, 0.2, 0.0, 0.2, 0.2},
{-0.2, -0.0, 0.2, -0.2, 0.1, 0.0, -0.1, 0.0, -0.1, -0.2, -0.1, -0.1, 0.1, 0.2, 0.1},
}
);
Matrix  transformer_layers_9_feed_forward_linear2_bias   (
{0.1, -0.2, 0.1, 0.2, 0.1, 0.1, 0.2, -0.1, -0.2, -0.2, 0.2, -0.2, 0.2, -0.1, 0.0, 0.1, -0.0, 0.2, 0.2, 0.1, 0.0, -0.1, -0.1, 0.0, -0.2}
);
Matrix  transformer_layers_9_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_9_feed_forward_ln1_layer_norm_bias   (
{0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0}
);
Matrix  transformer_layers_9_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_9_feed_forward_ln2_layer_norm_bias   (
{-0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_10_attention_query_weight   (
{{-0.2, -0.0, -0.2, -0.1, 0.1, -0.1, -0.0, 0.2, 0.1, 0.0, 0.1, -0.1, -0.1, 0.2, -0.2, -0.1, 0.2, 0.2, 0.1, -0.2, -0.2, -0.1, 0.1, 0.1, 0.0},
{0.1, 0.1, -0.1, 0.1, 0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.2, -0.1, -0.0, -0.1, 0.1, 0.0, -0.1, 0.2, 0.1, -0.1, 0.1, 0.1, -0.1},
{0.1, -0.2, 0.1, 0.1, -0.2, -0.0, -0.1, 0.0, 0.2, -0.0, 0.2, 0.1, 0.1, 0.0, -0.1, -0.1, -0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.2, 0.0, 0.1},
{0.0, 0.0, 0.2, 0.1, -0.1, -0.1, 0.0, -0.1, -0.0, 0.1, 0.2, -0.0, -0.0, -0.0, 0.1, 0.2, -0.1, 0.2, 0.2, 0.1, 0.0, -0.1, 0.1, -0.1, 0.1},
{0.2, -0.1, 0.0, 0.1, -0.2, 0.0, -0.0, 0.0, -0.2, -0.1, 0.1, 0.0, 0.0, -0.1, 0.1, -0.2, 0.1, 0.0, -0.2, 0.1, 0.1, -0.2, -0.0, -0.2, 0.1},
{0.1, 0.1, -0.2, -0.0, -0.1, -0.0, 0.0, 0.2, -0.1, 0.1, 0.2, -0.1, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.2, 0.1, -0.2, -0.0, 0.0},
{-0.2, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, -0.2, 0.1, -0.0, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, -0.0, -0.1, -0.1, -0.1},
{0.1, -0.1, 0.1, 0.1, -0.1, -0.2, -0.1, 0.2, 0.1, -0.2, 0.2, 0.1, 0.1, 0.1, 0.0, 0.2, -0.1, 0.1, 0.0, -0.1, -0.1, -0.2, 0.1, -0.0, 0.1},
{0.0, 0.1, 0.1, 0.1, -0.0, 0.1, -0.1, 0.1, -0.2, -0.2, -0.0, 0.1, 0.0, 0.1, 0.0, -0.1, 0.2, 0.1, 0.1, -0.1, 0.1, 0.1, 0.0, -0.2, 0.1},
{0.2, -0.1, -0.1, -0.1, -0.1, -0.2, -0.2, 0.1, -0.1, -0.0, -0.2, 0.1, -0.2, 0.0, 0.2, 0.0, 0.2, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.0},
{0.1, -0.2, 0.1, 0.1, 0.2, -0.1, 0.0, 0.1, 0.1, -0.2, 0.1, 0.1, 0.0, 0.1, 0.2, -0.0, 0.0, 0.0, -0.0, -0.0, 0.1, 0.1, -0.1, 0.1, 0.0},
{0.0, -0.0, -0.2, -0.1, -0.1, -0.1, -0.1, 0.1, -0.2, 0.2, -0.1, -0.2, -0.0, 0.2, -0.2, -0.1, -0.2, 0.2, 0.1, -0.1, 0.1, -0.1, 0.1, 0.2, -0.2},
{-0.0, 0.1, 0.1, 0.1, -0.0, -0.0, -0.1, 0.1, -0.2, 0.1, -0.1, 0.1, -0.0, 0.1, -0.1, -0.0, -0.1, 0.1, 0.0, -0.0, -0.0, -0.1, 0.2, 0.1, -0.1},
{0.2, -0.0, 0.1, 0.0, 0.1, 0.0, 0.1, -0.0, -0.1, 0.1, -0.0, -0.1, -0.0, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.2, -0.0, 0.1, -0.2, -0.0, -0.1},
{0.0, 0.0, -0.1, 0.0, -0.2, 0.2, 0.1, 0.2, 0.0, -0.2, -0.1, 0.2, 0.0, 0.0, 0.1, -0.2, -0.2, 0.1, 0.1, 0.0, -0.1, -0.0, -0.0, -0.1, 0.1},
{0.1, -0.1, 0.0, -0.0, -0.2, 0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.2, -0.1, 0.2, 0.1, -0.1, -0.2, 0.2, 0.0, -0.2, 0.1, -0.2, 0.2, 0.0, 0.1},
{0.1, -0.0, 0.1, -0.1, 0.1, 0.1, -0.0, -0.1, 0.0, -0.1, 0.2, 0.1, -0.0, -0.2, -0.1, -0.1, 0.1, 0.1, 0.0, 0.0, -0.2, -0.1, -0.1, 0.1, -0.1},
{-0.0, 0.1, 0.0, 0.0, 0.1, 0.2, -0.0, -0.0, 0.2, -0.2, -0.0, 0.1, -0.1, -0.1, 0.0, -0.0, 0.2, -0.0, -0.2, 0.2, -0.1, -0.0, -0.2, -0.2, -0.1},
{-0.0, 0.0, 0.2, 0.2, 0.0, 0.0, -0.2, -0.1, -0.1, -0.2, -0.2, 0.0, 0.2, 0.0, 0.1, -0.1, -0.1, 0.2, -0.1, -0.0, -0.1, -0.2, -0.1, -0.1, 0.0},
{-0.2, 0.1, 0.2, -0.1, -0.2, 0.1, 0.2, -0.0, -0.0, 0.1, 0.1, -0.1, -0.0, 0.0, 0.1, 0.0, -0.2, 0.1, -0.1, 0.2, -0.1, 0.1, 0.1, 0.2, -0.1},
{0.2, 0.0, 0.2, 0.1, 0.0, 0.1, 0.1, -0.1, -0.1, -0.2, 0.2, 0.1, 0.2, 0.1, 0.2, 0.0, -0.2, -0.2, -0.1, -0.1, 0.0, -0.1, -0.1, -0.0, -0.0},
{-0.1, 0.1, -0.1, 0.0, 0.1, 0.0, 0.1, 0.1, 0.1, -0.0, 0.1, -0.1, 0.1, -0.2, -0.0, -0.1, 0.1, 0.2, 0.0, -0.1, -0.1, -0.2, 0.0, -0.2, 0.1},
{0.0, 0.0, -0.0, 0.1, 0.2, 0.0, 0.2, 0.2, 0.1, -0.1, 0.1, -0.2, -0.0, 0.1, -0.0, -0.1, -0.1, 0.1, 0.1, 0.1, -0.1, -0.2, -0.2, -0.1, 0.1},
{0.2, 0.0, -0.1, -0.2, -0.2, -0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, -0.2, -0.1, 0.1, -0.0, 0.2, -0.1, -0.1, -0.2, 0.1, -0.2, 0.1, 0.1, 0.1},
{-0.0, 0.0, 0.2, 0.2, 0.1, 0.0, 0.2, -0.2, 0.2, 0.2, -0.2, -0.2, 0.1, -0.1, -0.1, 0.0, -0.1, 0.1, -0.1, -0.1, 0.0, -0.2, 0.1, -0.1, 0.0},
}
);
Matrix  transformer_layers_10_attention_query_bias   (
{0.0, 0.0, -0.2, 0.1, -0.1, 0.2, -0.1, -0.2, 0.1, 0.2, -0.0, 0.0, 0.1, -0.2, -0.0, -0.0, -0.1, 0.1, 0.0, 0.2, 0.1, -0.0, -0.1, -0.2, 0.1}
);
Matrix  transformer_layers_10_attention_key_weight   (
{{-0.1, 0.0, -0.1, -0.0, 0.0, 0.2, 0.1, 0.1, 0.2, -0.2, -0.1, 0.1, 0.1, 0.2, -0.1, 0.1, 0.2, -0.1, -0.0, -0.1, 0.0, -0.2, -0.0, 0.0, 0.0},
{-0.1, -0.2, 0.1, 0.2, -0.0, -0.1, -0.2, -0.1, 0.0, 0.2, -0.2, 0.0, 0.1, -0.0, -0.2, -0.1, -0.1, -0.0, -0.1, -0.2, -0.2, -0.1, -0.1, -0.1, -0.2},
{0.1, -0.1, -0.1, -0.0, 0.1, -0.1, -0.1, 0.1, 0.2, -0.1, -0.0, 0.1, -0.1, 0.2, 0.2, -0.1, -0.2, 0.1, 0.2, -0.1, 0.2, -0.2, 0.0, 0.1, -0.1},
{0.0, -0.1, -0.0, 0.2, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.0, 0.1, 0.1, 0.1, 0.2, -0.2, -0.0, 0.1, -0.1, 0.1, -0.2, -0.1, 0.1, -0.0},
{-0.1, -0.1, 0.1, -0.0, -0.1, -0.1, 0.1, -0.2, 0.1, 0.0, -0.0, 0.1, -0.0, -0.1, 0.2, -0.2, -0.0, 0.1, 0.1, -0.1, -0.2, -0.1, 0.0, 0.2, 0.2},
{-0.1, 0.1, -0.1, -0.0, 0.0, 0.2, 0.0, 0.2, -0.1, -0.1, -0.1, -0.0, 0.1, -0.2, 0.1, -0.2, -0.2, 0.1, 0.2, 0.2, -0.2, 0.1, 0.0, 0.2, 0.0},
{0.1, 0.1, 0.1, -0.1, -0.2, 0.1, -0.1, 0.2, -0.1, 0.1, 0.2, -0.1, 0.2, 0.2, -0.2, -0.1, 0.1, -0.0, 0.2, -0.1, -0.2, 0.0, 0.2, 0.1, -0.2},
{-0.1, 0.1, -0.0, -0.1, 0.1, 0.1, 0.1, -0.2, -0.1, -0.2, 0.2, -0.1, -0.1, -0.2, 0.2, 0.1, -0.2, -0.0, -0.1, 0.1, 0.1, -0.0, -0.2, -0.0, 0.1},
{-0.0, -0.2, 0.2, 0.0, -0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, -0.2, -0.0, -0.0, -0.1, 0.1, 0.1, 0.2, -0.0, 0.1, -0.2, 0.0, -0.1, 0.2, 0.1},
{-0.2, -0.1, 0.0, -0.1, 0.2, 0.1, -0.2, 0.0, -0.1, 0.0, -0.0, 0.1, 0.2, 0.0, 0.1, -0.0, 0.2, -0.1, 0.1, -0.1, -0.1, 0.0, -0.0, 0.2, -0.1},
{-0.0, -0.1, -0.2, 0.2, 0.0, 0.1, -0.1, -0.1, 0.2, 0.0, -0.2, 0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.2, 0.1, 0.2, 0.1, -0.1, 0.1, -0.1, -0.2},
{0.0, 0.0, 0.1, 0.2, -0.0, -0.1, -0.1, 0.1, -0.2, -0.1, -0.2, -0.2, -0.2, 0.0, -0.2, 0.0, 0.1, 0.2, 0.1, -0.1, 0.2, -0.0, 0.2, 0.2, 0.1},
{-0.1, 0.0, 0.1, -0.1, 0.0, 0.1, 0.0, -0.1, 0.2, -0.1, -0.0, -0.0, 0.2, 0.2, -0.0, -0.1, -0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, -0.0, -0.1},
{0.0, -0.1, 0.0, 0.0, -0.1, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.1, 0.1, -0.0, 0.0, -0.1, -0.0, -0.2, 0.0, 0.0, -0.2, 0.2, 0.0, 0.1},
{-0.1, 0.0, -0.0, -0.1, -0.0, -0.1, 0.1, 0.2, -0.0, 0.0, 0.2, 0.2, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1},
{-0.0, 0.1, -0.0, -0.1, 0.0, -0.1, -0.0, -0.1, -0.2, 0.2, -0.0, -0.2, -0.1, -0.1, -0.2, -0.1, 0.0, -0.2, 0.2, 0.1, 0.1, -0.1, -0.1, -0.2, 0.0},
{0.0, 0.1, 0.1, 0.1, -0.2, 0.1, -0.1, 0.0, -0.2, -0.0, 0.1, 0.1, 0.1, -0.2, -0.0, 0.0, -0.0, -0.0, 0.0, -0.1, -0.0, 0.0, -0.2, 0.1, -0.0},
{0.1, -0.0, 0.1, -0.1, 0.2, 0.2, -0.2, -0.2, 0.1, -0.2, 0.2, -0.2, 0.1, -0.2, -0.2, 0.0, 0.1, 0.0, 0.2, 0.2, -0.0, -0.2, -0.1, -0.2, -0.0},
{-0.2, -0.2, -0.0, -0.1, 0.2, 0.2, 0.0, -0.0, -0.1, 0.2, 0.2, 0.0, 0.1, 0.0, 0.1, 0.0, -0.1, -0.1, -0.1, 0.1, 0.0, 0.0, -0.1, 0.1, -0.0},
{-0.1, -0.2, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.0, -0.2, -0.0, 0.2, -0.2, -0.1, 0.0, -0.1, -0.1, 0.2, -0.1, -0.1, -0.1, -0.1, -0.0},
{0.0, -0.1, -0.1, 0.1, -0.1, -0.0, -0.2, -0.1, 0.1, 0.1, 0.1, -0.1, -0.0, -0.1, 0.0, 0.1, 0.2, -0.1, 0.0, 0.1, 0.1, 0.1, 0.2, 0.2, -0.0},
{0.1, 0.1, -0.0, -0.1, 0.1, 0.2, -0.1, -0.2, -0.1, 0.1, 0.0, 0.0, 0.2, -0.1, -0.1, -0.2, 0.1, 0.1, -0.0, 0.0, 0.1, 0.2, 0.1, -0.1, 0.0},
{-0.1, -0.0, -0.2, 0.1, -0.2, 0.0, -0.1, 0.1, 0.2, -0.2, -0.0, 0.1, -0.1, -0.0, 0.0, 0.2, 0.2, -0.0, -0.1, -0.1, -0.1, -0.0, -0.1, 0.1, -0.1},
{0.1, 0.2, 0.0, 0.2, 0.2, 0.1, -0.1, 0.0, -0.1, 0.1, -0.1, 0.2, -0.1, 0.0, -0.1, 0.2, -0.1, -0.1, 0.2, 0.1, 0.0, -0.2, 0.2, -0.0, -0.1},
{0.1, -0.2, 0.1, 0.1, -0.2, -0.1, -0.1, -0.0, -0.0, -0.1, -0.1, -0.2, 0.1, -0.1, 0.1, 0.1, -0.0, 0.2, -0.2, 0.1, -0.2, 0.0, -0.1, -0.1, -0.1},
}
);
Matrix  transformer_layers_10_attention_key_bias   (
{-0.1, -0.0, -0.0, 0.1, 0.1, -0.2, -0.0, 0.1, -0.0, -0.0, 0.0, -0.0, 0.1, -0.1, 0.2, -0.1, -0.1, 0.2, 0.1, -0.2, 0.1, 0.1, 0.2, 0.0, -0.1}
);
Matrix  transformer_layers_10_attention_value_weight   (
{{-0.2, 0.2, 0.1, 0.0, 0.0, -0.0, -0.1, -0.1, -0.1, 0.1, 0.1, 0.0, 0.1, -0.0, 0.1, 0.1, -0.1, 0.2, -0.1, 0.1, -0.1, -0.0, -0.2, -0.2, -0.0},
{0.0, 0.1, 0.2, -0.2, -0.1, -0.1, -0.2, 0.1, 0.2, -0.0, 0.1, 0.2, 0.2, 0.1, -0.1, 0.1, -0.2, 0.1, -0.1, -0.1, 0.1, -0.1, -0.0, -0.2, -0.2},
{-0.1, -0.2, -0.1, 0.0, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.2, -0.0, 0.1, 0.1, 0.2, -0.2, 0.2, -0.1, -0.2, -0.1, 0.2, -0.2, 0.0, -0.1},
{-0.2, -0.0, -0.2, 0.1, 0.2, 0.1, 0.2, -0.1, -0.1, 0.2, -0.1, -0.1, -0.2, 0.0, -0.1, 0.1, -0.1, 0.2, 0.0, -0.0, 0.0, 0.0, 0.1, -0.1, 0.1},
{0.1, -0.1, 0.2, -0.2, 0.1, 0.1, -0.2, -0.1, 0.1, 0.1, 0.1, -0.2, -0.1, -0.1, 0.1, 0.0, 0.2, -0.2, 0.0, 0.1, -0.1, -0.0, 0.2, 0.2, 0.2},
{0.1, 0.1, 0.2, 0.1, -0.0, 0.2, 0.1, 0.1, -0.1, 0.0, 0.0, 0.1, 0.0, -0.1, 0.0, 0.0, 0.1, 0.2, -0.2, -0.1, 0.1, 0.1, 0.2, -0.0, -0.0},
{0.1, 0.1, 0.0, -0.1, 0.1, 0.1, -0.2, -0.2, -0.0, -0.1, 0.2, -0.0, 0.2, 0.0, 0.0, -0.1, 0.2, 0.1, 0.0, 0.1, -0.2, 0.2, 0.0, 0.0, -0.2},
{0.1, 0.1, 0.2, -0.0, 0.2, -0.1, -0.1, 0.1, 0.0, -0.0, -0.0, 0.2, -0.0, -0.2, 0.1, -0.0, -0.2, -0.1, -0.0, -0.1, -0.0, -0.2, -0.1, 0.1, -0.2},
{0.2, 0.0, -0.1, -0.2, -0.2, -0.2, 0.1, 0.1, -0.1, 0.2, 0.0, -0.1, 0.1, 0.0, 0.1, 0.0, 0.1, -0.0, 0.2, -0.1, -0.0, -0.2, -0.2, 0.1, 0.1},
{0.0, 0.1, -0.2, 0.2, -0.1, -0.1, -0.0, -0.2, 0.1, -0.1, 0.1, -0.2, -0.1, 0.1, 0.1, -0.0, 0.1, -0.1, -0.1, -0.0, -0.0, -0.0, 0.1, 0.2, -0.1},
{0.2, 0.2, -0.1, -0.0, -0.0, -0.0, 0.0, 0.2, -0.2, 0.0, -0.2, 0.1, 0.0, 0.2, -0.0, -0.1, 0.1, -0.1, 0.1, 0.0, -0.2, 0.1, -0.1, -0.2, -0.2},
{-0.2, -0.2, -0.1, -0.2, -0.0, 0.2, -0.1, 0.2, 0.1, -0.0, 0.1, 0.2, -0.1, -0.2, 0.2, -0.0, 0.1, 0.1, -0.0, 0.1, -0.0, 0.2, 0.1, 0.1, 0.1},
{0.1, -0.2, 0.0, -0.0, 0.2, -0.0, -0.0, -0.1, 0.1, -0.2, -0.0, 0.0, -0.1, -0.0, 0.1, 0.2, -0.1, -0.1, -0.2, 0.1, 0.2, -0.2, 0.0, -0.0, -0.0},
{0.1, 0.1, 0.1, 0.1, -0.1, 0.0, 0.2, -0.0, 0.1, 0.1, 0.2, -0.0, -0.2, 0.0, 0.1, 0.2, 0.1, -0.1, -0.1, 0.1, -0.1, 0.0, 0.1, -0.1, -0.2},
{-0.2, 0.1, 0.0, -0.0, -0.1, 0.1, 0.0, 0.2, 0.0, -0.1, 0.0, -0.1, -0.1, 0.1, 0.1, 0.1, 0.0, 0.2, -0.1, 0.2, 0.1, -0.1, -0.1, -0.1, 0.0},
{-0.2, 0.1, 0.2, -0.1, 0.2, -0.2, 0.2, 0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.1, -0.2, -0.1, 0.1, 0.1, -0.0, 0.1, 0.1, -0.1, 0.0, 0.0, -0.0},
{-0.2, -0.2, -0.2, -0.2, -0.0, 0.2, -0.2, -0.0, 0.2, 0.1, 0.1, -0.1, -0.0, 0.2, -0.1, 0.2, 0.1, -0.0, -0.2, 0.1, 0.1, 0.0, 0.0, 0.1, 0.2},
{-0.2, -0.2, 0.2, -0.1, -0.0, 0.0, 0.0, 0.1, 0.1, 0.1, -0.1, -0.0, -0.1, -0.1, 0.2, -0.2, -0.1, -0.2, 0.2, -0.1, -0.1, 0.1, -0.1, -0.2, -0.1},
{0.2, -0.1, -0.1, 0.0, 0.2, -0.0, 0.1, -0.2, -0.2, 0.2, 0.0, 0.2, -0.1, -0.0, 0.2, -0.0, -0.2, 0.0, 0.2, -0.1, -0.0, -0.2, 0.2, 0.1, 0.0},
{0.1, 0.0, -0.1, 0.1, -0.0, -0.2, 0.0, 0.0, -0.1, 0.0, 0.1, -0.0, -0.1, -0.2, 0.1, 0.1, -0.2, -0.1, 0.2, -0.1, 0.1, -0.1, -0.1, 0.0, 0.1},
{-0.2, -0.0, -0.2, -0.1, 0.2, 0.0, -0.1, 0.1, -0.0, 0.0, 0.2, 0.1, -0.0, 0.1, 0.2, 0.2, 0.1, -0.2, -0.0, -0.0, 0.1, 0.1, -0.1, 0.1, 0.2},
{-0.1, 0.1, 0.1, 0.0, 0.1, -0.2, -0.0, 0.1, -0.1, 0.1, -0.2, 0.1, -0.1, -0.1, 0.2, -0.1, -0.1, 0.0, 0.1, -0.2, -0.1, 0.2, -0.1, 0.1, -0.1},
{0.2, 0.2, -0.1, 0.0, -0.1, -0.2, -0.0, 0.0, -0.1, 0.2, -0.1, -0.1, -0.1, -0.1, 0.0, 0.2, -0.2, 0.1, -0.1, 0.0, 0.2, -0.1, -0.1, 0.0, -0.1},
{0.1, -0.0, 0.0, 0.2, -0.1, -0.2, -0.1, 0.1, 0.2, -0.1, -0.0, -0.1, -0.2, 0.1, 0.1, 0.1, -0.1, -0.0, 0.1, -0.2, -0.2, 0.1, -0.2, -0.0, 0.2},
{-0.0, -0.1, 0.1, 0.1, 0.0, 0.2, 0.1, 0.2, -0.1, 0.0, -0.1, -0.0, 0.1, 0.2, 0.1, -0.1, 0.0, 0.0, 0.2, 0.2, 0.1, 0.2, -0.1, -0.2, 0.0},
}
);
Matrix  transformer_layers_10_attention_value_bias   (
{0.1, -0.0, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.2, -0.1, 0.1, -0.1, -0.1, 0.2, -0.1, -0.2, 0.2, -0.0, 0.1, 0.1, 0.2, -0.1, -0.1, 0.1, -0.1}
);
Matrix  transformer_layers_10_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_10_norm1_layer_norm_bias   (
{-0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0}
);
Matrix  transformer_layers_10_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_10_norm2_layer_norm_bias   (
{-0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_10_feed_forward_linear1_weight   (
{{0.0, 0.1, 0.2, 0.1, 0.2, 0.2, -0.1, 0.1, -0.1, -0.2, -0.0, -0.0, -0.2, -0.0, 0.1, -0.2, 0.1, 0.0, -0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.2},
{0.1, 0.1, 0.2, -0.2, 0.1, -0.1, 0.2, 0.2, -0.1, -0.1, 0.1, 0.1, -0.0, -0.1, -0.0, -0.2, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.0, 0.1},
{0.2, -0.2, -0.1, -0.1, -0.2, -0.2, -0.2, -0.1, -0.0, 0.0, 0.1, -0.0, -0.1, -0.1, -0.0, -0.0, -0.2, -0.1, 0.1, -0.0, 0.0, -0.1, -0.1, -0.1, -0.2},
{-0.2, -0.2, -0.1, 0.0, 0.1, -0.1, -0.0, 0.1, -0.0, 0.1, -0.1, -0.1, -0.0, -0.1, 0.2, -0.2, 0.1, -0.1, 0.1, -0.0, -0.2, -0.2, -0.1, -0.1, 0.0},
{0.1, 0.0, 0.0, -0.2, 0.1, -0.2, -0.0, -0.1, -0.1, -0.2, -0.2, -0.2, -0.1, -0.1, 0.2, -0.1, 0.2, 0.2, 0.1, 0.1, 0.1, -0.1, -0.0, -0.1, 0.0},
{0.0, -0.0, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1, 0.0, 0.1, -0.1, -0.1, -0.2, 0.1, -0.2, -0.1, 0.1, 0.2, -0.1, -0.0, 0.1, -0.0, 0.2},
{0.2, 0.1, 0.1, -0.2, 0.2, -0.0, 0.1, -0.2, 0.1, -0.2, -0.1, -0.2, -0.0, 0.2, 0.1, 0.0, -0.0, -0.1, 0.0, -0.2, -0.0, -0.0, 0.2, 0.0, -0.1},
{-0.2, 0.0, -0.1, -0.2, -0.0, -0.1, 0.1, -0.1, -0.1, -0.0, -0.1, -0.1, -0.1, -0.1, -0.2, -0.0, 0.1, 0.1, 0.1, -0.1, 0.1, 0.0, 0.1, -0.0, 0.0},
{-0.1, 0.1, 0.2, -0.1, 0.0, 0.1, 0.2, 0.1, 0.0, -0.0, 0.2, -0.0, 0.1, -0.0, 0.2, -0.1, -0.2, 0.0, 0.1, -0.1, -0.2, -0.1, 0.2, -0.1, -0.1},
{0.2, -0.2, -0.0, -0.1, -0.1, -0.0, -0.2, -0.2, 0.1, 0.0, -0.1, 0.1, -0.2, -0.0, -0.1, -0.2, -0.1, 0.1, -0.1, -0.2, -0.1, -0.2, 0.0, 0.1, 0.2},
{0.1, -0.1, -0.1, -0.2, 0.0, 0.2, 0.1, 0.0, 0.0, -0.2, -0.2, -0.2, 0.0, 0.2, 0.2, 0.0, -0.1, 0.0, -0.0, 0.0, 0.2, -0.2, -0.1, 0.1, -0.1},
{-0.1, -0.0, -0.1, -0.1, 0.1, -0.0, -0.2, 0.1, 0.1, -0.1, 0.1, 0.2, 0.1, 0.2, 0.1, 0.0, -0.0, -0.1, -0.2, 0.1, 0.2, 0.1, -0.0, 0.0, -0.1},
{-0.1, -0.1, -0.1, 0.2, -0.1, -0.0, 0.2, -0.1, -0.1, -0.0, -0.1, 0.2, -0.2, -0.0, 0.2, -0.0, -0.2, -0.1, -0.1, 0.1, 0.2, -0.0, 0.0, -0.0, 0.2},
{-0.0, 0.1, 0.2, 0.2, -0.1, 0.0, 0.1, 0.1, 0.0, -0.2, 0.0, 0.1, -0.1, -0.1, 0.0, 0.2, 0.1, -0.2, -0.1, 0.2, 0.1, 0.2, -0.1, -0.1, 0.1},
{-0.1, 0.1, -0.0, -0.1, -0.1, -0.0, -0.1, -0.2, 0.0, 0.1, 0.0, 0.0, 0.1, -0.2, 0.0, 0.1, -0.0, -0.0, 0.1, -0.2, -0.2, 0.2, -0.1, -0.1, -0.2},
}
);
Matrix  transformer_layers_10_feed_forward_linear1_bias   (
{-0.1, 0.1, 0.2, -0.2, -0.0, 0.1, -0.1, 0.1, -0.1, -0.2, -0.2, 0.1, 0.1, -0.1, -0.1}
);
Matrix  transformer_layers_10_feed_forward_linear2_weight   (
{{-0.2, 0.2, -0.2, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.0, 0.2, 0.0, -0.0, 0.1},
{-0.1, 0.2, -0.2, -0.0, 0.0, -0.1, -0.1, 0.2, 0.1, -0.2, 0.1, -0.2, -0.0, -0.2, 0.1},
{-0.2, 0.2, -0.3, -0.2, 0.0, -0.2, 0.1, -0.2, -0.2, 0.2, 0.0, 0.1, 0.0, -0.2, 0.2},
{0.1, 0.2, -0.1, 0.0, 0.2, 0.2, 0.1, 0.2, 0.1, -0.2, 0.1, -0.0, 0.1, -0.0, 0.2},
{0.0, 0.2, 0.0, -0.2, -0.2, -0.2, -0.1, 0.2, 0.1, 0.1, 0.0, 0.2, 0.1, 0.1, 0.0},
{0.0, 0.2, -0.2, 0.2, -0.0, 0.1, 0.2, -0.0, 0.2, -0.2, -0.2, 0.0, -0.2, -0.0, 0.0},
{-0.2, 0.0, 0.2, -0.1, -0.1, 0.0, -0.1, -0.2, 0.1, -0.2, 0.3, 0.1, -0.0, -0.1, -0.1},
{0.1, -0.2, 0.1, 0.2, -0.1, -0.1, -0.1, 0.1, -0.0, -0.2, 0.2, 0.2, 0.1, -0.1, 0.1},
{-0.2, -0.3, -0.2, -0.1, -0.1, -0.1, 0.1, -0.1, 0.3, -0.2, 0.0, 0.2, 0.2, 0.2, 0.2},
{0.1, -0.1, -0.0, 0.2, 0.1, 0.2, 0.2, -0.2, -0.3, 0.2, -0.1, 0.2, -0.1, 0.2, 0.2},
{0.1, 0.2, 0.1, 0.1, 0.0, -0.2, 0.1, 0.1, -0.0, -0.2, -0.2, 0.2, 0.1, 0.2, -0.0},
{0.1, 0.0, 0.0, 0.0, 0.3, -0.1, -0.2, 0.1, -0.2, 0.1, 0.1, 0.2, 0.1, -0.1, 0.2},
{-0.1, 0.1, 0.1, 0.1, -0.2, -0.1, -0.2, 0.1, -0.2, -0.1, 0.1, -0.0, -0.0, 0.1, -0.2},
{0.2, -0.1, 0.1, -0.1, -0.2, 0.1, -0.2, -0.1, 0.1, -0.0, 0.2, -0.1, -0.1, 0.1, 0.2},
{0.2, 0.2, -0.2, 0.2, 0.0, -0.1, 0.3, -0.2, 0.2, -0.2, 0.3, 0.2, -0.0, -0.1, 0.2},
{-0.2, 0.2, 0.2, -0.0, -0.1, 0.0, -0.1, -0.2, -0.1, -0.2, 0.2, -0.0, -0.0, 0.1, -0.2},
{0.1, -0.2, -0.2, 0.2, 0.1, -0.2, 0.0, 0.2, 0.1, -0.1, 0.2, 0.2, -0.0, -0.2, 0.2},
{0.1, -0.1, 0.2, -0.3, -0.1, -0.2, 0.3, -0.2, 0.1, -0.2, -0.2, 0.1, -0.0, 0.1, 0.1},
{-0.1, -0.3, -0.2, 0.2, -0.2, -0.0, 0.1, -0.0, -0.1, 0.2, -0.1, -0.2, 0.2, -0.1, 0.2},
{-0.2, -0.1, 0.2, -0.1, 0.2, 0.2, 0.0, 0.1, 0.0, -0.2, 0.1, -0.1, 0.1, -0.3, -0.2},
{0.2, -0.0, -0.0, -0.2, -0.1, -0.1, -0.1, 0.0, 0.1, -0.2, 0.1, -0.2, 0.3, 0.2, -0.0},
{0.0, -0.2, -0.2, 0.2, -0.2, 0.0, 0.1, -0.2, -0.1, 0.2, 0.1, -0.1, -0.1, -0.2, -0.2},
{-0.2, 0.1, 0.0, -0.2, 0.0, 0.2, -0.1, -0.1, 0.1, 0.1, 0.2, 0.1, -0.1, 0.2, -0.0},
{0.1, -0.1, 0.1, 0.2, 0.2, -0.1, -0.1, 0.2, 0.0, -0.1, 0.2, -0.1, -0.2, 0.0, 0.2},
{0.2, 0.1, 0.1, -0.2, -0.0, 0.1, -0.3, -0.0, -0.1, 0.1, 0.2, -0.2, 0.2, -0.2, -0.2},
}
);
Matrix  transformer_layers_10_feed_forward_linear2_bias   (
{0.0, -0.2, -0.2, 0.2, -0.2, -0.1, 0.0, 0.1, -0.1, -0.1, 0.2, -0.2, -0.1, -0.2, -0.1, -0.1, 0.0, 0.2, -0.1, 0.2, -0.0, -0.2, 0.2, 0.1, -0.2}
);
Matrix  transformer_layers_10_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_10_feed_forward_ln1_layer_norm_bias   (
{0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0}
);
Matrix  transformer_layers_10_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_10_feed_forward_ln2_layer_norm_bias   (
{-0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_11_attention_query_weight   (
{{0.1, 0.1, -0.1, -0.1, -0.2, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.2, 0.2, -0.2, -0.0, -0.0, 0.0, -0.1, -0.1, -0.1, -0.0, 0.2},
{-0.2, -0.1, -0.2, -0.0, 0.1, 0.2, -0.1, 0.2, -0.0, 0.0, 0.0, -0.1, 0.2, -0.1, -0.1, -0.2, 0.1, 0.1, 0.0, -0.1, 0.1, 0.2, -0.2, -0.1, 0.1},
{-0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, -0.1, -0.0, -0.1, -0.1, 0.2, -0.0, 0.1, 0.1, 0.0, -0.0, 0.2, 0.2, 0.2, -0.1, -0.1, -0.1, 0.2},
{0.0, -0.1, -0.1, 0.1, -0.1, -0.0, 0.0, -0.2, 0.0, -0.2, 0.0, -0.1, -0.1, 0.1, 0.0, -0.0, -0.2, 0.2, 0.1, 0.1, -0.1, 0.1, 0.2, -0.2, 0.1},
{0.1, 0.1, 0.2, 0.1, -0.1, -0.2, -0.1, 0.1, 0.1, 0.1, 0.0, -0.1, 0.0, 0.1, -0.2, 0.0, 0.0, 0.1, -0.2, -0.1, -0.1, -0.0, 0.2, 0.2, -0.0},
{-0.0, -0.1, -0.2, -0.1, -0.0, -0.2, -0.1, 0.1, 0.1, 0.1, -0.0, -0.1, 0.0, -0.2, -0.0, -0.2, -0.2, -0.1, 0.0, -0.1, -0.2, 0.1, -0.1, -0.2, 0.1},
{0.1, -0.1, 0.0, -0.0, 0.1, -0.0, -0.2, -0.2, -0.2, 0.1, 0.0, 0.0, -0.1, 0.1, 0.1, -0.2, -0.2, -0.0, 0.1, -0.2, -0.0, 0.2, -0.1, 0.2, 0.1},
{-0.2, -0.1, 0.0, 0.0, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.0, 0.2, -0.2, 0.2, -0.1, 0.1, -0.0, 0.2, -0.0, -0.0, -0.1, -0.1, -0.0, 0.1},
{0.0, 0.1, 0.1, -0.1, -0.2, 0.1, -0.0, 0.1, -0.2, 0.2, -0.2, 0.0, -0.0, -0.0, 0.1, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, -0.1, -0.0, 0.2, 0.1},
{0.1, 0.0, -0.1, 0.1, 0.2, 0.0, 0.1, -0.0, -0.2, -0.0, -0.0, 0.0, 0.1, -0.1, -0.0, -0.1, 0.1, -0.0, -0.1, -0.2, -0.1, 0.1, -0.1, -0.1, -0.1},
{-0.1, -0.1, -0.0, 0.1, 0.0, 0.2, 0.1, -0.2, -0.0, -0.0, -0.2, -0.2, 0.1, 0.1, -0.0, -0.1, 0.2, -0.1, -0.2, 0.1, -0.1, 0.1, 0.2, -0.1, 0.2},
{0.2, -0.1, -0.2, 0.1, -0.2, -0.0, 0.1, 0.1, -0.0, 0.1, 0.1, -0.0, -0.2, -0.1, 0.1, -0.0, -0.2, 0.1, 0.2, -0.0, 0.0, -0.0, -0.1, -0.2, -0.1},
{-0.1, 0.1, 0.2, -0.0, -0.0, 0.1, 0.2, -0.0, -0.1, -0.1, 0.1, -0.0, 0.2, 0.0, -0.0, 0.1, -0.1, 0.2, 0.1, -0.1, -0.1, 0.1, 0.1, -0.0, -0.2},
{0.0, -0.2, 0.1, 0.1, 0.2, 0.0, -0.0, -0.1, -0.1, 0.2, -0.1, -0.0, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, -0.0, 0.2, -0.2, 0.1, 0.1, -0.0, 0.1},
{-0.2, 0.0, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.0, 0.0, 0.2, -0.1, -0.0, 0.2, 0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.0, 0.2, 0.1, 0.1, 0.0},
{-0.1, 0.2, 0.0, -0.0, -0.1, 0.2, 0.1, 0.1, -0.0, -0.2, 0.1, -0.2, 0.1, 0.1, 0.0, -0.1, -0.1, 0.1, -0.2, -0.1, -0.2, -0.1, 0.2, 0.1, 0.1},
{0.2, 0.2, -0.0, -0.2, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.2, 0.2, -0.1, 0.1, -0.0, -0.1, -0.2, 0.2, -0.1, 0.1, 0.1, 0.0, 0.1, -0.1, -0.2},
{-0.0, 0.0, -0.2, 0.0, -0.0, -0.1, 0.1, -0.1, 0.0, -0.1, 0.1, 0.0, -0.1, 0.2, 0.0, -0.2, 0.1, 0.1, -0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.1},
{-0.0, -0.1, 0.1, -0.1, -0.2, -0.1, -0.2, -0.2, 0.1, 0.1, 0.2, -0.2, -0.0, -0.1, -0.2, -0.1, 0.1, -0.2, 0.1, 0.1, -0.1, -0.2, -0.2, -0.0, -0.1},
{0.1, -0.0, -0.2, -0.2, 0.1, -0.0, -0.0, -0.2, -0.2, 0.2, -0.1, -0.1, -0.1, 0.2, 0.1, 0.2, 0.0, 0.2, -0.1, 0.1, 0.2, 0.2, 0.1, 0.1, 0.2},
{-0.1, -0.0, 0.2, 0.1, 0.1, -0.1, 0.2, 0.1, -0.0, -0.1, -0.1, 0.1, -0.1, 0.1, -0.0, 0.1, -0.1, -0.1, -0.1, -0.1, -0.0, 0.2, 0.2, 0.1, 0.2},
{-0.0, -0.0, 0.1, -0.1, -0.1, -0.0, -0.0, -0.1, -0.0, -0.0, 0.2, -0.2, -0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.2, -0.2, 0.1, -0.2, -0.1, 0.2},
{-0.1, 0.1, -0.0, 0.0, 0.2, 0.1, 0.0, 0.1, -0.0, 0.1, -0.1, -0.1, 0.0, 0.2, -0.2, -0.0, -0.1, -0.1, 0.0, -0.1, 0.1, -0.2, -0.2, -0.0, -0.2},
{-0.1, -0.0, -0.1, -0.1, -0.1, 0.0, 0.2, -0.1, -0.1, 0.0, 0.1, 0.1, -0.1, -0.1, -0.1, -0.2, -0.1, 0.2, -0.1, 0.0, 0.1, -0.2, 0.0, -0.2, 0.0},
{0.0, 0.0, -0.2, -0.1, -0.1, 0.2, 0.1, 0.1, -0.1, -0.2, 0.0, 0.1, -0.1, -0.2, -0.1, 0.1, -0.2, 0.2, -0.2, 0.2, -0.1, -0.0, 0.2, 0.2, 0.2},
}
);
Matrix  transformer_layers_11_attention_query_bias   (
{0.0, 0.1, -0.1, -0.1, -0.2, -0.1, 0.1, -0.1, 0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.2, -0.0, 0.1, -0.0, 0.2, 0.2, -0.2, 0.0, -0.0, 0.1, -0.1}
);
Matrix  transformer_layers_11_attention_key_weight   (
{{0.2, 0.1, 0.1, -0.0, 0.2, 0.1, 0.1, -0.2, -0.2, -0.2, -0.0, 0.1, -0.1, 0.1, -0.2, 0.1, -0.1, -0.1, 0.1, -0.1, -0.1, -0.2, -0.1, -0.1, 0.0},
{-0.0, 0.2, -0.2, -0.0, 0.0, -0.2, -0.2, -0.1, -0.0, -0.1, -0.0, -0.1, -0.2, -0.1, 0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.0, -0.0, -0.2, 0.0},
{0.0, -0.1, -0.1, 0.1, 0.1, 0.1, 0.2, -0.2, -0.2, -0.2, 0.2, 0.1, -0.1, 0.1, -0.2, 0.2, -0.1, -0.1, -0.1, 0.1, 0.2, 0.0, -0.2, 0.0, -0.1},
{-0.1, -0.1, -0.2, 0.1, -0.2, -0.0, -0.0, 0.2, 0.1, 0.2, -0.0, 0.0, -0.1, 0.1, 0.1, 0.1, 0.2, 0.0, -0.1, -0.1, 0.0, 0.2, -0.1, -0.1, -0.2},
{0.1, -0.1, -0.0, 0.2, -0.0, -0.0, -0.1, -0.2, -0.0, 0.2, 0.1, -0.1, 0.1, -0.0, -0.0, -0.1, 0.1, -0.1, -0.1, -0.1, -0.0, 0.1, -0.2, 0.0, -0.2},
{-0.2, -0.1, 0.2, 0.1, -0.1, -0.0, 0.1, -0.2, 0.1, -0.1, 0.0, -0.2, 0.0, 0.2, -0.0, 0.2, 0.1, -0.1, 0.1, 0.2, -0.2, -0.2, 0.0, -0.1, -0.1},
{-0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, -0.2, 0.0, 0.1, 0.1, -0.2, 0.2, -0.1, 0.2, 0.1, 0.1, -0.1},
{0.1, 0.2, -0.0, -0.0, -0.1, 0.2, 0.1, -0.0, 0.0, 0.1, 0.1, 0.2, -0.2, 0.2, -0.1, -0.1, 0.1, -0.0, -0.2, 0.0, -0.1, -0.1, -0.2, -0.1, -0.1},
{-0.1, 0.1, -0.1, -0.1, 0.2, -0.0, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, -0.0, 0.1, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.0, 0.2, 0.0, -0.1, -0.2},
{0.0, -0.1, -0.2, 0.1, -0.2, -0.2, 0.0, 0.2, 0.1, 0.2, -0.0, 0.2, -0.1, -0.2, 0.0, 0.2, 0.1, -0.1, 0.1, 0.1, -0.2, -0.0, -0.1, -0.0, -0.1},
{0.1, 0.1, -0.1, 0.2, 0.1, -0.1, 0.1, -0.1, -0.2, 0.0, 0.1, -0.0, 0.1, -0.2, -0.1, 0.0, -0.1, -0.1, -0.2, 0.1, -0.1, 0.1, 0.2, -0.1, -0.0},
{0.2, -0.1, -0.0, -0.0, 0.1, 0.1, -0.1, -0.2, -0.0, -0.1, -0.1, 0.1, -0.0, -0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1},
{-0.1, 0.1, -0.1, -0.2, 0.1, -0.1, -0.2, 0.0, -0.0, -0.0, -0.0, 0.1, -0.1, -0.1, -0.1, -0.0, -0.1, -0.1, -0.2, 0.1, 0.2, -0.0, 0.1, 0.0, 0.2},
{0.0, 0.2, -0.0, -0.1, -0.1, -0.0, -0.1, -0.1, -0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.0, -0.0, 0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.2, 0.0},
{0.2, -0.2, 0.1, 0.1, 0.0, -0.1, 0.1, -0.2, 0.0, 0.0, -0.0, 0.2, 0.2, -0.2, -0.1, 0.2, 0.2, 0.0, -0.2, 0.1, 0.0, 0.1, -0.0, -0.0, 0.1},
{0.0, 0.1, -0.0, -0.0, -0.0, -0.1, -0.2, -0.2, -0.0, -0.1, 0.2, -0.1, -0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.0, 0.2, 0.0, 0.1, -0.0, -0.1, 0.1},
{-0.1, 0.1, 0.1, -0.1, 0.1, -0.1, -0.0, -0.1, -0.2, -0.1, -0.0, 0.1, -0.1, 0.1, -0.0, 0.1, 0.2, -0.0, 0.0, 0.1, -0.0, 0.1, 0.1, 0.0, 0.0},
{0.2, 0.0, -0.2, 0.2, 0.2, 0.1, -0.2, 0.1, 0.1, -0.1, -0.2, 0.2, 0.1, 0.1, 0.0, 0.0, -0.1, 0.2, -0.1, -0.1, -0.1, -0.1, 0.1, 0.2, -0.1},
{0.2, 0.0, -0.1, -0.1, -0.1, -0.1, -0.0, -0.1, 0.2, 0.1, -0.1, -0.1, 0.0, 0.1, 0.0, -0.1, -0.2, -0.1, -0.1, -0.1, 0.0, 0.0, 0.1, 0.1, -0.2},
{0.1, 0.2, -0.0, 0.2, -0.1, 0.1, 0.0, 0.0, 0.0, 0.1, -0.2, -0.2, 0.2, -0.0, -0.1, 0.2, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.2, 0.0},
{-0.1, 0.1, -0.0, 0.1, 0.1, -0.2, 0.1, 0.1, -0.1, -0.2, -0.0, -0.2, -0.1, 0.1, 0.1, 0.1, -0.0, 0.1, 0.1, -0.0, 0.1, -0.1, 0.1, 0.0, 0.1},
{-0.1, 0.2, -0.1, -0.1, -0.2, -0.0, -0.1, -0.1, 0.1, -0.1, -0.1, -0.0, 0.2, -0.1, -0.2, -0.2, -0.1, 0.2, -0.1, 0.2, 0.0, 0.1, 0.2, 0.1, -0.0},
{0.0, 0.2, 0.1, -0.1, -0.1, 0.0, 0.0, 0.1, 0.1, -0.1, 0.1, 0.0, -0.1, 0.0, 0.1, -0.1, 0.2, -0.1, 0.1, 0.2, -0.2, -0.1, 0.1, 0.0, -0.0},
{0.1, 0.1, 0.1, 0.2, 0.1, 0.2, 0.1, -0.0, -0.2, 0.0, 0.1, 0.0, -0.1, 0.2, 0.1, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.2, -0.1},
{-0.0, -0.2, -0.0, 0.2, 0.2, -0.1, 0.1, 0.1, -0.2, -0.1, 0.1, -0.1, -0.0, -0.2, -0.0, -0.0, -0.0, 0.2, 0.1, 0.1, 0.2, -0.1, -0.1, -0.2, 0.2},
}
);
Matrix  transformer_layers_11_attention_key_bias   (
{-0.1, 0.2, -0.1, -0.1, 0.2, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.0, -0.0, 0.0, -0.0, 0.1, 0.2, -0.1, -0.1, 0.1, -0.1, -0.1, -0.2, -0.1, -0.0}
);
Matrix  transformer_layers_11_attention_value_weight   (
{{0.1, -0.2, -0.2, 0.1, 0.1, -0.1, -0.2, 0.2, -0.1, -0.0, -0.1, -0.2, -0.0, -0.1, -0.1, -0.0, -0.1, 0.0, -0.0, -0.1, 0.1, 0.1, 0.1, -0.0, -0.1},
{-0.0, 0.1, 0.1, 0.1, -0.0, 0.0, -0.2, -0.1, 0.0, -0.1, -0.0, -0.0, 0.1, -0.2, -0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.1, 0.1, -0.2, -0.0, -0.2},
{0.1, 0.0, 0.1, 0.1, -0.1, 0.0, 0.0, -0.2, -0.1, 0.2, 0.2, -0.2, -0.1, 0.1, 0.0, 0.1, 0.2, -0.1, -0.1, -0.1, -0.0, 0.2, 0.2, -0.2, -0.1},
{-0.2, 0.1, -0.0, 0.1, 0.2, -0.2, -0.1, -0.1, 0.1, -0.1, 0.1, 0.0, 0.2, -0.0, -0.2, 0.0, 0.2, 0.1, -0.0, 0.1, -0.2, -0.0, 0.1, 0.1, 0.1},
{0.1, -0.2, -0.1, 0.1, -0.2, -0.0, 0.2, 0.0, 0.2, 0.1, 0.2, -0.2, -0.0, 0.1, -0.2, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, 0.2, -0.1, -0.2, -0.0},
{-0.0, 0.0, -0.2, -0.2, 0.1, -0.1, -0.0, 0.1, -0.1, -0.2, 0.1, -0.1, -0.0, 0.0, -0.1, 0.2, -0.2, -0.1, 0.1, 0.1, 0.1, 0.0, 0.2, 0.1, 0.1},
{0.2, 0.2, 0.2, -0.2, -0.2, 0.1, -0.1, 0.1, -0.1, 0.0, 0.0, 0.1, 0.1, -0.1, 0.1, -0.2, -0.1, -0.1, 0.0, -0.1, -0.1, -0.2, -0.2, -0.2, 0.1},
{0.1, -0.0, 0.1, -0.0, -0.0, -0.1, -0.1, -0.1, -0.0, -0.0, 0.1, -0.0, 0.1, -0.2, 0.0, 0.1, -0.1, -0.0, -0.0, 0.1, 0.2, -0.2, -0.1, 0.1, -0.1},
{0.0, 0.0, -0.1, 0.1, 0.1, 0.0, 0.0, 0.1, -0.2, -0.2, 0.1, -0.0, -0.1, -0.0, -0.2, 0.1, 0.1, 0.1, -0.0, 0.1, 0.1, 0.0, 0.1, -0.0, 0.1},
{-0.0, -0.1, -0.1, -0.2, -0.2, -0.2, 0.1, -0.1, 0.0, -0.2, -0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.1, 0.2, -0.2, -0.0, -0.1, 0.0, -0.1, -0.0, -0.0},
{-0.1, 0.0, 0.1, -0.1, 0.1, -0.0, -0.2, 0.2, 0.1, 0.1, -0.2, -0.1, -0.1, 0.1, 0.0, -0.0, 0.0, -0.1, -0.2, 0.1, -0.1, 0.1, -0.1, -0.1, -0.0},
{0.1, 0.2, -0.1, 0.0, -0.1, 0.0, -0.1, 0.2, -0.2, 0.1, 0.1, 0.1, -0.0, -0.1, 0.0, -0.1, 0.1, 0.2, 0.1, -0.1, 0.1, -0.1, -0.1, -0.0, -0.0},
{0.1, -0.0, -0.1, 0.2, 0.1, 0.1, -0.1, 0.2, -0.0, -0.1, 0.1, -0.1, 0.1, -0.0, -0.1, -0.0, -0.1, -0.1, -0.1, -0.1, 0.0, -0.1, 0.2, -0.1, -0.2},
{-0.1, -0.2, -0.2, 0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.0, -0.1, 0.2, -0.0, 0.0, -0.2, -0.1, 0.1, -0.1, -0.0, -0.0, 0.1, 0.0},
{0.1, -0.2, 0.1, -0.1, -0.1, 0.1, -0.0, -0.1, -0.1, 0.1, -0.0, 0.2, 0.1, -0.1, -0.1, -0.1, 0.1, -0.2, -0.1, 0.2, 0.1, 0.2, 0.1, -0.1, 0.1},
{-0.1, -0.1, 0.1, -0.2, -0.2, -0.2, -0.1, -0.1, 0.1, 0.0, -0.1, 0.1, -0.0, 0.0, 0.2, 0.1, 0.0, -0.1, 0.0, -0.1, -0.0, -0.0, -0.1, 0.2, -0.1},
{0.1, -0.1, -0.1, 0.2, -0.1, -0.1, 0.2, -0.2, 0.0, -0.1, -0.2, 0.0, -0.2, 0.0, -0.1, 0.1, 0.0, -0.1, -0.0, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1},
{0.1, 0.0, 0.2, 0.1, -0.2, -0.2, 0.1, -0.1, 0.2, -0.2, 0.0, -0.0, -0.2, -0.2, -0.0, 0.1, 0.2, 0.1, -0.1, 0.1, -0.2, 0.1, -0.1, 0.2, -0.1},
{-0.1, -0.0, 0.1, 0.0, 0.2, 0.2, -0.0, 0.1, 0.2, -0.0, -0.1, 0.1, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, -0.0, 0.2, -0.1, 0.1, 0.1, -0.1, 0.2},
{0.1, -0.1, -0.2, 0.0, 0.0, 0.0, 0.1, -0.1, 0.2, 0.2, 0.2, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.0, 0.1, 0.2, -0.2, 0.1, 0.2, 0.0, -0.1},
{-0.1, 0.1, 0.0, -0.0, 0.1, -0.1, -0.2, 0.1, -0.1, 0.1, -0.1, -0.1, -0.1, -0.2, 0.0, 0.2, -0.2, 0.2, 0.2, -0.1, -0.0, 0.2, -0.0, 0.0, -0.1},
{-0.2, -0.2, -0.1, 0.0, -0.0, -0.1, 0.1, 0.1, -0.1, -0.0, 0.0, -0.0, 0.2, -0.2, -0.2, -0.0, 0.1, 0.1, -0.1, -0.2, 0.0, -0.0, 0.1, -0.1, -0.0},
{0.1, -0.1, -0.1, -0.1, -0.0, 0.1, -0.1, 0.2, 0.1, 0.0, 0.1, 0.0, -0.1, 0.2, -0.1, 0.1, 0.1, -0.1, 0.2, 0.0, 0.1, -0.0, -0.0, -0.1, -0.0},
{0.2, 0.2, 0.0, -0.2, 0.1, 0.0, 0.0, -0.1, -0.2, -0.2, 0.0, 0.1, -0.2, 0.2, -0.1, 0.1, -0.1, -0.0, -0.2, 0.2, -0.1, 0.1, 0.1, -0.2, 0.2},
{0.2, 0.1, 0.2, -0.1, -0.0, -0.1, -0.1, 0.2, 0.1, -0.2, -0.1, -0.1, 0.0, 0.1, 0.2, 0.1, 0.1, 0.2, 0.2, -0.2, -0.1, -0.0, -0.2, -0.1, -0.1},
}
);
Matrix  transformer_layers_11_attention_value_bias   (
{-0.1, -0.2, -0.1, 0.0, -0.0, -0.0, -0.0, 0.0, -0.1, -0.2, 0.0, -0.0, 0.1, -0.2, 0.0, 0.1, -0.1, 0.1, -0.2, -0.1, 0.2, -0.1, -0.2, 0.1, 0.1}
);
Matrix  transformer_layers_11_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_11_norm1_layer_norm_bias   (
{-0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_11_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_11_norm2_layer_norm_bias   (
{0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_11_feed_forward_linear1_weight   (
{{0.0, -0.0, -0.1, -0.0, 0.1, -0.2, 0.1, -0.1, 0.1, -0.0, 0.1, -0.2, -0.2, -0.2, 0.1, 0.1, 0.1, -0.0, -0.2, 0.0, 0.0, 0.0, -0.2, -0.2, 0.1},
{0.2, 0.1, 0.2, 0.0, 0.2, 0.1, -0.2, -0.1, 0.1, 0.1, 0.0, -0.2, -0.2, 0.2, -0.1, 0.0, -0.2, -0.2, -0.0, -0.1, -0.1, 0.2, -0.2, 0.1, -0.1},
{-0.2, 0.0, 0.2, -0.1, -0.2, 0.1, -0.2, -0.2, 0.1, -0.0, -0.0, -0.0, -0.1, 0.0, 0.1, 0.2, -0.2, 0.0, -0.1, -0.1, -0.1, 0.0, -0.0, 0.1, 0.0},
{0.2, 0.0, -0.1, -0.2, -0.1, -0.0, -0.0, 0.2, 0.1, -0.1, 0.1, 0.2, -0.1, 0.1, 0.1, 0.0, -0.2, -0.2, -0.1, -0.2, -0.0, 0.1, 0.1, -0.2, 0.1},
{-0.1, 0.1, 0.0, -0.1, 0.1, -0.0, 0.1, 0.1, 0.1, 0.1, 0.0, -0.0, 0.1, -0.0, -0.0, 0.0, -0.0, -0.1, -0.2, -0.2, -0.2, 0.1, 0.0, -0.0, -0.0},
{0.0, -0.2, 0.0, -0.1, 0.2, -0.1, -0.0, -0.0, 0.0, 0.2, 0.1, -0.1, 0.1, -0.1, 0.2, 0.2, 0.1, 0.0, 0.0, -0.1, -0.1, 0.2, 0.0, -0.2, -0.1},
{0.0, 0.2, 0.2, -0.2, -0.2, -0.1, 0.0, -0.1, -0.2, -0.2, -0.1, -0.1, -0.2, -0.0, 0.0, 0.0, 0.0, -0.2, 0.1, 0.1, 0.1, 0.1, 0.0, 0.2, -0.0},
{-0.1, -0.2, 0.1, -0.1, 0.1, 0.0, -0.2, 0.1, 0.1, 0.1, -0.0, 0.0, -0.1, -0.2, -0.1, 0.0, 0.1, 0.1, 0.0, 0.1, -0.1, -0.2, -0.1, -0.0, -0.1},
{0.2, -0.1, 0.2, -0.1, 0.1, 0.2, 0.1, -0.0, -0.1, 0.0, -0.1, -0.2, 0.2, -0.1, 0.0, -0.1, -0.2, 0.1, 0.1, -0.2, 0.0, -0.1, 0.0, 0.1, -0.1},
{0.1, 0.2, 0.1, -0.1, 0.2, 0.0, 0.1, 0.1, -0.2, 0.1, -0.1, -0.1, 0.2, -0.1, -0.2, -0.1, 0.2, 0.1, 0.2, 0.2, -0.0, 0.1, 0.1, -0.1, 0.1},
{0.1, 0.2, -0.2, -0.1, -0.1, 0.1, -0.1, 0.0, 0.2, -0.2, -0.1, 0.2, -0.1, -0.1, 0.1, 0.2, -0.0, 0.1, 0.2, -0.1, 0.0, 0.2, -0.2, 0.2, -0.0},
{-0.1, -0.0, 0.0, 0.2, -0.1, -0.1, -0.1, 0.0, -0.0, 0.1, 0.1, -0.0, -0.0, -0.1, -0.2, -0.0, -0.2, 0.1, 0.0, -0.2, 0.1, 0.1, 0.0, -0.1, 0.1},
{-0.2, 0.1, -0.2, 0.0, 0.2, 0.1, -0.2, 0.1, 0.1, 0.1, -0.1, 0.2, 0.1, -0.1, -0.2, 0.0, -0.1, 0.1, -0.2, -0.1, 0.1, -0.0, -0.0, 0.2, -0.0},
{0.2, -0.2, 0.1, 0.2, -0.2, -0.2, -0.2, -0.2, 0.0, 0.2, 0.2, -0.1, -0.1, 0.2, -0.1, 0.1, -0.2, 0.0, 0.0, -0.2, -0.1, -0.2, -0.0, -0.2, 0.1},
{0.1, -0.1, -0.2, -0.0, 0.1, -0.0, 0.2, 0.0, -0.2, 0.1, 0.1, 0.0, 0.1, 0.1, -0.1, 0.0, 0.1, -0.0, 0.1, 0.1, 0.0, -0.1, -0.1, 0.0, 0.1},
}
);
Matrix  transformer_layers_11_feed_forward_linear1_bias   (
{-0.1, -0.2, -0.0, -0.1, -0.1, -0.2, -0.1, -0.0, 0.1, -0.2, -0.0, 0.1, 0.1, -0.0, 0.1}
);
Matrix  transformer_layers_11_feed_forward_linear2_weight   (
{{-0.2, -0.2, 0.0, 0.0, -0.2, -0.2, -0.0, 0.1, -0.2, -0.0, -0.1, -0.0, 0.0, 0.0, -0.2},
{0.2, 0.1, -0.1, -0.2, 0.1, -0.0, -0.1, -0.1, -0.2, -0.1, -0.2, -0.2, 0.1, -0.0, -0.2},
{-0.0, 0.2, 0.1, 0.2, -0.2, 0.1, 0.2, -0.2, 0.2, -0.2, -0.1, -0.2, 0.1, 0.3, 0.2},
{-0.1, 0.1, 0.0, 0.2, 0.1, 0.2, -0.0, -0.2, 0.2, -0.1, -0.2, -0.2, -0.0, 0.1, 0.2},
{0.2, 0.0, -0.2, -0.2, 0.2, -0.2, -0.1, -0.1, -0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1},
{-0.2, 0.2, -0.1, 0.0, -0.1, -0.1, -0.2, -0.0, 0.1, -0.2, -0.1, 0.1, 0.0, -0.2, 0.1},
{-0.0, 0.1, 0.1, 0.1, 0.1, -0.1, 0.2, -0.2, -0.2, 0.0, 0.1, 0.1, -0.3, -0.1, -0.2},
{0.2, 0.0, 0.1, 0.2, -0.3, 0.2, -0.0, -0.1, 0.2, -0.0, 0.2, 0.1, 0.2, 0.1, 0.2},
{0.2, -0.1, 0.1, -0.3, 0.1, 0.2, 0.2, 0.1, 0.2, 0.2, 0.2, 0.2, 0.1, -0.2, -0.2},
{-0.1, -0.3, 0.2, -0.0, -0.1, 0.2, 0.0, -0.1, -0.2, -0.1, -0.2, 0.2, 0.1, -0.1, -0.1},
{0.1, -0.1, 0.0, 0.1, -0.1, 0.1, -0.0, 0.2, -0.2, -0.3, 0.0, -0.0, -0.2, -0.0, 0.1},
{0.2, -0.3, -0.1, -0.2, 0.1, -0.1, -0.2, -0.1, -0.1, -0.1, -0.0, 0.1, -0.2, -0.2, 0.2},
{-0.2, 0.1, 0.1, -0.2, -0.0, 0.2, 0.3, 0.1, 0.2, 0.1, -0.1, -0.2, 0.1, -0.3, -0.1},
{0.1, 0.1, 0.0, 0.1, -0.2, 0.0, -0.2, -0.2, -0.2, 0.2, 0.2, -0.2, 0.2, 0.2, -0.2},
{-0.2, 0.2, -0.1, 0.1, -0.2, -0.0, -0.2, 0.0, -0.1, 0.2, -0.0, -0.1, 0.2, 0.0, -0.0},
{0.1, 0.1, 0.1, 0.1, -0.3, -0.2, 0.1, -0.0, 0.0, -0.1, -0.1, 0.2, 0.1, 0.2, 0.2},
{-0.3, -0.2, -0.2, -0.0, 0.1, -0.0, -0.0, 0.2, 0.2, -0.0, 0.2, -0.2, 0.2, -0.1, -0.1},
{0.1, -0.1, -0.1, 0.2, 0.3, -0.1, -0.0, -0.2, -0.3, -0.2, -0.1, 0.1, 0.0, -0.1, -0.2},
{-0.3, -0.3, -0.2, 0.3, -0.1, 0.0, -0.2, 0.2, -0.1, 0.1, 0.0, -0.2, 0.3, 0.2, 0.0},
{0.1, 0.2, -0.0, -0.1, -0.1, 0.0, 0.1, -0.1, 0.2, 0.3, 0.2, -0.0, -0.0, 0.2, -0.0},
{0.1, -0.1, -0.0, 0.2, -0.3, -0.2, 0.2, -0.1, 0.1, -0.1, 0.2, 0.2, -0.0, -0.1, -0.2},
{-0.2, 0.1, -0.2, -0.1, 0.2, 0.1, 0.2, 0.0, -0.2, -0.0, 0.1, -0.2, -0.1, 0.2, -0.3},
{-0.1, 0.2, 0.0, 0.1, 0.2, 0.3, 0.0, -0.1, 0.2, 0.1, -0.3, 0.2, 0.2, 0.0, 0.1},
{-0.2, 0.0, 0.0, -0.0, 0.2, 0.2, -0.1, -0.1, 0.0, -0.0, -0.1, 0.1, -0.1, 0.1, -0.2},
{0.1, -0.3, 0.1, 0.0, -0.2, 0.1, -0.2, -0.2, -0.2, 0.0, 0.1, -0.2, -0.1, 0.2, -0.0},
}
);
Matrix  transformer_layers_11_feed_forward_linear2_bias   (
{0.2, 0.2, 0.1, 0.2, 0.1, -0.0, -0.3, 0.1, 0.1, -0.2, -0.1, -0.2, 0.2, -0.2, 0.3, -0.1, 0.1, 0.2, -0.1, 0.0, -0.3, -0.1, 0.2, 0.1, -0.1}
);
Matrix  transformer_layers_11_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_11_feed_forward_ln1_layer_norm_bias   (
{0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_11_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_11_feed_forward_ln2_layer_norm_bias   (
{0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_12_attention_query_weight   (
{{-0.0, 0.1, -0.1, -0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.2, -0.1, -0.1, 0.2, 0.2, 0.1, -0.0, 0.0, -0.0, -0.1, 0.0, 0.2, -0.1, -0.1, 0.0},
{0.1, 0.1, 0.2, -0.1, -0.2, -0.1, 0.1, 0.0, 0.2, -0.0, -0.1, 0.1, -0.2, 0.1, -0.2, 0.2, 0.1, 0.1, -0.1, 0.0, -0.0, -0.0, -0.2, -0.1, -0.1},
{0.1, -0.1, 0.2, 0.1, -0.1, -0.1, 0.2, -0.2, 0.2, -0.1, 0.2, 0.2, 0.1, -0.0, 0.1, -0.2, 0.1, 0.1, 0.2, -0.1, 0.2, -0.1, -0.0, 0.1, 0.2},
{0.2, -0.2, -0.1, 0.0, -0.1, 0.0, -0.1, 0.1, 0.0, 0.0, 0.0, -0.1, 0.2, -0.1, 0.0, 0.2, 0.2, 0.0, 0.2, 0.1, 0.0, 0.1, 0.1, -0.1, -0.2},
{0.2, -0.2, -0.1, 0.0, 0.2, -0.2, -0.2, -0.1, -0.1, -0.1, 0.2, 0.1, -0.1, -0.1, -0.0, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.2, -0.2, -0.2},
{0.0, -0.0, 0.1, 0.0, 0.1, -0.0, -0.0, -0.2, 0.0, -0.0, 0.1, -0.1, -0.0, 0.0, -0.1, 0.0, -0.1, 0.1, -0.2, -0.2, 0.2, 0.1, 0.0, 0.1, 0.1},
{-0.1, -0.1, 0.2, 0.1, 0.1, 0.1, 0.0, 0.0, -0.1, -0.0, -0.0, 0.1, -0.2, 0.1, 0.2, 0.0, -0.2, -0.0, 0.0, 0.1, -0.1, 0.1, 0.1, 0.2, 0.2},
{-0.1, 0.2, 0.0, 0.0, -0.1, 0.1, 0.2, -0.1, 0.0, -0.1, 0.0, 0.0, -0.0, 0.2, 0.0, 0.1, -0.1, 0.1, 0.2, -0.1, -0.2, 0.0, 0.2, -0.1, -0.1},
{0.1, -0.1, -0.1, 0.0, -0.2, 0.0, -0.1, -0.2, 0.1, -0.0, -0.2, -0.1, 0.0, -0.0, -0.2, -0.2, -0.0, -0.1, 0.0, -0.1, -0.2, 0.0, -0.2, -0.0, -0.2},
{-0.2, -0.2, -0.2, 0.2, -0.2, 0.0, 0.1, -0.0, -0.1, -0.0, -0.1, 0.1, -0.0, 0.2, -0.1, -0.0, 0.1, -0.1, -0.1, 0.2, 0.1, -0.1, 0.0, -0.2, 0.1},
{0.0, 0.1, 0.2, 0.0, -0.1, -0.0, -0.2, 0.1, 0.2, 0.1, 0.2, -0.1, 0.2, 0.0, -0.0, 0.1, -0.1, 0.2, -0.1, 0.1, 0.1, 0.1, 0.0, -0.1, 0.1},
{-0.2, 0.0, -0.0, -0.1, 0.1, -0.2, -0.2, -0.1, -0.2, -0.2, -0.0, 0.0, 0.0, 0.2, -0.0, 0.2, 0.2, 0.1, -0.1, 0.1, 0.0, -0.0, 0.0, -0.2, -0.0},
{-0.1, -0.1, 0.2, 0.0, 0.0, 0.1, 0.1, 0.2, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.0, 0.2, -0.2, 0.0, 0.1, 0.0, -0.0, 0.2, -0.2, -0.1, -0.0},
{-0.1, -0.0, -0.1, 0.0, 0.2, -0.2, -0.0, 0.0, -0.1, 0.2, -0.1, -0.0, 0.2, -0.0, -0.1, 0.2, -0.2, 0.1, -0.1, -0.0, -0.1, 0.1, -0.0, -0.1, -0.2},
{0.0, -0.2, -0.0, 0.2, -0.0, 0.0, 0.2, -0.1, -0.1, 0.2, -0.1, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, -0.2, -0.0, -0.0, -0.2, 0.1, 0.0, 0.1, 0.0},
{-0.1, -0.2, -0.1, -0.2, -0.2, -0.2, 0.1, 0.1, 0.2, -0.0, -0.2, 0.2, 0.1, -0.2, -0.1, 0.0, -0.2, 0.1, 0.1, -0.0, -0.2, 0.0, -0.1, -0.0, 0.1},
{0.1, -0.1, -0.2, 0.2, -0.1, -0.0, -0.0, -0.1, -0.1, 0.1, -0.1, 0.0, -0.2, 0.1, 0.2, 0.1, -0.2, 0.1, 0.0, -0.1, 0.0, 0.1, -0.0, -0.0, 0.2},
{-0.1, -0.1, 0.1, 0.0, -0.1, -0.1, 0.2, 0.2, -0.2, 0.1, 0.2, 0.0, -0.1, -0.0, 0.0, -0.0, 0.2, -0.1, 0.0, 0.1, -0.0, -0.1, -0.2, 0.1, 0.1},
{0.2, -0.1, -0.1, -0.1, -0.2, -0.1, 0.2, 0.2, 0.2, -0.1, 0.2, -0.1, -0.2, -0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1},
{-0.0, -0.1, 0.0, -0.1, -0.1, -0.2, -0.2, -0.1, 0.1, 0.1, -0.2, -0.1, 0.2, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.2, 0.2, 0.1, 0.2, 0.1, 0.2},
{0.1, 0.1, -0.1, 0.2, 0.1, -0.2, -0.2, -0.1, -0.0, 0.1, 0.2, -0.1, -0.1, -0.2, 0.1, 0.2, -0.1, -0.1, -0.2, 0.1, -0.2, 0.1, 0.1, -0.2, -0.1},
{-0.1, -0.1, 0.1, -0.0, -0.0, -0.2, 0.0, -0.0, 0.2, -0.2, -0.1, -0.1, 0.0, 0.1, 0.0, -0.1, 0.2, -0.0, 0.1, 0.1, -0.2, -0.2, 0.0, -0.1, -0.1},
{-0.1, 0.0, -0.1, 0.0, -0.1, 0.1, -0.0, 0.0, 0.1, -0.1, -0.1, 0.1, -0.0, -0.0, 0.1, -0.0, 0.2, 0.2, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1},
{-0.2, -0.0, -0.2, 0.2, 0.1, -0.2, 0.2, -0.1, 0.1, -0.1, -0.1, 0.2, -0.1, 0.2, -0.2, 0.2, 0.1, 0.2, 0.1, -0.1, -0.0, -0.1, 0.2, 0.1, -0.1},
{0.2, 0.1, -0.1, 0.1, -0.2, 0.1, 0.1, -0.2, -0.2, -0.0, -0.0, 0.2, -0.1, -0.2, 0.0, 0.0, 0.1, -0.1, -0.1, -0.0, 0.1, 0.1, -0.0, -0.2, 0.2},
}
);
Matrix  transformer_layers_12_attention_query_bias   (
{0.1, 0.2, -0.0, -0.1, -0.2, -0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.2, -0.1, -0.1, -0.2, -0.0, 0.2, 0.2, 0.2, -0.2, -0.1, 0.0, -0.1, 0.1, -0.1}
);
Matrix  transformer_layers_12_attention_key_weight   (
{{0.1, 0.1, 0.2, 0.1, -0.1, 0.0, 0.1, -0.1, -0.1, -0.1, 0.0, 0.2, -0.0, 0.2, -0.0, 0.0, 0.1, -0.2, 0.2, -0.0, -0.2, 0.1, -0.1, -0.0, 0.1},
{0.1, 0.0, -0.1, 0.2, 0.0, 0.0, 0.0, 0.1, -0.2, 0.1, -0.1, -0.0, -0.1, -0.0, -0.0, 0.0, 0.0, -0.2, 0.2, 0.1, -0.1, 0.1, -0.0, -0.1, -0.1},
{-0.2, 0.0, 0.1, -0.0, -0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.0, -0.1, -0.1, 0.1, -0.0, 0.0, 0.2, -0.1, -0.1, -0.1, 0.1},
{0.1, 0.0, 0.2, -0.1, 0.2, 0.1, 0.2, 0.0, 0.0, 0.2, -0.0, -0.2, -0.2, 0.0, -0.1, -0.2, -0.1, 0.1, -0.1, -0.2, 0.1, 0.2, -0.1, 0.0, -0.1},
{-0.0, 0.1, 0.2, -0.2, 0.2, 0.0, -0.2, -0.1, -0.1, -0.0, -0.2, -0.2, -0.1, -0.1, 0.1, -0.1, 0.1, -0.0, 0.2, -0.2, -0.0, 0.1, 0.0, 0.2, -0.1},
{-0.2, -0.1, 0.0, -0.1, -0.1, -0.2, -0.2, -0.1, 0.0, 0.0, -0.1, 0.0, 0.2, 0.1, -0.1, -0.2, 0.1, -0.1, 0.0, 0.0, -0.0, -0.1, 0.1, -0.2, -0.0},
{0.0, -0.1, -0.0, -0.2, -0.0, -0.1, 0.1, 0.1, -0.2, -0.1, -0.2, -0.0, 0.2, 0.1, -0.2, -0.0, -0.1, -0.2, 0.0, 0.1, -0.1, -0.0, -0.0, -0.2, 0.1},
{-0.1, 0.2, -0.0, -0.1, 0.1, -0.1, 0.0, -0.0, 0.1, -0.1, -0.2, -0.1, 0.1, 0.2, 0.1, 0.2, -0.0, -0.0, -0.2, -0.0, -0.2, -0.2, -0.2, 0.1, 0.0},
{-0.1, 0.0, -0.2, -0.1, -0.2, -0.1, 0.1, -0.1, -0.1, -0.0, 0.0, -0.0, 0.0, -0.0, -0.1, -0.1, 0.0, 0.1, -0.2, 0.1, -0.0, -0.2, 0.2, -0.1, 0.2},
{0.0, 0.2, -0.0, -0.2, -0.1, 0.2, -0.1, 0.1, 0.1, 0.1, -0.2, -0.0, 0.1, -0.2, 0.2, 0.2, 0.0, 0.2, -0.0, 0.1, -0.0, -0.2, -0.0, -0.1, 0.1},
{-0.1, -0.1, -0.2, 0.1, 0.0, 0.2, 0.0, 0.2, -0.1, -0.1, 0.1, -0.2, -0.1, -0.2, 0.1, -0.1, 0.1, 0.2, -0.1, -0.1, 0.2, 0.1, -0.2, -0.1, 0.1},
{-0.1, -0.0, 0.1, -0.1, -0.1, -0.2, 0.2, 0.1, -0.1, 0.2, -0.2, -0.0, -0.0, -0.1, -0.2, 0.1, -0.1, -0.2, 0.1, 0.1, 0.2, 0.1, -0.0, 0.1, -0.0},
{0.2, 0.0, -0.1, 0.1, -0.1, -0.1, -0.2, -0.0, 0.0, 0.0, 0.2, -0.1, -0.2, -0.2, -0.1, 0.1, 0.1, 0.1, 0.2, 0.1, -0.0, 0.1, 0.2, 0.2, 0.0},
{0.2, 0.2, -0.1, 0.0, -0.1, 0.0, 0.1, -0.2, 0.1, 0.2, -0.0, 0.1, -0.1, -0.1, -0.0, 0.1, -0.1, 0.1, -0.0, -0.2, -0.0, 0.0, 0.1, 0.0, -0.0},
{0.1, -0.2, 0.1, 0.1, -0.2, 0.1, 0.1, 0.1, -0.1, -0.1, 0.0, -0.1, 0.1, -0.2, -0.1, -0.2, -0.1, 0.0, -0.2, -0.2, -0.1, -0.2, 0.1, -0.1, 0.1},
{-0.1, 0.2, -0.1, -0.2, -0.0, 0.1, 0.1, 0.1, -0.0, -0.2, 0.1, 0.0, -0.1, -0.1, 0.0, -0.1, 0.0, 0.2, 0.1, -0.1, 0.1, -0.1, 0.1, 0.0, 0.2},
{0.2, 0.2, 0.1, -0.2, -0.2, -0.0, -0.1, -0.0, 0.1, 0.1, -0.1, 0.2, -0.2, -0.1, -0.2, 0.1, 0.1, -0.0, -0.0, -0.1, -0.1, -0.2, -0.1, -0.0, -0.1},
{-0.1, 0.0, 0.1, 0.0, 0.1, -0.1, 0.1, -0.2, -0.1, 0.2, -0.2, -0.2, -0.0, -0.0, 0.0, 0.1, 0.1, -0.1, -0.0, 0.2, 0.1, -0.0, -0.2, 0.1, -0.0},
{0.2, -0.2, 0.2, 0.2, 0.1, 0.0, -0.2, 0.1, 0.2, 0.1, 0.1, 0.0, -0.1, -0.1, -0.2, -0.1, 0.1, 0.2, -0.1, -0.1, -0.1, 0.1, 0.0, -0.0, -0.0},
{-0.1, -0.1, 0.1, 0.1, 0.1, -0.2, 0.0, -0.1, -0.2, -0.0, 0.2, 0.1, -0.0, -0.1, 0.1, -0.2, -0.2, -0.1, -0.1, -0.2, -0.0, 0.2, -0.2, 0.1, -0.1},
{-0.2, -0.2, -0.1, -0.1, -0.0, 0.2, -0.0, 0.1, -0.2, -0.2, 0.1, 0.1, 0.1, 0.0, 0.1, -0.0, 0.1, 0.1, -0.1, 0.0, -0.0, 0.1, -0.1, -0.1, 0.0},
{0.2, 0.0, 0.1, 0.2, 0.1, 0.1, -0.2, -0.1, 0.2, 0.2, -0.1, -0.1, -0.1, -0.0, -0.2, 0.2, 0.1, -0.1, -0.0, -0.2, 0.1, 0.1, -0.1, -0.0, 0.0},
{-0.0, 0.1, 0.1, 0.1, -0.1, 0.0, 0.1, -0.1, -0.2, 0.2, 0.1, -0.1, 0.2, 0.1, -0.1, 0.2, -0.1, -0.1, -0.0, 0.1, -0.1, 0.1, -0.1, 0.0, 0.1},
{-0.2, 0.1, -0.2, 0.0, 0.2, 0.2, -0.0, 0.1, 0.1, -0.2, 0.0, 0.1, 0.0, -0.1, -0.1, -0.2, 0.0, -0.1, 0.0, -0.1, -0.0, 0.1, 0.1, -0.0, 0.1},
{-0.1, -0.1, 0.1, 0.0, -0.0, 0.1, 0.1, -0.1, 0.0, 0.1, 0.1, 0.2, 0.1, -0.1, -0.2, 0.0, -0.1, -0.2, -0.1, 0.1, -0.2, -0.1, 0.0, -0.1, 0.1},
}
);
Matrix  transformer_layers_12_attention_key_bias   (
{0.1, -0.1, -0.2, -0.1, 0.0, 0.1, -0.2, 0.1, -0.2, 0.0, -0.2, -0.1, 0.1, 0.1, -0.1, 0.0, -0.1, 0.2, -0.0, 0.1, -0.0, -0.0, 0.2, 0.1, -0.1}
);
Matrix  transformer_layers_12_attention_value_weight   (
{{-0.2, 0.1, 0.0, 0.1, 0.1, -0.1, 0.1, -0.0, 0.2, 0.1, 0.2, 0.2, 0.1, -0.1, 0.0, 0.1, -0.1, -0.2, -0.2, -0.2, 0.0, -0.0, -0.1, -0.1, -0.2},
{-0.0, 0.1, -0.1, -0.1, -0.2, -0.2, -0.1, 0.0, 0.0, 0.2, 0.0, 0.2, 0.1, 0.0, -0.2, -0.1, 0.1, -0.0, -0.1, 0.1, -0.0, -0.0, -0.1, -0.1, -0.2},
{-0.1, -0.2, -0.2, -0.0, 0.2, 0.0, 0.2, 0.2, -0.2, 0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.1, -0.1, 0.1, -0.1, -0.1, -0.1, 0.2, -0.0, 0.1, 0.0},
{-0.0, 0.1, 0.1, -0.0, -0.0, 0.2, 0.0, -0.1, 0.0, -0.1, -0.0, 0.1, 0.2, -0.2, 0.1, 0.2, 0.1, 0.0, -0.0, 0.0, -0.0, -0.2, -0.1, -0.1, 0.1},
{0.1, 0.0, 0.1, 0.1, -0.2, 0.1, -0.0, 0.1, -0.2, -0.2, 0.2, 0.2, 0.1, 0.2, 0.0, 0.0, -0.1, 0.1, 0.1, -0.2, 0.1, 0.1, 0.1, -0.1, 0.2},
{-0.1, -0.1, 0.1, -0.2, 0.1, 0.2, 0.1, 0.1, -0.1, -0.2, 0.0, 0.1, 0.2, -0.1, -0.1, 0.1, -0.0, 0.1, 0.1, 0.0, -0.2, 0.2, 0.2, -0.2, 0.2},
{0.0, -0.1, 0.0, -0.2, 0.0, 0.1, -0.1, 0.2, -0.1, -0.1, -0.1, -0.1, 0.2, 0.0, -0.2, 0.0, -0.1, 0.0, 0.1, 0.2, 0.1, 0.2, 0.1, 0.1, 0.1},
{0.1, 0.0, 0.1, 0.1, -0.0, -0.2, 0.0, -0.2, 0.1, -0.1, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, 0.2, -0.1, 0.1, 0.1, -0.1, 0.1, -0.0, -0.2, 0.2},
{-0.1, 0.0, 0.0, -0.1, 0.0, 0.0, -0.0, 0.0, -0.2, 0.0, -0.1, 0.1, -0.0, -0.2, 0.2, 0.2, -0.1, -0.2, -0.1, -0.1, 0.0, -0.2, 0.1, -0.1, -0.0},
{-0.1, 0.1, -0.1, 0.2, 0.1, -0.0, -0.0, -0.1, 0.1, -0.1, 0.1, -0.2, -0.1, 0.0, 0.1, -0.1, 0.1, -0.0, -0.1, -0.0, 0.0, 0.2, 0.0, -0.1, 0.2},
{-0.1, -0.1, -0.1, 0.2, -0.1, 0.1, -0.2, 0.0, -0.1, 0.0, -0.0, -0.2, -0.0, -0.2, 0.1, -0.1, 0.0, 0.0, -0.1, 0.0, 0.1, 0.1, -0.2, 0.0, -0.1},
{0.0, 0.0, 0.1, 0.1, -0.0, -0.1, -0.1, 0.1, 0.1, -0.2, 0.1, 0.2, -0.1, 0.1, -0.0, -0.0, -0.1, -0.2, 0.2, 0.2, -0.1, -0.0, -0.2, 0.2, -0.1},
{0.1, -0.1, 0.0, 0.1, 0.1, -0.1, -0.2, -0.0, -0.0, -0.0, -0.1, -0.1, 0.0, -0.1, -0.0, -0.0, 0.0, -0.2, -0.1, -0.1, 0.0, 0.1, -0.2, 0.0, 0.1},
{-0.2, -0.0, 0.0, -0.1, -0.1, 0.1, 0.0, 0.0, -0.1, -0.2, 0.1, -0.2, -0.0, -0.2, 0.1, -0.0, -0.2, -0.2, 0.1, 0.1, 0.2, 0.2, -0.0, -0.1, -0.2},
{-0.1, 0.0, 0.1, -0.1, 0.1, -0.1, -0.2, -0.1, 0.1, 0.0, -0.1, -0.0, -0.1, -0.0, 0.2, -0.2, -0.1, -0.1, 0.1, 0.0, -0.2, -0.2, 0.1, -0.1, -0.1},
{0.0, 0.0, -0.1, 0.2, -0.1, 0.1, 0.1, -0.1, -0.1, 0.2, 0.2, -0.2, 0.2, 0.0, -0.2, 0.1, -0.1, -0.1, 0.1, -0.0, -0.0, 0.2, 0.0, -0.1, 0.1},
{0.2, -0.1, -0.2, 0.0, -0.1, 0.1, 0.0, -0.1, -0.2, 0.2, -0.1, 0.2, 0.1, 0.1, -0.0, 0.2, 0.2, 0.2, 0.1, 0.1, -0.0, 0.0, 0.1, 0.1, -0.2},
{-0.2, -0.2, 0.0, 0.0, 0.0, -0.1, 0.0, 0.2, -0.1, -0.1, 0.0, -0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.0, -0.0, 0.2, -0.1, -0.1, 0.1},
{0.0, 0.1, 0.0, -0.2, -0.0, 0.2, -0.1, -0.1, 0.2, 0.1, 0.2, -0.0, -0.0, -0.1, 0.1, 0.2, -0.1, -0.1, -0.2, 0.0, 0.1, -0.0, -0.0, -0.2, 0.2},
{0.1, -0.2, 0.1, -0.1, 0.0, 0.1, 0.1, 0.2, 0.0, 0.1, 0.1, 0.0, -0.1, 0.2, 0.0, -0.1, -0.2, 0.2, -0.0, 0.1, -0.0, 0.2, 0.1, -0.0, 0.1},
{-0.0, 0.1, -0.2, -0.1, 0.1, -0.2, -0.2, 0.1, 0.1, -0.0, 0.1, -0.2, -0.0, 0.2, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.2, -0.1, -0.0, 0.1, -0.1},
{0.0, 0.2, -0.1, -0.2, -0.1, 0.2, -0.2, -0.1, 0.1, -0.2, -0.1, 0.1, -0.2, -0.1, -0.0, -0.0, 0.2, 0.1, 0.1, -0.1, -0.1, 0.1, -0.0, 0.2, 0.2},
{0.0, -0.1, 0.1, -0.1, -0.1, -0.0, 0.0, 0.0, -0.1, -0.1, 0.1, 0.0, -0.1, 0.1, -0.1, 0.0, -0.0, 0.1, -0.1, -0.1, -0.1, 0.1, 0.0, 0.1, 0.1},
{0.0, 0.1, -0.1, -0.1, 0.0, -0.1, 0.0, -0.2, -0.1, -0.0, 0.1, 0.0, -0.1, 0.0, 0.2, -0.0, 0.1, -0.1, -0.0, -0.2, 0.0, 0.1, 0.0, -0.1, -0.0},
{-0.2, 0.1, 0.1, 0.0, -0.1, 0.2, 0.1, -0.1, 0.1, 0.2, -0.0, 0.2, 0.1, 0.2, 0.1, 0.1, -0.1, -0.1, -0.2, 0.1, -0.2, -0.2, 0.2, -0.1, 0.1},
}
);
Matrix  transformer_layers_12_attention_value_bias   (
{-0.1, -0.2, 0.2, -0.1, -0.2, -0.1, 0.2, 0.0, -0.1, 0.2, -0.0, 0.0, -0.2, -0.2, -0.1, -0.1, -0.2, -0.0, 0.0, 0.1, -0.1, -0.1, 0.0, -0.1, -0.0}
);
Matrix  transformer_layers_12_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_12_norm1_layer_norm_bias   (
{0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_12_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_12_norm2_layer_norm_bias   (
{0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_12_feed_forward_linear1_weight   (
{{0.2, -0.1, -0.2, 0.1, -0.0, -0.0, 0.2, 0.0, -0.1, 0.1, -0.1, 0.2, -0.2, -0.1, 0.0, 0.0, 0.2, -0.1, 0.1, 0.1, -0.0, 0.1, 0.1, -0.2, -0.2},
{0.2, -0.0, -0.2, 0.2, -0.1, -0.2, 0.1, 0.0, -0.1, 0.1, -0.2, -0.2, -0.2, 0.1, -0.1, -0.1, -0.0, 0.0, -0.1, -0.1, -0.0, -0.1, 0.0, 0.1, 0.2},
{0.2, -0.0, 0.0, -0.1, 0.1, 0.0, 0.1, -0.1, -0.0, 0.1, -0.1, 0.2, 0.1, -0.2, -0.0, 0.0, -0.2, 0.1, -0.0, -0.0, 0.2, 0.1, -0.1, 0.2, -0.0},
{-0.2, -0.0, 0.0, -0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.2, -0.1, -0.2, 0.0, -0.1, 0.2, 0.0, -0.1, -0.1, -0.1, -0.1, 0.1, -0.1, 0.0, 0.1},
{-0.0, -0.2, 0.1, -0.2, -0.0, -0.1, 0.2, -0.1, -0.1, -0.0, -0.1, -0.1, -0.2, -0.2, 0.0, 0.1, 0.2, -0.0, 0.1, -0.0, -0.0, 0.0, -0.2, 0.2, -0.1},
{0.1, -0.0, -0.1, 0.0, -0.1, 0.2, -0.1, -0.1, -0.1, 0.0, 0.2, -0.1, 0.0, 0.1, -0.1, -0.1, 0.2, 0.0, 0.2, 0.2, 0.2, 0.1, 0.1, -0.1, 0.0},
{0.0, -0.0, -0.1, -0.0, 0.2, 0.0, 0.2, 0.0, 0.0, 0.1, -0.1, 0.2, -0.0, -0.2, 0.0, 0.2, -0.0, -0.0, 0.2, -0.0, -0.2, -0.1, 0.1, 0.2, 0.1},
{0.1, 0.2, -0.1, -0.2, 0.1, 0.0, -0.2, -0.2, -0.1, -0.1, -0.1, -0.0, 0.0, -0.1, -0.1, 0.1, -0.1, 0.1, 0.2, -0.0, -0.1, 0.0, 0.1, 0.0, 0.1},
{0.1, 0.0, -0.1, -0.0, -0.1, 0.1, 0.1, -0.2, -0.1, -0.2, -0.1, -0.2, 0.0, -0.0, 0.1, 0.0, 0.1, -0.2, 0.2, -0.1, -0.2, 0.1, -0.0, -0.2, 0.2},
{-0.1, -0.1, -0.1, 0.0, 0.1, 0.2, -0.1, -0.0, -0.0, 0.0, -0.1, -0.2, -0.0, 0.2, 0.1, 0.1, 0.0, -0.1, 0.2, 0.0, 0.0, -0.2, 0.1, -0.1, 0.2},
{0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.2, -0.1, 0.1, -0.2, -0.1, 0.1, -0.0, 0.0, 0.2, -0.1, 0.1, -0.2, 0.2, 0.1, -0.1, -0.0},
{-0.2, 0.0, 0.1, 0.1, 0.2, -0.1, -0.1, -0.1, 0.2, 0.1, 0.1, -0.2, -0.2, -0.1, -0.1, 0.1, 0.0, 0.0, 0.1, 0.2, -0.0, 0.1, -0.1, 0.1, -0.2},
{-0.0, -0.2, -0.1, -0.1, -0.1, 0.1, -0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.1, -0.1, 0.1, -0.2, 0.2, 0.0, 0.0, 0.1, -0.1, -0.1, -0.0, 0.1, -0.1},
{0.0, 0.1, -0.1, -0.0, -0.2, 0.2, 0.1, -0.2, -0.1, -0.2, -0.1, 0.0, -0.2, 0.1, 0.2, -0.0, 0.1, -0.1, -0.0, 0.1, 0.1, -0.0, -0.1, -0.1, 0.1},
{0.1, -0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.1, 0.1, -0.2, -0.0, 0.1, -0.2, 0.0, 0.2, -0.0, -0.2, 0.2, 0.1, -0.1, -0.0, -0.1, 0.1, 0.0},
}
);
Matrix  transformer_layers_12_feed_forward_linear1_bias   (
{-0.1, 0.1, -0.1, -0.1, 0.0, -0.1, 0.0, -0.1, 0.1, 0.2, 0.2, -0.2, 0.1, 0.1, 0.2}
);
Matrix  transformer_layers_12_feed_forward_linear2_weight   (
{{0.1, -0.2, 0.0, 0.2, -0.2, -0.1, 0.2, -0.2, 0.1, 0.1, 0.1, 0.2, -0.1, 0.2, 0.0},
{0.0, -0.2, -0.1, 0.3, 0.2, 0.1, 0.3, -0.2, -0.1, 0.2, 0.2, -0.0, 0.1, 0.2, 0.1},
{-0.0, -0.0, -0.1, 0.2, 0.1, 0.1, -0.2, 0.2, 0.2, -0.0, -0.3, 0.1, -0.2, 0.1, 0.0},
{0.1, -0.1, -0.1, -0.2, -0.2, -0.0, 0.1, 0.2, 0.1, 0.2, 0.2, -0.1, -0.1, 0.1, 0.3},
{-0.2, -0.2, -0.1, -0.1, -0.1, -0.1, 0.1, -0.2, 0.1, 0.2, -0.1, 0.0, -0.1, 0.0, 0.1},
{0.1, -0.2, -0.2, 0.2, 0.2, -0.0, 0.0, 0.2, -0.0, 0.1, 0.0, -0.1, -0.1, 0.2, -0.0},
{0.2, 0.1, -0.0, -0.2, 0.0, -0.0, -0.1, -0.0, -0.1, 0.1, -0.2, -0.2, -0.1, -0.0, -0.1},
{0.2, -0.3, -0.1, -0.1, 0.2, -0.0, 0.0, 0.1, -0.2, 0.0, 0.1, 0.2, 0.2, -0.2, 0.0},
{0.3, -0.0, -0.2, 0.2, -0.1, 0.0, -0.1, 0.2, 0.2, 0.2, -0.2, 0.2, 0.0, 0.2, 0.2},
{-0.1, 0.2, 0.2, -0.1, 0.0, -0.1, 0.1, 0.2, 0.0, -0.2, -0.0, -0.1, 0.1, 0.2, 0.2},
{-0.0, -0.2, -0.2, 0.1, -0.2, -0.2, -0.1, 0.2, 0.1, -0.1, 0.1, 0.1, 0.2, 0.0, 0.1},
{0.2, 0.1, -0.2, -0.1, 0.1, 0.1, -0.0, 0.2, -0.0, -0.1, 0.2, -0.1, 0.1, -0.2, 0.1},
{0.0, -0.2, 0.1, 0.1, 0.1, -0.0, 0.0, 0.3, -0.3, -0.3, 0.0, 0.0, 0.0, -0.0, 0.2},
{-0.0, 0.1, 0.2, -0.1, -0.0, 0.1, 0.0, -0.0, -0.2, -0.0, 0.0, -0.2, -0.0, 0.2, 0.2},
{0.2, -0.1, -0.2, -0.1, -0.2, -0.2, 0.1, -0.0, 0.0, -0.1, -0.2, 0.2, 0.2, 0.0, -0.3},
{-0.2, 0.0, -0.1, 0.2, 0.1, 0.0, -0.1, 0.2, 0.1, 0.2, 0.1, -0.2, 0.1, 0.0, -0.0},
{0.0, 0.0, -0.0, -0.1, 0.0, -0.2, -0.2, 0.1, 0.2, -0.1, 0.3, -0.2, 0.1, 0.0, 0.2},
{0.2, -0.1, 0.1, -0.1, 0.2, 0.1, 0.0, -0.1, 0.1, -0.2, -0.2, -0.2, -0.2, -0.2, -0.3},
{0.0, 0.2, 0.0, 0.1, 0.0, 0.2, -0.1, 0.1, -0.2, 0.0, 0.1, -0.2, 0.1, 0.0, -0.0},
{-0.0, -0.0, 0.2, 0.2, 0.2, -0.3, 0.1, -0.2, -0.1, 0.1, 0.1, -0.2, 0.2, -0.2, 0.1},
{0.2, 0.1, -0.2, -0.1, -0.2, 0.0, -0.2, -0.0, -0.2, -0.2, 0.3, 0.1, 0.2, 0.0, 0.2},
{0.0, 0.1, 0.1, 0.1, -0.2, -0.1, 0.2, -0.1, 0.1, 0.1, 0.1, 0.0, -0.1, -0.1, -0.1},
{-0.1, 0.1, -0.2, -0.2, 0.1, -0.2, 0.0, -0.0, -0.0, 0.2, -0.0, -0.1, 0.2, -0.1, 0.2},
{-0.0, -0.1, 0.1, -0.1, -0.1, 0.1, 0.0, 0.0, -0.2, 0.2, -0.0, -0.0, 0.2, -0.0, -0.0},
{-0.1, -0.1, 0.1, 0.0, -0.2, 0.2, -0.1, -0.2, 0.1, -0.0, -0.0, 0.2, -0.1, -0.0, -0.2},
}
);
Matrix  transformer_layers_12_feed_forward_linear2_bias   (
{0.0, 0.2, -0.2, 0.2, -0.2, -0.1, 0.1, 0.1, -0.1, -0.2, 0.3, 0.1, 0.0, -0.3, -0.1, -0.1, -0.2, -0.1, 0.2, 0.2, -0.1, -0.2, -0.0, -0.2, -0.0}
);
Matrix  transformer_layers_12_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_12_feed_forward_ln1_layer_norm_bias   (
{0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0}
);
Matrix  transformer_layers_12_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_12_feed_forward_ln2_layer_norm_bias   (
{0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, -0.0}
);
Matrix  transformer_layers_13_attention_query_weight   (
{{0.1, 0.1, -0.2, -0.0, -0.1, 0.1, -0.0, -0.1, -0.1, 0.1, -0.1, 0.0, 0.1, 0.1, -0.1, 0.1, -0.2, -0.1, -0.1, 0.0, 0.1, -0.1, -0.0, 0.1, 0.1},
{-0.2, -0.0, 0.2, 0.1, -0.1, 0.2, 0.0, -0.1, -0.0, -0.0, -0.2, -0.1, 0.1, -0.1, 0.1, 0.2, 0.0, 0.0, 0.0, -0.2, -0.0, 0.1, 0.1, -0.0, 0.1},
{-0.0, 0.2, -0.0, -0.0, 0.1, 0.0, 0.1, 0.1, -0.2, -0.1, 0.2, 0.1, -0.1, -0.0, -0.0, 0.1, -0.0, 0.1, 0.1, -0.1, -0.1, -0.2, -0.2, 0.1, -0.1},
{0.1, 0.1, -0.1, 0.1, -0.1, 0.0, -0.1, 0.1, 0.0, 0.1, -0.1, 0.2, -0.0, -0.2, 0.2, -0.2, -0.1, -0.1, 0.2, 0.2, 0.1, 0.2, 0.2, -0.1, -0.2},
{0.0, 0.1, 0.1, -0.0, -0.1, 0.0, -0.1, -0.1, 0.2, 0.1, 0.2, -0.1, 0.0, -0.1, 0.0, -0.0, -0.2, -0.0, 0.1, 0.1, 0.2, 0.2, 0.1, 0.2, -0.1},
{-0.2, 0.1, 0.0, 0.1, 0.1, -0.0, -0.0, 0.1, -0.2, 0.0, -0.2, -0.1, 0.1, 0.1, 0.2, 0.1, -0.1, -0.1, 0.2, 0.1, 0.1, -0.2, -0.0, 0.1, -0.1},
{0.1, -0.0, 0.0, 0.1, 0.1, -0.2, -0.1, 0.1, 0.0, 0.0, -0.1, 0.0, 0.0, 0.2, -0.1, -0.2, -0.0, -0.2, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, 0.0},
{0.1, -0.1, -0.1, 0.1, 0.0, 0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, -0.0, 0.2, 0.1, 0.1, -0.1, -0.1, 0.1, 0.2, -0.1, 0.1, 0.2, 0.1, -0.1},
{-0.2, -0.0, -0.1, -0.1, 0.1, 0.1, -0.0, -0.2, -0.0, -0.1, 0.0, -0.1, -0.2, -0.2, -0.2, -0.0, 0.1, -0.2, -0.0, 0.0, -0.1, 0.2, -0.2, -0.1, 0.1},
{0.1, -0.1, -0.0, 0.1, 0.1, 0.0, -0.1, 0.1, 0.2, -0.0, 0.1, -0.2, -0.0, 0.0, -0.1, -0.2, 0.0, -0.0, -0.1, -0.1, -0.1, 0.2, -0.1, -0.1, 0.1},
{0.1, -0.2, -0.1, 0.1, 0.0, -0.1, -0.1, -0.2, 0.0, -0.2, -0.1, 0.0, -0.1, 0.1, -0.2, 0.1, 0.2, 0.2, -0.1, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1},
{0.1, -0.2, -0.2, -0.1, 0.2, 0.0, -0.0, -0.1, -0.0, 0.0, 0.1, 0.1, 0.2, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, -0.1, 0.1, -0.1, -0.2, -0.2, -0.0},
{-0.1, 0.0, -0.0, -0.1, 0.2, 0.2, 0.1, 0.1, 0.1, -0.1, 0.0, 0.2, -0.1, 0.1, 0.1, 0.0, -0.1, -0.2, 0.1, -0.1, -0.1, 0.2, -0.2, -0.2, 0.0},
{0.1, 0.1, 0.2, 0.1, 0.0, 0.2, 0.2, 0.1, 0.1, 0.2, -0.0, 0.2, -0.0, 0.1, -0.1, -0.2, 0.1, 0.0, 0.0, 0.2, 0.2, 0.1, -0.2, -0.1, 0.2},
{-0.1, -0.0, 0.1, -0.1, 0.0, 0.1, 0.0, 0.1, -0.2, -0.1, 0.0, 0.1, -0.0, 0.1, -0.1, -0.1, -0.1, -0.2, 0.1, -0.2, -0.2, -0.1, -0.2, -0.0, 0.2},
{-0.1, -0.1, -0.0, 0.2, -0.1, 0.2, 0.1, -0.1, 0.2, -0.1, 0.2, -0.0, -0.1, 0.1, -0.0, 0.1, 0.2, 0.2, -0.2, 0.2, 0.1, -0.1, 0.1, -0.0, 0.1},
{0.2, 0.1, -0.0, 0.1, 0.1, 0.2, 0.0, 0.2, 0.1, 0.0, -0.1, 0.2, 0.1, 0.2, -0.2, 0.1, -0.1, 0.1, -0.2, 0.0, -0.1, 0.0, -0.2, 0.1, -0.1},
{-0.0, 0.1, -0.1, -0.1, 0.1, -0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, -0.1, 0.1, -0.1, -0.2, 0.1, 0.2, 0.0, 0.0, 0.1, -0.2, 0.0, 0.1, -0.0},
{-0.2, 0.2, -0.0, 0.0, 0.2, 0.1, 0.0, 0.1, 0.0, -0.2, -0.1, -0.2, 0.1, 0.2, 0.2, 0.1, 0.0, 0.0, 0.1, -0.2, -0.1, -0.2, 0.0, -0.0, -0.1},
{-0.2, 0.1, 0.1, 0.2, 0.1, 0.1, -0.2, 0.0, -0.0, 0.1, -0.2, 0.1, 0.1, 0.1, 0.2, -0.1, -0.0, 0.1, -0.2, 0.1, 0.1, 0.1, 0.0, -0.0, -0.1},
{-0.1, 0.2, -0.0, 0.2, 0.1, -0.1, -0.2, 0.1, -0.2, 0.1, -0.0, -0.0, 0.1, -0.0, -0.0, -0.0, 0.1, 0.1, 0.0, -0.2, 0.1, -0.0, 0.1, 0.2, 0.1},
{0.2, 0.1, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1, 0.0, -0.1, 0.1, -0.2, 0.0, -0.0, 0.2, -0.2, 0.2, 0.0, 0.1, 0.1},
{-0.0, 0.2, 0.1, 0.1, -0.1, -0.1, 0.0, 0.1, -0.1, 0.2, -0.0, -0.2, 0.1, 0.0, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.0, -0.1, -0.2, 0.1, 0.1},
{-0.2, 0.0, 0.0, 0.1, -0.0, -0.0, 0.1, 0.1, 0.1, 0.2, -0.1, -0.1, 0.0, 0.2, -0.2, -0.2, -0.2, -0.1, 0.2, -0.1, -0.0, -0.1, -0.1, 0.1, 0.1},
{0.1, -0.1, 0.1, -0.0, -0.2, 0.0, 0.2, 0.1, 0.1, 0.0, 0.2, 0.0, 0.1, -0.1, -0.1, -0.1, 0.0, -0.2, 0.1, 0.1, 0.1, 0.1, 0.2, -0.1, 0.0},
}
);
Matrix  transformer_layers_13_attention_query_bias   (
{-0.1, 0.1, -0.1, -0.1, 0.1, -0.2, 0.2, 0.1, 0.0, -0.1, -0.1, 0.1, -0.2, -0.1, -0.1, 0.0, -0.1, -0.1, -0.2, -0.1, 0.0, -0.0, 0.0, 0.1, -0.2}
);
Matrix  transformer_layers_13_attention_key_weight   (
{{-0.1, -0.1, -0.0, -0.0, -0.0, 0.1, -0.2, -0.2, 0.2, 0.1, 0.2, 0.2, 0.2, -0.2, 0.2, 0.2, 0.0, 0.1, -0.2, 0.1, -0.2, 0.0, -0.0, 0.1, 0.1},
{-0.2, 0.1, 0.1, -0.1, -0.2, 0.1, 0.1, 0.2, -0.2, -0.0, 0.0, 0.0, -0.0, -0.0, -0.1, 0.2, -0.2, 0.1, -0.2, 0.0, -0.0, 0.2, 0.1, -0.1, 0.2},
{-0.1, -0.0, 0.0, -0.1, -0.2, -0.2, 0.1, -0.2, 0.1, -0.2, 0.1, 0.1, -0.2, -0.2, 0.1, 0.1, 0.0, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.2, -0.1},
{0.0, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.0, -0.1, 0.2, -0.0, -0.2, -0.1, -0.0, 0.0, -0.0, 0.1, -0.0, -0.1, 0.1, -0.1, -0.2, 0.0, 0.2, 0.1},
{-0.2, 0.1, -0.1, -0.1, -0.1, 0.1, -0.1, -0.2, -0.2, 0.1, 0.1, -0.1, -0.2, 0.0, 0.1, 0.1, 0.0, 0.2, -0.0, -0.1, -0.0, 0.1, -0.1, 0.0, 0.2},
{0.2, 0.0, 0.2, 0.1, -0.1, 0.2, 0.2, 0.2, 0.2, -0.1, -0.1, 0.2, 0.1, 0.0, -0.0, 0.0, 0.2, 0.1, 0.1, -0.0, -0.2, -0.1, -0.2, 0.1, 0.1},
{-0.2, 0.0, -0.2, -0.1, 0.1, 0.1, -0.1, -0.2, -0.1, 0.1, -0.1, 0.1, -0.1, -0.0, 0.1, -0.2, -0.0, 0.2, 0.2, 0.2, -0.1, -0.0, 0.0, -0.1, 0.1},
{-0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.0, 0.0, 0.0, -0.2, -0.2, 0.1, -0.1, 0.1, 0.2, -0.0, -0.0, 0.2, 0.1, 0.1, -0.0, 0.1, -0.1, 0.0},
{0.1, -0.1, -0.2, -0.1, 0.0, -0.1, -0.1, 0.1, 0.2, -0.0, -0.0, -0.0, -0.2, 0.2, 0.2, -0.0, -0.1, 0.0, -0.2, -0.1, -0.0, -0.1, -0.1, -0.0, -0.2},
{-0.0, 0.1, 0.1, 0.2, 0.1, -0.2, -0.1, -0.1, 0.2, 0.1, -0.0, 0.0, 0.2, 0.1, 0.0, -0.1, -0.1, 0.2, -0.1, 0.0, -0.1, -0.0, -0.1, 0.1, 0.1},
{0.0, 0.1, -0.0, 0.1, -0.0, 0.0, -0.0, -0.2, -0.0, -0.2, 0.1, -0.2, 0.1, 0.1, -0.0, -0.1, -0.2, -0.2, 0.1, -0.1, -0.1, 0.2, -0.1, -0.2, -0.1},
{-0.1, -0.2, 0.2, 0.1, -0.1, 0.2, 0.1, 0.2, -0.0, 0.1, 0.1, 0.1, -0.0, -0.1, 0.2, -0.0, 0.2, 0.1, 0.2, -0.2, 0.0, 0.2, -0.0, 0.1, -0.0},
{-0.2, -0.1, 0.0, -0.1, 0.1, -0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.0, 0.2, 0.1, 0.0, -0.2, -0.1, 0.1, 0.1, -0.1, -0.0, -0.2, -0.1, 0.2, 0.2},
{-0.0, 0.1, 0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.2, -0.1, 0.0, 0.0, 0.1, -0.0, -0.1, 0.2, -0.1, 0.0, 0.1, 0.2, 0.1, -0.2, 0.2, -0.1, -0.2},
{-0.0, 0.2, -0.1, 0.0, -0.0, -0.1, 0.1, 0.1, 0.1, -0.2, -0.1, -0.1, 0.0, -0.0, -0.2, 0.1, 0.1, 0.0, -0.0, 0.1, 0.1, 0.0, -0.0, 0.1, 0.0},
{0.1, -0.1, 0.1, -0.0, -0.0, 0.2, -0.0, 0.0, -0.1, 0.0, -0.2, 0.2, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.1, -0.1, -0.1, -0.1, -0.0, -0.1, 0.0},
{-0.0, -0.1, 0.1, 0.2, -0.1, -0.1, 0.0, -0.0, -0.1, -0.2, 0.2, -0.1, 0.0, 0.2, -0.2, -0.0, -0.2, 0.0, -0.0, -0.1, 0.2, -0.0, 0.0, -0.1, -0.0},
{0.2, -0.1, -0.0, -0.1, 0.1, -0.0, -0.0, 0.0, 0.0, 0.0, 0.2, 0.0, -0.0, -0.1, -0.1, -0.0, 0.0, 0.2, 0.1, 0.1, 0.0, -0.1, -0.0, -0.1, 0.1},
{-0.0, -0.1, -0.0, -0.2, -0.0, -0.1, 0.0, -0.1, -0.0, 0.1, -0.0, -0.2, 0.0, -0.1, -0.2, 0.1, 0.1, -0.2, -0.0, -0.0, 0.2, -0.1, -0.2, -0.0, -0.0},
{-0.1, 0.0, -0.0, 0.2, 0.2, 0.1, -0.2, -0.0, -0.2, 0.1, 0.2, -0.2, 0.1, -0.0, 0.1, -0.1, 0.0, 0.1, -0.1, -0.1, -0.2, 0.1, 0.0, -0.2, 0.1},
{-0.0, 0.0, 0.1, 0.1, -0.0, -0.0, 0.1, -0.0, 0.0, -0.1, -0.1, -0.0, -0.1, -0.1, 0.0, 0.1, -0.2, -0.0, 0.2, -0.2, -0.1, -0.1, 0.1, 0.2, 0.1},
{0.1, 0.1, 0.0, -0.0, -0.2, 0.0, -0.1, 0.1, -0.2, 0.0, -0.1, -0.1, 0.2, 0.0, 0.1, -0.1, -0.0, 0.1, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.2},
{0.0, -0.2, 0.0, 0.1, -0.1, 0.1, -0.2, -0.2, 0.1, -0.2, -0.1, -0.1, -0.1, -0.0, 0.2, 0.1, 0.0, 0.1, -0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0},
{0.2, -0.2, 0.2, 0.2, -0.0, 0.1, 0.1, -0.0, -0.1, -0.1, -0.1, 0.2, 0.0, 0.0, 0.0, 0.1, -0.2, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1},
{-0.1, -0.2, 0.1, -0.2, -0.0, -0.2, -0.0, 0.1, 0.0, -0.0, -0.1, 0.0, 0.2, 0.2, 0.1, -0.0, 0.0, 0.1, 0.0, -0.2, -0.2, 0.2, -0.1, -0.2, -0.2},
}
);
Matrix  transformer_layers_13_attention_key_bias   (
{0.1, 0.1, -0.2, -0.1, -0.0, 0.0, 0.1, -0.2, -0.1, -0.2, -0.1, 0.0, -0.1, -0.1, 0.2, 0.1, 0.1, -0.1, -0.0, -0.1, -0.2, -0.1, -0.2, -0.1, 0.0}
);
Matrix  transformer_layers_13_attention_value_weight   (
{{-0.0, -0.1, 0.0, -0.1, -0.1, -0.1, 0.0, -0.0, 0.1, -0.1, 0.0, 0.1, 0.1, 0.0, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.1, -0.2, 0.2, -0.0},
{-0.0, 0.1, 0.2, -0.2, -0.1, 0.1, 0.1, -0.2, -0.2, 0.1, 0.1, 0.2, 0.1, 0.2, -0.1, 0.0, 0.2, 0.1, 0.2, -0.1, 0.0, 0.2, -0.0, 0.1, 0.1},
{-0.1, 0.2, 0.0, 0.1, -0.1, 0.2, -0.1, -0.1, 0.0, 0.0, -0.2, 0.2, 0.2, 0.0, -0.2, -0.1, -0.1, -0.1, -0.2, 0.2, -0.0, 0.1, -0.1, 0.0, -0.0},
{-0.0, 0.2, -0.2, 0.0, 0.1, -0.2, 0.1, -0.2, -0.2, -0.2, -0.1, -0.0, 0.1, -0.0, -0.2, 0.1, -0.2, 0.1, 0.0, -0.0, -0.1, 0.0, 0.1, -0.2, -0.1},
{0.1, 0.0, 0.0, -0.2, -0.2, -0.1, -0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.0, -0.2, 0.0, 0.1, 0.1, 0.1, -0.2, -0.2, -0.0, 0.2, -0.2, 0.1, -0.1},
{0.2, -0.0, 0.0, -0.1, -0.1, 0.2, 0.1, -0.2, 0.1, 0.2, -0.0, -0.0, 0.2, -0.1, -0.0, 0.1, 0.0, 0.2, 0.1, -0.0, -0.1, -0.1, 0.1, -0.1, -0.1},
{0.1, 0.1, 0.1, -0.2, 0.1, -0.1, -0.0, -0.1, -0.1, -0.0, 0.2, -0.1, 0.2, -0.2, 0.0, 0.1, -0.2, 0.1, 0.0, -0.1, -0.1, -0.1, 0.2, 0.2, 0.0},
{0.1, -0.2, -0.1, -0.1, -0.1, 0.2, -0.1, -0.1, 0.0, -0.1, 0.2, 0.2, -0.1, 0.1, -0.1, -0.0, -0.1, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.2, 0.2},
{0.1, -0.1, -0.2, 0.0, 0.0, -0.2, 0.1, -0.1, 0.1, 0.1, 0.0, -0.1, -0.2, 0.2, 0.1, -0.1, -0.1, -0.2, 0.1, -0.1, 0.0, 0.1, 0.1, -0.1, -0.0},
{0.2, 0.1, 0.0, 0.1, 0.2, 0.0, -0.1, 0.0, -0.0, 0.2, -0.0, 0.1, -0.1, 0.1, -0.2, 0.1, -0.1, -0.0, 0.2, -0.1, -0.0, -0.1, 0.0, -0.1, -0.1},
{-0.1, -0.1, 0.1, 0.2, -0.0, -0.0, 0.1, -0.2, -0.1, 0.2, 0.1, 0.1, -0.0, -0.1, -0.2, 0.2, -0.1, 0.2, 0.0, -0.0, 0.1, -0.1, -0.1, -0.2, 0.2},
{0.1, 0.2, -0.2, 0.2, 0.1, 0.1, 0.0, -0.1, -0.0, 0.0, 0.2, -0.2, 0.2, 0.1, -0.1, 0.0, 0.1, 0.2, -0.1, -0.1, 0.1, -0.1, 0.1, -0.0, 0.2},
{0.2, -0.1, -0.0, 0.1, -0.0, -0.2, 0.2, 0.1, 0.1, -0.2, -0.0, -0.1, 0.0, 0.2, -0.0, -0.1, -0.2, 0.1, -0.1, 0.1, -0.2, 0.2, 0.1, -0.1, -0.1},
{0.1, -0.2, 0.1, 0.1, 0.1, -0.0, -0.2, -0.1, -0.0, -0.0, -0.2, 0.2, -0.2, -0.0, 0.1, -0.1, -0.2, -0.1, 0.1, 0.0, 0.1, -0.2, -0.1, -0.2, -0.2},
{-0.1, 0.1, 0.0, -0.0, -0.1, -0.1, -0.2, 0.1, -0.1, 0.0, 0.2, -0.2, 0.0, -0.1, 0.1, 0.0, 0.0, 0.1, 0.2, 0.0, -0.2, -0.1, 0.1, 0.1, 0.0},
{-0.1, 0.0, 0.0, -0.2, 0.2, 0.0, 0.1, 0.0, 0.2, -0.0, -0.1, 0.0, 0.0, 0.0, -0.1, -0.1, 0.1, -0.1, -0.0, 0.0, 0.1, -0.0, -0.1, 0.1, -0.1},
{-0.1, 0.2, -0.2, -0.2, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.0, 0.0, -0.2, 0.2, -0.0, 0.2, 0.1},
{0.1, -0.2, -0.2, -0.2, 0.1, 0.1, 0.2, 0.1, -0.1, -0.1, -0.1, 0.0, 0.1, 0.1, 0.1, 0.2, 0.1, -0.1, -0.2, -0.1, 0.0, 0.1, -0.1, 0.2, -0.1},
{0.1, -0.1, -0.0, 0.1, 0.2, -0.1, -0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.2, 0.1, 0.2, 0.1, -0.2, -0.1, -0.1, -0.1, 0.0, 0.1},
{-0.1, -0.1, -0.0, -0.0, -0.0, 0.1, -0.1, 0.1, 0.2, 0.1, 0.0, 0.0, 0.1, 0.0, -0.0, -0.2, -0.2, 0.1, -0.1, -0.2, -0.0, -0.1, 0.1, -0.2, 0.1},
{-0.1, 0.2, -0.2, 0.0, -0.0, -0.0, 0.2, -0.2, 0.0, 0.2, 0.2, 0.2, 0.2, 0.0, 0.1, 0.2, -0.2, 0.2, -0.1, 0.0, 0.2, 0.1, -0.2, 0.1, 0.2},
{0.0, 0.1, 0.2, -0.0, -0.1, 0.2, -0.1, 0.2, -0.1, -0.1, 0.0, -0.1, -0.1, -0.0, -0.1, 0.0, -0.0, -0.2, -0.0, -0.2, 0.2, -0.2, -0.2, 0.1, -0.1},
{0.0, 0.2, -0.1, -0.1, -0.2, -0.1, -0.1, -0.2, -0.1, -0.1, -0.1, -0.1, 0.0, -0.0, -0.0, 0.1, 0.0, -0.0, -0.1, 0.1, -0.1, 0.1, -0.2, 0.1, 0.1},
{0.1, 0.1, 0.1, -0.0, 0.1, -0.0, 0.2, -0.2, 0.2, -0.1, -0.1, 0.0, -0.0, 0.2, -0.1, -0.2, 0.1, -0.1, 0.1, -0.0, 0.0, 0.0, -0.2, -0.0, 0.2},
{-0.1, 0.1, -0.0, 0.0, 0.0, -0.2, 0.1, -0.0, -0.1, 0.1, -0.2, -0.0, -0.1, -0.1, -0.2, -0.1, -0.2, -0.1, -0.1, -0.1, -0.1, 0.0, -0.1, -0.1, -0.1},
}
);
Matrix  transformer_layers_13_attention_value_bias   (
{-0.2, 0.2, 0.0, 0.1, 0.1, -0.1, 0.0, -0.0, -0.1, 0.1, -0.0, 0.1, 0.1, 0.2, -0.2, 0.1, -0.1, -0.1, -0.2, -0.1, -0.1, 0.1, 0.1, -0.1, -0.2}
);
Matrix  transformer_layers_13_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_13_norm1_layer_norm_bias   (
{0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_13_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_13_norm2_layer_norm_bias   (
{0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_13_feed_forward_linear1_weight   (
{{0.1, -0.0, -0.1, 0.1, 0.0, -0.0, 0.1, 0.0, -0.0, 0.1, 0.1, 0.1, 0.1, 0.1, -0.2, -0.2, -0.1, 0.2, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.0},
{0.2, 0.1, -0.0, -0.1, 0.1, 0.1, 0.1, 0.1, 0.2, -0.2, -0.0, -0.0, 0.0, 0.1, 0.1, -0.2, 0.0, 0.1, 0.1, 0.2, -0.2, -0.1, 0.2, 0.2, 0.1},
{0.0, -0.1, -0.2, 0.1, -0.1, -0.2, 0.2, 0.0, -0.0, 0.2, -0.1, 0.2, -0.1, 0.1, -0.2, -0.0, 0.1, -0.1, 0.1, -0.1, 0.2, -0.1, -0.0, -0.1, -0.0},
{0.2, 0.2, -0.1, 0.1, -0.0, -0.0, 0.1, -0.1, -0.1, 0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.1, 0.0, -0.2, 0.0, -0.1, 0.1, -0.2, 0.0, 0.1, -0.0},
{0.1, -0.2, -0.1, -0.2, -0.1, 0.1, -0.2, 0.1, -0.0, -0.1, 0.1, 0.1, -0.1, -0.0, 0.2, -0.1, 0.1, 0.1, 0.2, -0.2, -0.1, -0.2, 0.0, 0.1, 0.0},
{0.2, -0.1, -0.2, -0.2, -0.0, 0.1, 0.1, -0.1, -0.1, -0.0, -0.1, -0.2, -0.1, -0.0, 0.1, -0.1, 0.2, 0.2, -0.2, -0.1, -0.1, 0.2, -0.2, -0.1, -0.1},
{0.2, -0.2, -0.2, 0.1, -0.2, 0.0, 0.1, 0.0, -0.1, 0.1, 0.1, -0.0, 0.1, -0.1, -0.1, -0.2, 0.1, -0.0, 0.1, -0.2, -0.2, -0.1, 0.1, -0.2, 0.1},
{0.1, -0.1, 0.2, -0.1, -0.1, 0.2, -0.0, -0.1, 0.1, -0.1, -0.0, 0.1, -0.0, 0.1, 0.1, 0.2, -0.2, 0.2, 0.0, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1},
{0.1, 0.2, 0.1, -0.1, -0.0, 0.2, -0.0, 0.0, 0.2, -0.0, 0.1, 0.2, 0.0, 0.1, -0.2, 0.2, 0.1, 0.1, -0.1, -0.2, -0.1, 0.0, 0.0, 0.2, -0.0},
{-0.1, 0.2, 0.0, -0.1, -0.0, 0.1, -0.0, 0.0, -0.1, -0.1, 0.2, -0.2, -0.2, -0.2, 0.1, 0.0, 0.1, -0.1, 0.1, 0.2, -0.0, -0.0, -0.0, -0.1, -0.0},
{0.1, 0.1, -0.1, 0.1, -0.2, -0.0, 0.1, 0.1, -0.0, 0.1, -0.1, 0.0, -0.1, -0.1, 0.1, -0.1, 0.1, 0.2, -0.1, 0.2, 0.2, 0.1, -0.1, 0.0, 0.0},
{-0.2, -0.1, -0.1, 0.2, -0.1, -0.2, -0.1, -0.2, 0.0, 0.1, 0.1, 0.2, 0.0, 0.1, 0.1, 0.2, -0.0, 0.1, 0.1, -0.1, -0.0, -0.0, -0.1, -0.1, 0.2},
{0.0, -0.0, -0.1, 0.0, -0.1, 0.0, -0.2, 0.2, 0.2, -0.2, -0.2, -0.2, -0.1, 0.2, 0.0, 0.0, 0.2, 0.2, -0.1, 0.1, -0.1, 0.0, 0.1, -0.2, -0.2},
{-0.1, -0.1, -0.1, 0.2, -0.1, 0.1, 0.2, -0.2, 0.0, 0.1, 0.1, -0.1, 0.2, -0.2, -0.1, 0.0, -0.2, -0.1, -0.0, -0.1, -0.2, 0.1, 0.1, 0.1, 0.1},
{0.1, -0.1, 0.0, -0.0, -0.1, 0.0, -0.0, -0.2, 0.1, 0.1, 0.1, -0.1, -0.2, -0.0, 0.1, -0.0, -0.0, 0.0, -0.1, 0.0, -0.0, -0.0, 0.1, -0.1, 0.2},
}
);
Matrix  transformer_layers_13_feed_forward_linear1_bias   (
{-0.1, 0.2, 0.2, -0.1, 0.1, 0.0, -0.1, 0.1, -0.1, 0.0, 0.2, -0.2, 0.2, -0.0, 0.1}
);
Matrix  transformer_layers_13_feed_forward_linear2_weight   (
{{-0.1, -0.0, -0.1, -0.1, 0.1, -0.2, -0.3, 0.2, 0.0, 0.1, -0.2, 0.3, -0.2, -0.1, 0.2},
{-0.1, 0.3, 0.1, -0.1, 0.2, 0.1, 0.1, -0.0, 0.1, 0.0, 0.1, 0.2, 0.2, -0.2, 0.2},
{-0.1, 0.2, 0.2, 0.1, -0.1, 0.0, -0.0, -0.1, 0.0, -0.1, 0.0, 0.0, -0.0, 0.2, -0.2},
{0.1, 0.2, -0.1, 0.1, -0.0, 0.2, 0.1, 0.1, -0.1, -0.2, 0.1, -0.2, -0.2, -0.0, -0.0},
{-0.2, 0.0, 0.2, 0.2, -0.2, -0.2, -0.1, 0.2, 0.2, -0.3, 0.2, 0.0, 0.0, 0.1, 0.0},
{-0.1, 0.1, 0.1, -0.1, -0.2, 0.1, 0.0, 0.2, -0.1, 0.1, -0.2, 0.1, -0.3, 0.0, 0.2},
{0.1, 0.1, 0.1, -0.1, -0.2, -0.0, -0.0, 0.1, 0.0, 0.1, -0.1, 0.0, 0.0, -0.2, 0.2},
{-0.3, -0.2, -0.1, -0.3, 0.1, 0.2, 0.1, 0.2, 0.3, 0.1, -0.1, -0.2, -0.1, 0.1, 0.2},
{-0.1, 0.1, -0.1, -0.2, 0.1, 0.0, -0.2, -0.2, 0.2, -0.2, -0.1, -0.1, -0.2, 0.1, -0.0},
{-0.2, 0.1, 0.0, 0.1, -0.0, 0.1, 0.1, -0.2, -0.0, 0.2, 0.2, 0.0, -0.2, -0.1, 0.1},
{0.2, -0.2, -0.1, -0.1, -0.2, 0.2, 0.2, -0.1, -0.1, -0.0, 0.0, 0.1, 0.2, -0.0, 0.1},
{0.1, -0.2, -0.1, -0.2, -0.1, 0.1, 0.2, -0.1, 0.1, -0.0, 0.2, 0.2, 0.1, -0.1, -0.1},
{-0.2, 0.2, 0.2, 0.1, 0.0, 0.2, -0.0, 0.2, 0.0, 0.2, 0.0, 0.2, -0.2, -0.2, -0.1},
{0.2, -0.3, 0.1, -0.0, 0.2, -0.1, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, -0.1, -0.2, -0.1},
{-0.0, 0.0, -0.2, -0.1, -0.0, 0.2, -0.3, 0.2, -0.2, -0.2, -0.1, 0.2, -0.1, 0.0, -0.2},
{0.2, -0.1, 0.1, -0.1, 0.2, -0.2, 0.2, 0.0, -0.0, 0.0, -0.2, 0.1, 0.2, -0.3, -0.2},
{-0.1, 0.1, -0.0, 0.3, 0.2, -0.2, 0.3, -0.1, -0.1, -0.1, -0.2, 0.1, -0.2, 0.0, 0.2},
{-0.1, -0.0, 0.1, 0.1, 0.2, -0.1, -0.1, 0.2, 0.1, 0.2, -0.1, -0.0, 0.1, 0.1, -0.2},
{0.2, -0.0, -0.1, -0.0, 0.2, 0.2, 0.0, -0.2, -0.1, 0.2, -0.1, 0.3, 0.0, -0.1, -0.2},
{-0.0, 0.1, -0.1, -0.1, 0.2, 0.0, 0.3, 0.0, -0.1, 0.2, 0.1, 0.1, 0.1, -0.0, 0.2},
{-0.2, -0.1, 0.0, -0.1, -0.3, 0.2, 0.1, -0.2, -0.1, -0.2, -0.2, 0.2, -0.2, -0.0, -0.2},
{0.2, 0.2, 0.1, -0.3, -0.0, 0.2, -0.0, 0.2, -0.1, 0.1, 0.0, -0.0, 0.1, 0.1, 0.1},
{-0.1, 0.1, 0.1, -0.2, -0.1, -0.2, -0.0, -0.0, -0.1, 0.0, -0.1, 0.1, 0.2, -0.1, -0.2},
{0.1, 0.1, -0.2, 0.2, 0.0, 0.2, -0.1, 0.2, 0.2, 0.2, -0.0, 0.2, 0.0, 0.1, 0.1},
{0.1, -0.2, 0.0, 0.1, 0.2, -0.1, 0.1, -0.2, -0.2, 0.2, -0.2, 0.2, 0.2, -0.2, 0.2},
}
);
Matrix  transformer_layers_13_feed_forward_linear2_bias   (
{0.0, 0.0, -0.0, -0.3, -0.1, -0.1, -0.1, -0.1, -0.0, 0.1, 0.2, 0.2, 0.2, 0.0, 0.1, 0.2, -0.2, -0.2, 0.1, 0.1, 0.2, 0.2, 0.2, -0.2, -0.1}
);
Matrix  transformer_layers_13_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_13_feed_forward_ln1_layer_norm_bias   (
{-0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0}
);
Matrix  transformer_layers_13_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_13_feed_forward_ln2_layer_norm_bias   (
{0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_14_attention_query_weight   (
{{0.1, 0.2, -0.0, -0.1, -0.1, 0.1, 0.1, -0.2, -0.1, -0.0, 0.1, -0.1, 0.1, -0.2, 0.1, 0.1, -0.1, 0.2, 0.0, -0.1, 0.1, -0.2, 0.1, 0.1, 0.2},
{0.2, 0.1, 0.2, 0.1, -0.2, 0.1, -0.1, -0.1, -0.1, 0.2, 0.1, -0.1, 0.0, -0.0, -0.1, 0.1, 0.0, 0.1, -0.1, 0.1, 0.1, -0.0, 0.2, -0.1, 0.1},
{-0.1, 0.1, 0.0, -0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.1, 0.2, 0.2, -0.2, -0.1, 0.1, 0.1, -0.0, 0.2, 0.0, 0.1, 0.0, -0.0, -0.1, -0.1, 0.1},
{-0.2, -0.1, -0.0, 0.1, -0.1, 0.1, -0.1, -0.0, -0.1, 0.1, 0.0, -0.1, 0.0, -0.2, -0.0, 0.1, 0.1, 0.1, 0.1, -0.1, 0.2, 0.1, -0.0, -0.1, -0.1},
{0.0, 0.2, 0.1, 0.1, 0.0, 0.1, -0.1, 0.1, -0.0, -0.0, 0.0, -0.2, -0.2, -0.2, 0.2, -0.1, -0.1, 0.1, -0.1, -0.2, -0.2, -0.1, 0.2, -0.1, -0.2},
{-0.1, -0.1, -0.1, 0.0, 0.2, 0.0, -0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.0, -0.1, 0.1, -0.0, 0.1, 0.2, 0.2, -0.2, 0.2},
{-0.2, -0.1, -0.1, 0.2, -0.0, 0.0, 0.2, 0.1, -0.1, 0.1, -0.2, -0.0, 0.1, -0.1, 0.0, -0.1, 0.2, -0.1, -0.1, -0.1, -0.0, -0.1, 0.1, -0.2, -0.1},
{-0.1, -0.2, 0.0, -0.0, 0.0, -0.1, -0.0, -0.2, 0.1, 0.1, 0.1, -0.2, 0.0, 0.2, -0.0, 0.0, -0.1, 0.1, -0.0, 0.0, 0.1, 0.1, -0.1, -0.1, 0.0},
{-0.2, 0.1, 0.2, 0.1, -0.1, -0.1, 0.2, 0.2, -0.1, -0.0, 0.1, 0.1, 0.2, 0.2, -0.0, -0.0, -0.0, -0.1, -0.0, 0.1, 0.2, 0.2, -0.1, 0.0, 0.2},
{0.0, -0.0, 0.0, -0.1, -0.1, 0.1, -0.1, 0.2, -0.1, -0.1, -0.1, 0.1, 0.2, -0.0, -0.1, -0.2, 0.2, -0.1, 0.2, 0.0, -0.0, -0.2, -0.1, 0.1, 0.0},
{-0.1, -0.0, -0.1, 0.0, -0.1, 0.1, -0.1, 0.1, -0.2, -0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.0, -0.1, 0.0, 0.1, -0.1, 0.1, 0.2, 0.1},
{0.0, 0.1, -0.0, -0.1, 0.1, -0.1, -0.1, -0.1, -0.0, -0.2, 0.1, 0.0, 0.0, 0.1, 0.1, -0.0, -0.0, -0.1, 0.2, -0.0, 0.0, 0.2, 0.2, -0.1, 0.1},
{0.2, -0.2, -0.2, 0.2, -0.2, -0.2, 0.1, 0.1, 0.0, 0.2, -0.1, -0.2, -0.2, -0.2, 0.1, 0.1, -0.2, 0.0, -0.0, 0.0, -0.2, 0.2, -0.1, 0.0, 0.0},
{0.0, 0.2, -0.0, -0.2, 0.2, 0.1, 0.0, -0.0, -0.1, -0.0, -0.1, 0.2, -0.1, 0.0, -0.1, -0.1, 0.0, 0.1, 0.1, 0.2, -0.0, -0.1, 0.0, 0.0, 0.1},
{0.1, -0.0, 0.1, 0.0, 0.2, 0.2, 0.1, -0.1, 0.1, 0.1, 0.2, -0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.0, 0.2, -0.0, -0.1, -0.1, -0.1, 0.2, 0.0},
{-0.1, 0.1, 0.0, -0.1, 0.0, 0.2, -0.1, 0.1, 0.1, -0.1, 0.2, 0.2, -0.1, -0.0, 0.1, 0.1, 0.0, 0.0, -0.1, 0.2, 0.1, -0.0, -0.0, 0.1, -0.1},
{0.1, 0.1, -0.0, 0.1, -0.0, -0.2, -0.0, -0.0, 0.1, 0.1, 0.1, 0.0, -0.0, 0.0, -0.1, 0.1, -0.1, 0.1, -0.2, -0.1, -0.0, -0.1, 0.0, 0.1, 0.0},
{-0.1, -0.0, -0.1, 0.1, -0.0, 0.1, 0.2, -0.2, -0.1, -0.2, 0.2, 0.1, -0.2, 0.1, 0.1, -0.2, -0.1, 0.1, -0.2, 0.2, 0.0, -0.0, -0.0, 0.2, 0.1},
{0.0, -0.1, 0.1, -0.0, 0.1, -0.1, -0.1, 0.2, 0.2, -0.2, -0.0, -0.0, -0.1, 0.1, -0.2, 0.2, 0.1, 0.1, 0.2, -0.1, -0.1, 0.2, -0.1, 0.0, -0.1},
{0.2, 0.1, 0.1, -0.2, -0.1, -0.0, 0.0, 0.0, 0.1, -0.1, -0.1, 0.1, -0.2, 0.2, 0.0, 0.2, -0.1, 0.1, -0.0, -0.2, -0.1, -0.1, 0.0, 0.0, 0.0},
{0.1, 0.1, 0.0, -0.0, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, -0.0, 0.0, -0.0, 0.1, -0.2, 0.1, 0.2, 0.0, 0.1, 0.2, 0.1, -0.0, 0.2, -0.1, 0.2},
{0.0, 0.0, 0.0, -0.1, -0.0, -0.0, 0.2, -0.1, 0.0, 0.1, -0.2, 0.0, -0.2, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, 0.0, 0.2, -0.0, -0.1, 0.2, 0.2},
{-0.1, 0.0, -0.1, 0.2, 0.1, 0.1, 0.1, 0.1, -0.2, 0.1, 0.2, -0.2, -0.1, 0.1, -0.2, -0.0, 0.2, -0.1, -0.2, 0.1, 0.1, 0.2, 0.1, -0.1, 0.2},
{-0.1, -0.1, -0.0, 0.1, 0.0, -0.2, -0.0, 0.1, 0.0, -0.1, 0.2, -0.2, 0.0, -0.1, -0.2, 0.1, -0.2, -0.0, -0.2, 0.0, 0.0, 0.1, -0.2, 0.1, -0.1},
{0.1, 0.1, -0.0, 0.1, -0.0, -0.1, 0.0, -0.1, -0.0, -0.2, 0.2, -0.1, 0.2, 0.0, -0.2, -0.1, -0.1, -0.0, 0.1, 0.1, -0.0, 0.2, -0.2, 0.1, -0.1},
}
);
Matrix  transformer_layers_14_attention_query_bias   (
{-0.1, -0.2, 0.0, 0.2, -0.1, 0.2, 0.0, -0.1, -0.1, 0.2, -0.2, 0.1, -0.2, -0.1, -0.1, 0.1, -0.1, 0.0, -0.0, -0.1, 0.0, 0.1, 0.2, 0.2, 0.0}
);
Matrix  transformer_layers_14_attention_key_weight   (
{{0.1, 0.0, 0.2, -0.1, -0.1, -0.1, -0.1, 0.1, -0.1, -0.1, -0.0, -0.1, -0.1, -0.1, -0.2, 0.0, -0.0, 0.2, 0.1, -0.1, -0.1, 0.0, 0.1, 0.2, -0.1},
{0.0, 0.1, -0.1, -0.1, -0.0, 0.2, -0.2, 0.2, 0.1, -0.1, 0.1, 0.1, 0.0, -0.2, -0.2, 0.1, -0.2, 0.2, -0.2, -0.1, -0.2, 0.1, 0.1, 0.0, 0.0},
{-0.2, 0.2, 0.2, 0.2, 0.2, 0.1, -0.2, 0.2, 0.1, 0.0, -0.1, 0.1, 0.1, 0.1, -0.1, -0.0, -0.1, -0.2, -0.2, -0.1, 0.0, -0.0, -0.1, -0.2, -0.2},
{0.1, 0.2, 0.1, 0.1, -0.0, -0.2, 0.1, -0.2, -0.1, 0.1, -0.1, 0.1, 0.2, -0.2, 0.0, -0.1, 0.0, 0.1, -0.1, -0.2, -0.0, 0.0, 0.1, -0.0, 0.2},
{-0.1, 0.0, -0.0, -0.1, 0.1, 0.1, -0.2, -0.1, -0.0, 0.0, 0.1, -0.2, -0.1, 0.1, 0.1, 0.0, -0.2, -0.0, 0.0, -0.1, -0.1, -0.1, 0.1, 0.0, 0.1},
{0.1, 0.1, 0.0, 0.2, -0.2, 0.1, -0.0, 0.0, 0.0, -0.2, -0.1, 0.1, 0.1, -0.1, 0.2, -0.1, 0.0, -0.2, -0.0, -0.2, 0.1, -0.1, 0.0, 0.1, 0.1},
{-0.2, -0.1, 0.2, 0.0, 0.0, 0.0, 0.2, -0.2, -0.1, 0.2, -0.1, 0.1, 0.2, -0.1, 0.1, 0.2, -0.0, 0.1, -0.1, -0.0, 0.2, -0.0, 0.2, -0.2, 0.1},
{0.0, -0.0, 0.0, -0.1, -0.2, -0.1, -0.2, -0.0, -0.1, -0.2, 0.1, -0.1, -0.2, -0.1, 0.1, -0.0, -0.1, 0.0, 0.2, -0.2, -0.0, -0.1, 0.1, -0.2, 0.0},
{0.0, 0.2, -0.1, -0.2, 0.1, 0.0, -0.0, -0.1, -0.2, 0.0, -0.0, 0.2, 0.0, 0.1, -0.1, 0.1, 0.2, -0.1, 0.2, 0.1, -0.1, 0.2, -0.0, 0.1, 0.1},
{-0.2, 0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.0, -0.0, -0.1, -0.1, -0.1, 0.1, -0.0, 0.2, 0.2, -0.2, 0.0, 0.1, 0.2, 0.1, -0.1, -0.2, -0.2, -0.1},
{-0.0, 0.0, 0.1, 0.1, -0.1, -0.1, 0.1, 0.2, 0.1, -0.1, -0.0, 0.1, 0.2, -0.0, 0.1, -0.0, -0.1, -0.0, 0.0, -0.2, -0.1, -0.1, 0.1, -0.0, 0.1},
{-0.1, -0.2, 0.1, 0.1, 0.0, -0.2, -0.2, 0.1, 0.0, 0.1, -0.0, -0.0, -0.1, 0.1, 0.2, 0.0, 0.1, -0.2, -0.1, 0.1, 0.2, -0.2, 0.0, -0.0, -0.0},
{0.0, 0.1, -0.1, 0.0, 0.2, 0.2, -0.2, -0.2, -0.1, 0.1, 0.1, -0.0, 0.1, 0.0, -0.1, 0.1, -0.1, -0.1, -0.2, -0.1, 0.2, -0.1, -0.0, -0.1, -0.1},
{0.1, 0.1, -0.2, 0.0, -0.1, -0.2, -0.0, 0.1, -0.0, -0.0, -0.0, 0.0, -0.1, 0.1, -0.0, -0.0, 0.1, -0.2, -0.1, 0.1, -0.1, -0.0, 0.0, -0.1, 0.2},
{0.2, -0.0, 0.1, -0.1, -0.2, -0.2, 0.0, 0.1, -0.2, -0.0, 0.1, 0.0, 0.2, -0.0, 0.0, -0.1, 0.0, -0.1, 0.1, -0.2, -0.2, -0.1, -0.0, 0.1, 0.1},
{0.1, 0.2, -0.1, 0.1, -0.1, 0.0, 0.1, 0.1, -0.0, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.2, 0.2, -0.1, -0.0, -0.1, -0.1, 0.0},
{0.1, -0.2, -0.1, 0.0, 0.0, 0.0, -0.1, -0.2, 0.1, 0.2, 0.1, -0.1, 0.0, -0.1, -0.1, -0.0, 0.1, -0.2, -0.1, 0.0, 0.0, 0.1, -0.0, -0.2, 0.0},
{0.0, 0.1, -0.0, -0.1, -0.1, 0.2, 0.1, 0.0, 0.0, 0.1, -0.1, 0.1, -0.2, 0.1, -0.1, -0.0, 0.1, -0.1, 0.2, 0.1, 0.1, 0.1, -0.2, -0.2, -0.0},
{0.1, -0.0, -0.1, 0.1, -0.1, 0.1, 0.1, 0.2, -0.2, 0.0, 0.1, -0.1, 0.0, -0.1, 0.2, -0.1, -0.1, 0.1, 0.0, 0.0, 0.2, 0.2, -0.0, -0.0, -0.1},
{-0.0, 0.2, -0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.0, 0.1, 0.2, 0.0, 0.1, 0.1, -0.1, 0.2, 0.0, 0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1},
{0.0, 0.0, 0.1, -0.0, -0.1, -0.1, 0.1, -0.0, -0.2, -0.1, -0.0, -0.1, 0.1, -0.0, -0.1, -0.1, 0.0, 0.1, -0.1, 0.0, -0.1, -0.1, 0.2, -0.2, 0.0},
{-0.0, -0.0, 0.1, -0.2, -0.1, -0.0, 0.0, -0.2, -0.0, -0.2, -0.1, -0.1, -0.2, -0.2, -0.1, -0.0, -0.0, 0.1, -0.1, 0.1, 0.1, -0.2, 0.1, 0.1, 0.0},
{-0.2, -0.1, 0.1, -0.1, 0.0, -0.2, 0.0, -0.1, -0.1, 0.0, -0.1, -0.1, 0.1, 0.1, -0.2, -0.0, -0.1, -0.2, 0.0, -0.1, 0.2, -0.1, 0.0, 0.2, -0.1},
{-0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.0, 0.1, 0.1, 0.2, 0.2, -0.1, 0.2, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.0, 0.0, -0.1, -0.1, 0.2, 0.0},
{0.0, 0.1, -0.2, 0.0, 0.1, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.1, 0.1, 0.0, 0.1, 0.0, -0.1, -0.0, -0.1, 0.2, -0.1, 0.1, -0.1, -0.1, 0.1},
}
);
Matrix  transformer_layers_14_attention_key_bias   (
{0.0, 0.2, -0.1, -0.1, 0.1, -0.2, -0.1, 0.1, 0.0, -0.1, -0.0, -0.1, 0.1, -0.1, 0.0, 0.2, 0.2, -0.0, -0.1, 0.0, 0.1, 0.2, -0.2, -0.1, 0.2}
);
Matrix  transformer_layers_14_attention_value_weight   (
{{-0.2, -0.1, -0.1, 0.0, -0.1, 0.1, -0.0, -0.1, 0.1, -0.1, -0.2, -0.1, -0.0, -0.2, 0.0, -0.0, 0.1, -0.0, -0.2, -0.1, 0.1, 0.0, 0.2, 0.1, -0.2},
{-0.0, 0.1, -0.1, 0.1, -0.2, -0.2, 0.1, 0.1, -0.1, -0.1, 0.2, 0.0, 0.2, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.1, 0.0, -0.1, -0.0, 0.0, 0.2},
{-0.2, 0.0, -0.1, 0.1, -0.1, -0.1, 0.2, 0.0, 0.1, 0.1, 0.1, -0.1, -0.2, -0.1, -0.1, 0.0, -0.2, -0.2, -0.1, 0.1, 0.1, -0.0, -0.1, -0.1, -0.2},
{-0.1, 0.0, -0.0, 0.1, -0.2, -0.2, 0.2, 0.1, 0.1, 0.0, -0.1, 0.1, 0.2, 0.1, 0.1, 0.1, -0.0, -0.2, -0.1, 0.1, -0.0, 0.2, -0.2, -0.0, -0.1},
{-0.1, -0.2, -0.2, 0.2, 0.1, -0.0, -0.1, -0.0, -0.1, 0.1, -0.1, -0.2, 0.2, -0.1, -0.0, -0.1, 0.1, -0.0, -0.1, -0.0, -0.1, -0.0, 0.0, -0.1, 0.2},
{0.2, 0.1, -0.1, 0.0, -0.1, 0.1, 0.0, 0.0, 0.1, 0.1, -0.0, -0.0, -0.2, -0.0, -0.1, -0.1, -0.1, -0.1, -0.0, 0.1, 0.0, -0.2, -0.2, 0.2, -0.1},
{0.2, 0.2, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.2, 0.1, 0.1, 0.1, -0.0, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.2, -0.0, 0.1, 0.1, 0.0},
{-0.1, -0.2, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.2, 0.0, -0.1, 0.1, 0.0, -0.1, -0.0, 0.2, 0.1, -0.1, 0.0, 0.1, -0.1, -0.1, 0.1},
{0.0, 0.1, -0.1, 0.0, -0.2, -0.1, 0.1, 0.1, -0.1, 0.2, -0.1, -0.1, 0.1, 0.1, -0.2, 0.2, 0.1, 0.2, 0.0, 0.1, -0.1, -0.1, 0.1, 0.2, -0.0},
{-0.1, -0.0, 0.2, 0.1, -0.1, 0.1, 0.1, -0.1, -0.0, 0.0, -0.2, 0.1, -0.1, 0.1, 0.0, 0.1, 0.1, 0.0, 0.1, -0.1, -0.0, 0.2, -0.2, 0.1, -0.2},
{0.2, 0.1, 0.1, -0.1, -0.2, -0.0, -0.1, -0.2, 0.0, 0.0, -0.2, -0.1, 0.1, 0.0, 0.1, -0.2, 0.2, 0.2, -0.1, 0.0, 0.1, -0.2, -0.0, 0.0, 0.2},
{-0.1, -0.1, -0.0, 0.2, -0.1, -0.0, -0.2, -0.1, 0.1, -0.2, -0.0, -0.2, 0.2, 0.0, 0.1, -0.0, -0.0, -0.1, -0.1, 0.0, -0.1, -0.1, -0.1, 0.1, -0.0},
{-0.2, -0.1, -0.1, 0.1, 0.1, -0.0, 0.1, -0.1, 0.2, 0.2, -0.2, 0.1, -0.0, -0.1, 0.2, 0.2, 0.1, -0.1, -0.1, -0.0, 0.0, 0.1, 0.2, 0.1, -0.1},
{-0.1, -0.1, -0.1, -0.0, 0.1, -0.1, 0.1, -0.0, -0.1, 0.2, 0.1, -0.1, -0.1, 0.0, 0.0, 0.1, 0.1, -0.1, 0.2, 0.1, -0.2, -0.0, -0.1, -0.2, 0.1},
{-0.1, -0.1, -0.0, 0.1, 0.2, -0.0, -0.2, 0.1, 0.2, 0.2, 0.1, 0.0, 0.1, -0.1, -0.1, 0.1, 0.1, 0.2, 0.2, -0.1, -0.1, 0.1, 0.0, -0.1, 0.0},
{0.1, -0.1, -0.2, -0.2, 0.1, -0.2, -0.2, 0.1, 0.1, 0.1, -0.0, 0.1, -0.0, -0.2, -0.2, 0.1, -0.0, -0.2, -0.2, 0.2, -0.1, 0.1, 0.2, -0.2, 0.2},
{0.1, 0.1, -0.1, 0.2, 0.2, 0.0, -0.1, 0.1, -0.1, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, -0.1, 0.2, 0.1, 0.0, 0.0, 0.1, 0.2, 0.0, -0.0, -0.2},
{0.0, 0.0, -0.2, -0.0, 0.1, 0.1, -0.1, 0.0, 0.2, 0.2, 0.2, -0.0, -0.1, -0.1, 0.0, -0.0, 0.1, 0.0, -0.1, 0.2, 0.0, -0.1, 0.2, 0.0, 0.2},
{0.0, -0.1, -0.0, 0.2, 0.1, -0.0, 0.0, -0.1, 0.2, 0.0, -0.2, -0.2, -0.1, 0.1, 0.1, 0.1, 0.1, -0.1, -0.2, 0.1, -0.1, -0.1, -0.0, -0.1, 0.2},
{0.2, 0.1, -0.1, -0.1, -0.1, 0.1, -0.0, 0.1, 0.1, -0.1, 0.0, 0.1, 0.1, 0.0, -0.0, 0.1, -0.1, 0.2, 0.1, 0.1, 0.1, 0.0, -0.2, 0.1, -0.0},
{-0.0, 0.2, -0.1, 0.1, 0.1, 0.1, 0.2, -0.1, 0.1, 0.1, 0.2, -0.1, 0.1, 0.1, -0.0, 0.0, -0.2, -0.0, 0.2, 0.2, -0.0, -0.1, -0.2, 0.0, -0.1},
{0.1, 0.1, 0.2, -0.0, -0.1, 0.1, -0.2, -0.1, 0.0, -0.1, -0.2, -0.0, -0.0, -0.1, -0.1, -0.1, -0.0, 0.0, -0.0, 0.1, 0.1, 0.1, -0.1, -0.2, 0.0},
{-0.1, -0.1, -0.1, 0.1, 0.1, 0.0, 0.0, -0.0, 0.1, 0.1, -0.1, 0.1, 0.1, 0.2, -0.1, -0.1, -0.1, -0.2, -0.1, -0.1, -0.1, 0.2, 0.1, -0.1, 0.2},
{0.0, -0.0, 0.2, 0.1, 0.1, 0.0, 0.2, 0.2, 0.0, -0.1, 0.0, -0.1, 0.2, 0.2, -0.2, -0.1, 0.0, -0.1, -0.1, -0.1, -0.2, -0.2, 0.1, 0.0, -0.2},
{0.1, -0.2, -0.1, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1, 0.2, 0.1, 0.2, 0.2, -0.1, 0.0, 0.1, -0.0, 0.1, -0.1, 0.1, 0.0, 0.2, 0.1, -0.0, 0.1},
}
);
Matrix  transformer_layers_14_attention_value_bias   (
{0.0, -0.0, 0.1, 0.1, 0.2, -0.2, 0.1, 0.1, 0.1, -0.1, 0.0, 0.1, -0.1, 0.1, -0.1, 0.2, -0.0, 0.1, 0.1, 0.1, -0.1, -0.2, -0.1, -0.0, 0.1}
);
Matrix  transformer_layers_14_norm1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_14_norm1_layer_norm_bias   (
{0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_14_norm2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_14_norm2_layer_norm_bias   (
{0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  transformer_layers_14_feed_forward_linear1_weight   (
{{-0.2, -0.1, -0.0, -0.2, -0.2, 0.1, 0.1, 0.2, -0.0, -0.1, 0.0, 0.1, 0.1, 0.0, -0.1, 0.1, 0.0, 0.1, 0.2, 0.0, -0.0, -0.0, -0.0, -0.2, -0.0},
{0.1, -0.1, 0.1, 0.1, -0.0, 0.1, 0.0, 0.1, -0.1, -0.1, 0.2, -0.2, 0.1, 0.2, -0.2, 0.1, -0.1, 0.1, -0.1, -0.2, -0.1, -0.1, -0.0, -0.1, 0.1},
{-0.0, -0.0, 0.0, 0.2, -0.1, -0.0, -0.1, 0.1, 0.0, 0.2, -0.2, -0.2, -0.2, 0.1, 0.1, 0.1, -0.2, 0.1, 0.1, -0.0, -0.0, 0.2, 0.1, 0.0, 0.0},
{0.1, -0.1, -0.0, 0.0, 0.1, 0.0, 0.2, 0.1, 0.1, 0.2, -0.0, 0.0, 0.0, -0.2, -0.0, -0.2, -0.1, 0.1, 0.1, -0.1, 0.1, -0.0, 0.1, 0.2, 0.2},
{-0.1, -0.1, 0.0, -0.1, -0.2, -0.1, 0.0, 0.1, -0.2, 0.1, -0.0, -0.1, -0.1, 0.1, -0.1, 0.0, 0.1, -0.2, 0.1, 0.0, 0.0, -0.2, 0.1, 0.0, 0.1},
{0.1, -0.1, 0.0, -0.1, 0.2, -0.0, -0.1, -0.0, 0.0, -0.2, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.2, 0.2, 0.1, -0.1, -0.0, 0.1, -0.1, -0.1, 0.1},
{0.2, 0.1, 0.1, -0.2, -0.1, 0.2, 0.2, 0.1, 0.0, -0.1, -0.2, 0.1, 0.0, 0.2, -0.1, 0.0, -0.2, 0.1, 0.0, -0.1, 0.1, 0.1, -0.2, 0.0, 0.2},
{0.2, -0.1, -0.2, 0.1, 0.1, -0.1, -0.1, -0.0, 0.2, -0.1, -0.1, 0.2, 0.1, -0.2, -0.0, 0.1, -0.2, 0.0, 0.1, -0.1, -0.0, -0.1, 0.1, 0.1, -0.1},
{-0.1, -0.1, -0.1, 0.2, 0.0, -0.2, 0.1, 0.1, -0.1, -0.2, 0.2, -0.0, 0.0, 0.2, -0.2, 0.1, 0.1, -0.1, -0.2, 0.1, -0.0, 0.1, -0.1, -0.1, 0.2},
{-0.0, -0.2, 0.1, -0.1, 0.1, 0.2, -0.0, 0.0, 0.0, 0.2, 0.0, -0.1, 0.1, 0.0, -0.2, 0.1, -0.1, 0.2, 0.2, -0.1, -0.1, 0.1, -0.1, 0.0, -0.1},
{-0.0, 0.0, 0.1, 0.0, 0.2, 0.2, 0.1, -0.2, 0.2, -0.2, -0.1, 0.0, -0.1, -0.1, 0.1, 0.1, -0.2, -0.0, -0.1, 0.1, -0.1, 0.1, -0.0, 0.0, -0.1},
{-0.0, -0.1, -0.2, -0.1, -0.0, 0.2, 0.2, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.1, -0.1, 0.2, 0.1, -0.0, 0.2, 0.1, -0.1, 0.1, 0.1, -0.0, 0.0},
{-0.1, -0.1, 0.1, 0.0, 0.1, -0.2, 0.2, 0.0, -0.0, -0.0, 0.1, -0.0, -0.2, 0.1, 0.0, 0.1, -0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.1, 0.1, -0.2},
{-0.0, -0.2, 0.1, -0.1, 0.0, -0.1, -0.2, -0.2, -0.2, -0.1, -0.1, -0.1, -0.2, 0.0, 0.1, -0.0, -0.0, 0.1, -0.0, -0.1, 0.1, -0.0, 0.1, 0.1, 0.1},
{-0.0, -0.2, 0.2, -0.2, -0.1, 0.1, 0.0, -0.2, 0.1, 0.1, 0.0, 0.1, 0.1, -0.1, -0.2, 0.1, 0.0, 0.2, -0.2, -0.1, 0.0, 0.1, -0.1, -0.2, -0.0},
}
);
Matrix  transformer_layers_14_feed_forward_linear1_bias   (
{0.0, -0.2, -0.1, 0.1, 0.2, -0.0, 0.1, -0.2, 0.1, -0.0, -0.1, 0.1, 0.1, 0.0, -0.2}
);
Matrix  transformer_layers_14_feed_forward_linear2_weight   (
{{0.2, -0.0, -0.0, -0.0, 0.3, 0.2, 0.1, 0.2, 0.1, -0.1, -0.1, -0.2, 0.2, 0.1, -0.0},
{-0.0, 0.2, 0.0, -0.1, 0.1, -0.0, -0.2, -0.1, -0.0, -0.0, 0.1, 0.2, 0.2, 0.0, -0.1},
{0.0, -0.1, -0.2, 0.0, -0.1, 0.1, -0.1, 0.2, 0.1, 0.0, -0.1, -0.2, -0.2, -0.1, 0.2},
{0.0, -0.2, -0.1, 0.2, 0.2, 0.2, 0.0, 0.2, -0.2, -0.2, 0.1, -0.2, 0.2, -0.1, 0.2},
{-0.2, 0.2, -0.1, 0.2, -0.3, -0.2, 0.1, -0.1, 0.0, -0.1, 0.1, -0.1, -0.2, -0.1, -0.1},
{-0.0, -0.0, 0.0, 0.0, -0.2, -0.0, 0.2, 0.0, 0.2, -0.1, 0.2, 0.2, -0.0, -0.2, -0.2},
{-0.0, 0.2, -0.2, 0.1, 0.1, 0.2, 0.2, -0.2, -0.2, -0.0, -0.0, 0.2, 0.2, 0.0, 0.2},
{-0.1, -0.2, 0.0, -0.0, -0.1, -0.1, -0.2, 0.0, 0.0, -0.2, -0.2, 0.2, 0.2, -0.2, 0.1},
{-0.1, -0.0, -0.1, 0.2, -0.2, 0.0, -0.1, -0.2, 0.1, -0.2, -0.0, 0.2, 0.1, -0.1, -0.1},
{-0.2, -0.1, -0.2, 0.1, 0.2, -0.1, 0.2, 0.1, 0.0, 0.2, -0.0, 0.2, -0.1, 0.1, -0.1},
{0.1, 0.0, 0.0, -0.0, 0.1, 0.1, 0.1, 0.0, 0.1, 0.2, -0.2, -0.0, -0.2, 0.1, 0.1},
{-0.2, -0.0, -0.1, -0.1, 0.2, 0.2, 0.2, -0.0, -0.1, -0.1, 0.2, -0.1, 0.2, -0.1, 0.2},
{-0.0, -0.0, -0.1, 0.1, 0.1, -0.1, 0.0, 0.1, 0.1, -0.0, 0.1, -0.2, 0.1, -0.0, -0.0},
{-0.1, -0.1, 0.1, 0.2, -0.2, 0.2, 0.0, -0.1, -0.1, 0.2, -0.1, 0.1, -0.2, -0.2, -0.2},
{0.0, -0.0, 0.2, 0.2, 0.1, -0.2, -0.0, 0.1, 0.1, 0.2, -0.2, -0.1, -0.2, 0.1, -0.2},
{0.2, 0.0, 0.0, -0.0, -0.0, 0.3, 0.2, 0.2, -0.2, 0.2, -0.1, 0.2, -0.1, -0.1, -0.1},
{0.1, 0.1, 0.1, -0.2, 0.3, 0.2, -0.1, 0.2, -0.1, 0.1, -0.1, 0.2, 0.3, 0.2, 0.2},
{0.2, 0.1, -0.1, -0.1, 0.2, -0.2, -0.1, -0.2, -0.0, -0.2, -0.1, 0.2, 0.1, -0.2, 0.2},
{-0.1, -0.0, 0.2, -0.2, 0.1, 0.0, 0.2, 0.1, -0.1, -0.0, 0.2, -0.1, -0.1, 0.2, -0.1},
{0.0, 0.2, -0.1, -0.3, -0.1, 0.2, -0.2, -0.3, -0.1, -0.1, -0.0, -0.2, 0.1, -0.2, 0.1},
{0.2, -0.2, -0.1, 0.0, -0.1, -0.2, 0.0, 0.2, 0.1, 0.1, 0.1, -0.0, -0.2, -0.0, 0.0},
{-0.2, 0.1, 0.1, 0.2, -0.0, 0.2, -0.1, -0.1, -0.1, -0.1, -0.1, 0.2, 0.3, -0.0, -0.1},
{-0.2, 0.1, 0.2, 0.1, 0.1, -0.1, -0.0, -0.2, -0.2, -0.1, 0.0, -0.1, 0.1, 0.2, -0.1},
{-0.2, 0.1, 0.1, -0.2, -0.2, -0.2, -0.2, 0.0, 0.1, 0.2, -0.3, -0.1, 0.2, -0.3, -0.1},
{0.1, 0.0, -0.2, -0.2, 0.2, 0.1, 0.0, 0.1, 0.1, 0.0, 0.2, -0.1, 0.1, -0.2, -0.1},
}
);
Matrix  transformer_layers_14_feed_forward_linear2_bias   (
{-0.0, -0.2, -0.1, -0.1, -0.1, -0.2, -0.0, 0.2, -0.2, -0.1, 0.2, 0.2, -0.0, -0.2, -0.1, -0.3, -0.0, -0.0, 0.1, 0.2, 0.2, 0.2, 0.0, 0.0, -0.2}
);
Matrix  transformer_layers_14_feed_forward_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_14_feed_forward_ln1_layer_norm_bias   (
{-0.0, 0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0}
);
Matrix  transformer_layers_14_feed_forward_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  transformer_layers_14_feed_forward_ln2_layer_norm_bias   (
{0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.0, -0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.0}
);
Matrix  inputff_linear1_weight   (
{{0.1, -0.0, -0.1, -0.1, 0.0, 0.0, -0.0, 0.1, 0.0, -0.0, 0.1, -0.0, 0.1, 0.1, -0.1, -0.0, -0.1, 0.1, -0.0, 0.0, 0.1, -0.1, 0.0, -0.1, 0.0, -0.1, -0.1, 0.1, 0.0, 0.1, 0.1, 0.1, -0.1, -0.1, -0.0, -0.1, 0.1, -0.0, -0.1, -0.1, 0.1, -0.0, 0.1, -0.1, -0.1, 0.1, -0.0, -0.0, 0.1, -0.0, -0.0, -0.1, 0.1, -0.1, 0.1, -0.1, 0.0, -0.0, -0.0, -0.0, -0.1, -0.1, 0.1, 0.0},
{-0.1, 0.1, 0.1, -0.1, -0.1, -0.0, 0.0, 0.1, 0.0, 0.0, -0.0, 0.1, -0.0, 0.0, -0.1, 0.1, -0.1, 0.1, -0.0, 0.1, -0.1, 0.1, -0.0, -0.1, -0.1, -0.1, 0.1, -0.1, -0.0, 0.1, -0.0, -0.1, 0.0, 0.1, 0.0, -0.1, -0.0, 0.1, 0.0, 0.1, 0.1, -0.1, 0.1, 0.1, -0.0, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.0, 0.0, -0.1, -0.1, -0.1, -0.1, 0.0, 0.1},
{0.1, -0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.1, 0.0, -0.1, 0.1, 0.0, -0.0, 0.0, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.0, -0.0, -0.1, 0.1, 0.0, -0.1, 0.0, -0.1, -0.1, -0.1, 0.1, -0.1, 0.0, -0.1, 0.1, -0.0, 0.0, -0.1, 0.0, 0.0, 0.1, 0.1, -0.1, -0.1, 0.1, 0.0, -0.1, 0.0, -0.1, -0.0, -0.0, -0.0, -0.1, -0.0, -0.1, 0.0, -0.1, 0.1, 0.0},
{-0.1, 0.0, 0.0, 0.0, 0.1, -0.1, 0.1, -0.0, -0.1, 0.1, 0.1, 0.0, -0.0, 0.0, -0.1, 0.1, 0.1, -0.1, 0.1, 0.0, -0.0, 0.1, 0.0, 0.0, -0.1, -0.0, 0.1, -0.1, -0.1, -0.0, -0.1, -0.0, -0.0, -0.1, 0.1, -0.1, 0.0, -0.0, 0.1, -0.1, -0.1, -0.1, -0.1, 0.0, 0.1, -0.0, 0.0, -0.1, -0.1, 0.1, 0.0, 0.1, 0.0, 0.0, -0.0, -0.1, -0.0, -0.0, -0.1, 0.1, 0.1, -0.0, 0.1, -0.0},
{-0.1, -0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.0, -0.1, 0.1, 0.1, 0.1, -0.1, -0.0, -0.0, 0.1, -0.1, 0.1, -0.0, 0.0, -0.1, -0.1, -0.0, 0.0, 0.0, -0.1, -0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.1, 0.0, -0.1, 0.0, -0.1, 0.0, 0.1, 0.1, 0.1, 0.0, 0.1, -0.1, -0.0, 0.0, -0.1, 0.1, -0.1, 0.0, -0.0, -0.0, -0.1, 0.1, -0.1, 0.1, -0.0, -0.0, -0.1, -0.1, -0.0, 0.1},
{0.1, -0.1, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.1, -0.1, 0.1, -0.0, 0.0, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.0, 0.1, 0.0, -0.1, 0.1, 0.1, -0.0, 0.1, 0.1, 0.0, 0.1, 0.1, 0.1, 0.1, -0.1, -0.0, -0.0, -0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.0, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, 0.0, -0.0, 0.1, -0.1, 0.1, 0.1, -0.1},
{0.0, -0.0, -0.1, 0.0, 0.0, 0.1, -0.0, -0.1, 0.1, 0.0, -0.1, -0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.1, 0.0, -0.0, -0.1, 0.1, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, -0.1, 0.1, -0.1, 0.0, -0.1, 0.0, -0.1, -0.1, 0.1, -0.1, 0.0, -0.0, 0.0, -0.0, -0.1, 0.1, 0.0, 0.1, -0.0, -0.1, 0.0, 0.0, 0.1, -0.1, -0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0, -0.0, -0.1, 0.0, 0.1},
{-0.1, 0.1, -0.1, -0.0, -0.1, 0.1, 0.1, 0.0, -0.1, -0.1, 0.0, 0.1, -0.0, 0.0, 0.1, 0.0, -0.0, 0.1, -0.1, 0.1, 0.0, 0.1, 0.1, -0.1, 0.0, -0.1, -0.0, 0.1, -0.1, 0.1, 0.1, -0.0, 0.0, -0.1, 0.0, -0.0, -0.1, 0.1, 0.1, 0.1, 0.1, -0.0, -0.1, 0.1, 0.0, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.0, -0.1, -0.0, -0.1, -0.1, -0.1, -0.0, 0.1, 0.1, -0.1, -0.1},
{0.1, 0.1, 0.1, 0.1, -0.1, 0.1, -0.1, -0.1, -0.1, -0.0, 0.1, 0.1, 0.0, 0.1, -0.1, 0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.1, 0.1, -0.1, 0.1, -0.1, -0.1, 0.0, 0.0, 0.0, -0.1, -0.0, 0.1, 0.1, 0.1, 0.1, -0.1, -0.0, -0.0, 0.1, -0.0, 0.1, 0.1, 0.0, 0.1, 0.1, -0.0, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.1, -0.1, 0.0},
{0.1, -0.0, -0.0, -0.0, 0.1, 0.1, -0.0, -0.1, 0.1, 0.1, 0.0, -0.1, -0.1, -0.1, -0.0, 0.1, 0.1, 0.1, -0.1, -0.0, -0.0, -0.0, 0.0, 0.1, 0.0, -0.1, -0.1, -0.0, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.0, -0.1, 0.0, -0.1, 0.0, 0.1, 0.0, -0.0, -0.0, 0.1, -0.1, 0.1, 0.1, -0.0, -0.1, -0.1, -0.0, -0.0, 0.1, -0.0, -0.0, 0.0, -0.1, -0.0, -0.1, 0.0, -0.0, 0.1, -0.1, -0.0},
{0.0, -0.0, 0.1, 0.1, -0.0, 0.1, 0.1, 0.0, -0.1, -0.1, -0.1, -0.1, 0.1, 0.0, 0.1, -0.1, 0.0, -0.1, 0.1, -0.1, 0.1, -0.0, 0.0, 0.1, -0.1, 0.0, -0.0, 0.0, 0.1, -0.1, 0.0, -0.1, 0.0, -0.1, -0.1, -0.1, -0.1, -0.0, 0.0, 0.1, -0.0, 0.1, -0.1, -0.0, -0.0, 0.1, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, -0.0, 0.1, -0.0},
{-0.1, 0.1, 0.1, -0.1, 0.1, 0.1, -0.0, 0.1, -0.1, 0.1, 0.1, -0.0, 0.0, 0.0, -0.0, 0.0, 0.1, -0.1, 0.1, 0.0, 0.0, -0.0, 0.0, 0.1, -0.0, -0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, 0.0, 0.0, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, 0.0, -0.0, -0.0, 0.1, 0.1, -0.1, -0.0, 0.0, -0.1, -0.0, -0.1, 0.1, 0.0, -0.0, 0.1, -0.1, -0.0, -0.1, 0.1, 0.1, -0.1},
{0.0, 0.1, -0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.0, 0.1, -0.1, 0.0, -0.0, 0.1, 0.1, -0.0, 0.1, -0.0, -0.0, 0.0, -0.0, 0.1, -0.0, -0.0, 0.1, 0.1, -0.0, 0.1, -0.1, -0.1, 0.0, 0.1, -0.0, -0.1, 0.1, -0.0, 0.0, 0.1, -0.1, -0.1, -0.1, -0.1, 0.0, 0.1, 0.0, -0.1, -0.1, -0.1, 0.1, 0.0, 0.0, -0.1, -0.1, 0.1, -0.0, -0.1, -0.1, 0.1, -0.0, -0.1, -0.0, 0.1, 0.0, -0.1},
{-0.0, -0.1, 0.0, 0.0, -0.1, -0.1, -0.1, -0.0, 0.0, 0.1, -0.0, -0.1, 0.1, 0.1, -0.1, 0.1, 0.0, 0.0, 0.1, 0.1, 0.0, 0.0, 0.0, 0.1, -0.0, -0.0, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, 0.0, -0.1, -0.0, 0.1, 0.1, 0.1, -0.0, 0.1, 0.0, 0.0, -0.1, -0.0, 0.1, -0.1, 0.1, -0.0, 0.1, -0.1, -0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.0, -0.1, -0.1, 0.1, 0.0, 0.0, 0.0, -0.1},
{0.0, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.0, 0.0, -0.1, -0.0, -0.1, -0.0, 0.1, 0.0, 0.1, -0.0, 0.1, 0.0, 0.0, -0.1, 0.1, -0.1, 0.1, -0.0, -0.1, -0.0, -0.1, 0.1, 0.0, -0.1, -0.1, 0.0, 0.1, 0.0, -0.1, -0.0, 0.1, 0.0, -0.0, 0.1, -0.1, 0.1, 0.0, -0.0, -0.0, -0.0, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.0, -0.0, -0.1, -0.1, 0.1, 0.1, -0.1, -0.0, 0.0},
}
);
Matrix  inputff_linear1_bias   (
{-0.0, -0.1, -0.1, -0.1, 0.1, -0.0, -0.0, 0.1, -0.1, -0.1, -0.1, 0.0, 0.1, -0.1, -0.1}
);
Matrix  inputff_linear2_weight   (
{{-0.2, -0.0, -0.3, 0.1, 0.1, -0.1, -0.0, -0.1, -0.0, 0.1, -0.2, -0.1, 0.2, -0.1, 0.2},
{0.1, -0.2, -0.1, -0.2, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, 0.2, -0.1, 0.1, -0.2, -0.0},
{-0.1, 0.0, 0.1, -0.1, 0.1, 0.1, 0.3, -0.2, -0.0, -0.2, -0.1, -0.3, 0.1, -0.1, -0.1},
{-0.2, -0.1, 0.1, 0.2, -0.2, -0.1, 0.2, 0.1, 0.0, 0.2, -0.2, -0.2, -0.2, 0.1, 0.3},
{0.1, -0.1, 0.1, -0.1, 0.1, 0.2, 0.2, -0.2, 0.2, 0.1, -0.1, 0.2, -0.1, 0.1, -0.2},
{0.1, 0.2, 0.0, 0.2, 0.0, -0.2, 0.0, 0.1, 0.1, 0.2, 0.2, 0.0, -0.1, 0.2, 0.1},
{-0.3, 0.1, -0.2, -0.2, -0.1, -0.2, 0.1, -0.2, -0.2, 0.0, 0.1, 0.2, -0.1, 0.2, -0.0},
{-0.2, -0.0, 0.2, -0.2, 0.1, -0.0, 0.1, 0.2, -0.1, -0.2, -0.0, -0.0, -0.0, -0.1, -0.1},
{-0.1, 0.2, -0.0, -0.2, -0.1, 0.1, -0.2, 0.0, 0.2, -0.1, 0.2, -0.1, 0.2, 0.2, 0.2},
{-0.2, 0.0, 0.0, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.2, 0.0, 0.1, -0.2, 0.2},
{0.1, 0.1, 0.2, 0.2, -0.1, -0.0, 0.0, -0.0, -0.1, 0.1, -0.0, 0.2, 0.2, 0.2, 0.2},
{-0.1, -0.2, 0.2, -0.2, -0.1, 0.2, -0.2, 0.0, 0.0, -0.2, 0.2, 0.2, -0.2, -0.1, 0.2},
{-0.0, -0.2, -0.1, -0.1, -0.0, -0.0, 0.2, -0.1, 0.2, -0.2, -0.2, 0.2, 0.2, 0.1, -0.0},
{-0.0, 0.1, -0.2, 0.0, -0.1, 0.2, 0.2, -0.2, -0.1, 0.1, 0.1, -0.0, 0.2, 0.2, 0.2},
{0.2, -0.0, 0.1, -0.0, 0.2, 0.2, 0.0, 0.2, 0.0, -0.1, 0.2, -0.2, 0.1, 0.1, -0.2},
{0.1, 0.2, 0.1, 0.2, 0.1, -0.1, -0.2, 0.0, -0.0, 0.3, -0.0, -0.3, -0.1, 0.1, 0.2},
{0.0, -0.0, 0.2, -0.2, -0.2, 0.2, -0.1, 0.2, 0.2, -0.1, -0.2, -0.2, 0.2, -0.1, 0.0},
{-0.1, 0.1, 0.0, 0.1, -0.2, 0.1, 0.2, 0.2, 0.0, 0.1, -0.1, 0.2, -0.2, -0.1, -0.2},
{-0.2, 0.2, -0.2, -0.1, 0.2, -0.2, -0.1, 0.2, 0.1, -0.2, 0.1, -0.1, 0.1, -0.1, -0.2},
{0.1, 0.1, 0.1, 0.1, 0.2, -0.2, -0.2, 0.1, 0.0, 0.1, 0.1, -0.2, -0.3, 0.0, 0.2},
{-0.2, -0.0, -0.0, -0.1, -0.1, 0.2, 0.1, -0.0, 0.2, 0.2, 0.1, -0.2, 0.0, 0.1, 0.2},
{0.1, 0.2, 0.1, 0.2, -0.2, -0.2, -0.2, -0.2, -0.0, -0.0, -0.0, 0.1, 0.0, 0.1, -0.0},
{0.2, 0.2, 0.1, -0.1, 0.1, -0.2, 0.3, 0.2, -0.1, 0.1, 0.2, -0.0, 0.1, -0.0, 0.0},
{-0.1, -0.2, -0.2, 0.2, -0.2, -0.2, -0.2, 0.1, -0.1, -0.1, -0.3, 0.2, -0.2, -0.1, 0.2},
{-0.0, -0.1, -0.2, -0.1, 0.0, 0.1, -0.2, 0.2, -0.0, -0.0, 0.0, -0.2, -0.1, 0.1, 0.2},
}
);
Matrix  inputff_linear2_bias   (
{0.0, 0.2, 0.2, -0.0, -0.2, 0.1, 0.0, 0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.2, 0.2, -0.1, 0.2, 0.2, -0.1, -0.1, -0.1, -0.0, 0.0, -0.2, -0.2}
);
Matrix  inputff_ln1_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  inputff_ln1_layer_norm_bias   (
{-0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, -0.0, 0.0, -0.0, -0.0, 0.0, 0.0}
);
Matrix  inputff_ln2_layer_norm_weight   (
{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
);
Matrix  inputff_ln2_layer_norm_bias   (
{-0.0, 0.0, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0, 0.0, -0.0, 0.0, 0.0, 0.0, -0.0, 0.0, 0.0, -0.0, -0.0, 0.0}
);
Matrix  value_head_weight   (
{0.0, -0.0, -0.2, -0.0, 0.1, 0.1, 0.1, -0.2, -0.0, -0.1, 0.1, 0.2, 0.1, -0.2, -0.2, -0.2, 0.0, -0.2, -0.2, -0.0, 0.1, 0.0, 0.1, 0.0, -0.1}
);
// Matrix  value_head_bias   (
// {-0.1}
// );
Matrix  policy_head_weight   (
{{-0.0, 0.1, -0.1, -0.1, 0.1, -0.2, 0.1, 0.1, -0.2, -0.1, 0.1, -0.1, 0.1, 0.2, 0.0, 0.2, 0.1, -0.2, 0.2, 0.1, 0.1, -0.1, -0.0, 0.1, -0.2},
{-0.1, 0.0, 0.1, 0.2, -0.1, -0.1, -0.0, 0.0, -0.2, 0.1, 0.1, -0.1, -0.0, -0.0, -0.0, -0.2, 0.1, 0.1, -0.0, -0.1, 0.1, -0.1, -0.2, 0.1, -0.2},
{-0.1, -0.2, -0.1, 0.1, -0.2, -0.2, 0.0, -0.0, 0.0, 0.0, 0.0, -0.2, -0.1, 0.2, 0.1, -0.1, 0.2, 0.1, 0.1, 0.1, -0.1, -0.1, 0.0, 0.1, 0.1},
{0.0, 0.1, -0.1, 0.0, 0.0, -0.2, -0.1, 0.2, -0.1, -0.2, 0.0, 0.0, -0.1, 0.1, 0.2, -0.1, -0.2, 0.2, -0.1, -0.1, 0.1, -0.0, 0.1, 0.1, -0.1},
{-0.2, 0.0, 0.1, 0.2, 0.1, 0.1, -0.0, 0.0, 0.1, 0.1, -0.2, -0.1, -0.1, -0.2, -0.1, 0.1, 0.0, -0.1, 0.2, 0.1, -0.0, -0.1, -0.1, 0.2, -0.2},
{0.2, 0.2, 0.1, 0.1, -0.0, -0.2, 0.0, -0.1, 0.2, 0.2, 0.1, -0.1, 0.0, 0.1, 0.0, 0.2, 0.1, 0.0, -0.2, -0.1, -0.1, -0.1, 0.1, 0.1, -0.1},
{0.1, 0.1, 0.0, -0.1, -0.1, -0.1, 0.2, 0.0, 0.1, 0.2, 0.1, -0.2, -0.1, -0.1, -0.2, 0.1, 0.1, -0.1, 0.0, 0.1, -0.0, 0.1, 0.1, 0.1, -0.1},
{-0.2, -0.2, 0.1, -0.1, -0.1, 0.0, 0.1, -0.1, -0.2, -0.1, 0.1, -0.1, 0.0, 0.1, -0.1, 0.2, 0.1, -0.1, -0.1, 0.1, 0.1, 0.2, 0.1, 0.2, -0.0},
{-0.0, -0.1, 0.1, -0.0, 0.2, -0.1, 0.1, 0.1, 0.2, -0.0, -0.2, -0.0, 0.0, 0.2, -0.1, -0.1, 0.2, 0.1, -0.1, 0.0, 0.1, -0.0, 0.0, -0.2, 0.1},
{0.1, -0.1, 0.1, 0.0, 0.1, 0.0, -0.1, -0.1, -0.1, 0.2, 0.2, -0.1, 0.1, 0.2, -0.1, 0.0, -0.2, -0.1, 0.2, -0.2, 0.2, 0.2, -0.2, -0.1, 0.1},
{-0.1, 0.0, 0.2, 0.1, -0.0, 0.1, 0.2, 0.2, -0.2, -0.1, -0.1, -0.1, -0.2, 0.1, 0.1, -0.0, -0.1, -0.1, 0.1, 0.1, 0.1, -0.1, -0.1, 0.1, -0.2},
{0.2, -0.2, 0.0, -0.1, 0.0, -0.0, 0.1, -0.0, 0.2, -0.0, -0.1, 0.0, 0.2, 0.1, -0.0, -0.2, 0.2, -0.1, -0.2, -0.0, -0.1, -0.2, 0.1, 0.2, 0.2},
{-0.0, 0.0, 0.1, 0.2, 0.0, -0.2, -0.1, 0.2, -0.1, 0.0, -0.1, -0.0, -0.1, 0.1, 0.1, -0.1, -0.2, 0.2, 0.1, -0.1, -0.1, -0.2, -0.1, -0.0, -0.2},
{0.1, 0.1, -0.0, 0.2, 0.1, 0.1, 0.1, 0.1, -0.1, 0.2, 0.0, 0.1, -0.1, -0.0, -0.0, -0.1, 0.2, -0.2, 0.2, 0.0, 0.1, 0.1, 0.1, -0.0, 0.2},
{0.1, 0.1, 0.1, 0.0, -0.2, 0.1, -0.1, 0.1, 0.1, -0.0, -0.2, -0.2, 0.0, 0.1, -0.1, 0.1, -0.1, -0.2, -0.1, -0.2, -0.1, 0.1, -0.1, -0.1, 0.1},
{-0.0, 0.2, 0.2, -0.2, 0.0, 0.0, -0.0, -0.1, 0.1, -0.1, -0.1, 0.1, 0.2, -0.1, -0.1, 0.0, -0.0, -0.1, -0.1, 0.1, -0.1, 0.0, -0.2, -0.2, -0.1},
{0.1, -0.1, -0.2, -0.0, -0.1, 0.1, -0.1, -0.1, -0.2, 0.1, -0.0, -0.0, -0.2, 0.1, 0.1, 0.1, 0.1, -0.2, -0.2, -0.1, -0.1, -0.2, 0.1, 0.0, 0.1},
{0.2, -0.2, -0.1, -0.2, -0.0, 0.0, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.2, -0.1, 0.0, 0.1, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.0, -0.1, 0.1},
{-0.1, 0.2, -0.2, -0.1, -0.1, -0.2, 0.2, 0.0, 0.1, -0.1, 0.2, 0.2, -0.1, 0.0, 0.2, -0.0, 0.0, 0.0, -0.2, 0.0, 0.1, 0.1, -0.1, -0.0, -0.1},
{0.1, 0.2, -0.1, -0.2, 0.1, -0.0, -0.2, 0.0, -0.0, 0.1, 0.0, -0.1, 0.0, -0.2, 0.1, 0.1, 0.0, 0.1, -0.1, 0.1, -0.2, -0.2, -0.2, 0.0, 0.1},
{0.2, -0.1, -0.0, -0.1, 0.2, 0.1, -0.1, 0.1, 0.2, 0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.2, -0.0, -0.0, -0.0, 0.2, 0.2, -0.1, 0.0},
{0.1, -0.2, 0.2, 0.1, 0.1, -0.0, 0.1, 0.1, 0.0, 0.2, 0.0, 0.2, -0.1, 0.1, 0.0, 0.1, -0.2, -0.0, -0.1, 0.2, -0.0, 0.1, 0.0, -0.2, -0.1},
{0.1, 0.2, -0.0, -0.1, -0.2, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, -0.0, -0.1, -0.1, 0.1, -0.1, 0.1, -0.2, -0.1, -0.1, 0.2, 0.0, -0.1, 0.0, 0.1},
{-0.0, 0.1, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, -0.0, 0.2, -0.1, -0.2, 0.1, -0.1, 0.2, -0.1, -0.1, -0.0, 0.2, -0.0, 0.0, 0.1, -0.1},
{0.0, 0.0, -0.1, -0.1, 0.1, 0.0, -0.1, 0.2, -0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.2, -0.0, -0.1, -0.0, 0.1, -0.2, 0.0, 0.2, 0.1, -0.1},
{-0.1, 0.0, 0.2, -0.1, 0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, -0.2, -0.2, 0.2, -0.1, -0.1, 0.2, 0.2, -0.1, -0.1, 0.2, -0.0, 0.2, 0.1, 0.2},
{-0.2, 0.1, 0.0, -0.1, -0.0, -0.1, -0.2, -0.1, -0.0, 0.1, 0.1, 0.0, 0.0, -0.0, 0.1, -0.1, 0.1, 0.2, -0.2, 0.0, 0.0, -0.2, -0.2, -0.0, 0.1},
{-0.1, -0.2, -0.2, -0.0, -0.1, 0.1, 0.1, -0.2, -0.0, 0.1, -0.2, 0.2, -0.1, 0.2, -0.0, -0.2, -0.2, 0.0, 0.1, -0.2, -0.1, 0.1, -0.0, 0.2, -0.1},
{0.1, 0.1, 0.1, -0.1, -0.1, -0.2, 0.1, 0.1, -0.0, -0.2, 0.2, -0.1, -0.1, -0.0, -0.2, 0.0, 0.0, -0.1, 0.1, -0.1, -0.2, -0.0, -0.0, 0.1, 0.0},
{0.2, -0.1, -0.1, -0.0, 0.1, 0.1, -0.0, 0.1, 0.1, 0.1, 0.2, -0.1, 0.2, -0.2, -0.2, 0.1, 0.0, 0.0, 0.1, 0.1, 0.2, 0.1, -0.2, 0.2, 0.0},
{0.1, 0.1, 0.1, -0.2, -0.2, 0.2, -0.1, -0.1, -0.1, 0.0, 0.1, -0.1, 0.1, 0.1, -0.1, -0.2, 0.1, -0.2, -0.1, 0.0, -0.1, 0.0, 0.1, -0.1, -0.1},
{-0.0, 0.1, 0.2, 0.0, -0.2, 0.0, 0.2, 0.0, -0.0, 0.1, 0.0, 0.2, -0.1, 0.1, -0.1, -0.0, 0.1, 0.0, -0.1, -0.2, -0.1, -0.0, 0.0, -0.1, -0.2},
{0.2, 0.2, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.0, 0.1, 0.2, 0.1, -0.0, -0.1, 0.2, 0.1, 0.1, 0.2, -0.2, 0.2, 0.2, 0.1, -0.0, 0.1},
{-0.1, 0.1, -0.1, -0.1, 0.1, -0.0, -0.1, 0.1, -0.0, -0.1, 0.2, -0.1, -0.1, 0.0, -0.1, -0.0, 0.0, 0.1, 0.1, -0.1, -0.2, -0.1, -0.0, 0.0, 0.1},
{-0.0, 0.2, -0.1, 0.2, 0.2, -0.1, 0.2, 0.2, -0.2, -0.2, -0.2, -0.0, -0.1, -0.1, 0.1, -0.2, 0.2, 0.0, -0.0, 0.1, 0.2, -0.2, 0.0, -0.1, 0.2},
{-0.1, -0.1, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.0, -0.0, -0.1, 0.1, 0.2, -0.2, 0.0, 0.2, -0.0, 0.1, 0.1, 0.2, -0.0, -0.1, 0.2, 0.0, 0.0},
{0.2, -0.1, -0.2, 0.1, -0.1, 0.2, 0.1, -0.2, -0.2, -0.1, 0.2, -0.2, -0.2, -0.1, 0.2, 0.0, -0.0, -0.1, -0.2, -0.0, -0.0, 0.1, 0.2, 0.2, 0.2},
{-0.0, -0.1, 0.1, -0.1, 0.1, 0.2, 0.1, 0.1, -0.0, 0.1, -0.1, -0.1, -0.2, 0.0, -0.1, 0.1, -0.0, 0.1, 0.1, 0.1, 0.2, -0.1, 0.2, 0.2, -0.0},
{-0.2, 0.2, 0.2, 0.0, -0.1, -0.1, -0.1, 0.1, -0.2, -0.0, -0.1, 0.1, 0.1, -0.1, -0.0, -0.2, 0.0, 0.2, -0.2, -0.1, 0.0, -0.1, 0.0, 0.1, 0.0},
{-0.1, 0.1, 0.2, 0.1, 0.1, -0.2, -0.0, -0.1, -0.0, 0.2, -0.1, 0.0, -0.0, -0.2, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1},
{-0.1, -0.1, 0.0, 0.1, -0.2, -0.0, 0.2, 0.1, -0.0, 0.2, -0.1, 0.0, -0.2, 0.1, -0.2, -0.1, 0.0, 0.1, 0.1, 0.2, 0.1, 0.1, -0.1, 0.1, 0.1},
{0.1, -0.1, 0.0, -0.2, 0.0, -0.2, 0.0, 0.2, -0.1, -0.0, 0.2, 0.1, 0.2, 0.0, 0.1, 0.1, 0.2, -0.1, 0.1, -0.1, -0.1, 0.1, 0.1, 0.1, 0.2},
{-0.2, -0.1, -0.0, 0.0, 0.2, 0.0, 0.2, -0.0, 0.1, -0.0, -0.1, 0.1, -0.0, 0.1, -0.1, 0.1, 0.2, 0.1, -0.2, -0.1, 0.0, 0.1, -0.1, 0.0, 0.1},
{0.0, -0.2, 0.1, -0.1, -0.0, -0.1, 0.0, -0.1, 0.2, -0.0, 0.0, 0.0, 0.1, -0.0, 0.1, -0.0, 0.2, 0.2, -0.1, -0.1, 0.2, -0.1, 0.1, -0.1, -0.1},
{-0.1, -0.0, -0.1, -0.2, -0.0, 0.2, -0.0, 0.0, -0.2, -0.0, 0.0, -0.1, 0.2, -0.2, 0.1, 0.2, 0.0, 0.1, -0.1, -0.1, -0.0, -0.0, -0.0, -0.1, -0.2},
{-0.2, 0.2, -0.0, 0.1, -0.1, 0.2, 0.1, 0.1, -0.0, 0.0, -0.2, -0.0, -0.0, -0.1, -0.0, -0.2, 0.2, 0.1, -0.0, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1},
{-0.2, 0.1, -0.1, 0.1, 0.0, 0.2, -0.0, -0.1, 0.1, -0.1, -0.0, -0.1, -0.1, -0.0, 0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.2, -0.0, -0.0, 0.2},
{-0.0, 0.2, -0.0, -0.2, 0.2, 0.1, -0.1, -0.0, -0.1, -0.2, 0.1, -0.1, 0.1, -0.2, 0.1, -0.2, 0.1, -0.1, -0.1, 0.2, -0.1, 0.1, 0.1, -0.1, 0.1},
{0.2, 0.1, -0.1, -0.2, -0.0, 0.2, 0.1, -0.1, -0.0, -0.1, 0.0, 0.0, -0.1, -0.1, 0.2, 0.0, -0.0, -0.0, 0.1, -0.1, -0.2, -0.0, -0.1, 0.0, -0.2},
{-0.1, -0.2, -0.2, 0.2, -0.0, -0.2, 0.1, -0.0, -0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.1, -0.0, 0.1, -0.0, -0.1, 0.1, -0.1, 0.1, 0.2, -0.2, -0.2},
{-0.1, 0.0, -0.1, 0.2, 0.0, -0.1, -0.1, 0.1, 0.2, -0.0, 0.0, 0.2, 0.2, -0.0, 0.1, -0.2, 0.0, -0.1, 0.0, -0.2, 0.0, 0.1, 0.0, -0.0, -0.1},
{-0.2, -0.1, -0.0, -0.1, 0.1, -0.1, 0.1, 0.1, 0.0, 0.0, -0.2, 0.2, 0.1, 0.1, -0.1, -0.0, -0.2, -0.0, -0.2, -0.1, -0.0, 0.1, -0.1, -0.1, -0.2},
{-0.2, -0.1, -0.2, 0.1, 0.1, -0.1, 0.2, 0.2, -0.2, 0.1, -0.0, 0.2, 0.0, 0.1, 0.1, 0.1, 0.2, 0.1, -0.2, 0.0, 0.1, 0.2, 0.1, -0.1, 0.1},
{0.1, 0.2, 0.1, -0.1, -0.1, -0.1, 0.2, 0.1, 0.2, 0.1, -0.2, -0.0, -0.1, -0.1, -0.2, -0.2, 0.2, -0.1, 0.0, 0.0, -0.0, 0.0, 0.1, 0.0, 0.1},
{0.1, 0.0, 0.2, 0.0, 0.1, 0.1, 0.1, 0.1, -0.0, 0.2, -0.2, -0.1, -0.0, 0.1, -0.0, 0.0, 0.1, -0.1, 0.2, -0.1, -0.2, -0.2, 0.1, -0.0, 0.1},
{-0.0, -0.1, 0.1, -0.1, -0.2, 0.1, 0.1, 0.2, -0.0, 0.1, 0.1, -0.0, 0.0, -0.0, -0.1, -0.1, 0.2, 0.1, -0.0, 0.1, -0.2, -0.0, -0.2, 0.1, -0.1},
{0.0, -0.0, -0.1, -0.1, 0.2, -0.0, -0.1, -0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.2, -0.2, 0.1, -0.2, -0.1, 0.1, 0.0, 0.2, 0.1, 0.1, 0.1, -0.1},
{0.1, -0.1, -0.1, -0.1, 0.0, -0.1, 0.1, 0.1, 0.2, 0.2, 0.2, -0.2, 0.1, 0.1, -0.0, -0.0, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1, -0.1, -0.1},
{0.1, -0.2, 0.0, -0.1, -0.1, 0.1, -0.1, -0.2, -0.1, 0.0, 0.1, -0.0, -0.1, 0.2, 0.1, 0.2, -0.2, 0.0, 0.2, -0.0, 0.1, 0.0, -0.2, -0.2, -0.1},
{-0.0, 0.0, -0.1, 0.2, -0.1, -0.1, -0.0, -0.2, -0.0, -0.0, -0.0, -0.2, 0.0, -0.1, 0.0, 0.0, 0.1, 0.0, 0.1, 0.1, 0.2, -0.0, 0.2, -0.1, -0.0},
{-0.1, -0.1, -0.0, 0.1, -0.1, 0.2, -0.1, -0.0, 0.2, -0.2, -0.1, -0.0, 0.1, -0.1, -0.1, 0.1, -0.0, 0.2, 0.1, -0.2, 0.1, -0.0, -0.2, -0.1, -0.1},
{-0.2, -0.1, -0.1, 0.2, -0.1, -0.2, 0.1, 0.1, 0.1, -0.1, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, -0.0, -0.1, -0.2, -0.1, 0.1, -0.0, -0.2, -0.1, 0.1},
{-0.1, -0.1, -0.1, -0.1, -0.1, 0.1, 0.0, -0.1, -0.0, 0.1, 0.1, 0.1, -0.2, 0.2, -0.1, 0.0, -0.2, 0.1, -0.1, 0.2, 0.0, 0.1, -0.2, 0.1, -0.2},
{-0.1, -0.1, 0.1, 0.2, 0.1, -0.1, 0.1, -0.1, 0.1, -0.0, 0.1, 0.1, -0.0, 0.1, 0.0, -0.1, -0.1, -0.1, -0.1, -0.0, -0.1, 0.2, 0.2, 0.1, 0.1},
{-0.1, -0.1, 0.0, 0.1, 0.0, -0.1, -0.2, 0.2, -0.0, 0.2, 0.1, -0.0, 0.2, 0.1, -0.0, 0.2, -0.2, -0.1, -0.0, -0.1, 0.1, 0.1, 0.0, 0.1, 0.2},
{0.1, -0.2, -0.1, 0.1, 0.1, -0.1, 0.1, -0.1, 0.2, -0.1, 0.1, -0.0, 0.2, -0.1, 0.2, 0.1, -0.0, -0.2, -0.2, -0.1, 0.1, 0.2, -0.1, -0.1, -0.1},
{-0.2, 0.2, 0.0, 0.1, -0.0, 0.1, 0.2, 0.1, -0.1, -0.1, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, -0.2, 0.1, -0.2, 0.2, -0.1, 0.1, 0.0, -0.1, 0.1},
{-0.1, 0.1, -0.1, 0.0, -0.1, -0.0, -0.2, 0.1, -0.1, 0.2, -0.2, 0.1, -0.2, 0.2, 0.2, 0.1, 0.1, 0.2, 0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1},
{0.0, -0.1, -0.1, 0.0, -0.1, 0.2, 0.2, -0.2, -0.1, -0.0, -0.2, 0.1, -0.0, 0.2, 0.2, 0.1, -0.1, -0.0, -0.1, 0.1, -0.2, 0.1, 0.1, -0.2, -0.1},
{-0.0, -0.1, -0.1, 0.1, 0.0, 0.2, -0.0, 0.1, 0.1, -0.0, -0.1, -0.2, -0.1, -0.1, -0.2, -0.1, -0.1, 0.0, 0.1, 0.2, 0.1, 0.2, -0.2, -0.0, 0.0},
{-0.2, 0.2, -0.1, 0.2, -0.2, 0.1, 0.2, 0.0, 0.0, 0.2, 0.2, 0.1, -0.1, -0.1, -0.2, -0.2, -0.0, 0.1, -0.0, 0.1, 0.1, 0.0, -0.2, 0.0, -0.1},
{0.1, 0.1, 0.0, 0.0, 0.0, -0.0, -0.0, 0.0, -0.0, 0.1, 0.2, -0.0, -0.1, -0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.0, -0.0, -0.1, 0.0, 0.2, 0.1},
{0.1, -0.1, -0.1, -0.2, 0.1, 0.1, 0.2, 0.2, 0.0, 0.1, 0.1, 0.2, -0.0, 0.1, 0.0, 0.2, 0.1, 0.2, 0.1, -0.2, -0.0, 0.0, -0.1, -0.0, -0.2},
{0.2, -0.0, -0.2, 0.0, -0.0, -0.0, 0.2, 0.1, 0.1, -0.2, -0.2, -0.1, -0.2, 0.2, 0.1, 0.0, 0.1, -0.0, 0.1, -0.2, 0.0, 0.2, -0.1, 0.0, 0.1},
{-0.1, -0.2, -0.1, 0.2, 0.2, 0.1, -0.1, -0.1, 0.2, 0.0, -0.1, 0.0, -0.2, -0.0, 0.1, -0.2, -0.0, -0.2, -0.1, 0.1, -0.1, 0.2, 0.2, -0.0, 0.2},
{0.1, -0.1, -0.0, -0.2, 0.0, 0.1, -0.1, 0.0, 0.1, -0.2, 0.2, -0.1, 0.1, 0.2, -0.1, 0.2, -0.0, -0.2, -0.1, -0.1, -0.1, 0.2, 0.0, 0.0, -0.2},
{0.1, 0.1, -0.1, -0.0, 0.1, 0.2, -0.1, -0.2, -0.1, -0.2, -0.2, 0.2, 0.2, -0.1, -0.1, -0.1, 0.0, 0.0, 0.0, -0.2, 0.0, -0.1, -0.2, -0.0, 0.2},
{0.0, 0.1, -0.1, -0.2, -0.0, 0.2, 0.1, 0.0, -0.1, 0.0, 0.1, -0.2, 0.1, 0.2, 0.1, 0.1, 0.1, 0.1, -0.1, -0.2, 0.0, -0.1, 0.1, 0.1, 0.0},
{0.2, -0.1, -0.2, -0.1, 0.0, -0.0, 0.1, 0.2, 0.0, -0.1, -0.1, -0.1, 0.0, 0.1, 0.1, 0.1, -0.2, -0.2, 0.1, 0.0, -0.1, -0.0, 0.2, -0.1, -0.1},
{-0.1, -0.1, -0.1, 0.1, 0.2, 0.0, 0.0, -0.1, -0.0, 0.2, -0.1, 0.1, 0.0, 0.2, -0.1, 0.1, -0.1, -0.2, 0.0, 0.0, 0.0, 0.2, 0.2, 0.1, 0.1},
{0.1, 0.1, 0.1, -0.0, 0.2, -0.1, 0.2, -0.1, -0.2, -0.2, 0.0, 0.1, 0.1, 0.1, -0.1, -0.0, -0.2, -0.1, 0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.2},
{0.2, 0.1, 0.2, 0.2, -0.0, 0.1, -0.0, 0.2, 0.0, 0.1, 0.2, 0.0, -0.2, -0.1, -0.1, 0.2, -0.0, -0.0, -0.1, 0.1, 0.0, -0.2, 0.1, -0.1, -0.1},
{0.1, 0.2, 0.1, 0.1, -0.2, -0.0, 0.2, 0.0, -0.2, 0.0, 0.1, 0.0, 0.0, 0.1, -0.1, 0.1, 0.0, -0.1, -0.1, -0.0, 0.1, -0.1, -0.0, -0.2, 0.2},
{-0.1, -0.2, 0.1, -0.0, 0.1, -0.1, -0.1, 0.0, 0.2, 0.1, -0.0, -0.0, 0.1, 0.0, -0.0, -0.1, 0.0, 0.1, -0.2, -0.1, 0.1, -0.2, 0.0, -0.0, 0.0},
{-0.0, -0.0, -0.2, 0.0, -0.1, 0.2, -0.2, 0.1, -0.0, 0.0, -0.1, -0.1, 0.1, -0.1, -0.0, -0.0, -0.0, 0.1, 0.0, -0.0, 0.0, 0.0, -0.0, 0.1, -0.0},
{0.2, 0.1, -0.1, 0.1, -0.2, -0.1, -0.2, -0.1, 0.1, 0.1, 0.1, -0.1, 0.0, 0.2, 0.2, 0.1, 0.2, -0.1, -0.2, -0.1, -0.2, 0.2, 0.0, -0.0, 0.1},
{-0.1, 0.2, -0.0, -0.0, 0.1, 0.1, 0.0, -0.2, -0.2, -0.1, 0.0, -0.0, 0.0, 0.2, -0.0, 0.0, 0.1, 0.1, 0.2, -0.1, -0.2, -0.2, -0.1, -0.2, 0.0},
{-0.0, 0.2, 0.1, 0.2, 0.0, 0.2, -0.0, 0.1, 0.2, -0.1, 0.1, -0.2, 0.1, -0.1, -0.1, -0.1, 0.1, -0.1, -0.0, -0.2, -0.2, -0.1, -0.2, -0.1, 0.1},
{0.0, 0.2, 0.1, 0.1, -0.1, 0.1, -0.2, 0.1, -0.0, 0.1, -0.1, -0.0, -0.2, -0.1, 0.2, -0.0, 0.0, 0.2, -0.2, 0.1, -0.0, -0.2, 0.1, -0.1, -0.0},
{-0.1, 0.2, 0.2, 0.1, 0.1, -0.2, -0.2, -0.1, 0.1, 0.1, -0.2, 0.1, 0.1, -0.1, -0.2, -0.2, -0.1, -0.0, -0.1, 0.2, 0.0, -0.2, -0.0, -0.1, 0.2},
{-0.0, 0.1, -0.2, -0.1, 0.1, 0.0, -0.1, -0.1, 0.0, 0.0, -0.1, 0.2, -0.1, -0.1, -0.1, 0.1, 0.0, 0.0, 0.2, -0.1, -0.1, -0.1, -0.0, -0.1, 0.2},
{-0.0, 0.0, -0.0, 0.2, 0.1, 0.1, -0.2, -0.2, -0.2, 0.1, 0.0, -0.1, 0.0, -0.2, 0.1, -0.1, 0.1, 0.1, -0.0, 0.0, 0.2, 0.1, 0.0, 0.1, 0.1},
{-0.1, -0.1, -0.1, 0.1, 0.1, 0.0, 0.0, -0.2, -0.1, 0.1, 0.0, 0.1, -0.1, -0.0, 0.2, -0.1, -0.1, 0.0, 0.1, 0.0, -0.2, -0.2, -0.1, 0.1, 0.1},
{0.1, -0.1, 0.0, -0.1, 0.0, -0.1, -0.1, 0.1, -0.1, 0.1, -0.0, -0.2, -0.1, 0.2, -0.2, -0.1, -0.1, -0.1, 0.1, -0.2, -0.0, 0.1, 0.1, -0.1, 0.2},
{-0.1, -0.1, -0.2, -0.1, 0.0, 0.2, 0.0, -0.1, -0.1, 0.2, -0.1, -0.0, -0.0, 0.1, 0.1, 0.1, 0.1, 0.0, 0.2, -0.0, 0.0, 0.0, 0.1, -0.1, -0.1},
{0.2, 0.1, -0.2, 0.1, -0.0, -0.1, -0.0, 0.1, -0.2, -0.2, 0.2, 0.0, 0.0, -0.1, 0.1, -0.0, 0.0, -0.1, 0.1, 0.1, -0.0, -0.1, -0.1, -0.1, 0.1},
{-0.1, 0.1, 0.1, 0.2, 0.1, 0.1, -0.1, 0.0, 0.0, -0.1, 0.1, -0.1, -0.0, -0.0, 0.0, 0.1, 0.2, -0.1, 0.1, 0.2, 0.1, 0.1, 0.1, -0.2, -0.2},
{-0.2, 0.1, 0.2, 0.1, 0.1, 0.1, -0.1, -0.2, 0.1, -0.1, 0.0, 0.1, -0.0, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1, -0.0, 0.1, 0.1, 0.1, 0.1, 0.2},
{-0.1, -0.2, -0.0, 0.0, 0.1, 0.1, -0.1, -0.1, -0.1, 0.0, -0.1, -0.1, -0.1, -0.2, 0.2, -0.1, 0.0, 0.1, -0.0, 0.2, -0.0, -0.1, -0.2, 0.2, -0.0},
{-0.1, 0.2, -0.1, 0.1, 0.0, -0.1, 0.0, -0.0, -0.1, 0.1, -0.0, -0.1, 0.1, -0.0, -0.1, -0.2, 0.2, -0.0, 0.1, 0.2, 0.0, 0.1, 0.2, 0.0, 0.1},
}
);
Matrix  policy_head_bias   (
{-0.1, -0.1, 0.0, 0.1, 0.1, -0.0, -0.1, 0.0, -0.2, -0.2, 0.2, 0.2, -0.0, -0.1, 0.2, 0.2, 0.1, -0.1, 0.1, -0.0, -0.1, 0.1, -0.1, -0.2, -0.2, 0.2, -0.1, 0.0, 0.2, -0.1, -0.2, 0.0, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, 0.1, 0.1, -0.1, -0.1, -0.1, -0.1, -0.2, 0.2, 0.2, -0.2, -0.2, 0.1, 0.1, -0.2, 0.1, -0.0, -0.1, 0.1, 0.1, -0.2, -0.0, -0.1, 0.2, 0.1, -0.2, -0.1, 0.1, -0.0, 0.1, 0.2, -0.1, -0.0, 0.1, 0.1, 0.2, 0.0, 0.1, -0.1, 0.1, 0.1, -0.0, -0.2, 0.1, 0.0, 0.0, -0.2, -0.2, 0.0, 0.2, -0.2, 0.1, 0.2, 0.1, 0.1, 0.0, 0.1, 0.1, 0.0, -0.1, 0.2, 0.1, 0.0}
);





Transformer transformer(
    {
        TransformerBlock (
            SelfAttention(
                transformer_layers_0_attention_query_weight,
                transformer_layers_0_attention_query_bias,
                transformer_layers_0_attention_key_weight,
                transformer_layers_0_attention_key_bias,
                transformer_layers_0_attention_value_weight,
                transformer_layers_0_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_0_norm1_layer_norm_weight,
                transformer_layers_0_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_0_norm2_layer_norm_weight,
                transformer_layers_0_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_0_feed_forward_linear1_weight,
                transformer_layers_0_feed_forward_linear1_bias,
                transformer_layers_0_feed_forward_linear2_weight,
                transformer_layers_0_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_0_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_0_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_0_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_0_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_1_attention_query_weight,
                transformer_layers_1_attention_query_bias,
                transformer_layers_1_attention_key_weight,
                transformer_layers_1_attention_key_bias,
                transformer_layers_1_attention_value_weight,
                transformer_layers_1_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_1_norm1_layer_norm_weight,
                transformer_layers_1_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_1_norm2_layer_norm_weight,
                transformer_layers_1_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_1_feed_forward_linear1_weight,
                transformer_layers_1_feed_forward_linear1_bias,
                transformer_layers_1_feed_forward_linear2_weight,
                transformer_layers_1_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_1_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_1_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_1_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_1_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_2_attention_query_weight,
                transformer_layers_2_attention_query_bias,
                transformer_layers_2_attention_key_weight,
                transformer_layers_2_attention_key_bias,
                transformer_layers_2_attention_value_weight,
                transformer_layers_2_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_2_norm1_layer_norm_weight,
                transformer_layers_2_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_2_norm2_layer_norm_weight,
                transformer_layers_2_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_2_feed_forward_linear1_weight,
                transformer_layers_2_feed_forward_linear1_bias,
                transformer_layers_2_feed_forward_linear2_weight,
                transformer_layers_2_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_2_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_2_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_2_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_2_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_3_attention_query_weight,
                transformer_layers_3_attention_query_bias,
                transformer_layers_3_attention_key_weight,
                transformer_layers_3_attention_key_bias,
                transformer_layers_3_attention_value_weight,
                transformer_layers_3_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_3_norm1_layer_norm_weight,
                transformer_layers_3_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_3_norm2_layer_norm_weight,
                transformer_layers_3_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_3_feed_forward_linear1_weight,
                transformer_layers_3_feed_forward_linear1_bias,
                transformer_layers_3_feed_forward_linear2_weight,
                transformer_layers_3_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_3_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_3_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_3_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_3_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_4_attention_query_weight,
                transformer_layers_4_attention_query_bias,
                transformer_layers_4_attention_key_weight,
                transformer_layers_4_attention_key_bias,
                transformer_layers_4_attention_value_weight,
                transformer_layers_4_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_4_norm1_layer_norm_weight,
                transformer_layers_4_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_4_norm2_layer_norm_weight,
                transformer_layers_4_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_4_feed_forward_linear1_weight,
                transformer_layers_4_feed_forward_linear1_bias,
                transformer_layers_4_feed_forward_linear2_weight,
                transformer_layers_4_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_4_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_4_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_4_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_4_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_5_attention_query_weight,
                transformer_layers_5_attention_query_bias,
                transformer_layers_5_attention_key_weight,
                transformer_layers_5_attention_key_bias,
                transformer_layers_5_attention_value_weight,
                transformer_layers_5_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_5_norm1_layer_norm_weight,
                transformer_layers_5_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_5_norm2_layer_norm_weight,
                transformer_layers_5_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_5_feed_forward_linear1_weight,
                transformer_layers_5_feed_forward_linear1_bias,
                transformer_layers_5_feed_forward_linear2_weight,
                transformer_layers_5_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_5_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_5_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_5_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_5_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_6_attention_query_weight,
                transformer_layers_6_attention_query_bias,
                transformer_layers_6_attention_key_weight,
                transformer_layers_6_attention_key_bias,
                transformer_layers_6_attention_value_weight,
                transformer_layers_6_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_6_norm1_layer_norm_weight,
                transformer_layers_6_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_6_norm2_layer_norm_weight,
                transformer_layers_6_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_6_feed_forward_linear1_weight,
                transformer_layers_6_feed_forward_linear1_bias,
                transformer_layers_6_feed_forward_linear2_weight,
                transformer_layers_6_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_6_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_6_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_6_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_6_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_7_attention_query_weight,
                transformer_layers_7_attention_query_bias,
                transformer_layers_7_attention_key_weight,
                transformer_layers_7_attention_key_bias,
                transformer_layers_7_attention_value_weight,
                transformer_layers_7_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_7_norm1_layer_norm_weight,
                transformer_layers_7_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_7_norm2_layer_norm_weight,
                transformer_layers_7_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_7_feed_forward_linear1_weight,
                transformer_layers_7_feed_forward_linear1_bias,
                transformer_layers_7_feed_forward_linear2_weight,
                transformer_layers_7_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_7_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_7_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_7_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_7_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_8_attention_query_weight,
                transformer_layers_8_attention_query_bias,
                transformer_layers_8_attention_key_weight,
                transformer_layers_8_attention_key_bias,
                transformer_layers_8_attention_value_weight,
                transformer_layers_8_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_8_norm1_layer_norm_weight,
                transformer_layers_8_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_8_norm2_layer_norm_weight,
                transformer_layers_8_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_8_feed_forward_linear1_weight,
                transformer_layers_8_feed_forward_linear1_bias,
                transformer_layers_8_feed_forward_linear2_weight,
                transformer_layers_8_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_8_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_8_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_8_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_8_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_9_attention_query_weight,
                transformer_layers_9_attention_query_bias,
                transformer_layers_9_attention_key_weight,
                transformer_layers_9_attention_key_bias,
                transformer_layers_9_attention_value_weight,
                transformer_layers_9_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_9_norm1_layer_norm_weight,
                transformer_layers_9_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_9_norm2_layer_norm_weight,
                transformer_layers_9_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_9_feed_forward_linear1_weight,
                transformer_layers_9_feed_forward_linear1_bias,
                transformer_layers_9_feed_forward_linear2_weight,
                transformer_layers_9_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_9_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_9_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_9_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_9_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_10_attention_query_weight,
                transformer_layers_10_attention_query_bias,
                transformer_layers_10_attention_key_weight,
                transformer_layers_10_attention_key_bias,
                transformer_layers_10_attention_value_weight,
                transformer_layers_10_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_10_norm1_layer_norm_weight,
                transformer_layers_10_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_10_norm2_layer_norm_weight,
                transformer_layers_10_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_10_feed_forward_linear1_weight,
                transformer_layers_10_feed_forward_linear1_bias,
                transformer_layers_10_feed_forward_linear2_weight,
                transformer_layers_10_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_10_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_10_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_10_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_10_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_11_attention_query_weight,
                transformer_layers_11_attention_query_bias,
                transformer_layers_11_attention_key_weight,
                transformer_layers_11_attention_key_bias,
                transformer_layers_11_attention_value_weight,
                transformer_layers_11_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_11_norm1_layer_norm_weight,
                transformer_layers_11_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_11_norm2_layer_norm_weight,
                transformer_layers_11_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_11_feed_forward_linear1_weight,
                transformer_layers_11_feed_forward_linear1_bias,
                transformer_layers_11_feed_forward_linear2_weight,
                transformer_layers_11_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_11_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_11_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_11_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_11_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_12_attention_query_weight,
                transformer_layers_12_attention_query_bias,
                transformer_layers_12_attention_key_weight,
                transformer_layers_12_attention_key_bias,
                transformer_layers_12_attention_value_weight,
                transformer_layers_12_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_12_norm1_layer_norm_weight,
                transformer_layers_12_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_12_norm2_layer_norm_weight,
                transformer_layers_12_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_12_feed_forward_linear1_weight,
                transformer_layers_12_feed_forward_linear1_bias,
                transformer_layers_12_feed_forward_linear2_weight,
                transformer_layers_12_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_12_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_12_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_12_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_12_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_13_attention_query_weight,
                transformer_layers_13_attention_query_bias,
                transformer_layers_13_attention_key_weight,
                transformer_layers_13_attention_key_bias,
                transformer_layers_13_attention_value_weight,
                transformer_layers_13_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_13_norm1_layer_norm_weight,
                transformer_layers_13_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_13_norm2_layer_norm_weight,
                transformer_layers_13_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_13_feed_forward_linear1_weight,
                transformer_layers_13_feed_forward_linear1_bias,
                transformer_layers_13_feed_forward_linear2_weight,
                transformer_layers_13_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_13_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_13_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_13_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_13_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    
        TransformerBlock (
            SelfAttention(
                transformer_layers_14_attention_query_weight,
                transformer_layers_14_attention_query_bias,
                transformer_layers_14_attention_key_weight,
                transformer_layers_14_attention_key_bias,
                transformer_layers_14_attention_value_weight,
                transformer_layers_14_attention_value_bias
                ),

            LayerNorm(
                transformer_layers_14_norm1_layer_norm_weight,
                transformer_layers_14_norm1_layer_norm_bias
                ),
            LayerNorm(
                transformer_layers_14_norm2_layer_norm_weight,
                transformer_layers_14_norm2_layer_norm_bias
            ),
            FeedForward(
                transformer_layers_14_feed_forward_linear1_weight,
                transformer_layers_14_feed_forward_linear1_bias,
                transformer_layers_14_feed_forward_linear2_weight,
                transformer_layers_14_feed_forward_linear2_bias,
                LayerNorm(
                    transformer_layers_14_feed_forward_ln1_layer_norm_weight,
                    transformer_layers_14_feed_forward_ln1_layer_norm_bias
                ),
                LayerNorm(
                    transformer_layers_14_feed_forward_ln2_layer_norm_weight,
                    transformer_layers_14_feed_forward_ln2_layer_norm_bias
                    )
            )
        ),

    }

);


int main (){

    Matrix a({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,
             17,18,19,20,21,22,23,24,25});

    Matrix b = transformer.forward(a);

    b.print();

}