#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
// use std as namespace
using namespace std;


class Matrix {
public:
    // Constructor, destructor, copy constructor, and other utility methods

    // Methods for matrix operations (addition, multiplication, etc.)
    Matrix dot(const Matrix& other) const;
    Matrix add(const Matrix& other) const;
    Matrix subtract(const Matrix& other) const;
    Matrix transpose() const;
    Matrix dot_transpose(const Matrix& other) const;
    size_t rows, cols;
    std::vector<std::vector<float>> data;

Matrix(int rows, int cols) : rows(rows), cols(cols) {
    int size = rows * cols;
    #if defined(_WIN32) // For Windows
        data = reinterpret_cast<float*>(_aligned_malloc(size * sizeof(float), 64));
    #else // For Unix-based systems
        if (posix_memalign(reinterpret_cast<void**>(&data), 64, size * sizeof(float)) != 0) {
            data = nullptr;
        }
    #endif
    if (data) {
        std::memset(data, 0, size * sizeof(float));
    } else {
        // Handle allocation failure
    }
}

~Matrix() {
    #if defined(_WIN32)
        _aligned_free(data);
    #else
        free(data);
    #endif
}



    Matrix dot (const Matrix& other) const {
        Matrix result(rows, other.cols);
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < other.cols; j++) {
                float sum = 0;
                for (size_t k = 0; k < cols; k++) {
                    sum += data[i][k] * other.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }
        return result;
    }
};


class LayerNorm {
public:
    LayerNorm(size_t featureSize);
    Matrix forward(const Matrix& x) const;

private:
    Matrix weights, bias;
};

class FeedForward {
public:
    FeedForward(size_t inputSize, size_t hiddenSize, size_t outputSize);
    Matrix forward(const Matrix& x) const;

private:
    Matrix weight1, weight2, bias1, bias2;
};


class SelfAttention {
public:
    SelfAttention(size_t numHeads, size_t modelSize);
    Matrix forward(const Matrix& x) const;

private:
    Matrix queryWeights, keyWeights, valueWeights;
    size_t numHeads;
    Matrix splitHeads(const Matrix& x, size_t headSize) const;
    Matrix scaledDotProductAttention(const Matrix& queries, const Matrix& keys, const Matrix& values) const;
};


class TransformerBlock {
public:
    TransformerBlock(size_t modelSize, size_t numHeads, size_t feedForwardSize);
    Matrix forward(const Matrix& x) const;

private:
    SelfAttention attention;
    LayerNorm norm1, norm2;
    FeedForward feedForward;
};


class Transformer {
public:
    Transformer(size_t numLayers, size_t modelSize, size_t numHeads, size_t feedForwardSize);
    Matrix forward(const Matrix& x) const;

private:
    vector<TransformerBlock> blocks;
};
