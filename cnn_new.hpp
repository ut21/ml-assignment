#ifndef CNN_NEW_HPP
#define CNN_NEW_HPP

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cassert>
// #include <opencv2/opencv.hpp>

using namespace std;
// using namespace cv;

const int IMAGE_WIDTH = 32;
const int IMAGE_HEIGHT = 32;
const int IMAGE_CHANNELS = 3;
const int NUM_CLASSES = 10;
const float LEARNING_RATE = 0.001f;

// ---------- Helper Functions ----------

// ReLU activation and its derivative.
inline float relu(float x) {
    return x > 0 ? x : 0;
}
inline float relu_derivative(float x) {
    return x > 0 ? 1 : 0;
}

// Softmax on a vector.
inline vector<float> softmax(const vector<float>& x) {
    vector<float> result(x.size());
    float maxElem = *max_element(x.begin(), x.end());
    float sum = 0;
    for (float val : x) {
        sum += exp(val - maxElem);
    }
    for (size_t i = 0; i < x.size(); i++) {
        result[i] = exp(x[i] - maxElem) / sum;
    }
    return result;
}

// One-hot encoding for labels.
inline vector<float> one_hot(uint8_t label) {
    vector<float> vec(NUM_CLASSES, 0.0f);
    vec[label] = 1.0f;
    return vec;
}

// ---------- Data Structure ----------

struct Image {
    // Pixels stored as normalized floats in channelwise order: [R(1024), G(1024), B(1024)]
    vector<float> pixels;
    uint8_t label;
};

// ---------- Convolutional Layer ----------

class ConvLayer {
public:
    int inChannels, outChannels, kernelSize;
    // Weights dimensions: [outChannels][inChannels][kernelSize][kernelSize]
    vector<vector<vector<vector<float>>>> weights;
    vector<float> biases;
    // To store intermediate values needed for backpropagation.
    vector<vector<vector<float>>> last_input;  // shape: [inChannels][H][W]
    vector<vector<vector<float>>> last_output; // after ReLU, shape: [outChannels][H][W]

    ConvLayer(int inChannels, int outChannels, int kernelSize)
        : inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize)
    {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> dist(0, 0.01f);
        weights.resize(outChannels, vector<vector<vector<float>>>(inChannels,
                     vector<vector<float>>(kernelSize, vector<float>(kernelSize))));
        biases.resize(outChannels, 0.0f);
        for (int oc = 0; oc < outChannels; ++oc) {
            for (int ic = 0; ic < inChannels; ++ic) {
                for (int i = 0; i < kernelSize; ++i) {
                    for (int j = 0; j < kernelSize; ++j) {
                        weights[oc][ic][i][j] = dist(gen);
                    }
                }
            }
        }
    }

    // Forward pass: input is a volume of shape [inChannels][H][W]
    vector<vector<vector<float>>> forward(const vector<vector<vector<float>>>& input) {
        last_input = input;  // store for backward
        int H = input[0].size();
        int W = input[0][0].size();
        int pad = kernelSize / 2;
        // Output volume: [outChannels][H][W]
        vector<vector<vector<float>>> output(outChannels,
            vector<vector<float>>(H, vector<float>(W, 0.0f)));

        for (int oc = 0; oc < outChannels; ++oc) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    float sum = biases[oc];
                    for (int ic = 0; ic < inChannels; ++ic) {
                        for (int m = 0; m < kernelSize; ++m) {
                            for (int n = 0; n < kernelSize; ++n) {
                                int ii = i + m - pad;
                                int jj = j + n - pad;
                                float val = 0.0f;
                                if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                                    val = input[ic][ii][jj];
                                }
                                sum += weights[oc][ic][m][n] * val;
                            }
                        }
                    }
                    output[oc][i][j] = relu(sum);
                }
            }
        }
        last_output = output; // store for backward (after activation)
        return output;
    }

    // Backward pass: computes gradients and updates weights & biases.
    // dL_dout: gradient with respect to output of this layer (same shape as last_output).
    // Returns gradient with respect to input.
    vector<vector<vector<float>>> backward(const vector<vector<vector<float>>>& dL_dout) {
        int H = last_input[0].size();
        int W = last_input[0][0].size();
        int pad = kernelSize / 2;
        // Initialize gradient with respect to input.
        vector<vector<vector<float>>> dL_dinput(inChannels, vector<vector<float>>(H, vector<float>(W, 0.0f)));
        // Initialize gradients for weights and biases.
        vector<vector<vector<vector<float>>>> gradW(outChannels,
            vector<vector<vector<float>>>(inChannels,
                vector<vector<float>>(kernelSize, vector<float>(kernelSize, 0.0f))));
        vector<float> gradB(outChannels, 0.0f);

        // Loop over output dimensions.
        for (int oc = 0; oc < outChannels; ++oc) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    // Backprop through ReLU.
                    float delta = dL_dout[oc][i][j] * relu_derivative(last_output[oc][i][j]);
                    gradB[oc] += delta;
                    for (int ic = 0; ic < inChannels; ++ic) {
                        for (int m = 0; m < kernelSize; ++m) {
                            for (int n = 0; n < kernelSize; ++n) {
                                int ii = i + m - pad;
                                int jj = j + n - pad;
                                if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                                    gradW[oc][ic][m][n] += last_input[ic][ii][jj] * delta;
                                    dL_dinput[ic][ii][jj] += weights[oc][ic][m][n] * delta;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Update weights and biases.
        for (int oc = 0; oc < outChannels; ++oc) {
            for (int ic = 0; ic < inChannels; ++ic) {
                for (int m = 0; m < kernelSize; ++m) {
                    for (int n = 0; n < kernelSize; ++n) {
                        weights[oc][ic][m][n] -= LEARNING_RATE * gradW[oc][ic][m][n];
                    }
                }
            }
            biases[oc] -= LEARNING_RATE * gradB[oc];
        }
        return dL_dinput;
    }
};

// ---------- Max Pooling Layer ----------

class MaxPoolLayer {
public:
    int poolSize;
    // To store indices of maximum values from forward pass.
    // Dimensions: [channels][outH][outW] each element is a pair {row_offset, col_offset} within the window.
    vector<vector<vector<pair<int,int>>>> maxIndices;
    // Also store the shape of the input.
    vector<vector<vector<float>>> last_input;

    MaxPoolLayer(int poolSize) : poolSize(poolSize) {}

    // Forward pass: input shape: [channels][H][W]
    // Output shape: [channels][H/poolSize][W/poolSize]
    vector<vector<vector<float>>> forward(const vector<vector<vector<float>>>& input) {
        last_input = input;
        int channels = input.size();
        int H = input[0].size();
        int W = input[0][0].size();
        int outH = H / poolSize;
        int outW = W / poolSize;
        vector<vector<vector<float>>> output(channels,
            vector<vector<float>>(outH, vector<float>(outW, 0.0f)));
        maxIndices.resize(channels, vector<vector<pair<int,int>>>(outH, vector<pair<int,int>>(outW, {0,0})));
        
        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < outH; ++i) {
                for (int j = 0; j < outW; ++j) {
                    float maxVal = -1e9;
                    int max_m = 0, max_n = 0;
                    for (int m = 0; m < poolSize; ++m) {
                        for (int n = 0; n < poolSize; ++n) {
                            int ii = i * poolSize + m;
                            int jj = j * poolSize + n;
                            if (input[c][ii][jj] > maxVal) {
                                maxVal = input[c][ii][jj];
                                max_m = m;
                                max_n = n;
                            }
                        }
                    }
                    output[c][i][j] = maxVal;
                    maxIndices[c][i][j] = {max_m, max_n};
                }
            }
        }
        return output;
    }

    // Backward pass: propagate gradients through the max pooling layer.
    // dL_dout: gradient w.r.t. the output of pooling layer (shape: same as forward output)
    // Returns: gradient w.r.t. the input (shape: same as last_input)
    vector<vector<vector<float>>> backward(const vector<vector<vector<float>>>& dL_dout) {
        int channels = last_input.size();
        int H = last_input[0].size();
        int W = last_input[0][0].size();
        int outH = H / poolSize;
        int outW = W / poolSize;
        vector<vector<vector<float>>> dL_dinput(channels,
            vector<vector<float>>(H, vector<float>(W, 0.0f)));
        
        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < outH; ++i) {
                for (int j = 0; j < outW; ++j) {
                    // Get the max index within the window.
                    pair<int,int> idx = maxIndices[c][i][j];
                    int ii = i * poolSize + idx.first;
                    int jj = j * poolSize + idx.second;
                    dL_dinput[c][ii][jj] = dL_dout[c][i][j];
                }
            }
        }
        return dL_dinput;
    }
};

// ---------- Fully Connected Layer ----------

class FullyConnectedLayer {
public:
    int inputSize, outputSize;
    // weights dimensions: [outputSize][inputSize]
    vector<vector<float>> weights;
    vector<float> biases;

    FullyConnectedLayer(int inputSize, int outputSize)
        : inputSize(inputSize), outputSize(outputSize)
    {
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<float> dist(0, 0.01f);
        weights.resize(outputSize, vector<float>(inputSize, 0.0f));
        biases.resize(outputSize, 0.0f);
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] = dist(gen);
            }
        }
    }

    // Forward pass: input is a flat vector.
    vector<float> forward(const vector<float>& input) {
        assert(input.size() == inputSize);
        vector<float> output(outputSize, 0.0f);
        for (int i = 0; i < outputSize; ++i) {
            float sum = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                sum += weights[i][j] * input[j];
            }
            output[i] = sum;
        }
        return output;
    }

    // Backward pass:
    // dL_dout: gradient of loss with respect to output of FC layer (size outputSize)
    // input: the original input vector that was fed to forward()
    // Returns gradient with respect to the input (size inputSize).
    vector<float> backward(const vector<float>& dL_dout, const vector<float>& input) {
        vector<float> dL_din(inputSize, 0.0f);
        // Update weights and biases, and compute gradient w.r.t. input.
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                dL_din[j] += weights[i][j] * dL_dout[i];
                float grad = dL_dout[i] * input[j];
                weights[i][j] -= LEARNING_RATE * grad;
            }
            biases[i] -= LEARNING_RATE * dL_dout[i];
        }
        return dL_din;
    }
};

// ---------- CNN Model ----------

class CNN {
public:
    ConvLayer conv;
    MaxPoolLayer pool;
    FullyConnectedLayer fc;

    CNN()
        // Example architecture: one conv layer with 8 filters (3x3 kernel),
        // followed by 2x2 max pooling, then a fully connected layer.
        : conv(IMAGE_CHANNELS, 8, 3),
          pool(2),
          fc(8 * (IMAGE_HEIGHT/2) * (IMAGE_WIDTH/2), NUM_CLASSES)
    {}

    // Convert flat image vector (length 3072) to volume shape: [channels][height][width]
    vector<vector<vector<float>>> imageToVolume(const Image& img) {
        vector<vector<vector<float>>> volume(IMAGE_CHANNELS,
            vector<vector<float>>(IMAGE_HEIGHT, vector<float>(IMAGE_WIDTH, 0.0f)));
        int channelSize = IMAGE_HEIGHT * IMAGE_WIDTH;
        for (int c = 0; c < IMAGE_CHANNELS; ++c) {
            for (int i = 0; i < IMAGE_HEIGHT; ++i) {
                for (int j = 0; j < IMAGE_WIDTH; ++j) {
                    volume[c][i][j] = img.pixels[c * channelSize + i * IMAGE_WIDTH + j];
                }
            }
        }
        return volume;
    }

    // Forward pass through the network.
    // Returns the softmax probabilities.
    vector<float> forward(const Image& img, 
                          vector<vector<vector<float>>>& convOutRef,
                          vector<vector<vector<float>>>& poolOutRef,
                          vector<float>& flatRef) 
    {
        // Convert image to volume.
        auto inputVolume = imageToVolume(img);
        // Convolutional layer.
        auto convOut = conv.forward(inputVolume);
        convOutRef = convOut; // store for backpropagation
        // Pooling layer.
        auto poolOut = pool.forward(convOut);
        poolOutRef = poolOut; // store for backpropagation

        // Flatten pooled output.
        flatRef.clear();
        for (auto& channel : poolOut) {
            for (auto& row : channel) {
                for (float val : row) {
                    flatRef.push_back(val);
                }
            }
        }
        // Fully connected layer.
        auto fcOut = fc.forward(flatRef);
        auto probs = softmax(fcOut);
        return probs;
    }

    // Backward pass through the network.
    // target: one-hot encoded label.
    void backward(const Image& img, const vector<float>& target) {
        // --- Forward pass (store intermediate values) ---
        vector<vector<vector<float>>> convOut, poolOut;
        vector<float> flat;
        auto probs = forward(img, convOut, poolOut, flat);

        // --- Compute gradient at FC layer (softmax + cross-entropy) ---
        vector<float> dL_dfc(probs.size(), 0.0f);
        for (size_t i = 0; i < probs.size(); ++i) {
            dL_dfc[i] = probs[i] - target[i];
        }
        // Backprop through fully connected layer.
        vector<float> dL_dflat = fc.backward(dL_dfc, flat);

        // Reshape dL_dflat to the shape of poolOut.
        int poolChannels = poolOut.size();
        int poolH = poolOut[0].size();
        int poolW = poolOut[0][0].size();
        vector<vector<vector<float>>> dL_dpool(poolChannels,
            vector<vector<float>>(poolH, vector<float>(poolW, 0.0f)));
        int idx = 0;
        for (int c = 0; c < poolChannels; c++) {
            for (int i = 0; i < poolH; i++) {
                for (int j = 0; j < poolW; j++) {
                    dL_dpool[c][i][j] = dL_dflat[idx++];
                }
            }
        }
        // Backprop through pooling layer.
        auto dL_dconv = pool.backward(dL_dpool);
        // Backprop through convolutional layer.
        conv.backward(dL_dconv);
    }

    // Train on a single example.
    void trainOnExample(const Image& img) {
        vector<float> target = one_hot(img.label);
        backward(img, target);
    }
};

#endif  // CNN_HPP
