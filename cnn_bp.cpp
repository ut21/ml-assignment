#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>

using namespace std;

const int IMAGE_SIZE = 32 * 32 * 3;
const int NUM_IMAGES = 1000;
const int IMAGE_WIDTH = 32;
const int IMAGE_HEIGHT = 32;
const int IMAGE_CHANNELS = 3;
const int NUM_CLASSES = 10;
const float LEARNING_RATE = 0.001f;
const int EPOCHS = 5;

struct Image {
    vector<float> pixels; // normalized pixel values
    uint8_t label;
};

vector<Image> loadCIFAR10Batch(const string& filename) {
    vector<Image> images;
    ifstream file(filename, ios::binary);
    const int CHANNEL_SIZE = 1024; // 32*32
    const int IMAGE_SIZE = CHANNEL_SIZE * 3;
    
    if (file.is_open()) {
        for (int i = 0; i < NUM_IMAGES; ++i) {
            Image img;
            img.pixels.resize(IMAGE_SIZE);
            // Read label
            file.read(reinterpret_cast<char*>(&img.label), sizeof(img.label));

            // Create a temporary buffer for the raw data
            vector<uint8_t> buffer(IMAGE_SIZE);
            file.read(reinterpret_cast<char*>(buffer.data()), IMAGE_SIZE);
            
            // Convert buffer to float (and optionally normalize later)
            for (int j = 0; j < IMAGE_SIZE; ++j) {
                img.pixels[j] = static_cast<float>(buffer[j]);
            }
            
            images.push_back(img);
        }
        file.close();
    } else {
        cout << "Failed to open file: " << filename << endl;
    }
    return images;
}

void loadCIFAR10Data(const string& folder, vector<Image>& train_images, vector<Image>& test_images) {
    for (int i = 1; i <= 5; ++i) {
        string filename = folder + "/data_batch_" + to_string(i) + ".bin";
        vector<Image> batch = loadCIFAR10Batch(filename);
        train_images.insert(train_images.end(), batch.begin(), batch.end());
    }
    test_images = loadCIFAR10Batch(folder + "/test_batch.bin");
}

void normalizeImages(vector<Image>& images) {
    for (auto& img : images) {
        for (auto& pixel : img.pixels) {
            pixel = pixel / 255.0f;
        }
    }
}

float relu(float x) {
    return max(0.0f, x);
}

float relu_derivative(float x) {
    return x > 0 ? 1.0f : 0.0f;
}

vector<float> softmax(const vector<float>& x) {
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

vector<float> one_hot(uint8_t label) {
    vector<float> vec(NUM_CLASSES, 0.0f);
    vec[label] = 1.0f;
    return vec;
}

// Type alias for a 3D volume: [channels][height][width]
using Volume = vector<vector<vector<float>>>;

class ConvLayer {
public:
    int inChannels, outChannels;
    int kernelSize;
    // weights dimensions: [outChannels][inChannels][kernelSize][kernelSize]
    vector<vector<vector<vector<float>>>> weights;
    vector<float> biases;
    
    // Variables to store forward pass results
    Volume last_input;
    Volume last_output; // after ReLU activation

    ConvLayer(int inChannels, int outChannels, int kernelSize) 
        : inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize) {
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

    Volume forward(const Volume& input) {
        last_input = input;
        int H = input[0].size();
        int W = input[0][0].size();
        int pad = kernelSize / 2;
        // Initialize output volume [outChannels][H][W]
        Volume output(outChannels, vector<vector<float>>(H, vector<float>(W, 0.0f)));

        // Convolution + ReLU activation:
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
        last_output = output;
        return output;
    }

    // Backward pass for convolution layer.
    // d_out: gradient from the next layer with shape [outChannels][H][W].
    // Returns the gradient with respect to the input of this layer.
    Volume backward(const Volume& d_out) {
        int H = last_input[0].size();
        int W = last_input[0][0].size();
        int pad = kernelSize / 2;
        Volume d_input(inChannels, vector<vector<float>>(H, vector<float>(W, 0.0f)));
        // Initialize gradients for weights and biases.
        vector<vector<vector<vector<float>>>> d_weights(outChannels,
            vector<vector<vector<float>>>(inChannels, vector<vector<float>>(kernelSize, vector<float>(kernelSize, 0.0f))));
        vector<float> d_biases(outChannels, 0.0f);

        // Loop over output gradients.
        for (int oc = 0; oc < outChannels; ++oc) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    float grad = d_out[oc][i][j];
                    // Backprop through ReLU: if the activation was 0, no gradient passes.
                    if (last_output[oc][i][j] <= 0)
                        grad = 0;
                    d_biases[oc] += grad;
                    for (int ic = 0; ic < inChannels; ++ic) {
                        for (int m = 0; m < kernelSize; ++m) {
                            for (int n = 0; n < kernelSize; ++n) {
                                int ii = i + m - pad;
                                int jj = j + n - pad;
                                if (ii >= 0 && ii < H && jj >= 0 && jj < W) {
                                    d_weights[oc][ic][m][n] += last_input[ic][ii][jj] * grad;
                                    d_input[ic][ii][jj] += weights[oc][ic][m][n] * grad;
                                }
                            }
                        }
                    }
                }
            }
        }
        // Update weights and biases.
        for (int oc = 0; oc < outChannels; ++oc) {
            biases[oc] -= LEARNING_RATE * d_biases[oc];
            for (int ic = 0; ic < inChannels; ++ic) {
                for (int m = 0; m < kernelSize; ++m) {
                    for (int n = 0; n < kernelSize; ++n) {
                        weights[oc][ic][m][n] -= LEARNING_RATE * d_weights[oc][ic][m][n];
                    }
                }
            }
        }
        return d_input;
    }
};

class MaxPoolLayer {
public:
    int poolSize;
    // To store inputs and mask for backpropagation.
    Volume last_input;
    Volume mask; // same shape as input; 1.0f at positions of the maximum, else 0.

    MaxPoolLayer(int poolSize) : poolSize(poolSize) {}

    // Forward pass for max pooling. Input shape: [channels][H][W]
    Volume forward(const Volume& input) {
        last_input = input;
        int channels = input.size();
        int H = input[0].size();
        int W = input[0][0].size();
        int outH = H / poolSize;
        int outW = W / poolSize;
        Volume output(channels, vector<vector<float>>(outH, vector<float>(outW, 0.0f)));
        // Initialize mask with zeros.
        mask = Volume(channels, vector<vector<float>>(H, vector<float>(W, 0.0f)));
        
        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < outH; ++i) {
                for (int j = 0; j < outW; ++j) {
                    float maxVal = -1e9;
                    int max_i = -1, max_j = -1;
                    for (int m = 0; m < poolSize; ++m) {
                        for (int n = 0; n < poolSize; ++n) {
                            int ii = i * poolSize + m;
                            int jj = j * poolSize + n;
                            if (input[c][ii][jj] > maxVal) {
                                maxVal = input[c][ii][jj];
                                max_i = ii;
                                max_j = jj;
                            }
                        }
                    }
                    output[c][i][j] = maxVal;
                    mask[c][max_i][max_j] = 1.0f; // mark the position of the max
                }
            }
        }
        return output;
    }

    // Backward pass for max pooling.
    // d_out has shape [channels][H/poolSize][W/poolSize]
    // Returns gradient with respect to the input.
    Volume backward(const Volume& d_out) {
        int channels = last_input.size();
        int H = last_input[0].size();
        int W = last_input[0][0].size();
        Volume d_input(channels, vector<vector<float>>(H, vector<float>(W, 0.0f)));
        int outH = H / poolSize;
        int outW = W / poolSize;
        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < outH; ++i) {
                for (int j = 0; j < outW; ++j) {
                    // In each pooling window, only the position with mask==1 receives the gradient.
                    for (int m = 0; m < poolSize; ++m) {
                        for (int n = 0; n < poolSize; ++n) {
                            int ii = i * poolSize + m;
                            int jj = j * poolSize + n;
                            if (mask[c][ii][jj] == 1.0f) {
                                d_input[c][ii][jj] = d_out[c][i][j];
                            }
                        }
                    }
                }
            }
        }
        return d_input;
    }
};

class FullyConnectedLayer {
public:
    int inputSize, outputSize;
    // weights dimensions: [outputSize][inputSize]
    vector<vector<float>> weights;
    vector<float> biases;
    // Variables to store forward pass inputs/outputs.
    vector<float> last_input;
    vector<float> last_output;

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
        last_input = input;
        vector<float> output(outputSize, 0.0f);
        for (int i = 0; i < outputSize; ++i) {
            float sum = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                sum += weights[i][j] * input[j];
            }
            output[i] = sum;
        }
        last_output = output;
        return output;
    }

    // Backward pass for the fully connected layer.
    // d_out: gradient with respect to the layer output (size = outputSize)
    // Returns gradient with respect to the layer input.
    vector<float> backward(const vector<float>& d_out) {
        vector<vector<float>> dW(outputSize, vector<float>(inputSize, 0.0f));
        vector<float> dB(outputSize, 0.0f);
        vector<float> d_input(inputSize, 0.0f);
        for (int i = 0; i < outputSize; i++) {
            dB[i] = d_out[i];
            for (int j = 0; j < inputSize; j++) {
                dW[i][j] = d_out[i] * last_input[j];
                d_input[j] += weights[i][j] * d_out[i];
            }
        }
        // Update weights and biases.
        for (int i = 0; i < outputSize; i++) {
            biases[i] -= LEARNING_RATE * dB[i];
            for (int j = 0; j < inputSize; j++) {
                weights[i][j] -= LEARNING_RATE * dW[i][j];
            }
        }
        return d_input;
    }
};

class CNN {
public:
    ConvLayer conv;
    MaxPoolLayer pool;
    FullyConnectedLayer fc;

    CNN() 
        // Example architecture: conv layer with 8 filters (3x3 kernel); pool with 2x2; fc from flattened pooled features to 10 classes.
        : conv(IMAGE_CHANNELS, 8, 3),
          pool(2),
          fc(8 * (IMAGE_HEIGHT/2) * (IMAGE_WIDTH/2), NUM_CLASSES)
    {}

    // Utility: Convert the flat image vector (length 3072) into a 3D volume [channels][height][width]
    Volume imageToVolume(const Image& img) {
        Volume volume(IMAGE_CHANNELS, vector<vector<float>>(IMAGE_HEIGHT, vector<float>(IMAGE_WIDTH, 0.0f)));
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

    // Forward pass through the network. Returns the softmax probabilities.
    vector<float> forward(const Image& img) {
        auto inputVolume = imageToVolume(img);
        auto convOut = conv.forward(inputVolume);
        auto poolOut = pool.forward(convOut);
        vector<float> flattened;
        for (auto& channel : poolOut) {
            for (auto& row : channel) {
                for (float val : row) {
                    flattened.push_back(val);
                }
            }
        }
        auto fcOut = fc.forward(flattened);
        auto probs = softmax(fcOut);
        return probs;
    }

    // Backward pass through the network.
    // target: one-hot vector for the true class.
    void backward(const Image& img, const vector<float>& target) {
        // Re-run forward pass to ensure all "last_*" variables are updated.
        auto inputVolume = imageToVolume(img);
        auto convOut = conv.forward(inputVolume);        // conv.last_input and conv.last_output set here.
        auto poolOut = pool.forward(convOut);              // pool.last_input and mask set here.
        vector<float> flattened;
        for (auto& channel : poolOut) {
            for (auto& row : channel) {
                for (float val : row) {
                    flattened.push_back(val);
                }
            }
        }
        auto fcOut = fc.forward(flattened);                // fc.last_input and fc.last_output set here.
        auto probs = softmax(fcOut);

        // Compute gradient at the fully connected layer output:
        // For softmax with cross-entropy loss, gradient is (probs - target).
        vector<float> d_fc_out(NUM_CLASSES, 0.0f);
        for (int i = 0; i < NUM_CLASSES; i++) {
            d_fc_out[i] = probs[i] - target[i];
        }
        // Backprop through fully connected layer.
        vector<float> d_flatten = fc.backward(d_fc_out);

        // Reshape d_flatten to the shape of the pooled output.
        int channels = poolOut.size();
        int pooledH = poolOut[0].size();
        int pooledW = poolOut[0][0].size();
        Volume d_pool(channels, vector<vector<float>>(pooledH, vector<float>(pooledW, 0.0f)));
        int idx = 0;
        for (int c = 0; c < channels; c++) {
            for (int i = 0; i < pooledH; i++) {
                for (int j = 0; j < pooledW; j++) {
                    d_pool[c][i][j] = d_flatten[idx++];
                }
            }
        }
        // Backprop through max pooling layer.
        Volume d_conv = pool.backward(d_pool);
        // Backprop through convolutional layer.
        conv.backward(d_conv);
    }

    // Train on one example.
    void trainOnExample(const Image& img) {
        vector<float> target = one_hot(img.label);
        // Forward pass is not used separately because backward() re–runs it.
        backward(img, target);
    }
};

int main() {
    string folder = "cifar-10-batches-bin"; // Path to CIFAR-10 data
    vector<Image> train_images, test_images;

    loadCIFAR10Data(folder, train_images, test_images);
    normalizeImages(train_images);
    normalizeImages(test_images);

    CNN model;

    // Training loop (very basic, one image at a time)
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        cout << "Epoch " << epoch + 1 << "/" << EPOCHS << endl;
        int count = 0;
        for (const auto& img : train_images) {
            model.trainOnExample(img);
            count++;
            if (count % 1000 == 0) {
                cout << "Processed " << count << " images" << endl;
            }
        }
    }

    // Test on one image and display the prediction.
    if (!test_images.empty()) {
        auto probs = model.forward(test_images[0]);
        cout << "Predicted probabilities: ";
        for (auto p : probs) {
            cout << p << " ";
        }
        cout << "\nActual label: " << static_cast<int>(test_images[0].label) << endl;
    }

    return 0;
}
