#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <numeric> // For std::accumulate

using namespace std;

// --- Constants (Consider making some configurable if needed) ---
const int IMAGE_WIDTH = 32;
const int IMAGE_HEIGHT = 32;
const int IMAGE_CHANNELS = 3;
const int IMAGE_SIZE = IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNELS;
const int NUM_IMAGES_PER_BATCH = 10000; // CIFAR-10 specific
const int NUM_TRAIN_BATCHES = 5;
const int NUM_CLASSES = 10;
float LEARNING_RATE = 0.001f; // Make it non-const if you plan to decay it
const int EPOCHS = 15; // Increased epochs for deeper network
const int BATCH_SIZE = 1; // Current implementation uses SGD (batch size 1)

// --- Data Structures and Loading (Mostly unchanged) ---
struct Image {
    vector<float> pixels;
    uint8_t label;
};

// Type alias for a 3D volume: [channels][height][width]
using Volume = vector<vector<vector<float>>>;


vector<Image> loadCIFAR10Batch(const string& filename) {
    vector<Image> images;
    ifstream file(filename, ios::binary);
    const int CHANNEL_SIZE = IMAGE_HEIGHT * IMAGE_WIDTH;
    const int CURRENT_IMAGE_SIZE = CHANNEL_SIZE * IMAGE_CHANNELS; // Use constant IMAGE_SIZE

    if (file.is_open()) {
        for (int i = 0; i < NUM_IMAGES_PER_BATCH; ++i) {
            Image img;
            img.pixels.resize(CURRENT_IMAGE_SIZE);
            // Read label
            file.read(reinterpret_cast<char*>(&img.label), sizeof(img.label));
            // Read pixel data (R, G, B channels separately)
            vector<uint8_t> buffer(CURRENT_IMAGE_SIZE);
            file.read(reinterpret_cast<char*>(buffer.data()), CURRENT_IMAGE_SIZE);
            // Interleave is NOT needed if loading directly like this.
            // The file format is [label][R channel][G channel][B channel]
            // We need to reorder to [channels][height][width] format later
            // For now, store flat: RRR...GGG...BBB...
            for (int j = 0; j < CURRENT_IMAGE_SIZE; ++j) {
                 img.pixels[j] = static_cast<float>(buffer[j]);
            }

            // --- Correction: Reorder pixels to [C][H][W] format conceptually ---
            // The original code stored pixels as RRR... GGG... BBB... flat.
            // Let's keep it that way for loading, and imageToVolume will handle the reshaping.
            // vector<float> reordered_pixels(CURRENT_IMAGE_SIZE);
            // for(int c=0; c < IMAGE_CHANNELS; ++c) {
            //     for(int y=0; y < IMAGE_HEIGHT; ++y) {
            //         for(int x=0; x < IMAGE_WIDTH; ++x) {
            //             reordered_pixels[c * CHANNEL_SIZE + y * IMAGE_WIDTH + x] =
            //                 static_cast<float>(buffer[c * CHANNEL_SIZE + y * IMAGE_WIDTH + x]);
            //         }
            //     }
            // }
            // img.pixels = reordered_pixels; // Use reordered if preferred storage format changes

            images.push_back(img);
        }
        file.close();
    } else {
        cerr << "Error: Failed to open file: " << filename << endl; // Use cerr for errors
    }
    return images;
}

void loadCIFAR10Data(const string& folder, vector<Image>& train_images, vector<Image>& test_images) {
    for (int i = 1; i <= NUM_TRAIN_BATCHES; ++i) {
        string filename = folder + "/data_batch_" + to_string(i) + ".bin";
        cout << "Loading " << filename << endl;
        vector<Image> batch = loadCIFAR10Batch(filename);
        if (batch.empty()) {
             cerr << "Warning: Loaded empty batch from " << filename << endl;
        }
        train_images.insert(train_images.end(), batch.begin(), batch.end());
    }
     string test_filename = folder + "/test_batch.bin";
     cout << "Loading " << test_filename << endl;
    test_images = loadCIFAR10Batch(test_filename);
     if (test_images.empty()) {
         cerr << "Warning: Loaded empty test batch from " << test_filename << endl;
     }
     cout << "Loaded " << train_images.size() << " training images and " << test_images.size() << " test images." << endl;
}

void normalizeImages(vector<Image>& images) {
    // Calculate mean and std deviation per channel (more robust normalization)
    // Or simply divide by 255 as before for simplicity
     if (images.empty()) return;
    for (auto& img : images) {
        for (auto& pixel : img.pixels) {
            pixel = pixel / 255.0f;
            // Optional: Subtract mean, divide by std dev (calculated across dataset)
        }
    }
}

// --- Activation Functions ---
float relu(float x) {
    return max(0.0f, x);
}

float relu_derivative(float activated_output) { // Derivative depends on the *output* of ReLU
    return activated_output > 0 ? 1.0f : 0.0f;
}

vector<float> softmax(const vector<float>& x) {
    vector<float> result(x.size());
     if (x.empty()) return result; // Handle empty input
    float maxElem = *max_element(x.begin(), x.end());
    float sum = 0.0f;
     vector<float> exp_x(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        exp_x[i] = exp(x[i] - maxElem); // Subtract max for numerical stability
        sum += exp_x[i];
    }

    if (sum == 0.0f) { // Avoid division by zero
        // Handle this case, e.g., return uniform distribution or error
         fill(result.begin(), result.end(), 1.0f / x.size());
         return result;
    }

    for (size_t i = 0; i < x.size(); i++) {
        result[i] = exp_x[i] / sum;
    }
    return result;
}

// --- Utility Functions ---
vector<float> one_hot(uint8_t label) {
    vector<float> vec(NUM_CLASSES, 0.0f);
     if (label < NUM_CLASSES) { // Basic bounds check
        vec[label] = 1.0f;
     } else {
         cerr << "Warning: Label out of bounds: " << (int)label << endl;
     }
    return vec;
}

// --- Layer Implementations ---

class ConvLayer {
public:
    int inChannels, outChannels;
    int kernelSize;
    int stride; // Added stride
    int padding; // Added padding
    vector<vector<vector<vector<float>>>> weights; // [outC][inC][kH][kW]
    vector<float> biases; // [outC]

    // Caching for backward pass
    Volume last_input;
    Volume last_z; // Output *before* ReLU

    ConvLayer(int inC, int outC, int kSize, int s = 1, int p = -1) // Default stride 1, padding calculated if -1
        : inChannels(inC), outChannels(outC), kernelSize(kSize), stride(s) {

        // Calculate 'same' padding if p is -1
        padding = (p == -1) ? (kernelSize - 1) / 2 : p;

        // Initialization (using He initialization is often better for ReLU)
        random_device rd;
        mt19937 gen(rd());
        // He initialization variance: 2.0 / (input_fan_in)
        float stddev = sqrt(2.0f / (inChannels * kernelSize * kernelSize));
        normal_distribution<float> dist(0.0f, stddev);

        weights.resize(outChannels, vector<vector<vector<float>>>(inChannels,
                     vector<vector<float>>(kernelSize, vector<float>(kernelSize))));
        biases.resize(outChannels, 0.0f); // Initialize biases to zero

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
        last_input = input; // Cache input
        assert(!input.empty() && !input[0].empty() && !input[0][0].empty()); // Basic validation

        int H = input[0].size();
        int W = input[0][0].size();

        // Calculate output dimensions
        int outH = (H + 2 * padding - kernelSize) / stride + 1;
        int outW = (W + 2 * padding - kernelSize) / stride + 1;
        assert(outH > 0 && outW > 0); // Ensure valid output dimensions

        Volume output_z(outChannels, vector<vector<float>>(outH, vector<float>(outW, 0.0f)));

        // --- Convolution Operation ---
        for (int oc = 0; oc < outChannels; ++oc) { // Output channel
            for (int i = 0; i < outH; ++i) {       // Output height
                for (int j = 0; j < outW; ++j) {   // Output width
                    float sum = biases[oc];
                    for (int ic = 0; ic < inChannels; ++ic) { // Input channel
                        for (int m = 0; m < kernelSize; ++m) { // Kernel height
                            for (int n = 0; n < kernelSize; ++n) { // Kernel width
                                // Input coordinates corresponding to kernel position
                                int input_h = i * stride + m - padding;
                                int input_w = j * stride + n - padding;

                                // Check bounds (for padding)
                                if (input_h >= 0 && input_h < H && input_w >= 0 && input_w < W) {
                                    sum += weights[oc][ic][m][n] * input[ic][input_h][input_w];
                                }
                                // else: implicitly multiply by 0 (padded value)
                            }
                        }
                    }
                    output_z[oc][i][j] = sum; // Store pre-activation value
                }
            }
        }

        last_z = output_z; // Cache pre-activation output

        // --- Apply ReLU Activation ---
        Volume output_activated = output_z; // Copy structure
        for (int c = 0; c < outChannels; ++c) {
            for (int i = 0; i < outH; ++i) {
                for (int j = 0; j < outW; ++j) {
                    output_activated[c][i][j] = relu(output_z[c][i][j]);
                }
            }
        }

        return output_activated; // Return activated output
    }

     Volume backward(const Volume& d_out_activated) {
         // d_out_activated is the gradient of the loss w.r.t. the *activated* output of this layer

         assert(!last_input.empty());
         assert(d_out_activated.size() == outChannels);
         int H = last_input[0].size();
         int W = last_input[0][0].size();
         int outH = d_out_activated[0].size();
         int outW = d_out_activated[0][0].size();

         // --- Step 1: Gradient w.r.t. pre-activation output (dZ) ---
         // Apply derivative of ReLU: dZ = d_out_activated * relu_derivative(Z)
         Volume dZ = d_out_activated; // Copy structure
         for(int c = 0; c < outChannels; ++c) {
             for(int i = 0; i < outH; ++i) {
                 for(int j = 0; j < outW; ++j) {
                      // relu_derivative needs the *output* of relu (which is the input to the next layer)
                      // A simpler way: check the pre-activation value 'last_z'
                     dZ[c][i][j] *= (last_z[c][i][j] > 0 ? 1.0f : 0.0f);
                 }
             }
         }

         // --- Step 2: Gradients for weights (dW), biases (dB), and input (dX) ---
         Volume d_input(inChannels, vector<vector<float>>(H, vector<float>(W, 0.0f)));
         vector<vector<vector<vector<float>>>> d_weights(outChannels,
             vector<vector<vector<float>>>(inChannels, vector<vector<float>>(kernelSize, vector<float>(kernelSize, 0.0f))));
         vector<float> d_biases(outChannels, 0.0f);

         // Loop structure similar to forward pass, but calculating gradients
         for (int oc = 0; oc < outChannels; ++oc) {
             for (int i = 0; i < outH; ++i) {
                 for (int j = 0; j < outW; ++j) {
                     float grad_z = dZ[oc][i][j]; // Gradient from downstream, passed through ReLU derivative

                     // --- Bias Gradient ---
                     d_biases[oc] += grad_z;

                     // --- Weight and Input Gradients ---
                     for (int ic = 0; ic < inChannels; ++ic) {
                         for (int m = 0; m < kernelSize; ++m) {
                             for (int n = 0; n < kernelSize; ++n) {
                                 int input_h = i * stride + m - padding;
                                 int input_w = j * stride + n - padding;

                                 if (input_h >= 0 && input_h < H && input_w >= 0 && input_w < W) {
                                     // Weight Gradient: dW[oc][ic][m][n] += dZ[oc][i][j] * X[ic][input_h][input_w]
                                     d_weights[oc][ic][m][n] += grad_z * last_input[ic][input_h][input_w];

                                     // Input Gradient: dX[ic][input_h][input_w] += dZ[oc][i][j] * W[oc][ic][m][n]
                                     d_input[ic][input_h][input_w] += grad_z * weights[oc][ic][m][n];
                                 }
                             }
                         }
                     }
                 }
             }
         }

         // --- Step 3: Update weights and biases ---
         // Note: If using batching, average gradients before updating
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

         return d_input; // Return gradient w.r.t. the input of this layer
     }
};

class MaxPoolLayer {
public:
    int poolSize;
    int stride;
    Volume last_input;
    Volume mask; // Store locations of max values

    // Constructor allowing different pool size and stride
    MaxPoolLayer(int pSize, int s = -1) : poolSize(pSize) {
        stride = (s == -1) ? pSize : s; // Default stride equals pool size
         assert(poolSize > 0 && stride > 0);
    }

    Volume forward(const Volume& input) {
        last_input = input;
         assert(!input.empty() && !input[0].empty() && !input[0][0].empty());
        int channels = input.size();
        int H = input[0].size();
        int W = input[0][0].size();

        // Calculate output dimensions
        // Assuming no padding for standard max pooling
        int outH = (H - poolSize) / stride + 1;
        int outW = (W - poolSize) / stride + 1;
         assert(outH > 0 && outW > 0);

        Volume output(channels, vector<vector<float>>(outH, vector<float>(outW, 0.0f)));
        // Mask needs to have the same shape as the *input* to mark max locations
        mask = Volume(channels, vector<vector<float>>(H, vector<float>(W, 0.0f)));

        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < outH; ++i) {
                for (int j = 0; j < outW; ++j) {
                    float maxVal = -numeric_limits<float>::infinity(); // Use proper min value
                    int max_h = -1, max_w = -1;
                    // Loop over the pooling window
                    for (int m = 0; m < poolSize; ++m) {
                        for (int n = 0; n < poolSize; ++n) {
                            int input_h = i * stride + m;
                            int input_w = j * stride + n;
                            // Check bounds (should usually be within H, W if calculated correctly)
                            if (input_h < H && input_w < W) {
                                if (input[c][input_h][input_w] > maxVal) {
                                    maxVal = input[c][input_h][input_w];
                                    max_h = input_h;
                                    max_w = input_w;
                                }
                            }
                        }
                    }
                    output[c][i][j] = maxVal;
                    if (max_h != -1) { // Mark the location of the max value
                        mask[c][max_h][max_w] = 1.0f;
                    } else {
                        // This should not happen if input dimensions are valid
                        cerr << "Warning: Max pooling found no max value (check dimensions/input)" << endl;
                    }
                }
            }
        }
        return output;
    }

    Volume backward(const Volume& d_out) {
        // d_out is the gradient of the loss w.r.t the output of this pooling layer
        assert(!last_input.empty() && !mask.empty());
        int channels = last_input.size();
        int H = last_input[0].size();
        int W = last_input[0][0].size();
        assert(d_out.size() == channels);
        int outH = d_out[0].size();
        int outW = d_out[0][0].size();

        Volume d_input(channels, vector<vector<float>>(H, vector<float>(W, 0.0f)));

        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < outH; ++i) { // Loop through output gradient
                for (int j = 0; j < outW; ++j) {
                    float grad_out = d_out[c][i][j];
                    // Loop through the corresponding input window
                    for (int m = 0; m < poolSize; ++m) {
                        for (int n = 0; n < poolSize; ++n) {
                            int input_h = i * stride + m;
                            int input_w = j * stride + n;
                            if (input_h < H && input_w < W) {
                                // If this input location was the max, pass the gradient
                                if (mask[c][input_h][input_w] == 1.0f) {
                                     // Gradient adds up if an input location is max for multiple output locations (can happen with stride < poolSize)
                                    d_input[c][input_h][input_w] += grad_out;
                                }
                            }
                        }
                    }
                }
            }
        }
        return d_input; // Gradient w.r.t. the input of this layer
    }
};

class FullyConnectedLayer {
public:
    int inputSize, outputSize;
    bool apply_relu; // Flag to control ReLU activation
    vector<vector<float>> weights; // [outputSize][inputSize]
    vector<float> biases; // [outputSize]

    // Caching for backward pass
    vector<float> last_input;
    vector<float> last_z; // Output *before* activation

    FullyConnectedLayer(int inSize, int outSize, bool use_relu = false) // Default: no ReLU (e.g., for final layer)
        : inputSize(inSize), outputSize(outSize), apply_relu(use_relu)
    {
        assert(inputSize > 0 && outputSize > 0);
        random_device rd;
        mt19937 gen(rd());
        // He init for ReLU, Xavier/Glorot for linear/tanh (or use small stddev like before)
        float stddev = sqrt(2.0f / inputSize); // He init variance
        if (!apply_relu) {
            stddev = sqrt(1.0f / inputSize); // Xavier/Glorot variance approx
        }
        normal_distribution<float> dist(0.0f, stddev);

        weights.resize(outputSize, vector<float>(inputSize));
        biases.resize(outputSize, 0.0f); // Init biases to zero
        for (int i = 0; i < outputSize; ++i) {
            for (int j = 0; j < inputSize; ++j) {
                weights[i][j] = dist(gen);
            }
        }
    }

    vector<float> forward(const vector<float>& input) {
        assert((int)input.size() == inputSize);
        last_input = input; // Cache input

        vector<float> output_z(outputSize);
        for (int i = 0; i < outputSize; ++i) {
            float sum = biases[i];
            for (int j = 0; j < inputSize; ++j) {
                sum += weights[i][j] * input[j];
            }
            output_z[i] = sum; // Pre-activation output
        }

        last_z = output_z; // Cache pre-activation

        if (apply_relu) {
            vector<float> output_activated(outputSize);
             for(int i=0; i<outputSize; ++i) {
                 output_activated[i] = relu(output_z[i]);
             }
            return output_activated;
        } else {
            return output_z; // Return pre-activation output if no ReLU
        }
    }

     vector<float> backward(const vector<float>& d_out) {
         // d_out is the gradient of the loss w.r.t the output of this layer
         // (either activated or pre-activated, depending on apply_relu)
         assert((int)d_out.size() == outputSize);

         vector<float> dZ = d_out; // Gradient w.r.t pre-activation output

         // If ReLU was applied, backpropagate through it first
         if (apply_relu) {
             for (int i = 0; i < outputSize; ++i) {
                 dZ[i] *= (last_z[i] > 0 ? 1.0f : 0.0f); // Gradient * derivative_of_relu(Z)
             }
         }
         // Now dZ holds the gradient dL/dZ

         // --- Calculate Gradients: dW, dB, dX ---
         vector<vector<float>> dW(outputSize, vector<float>(inputSize, 0.0f));
         vector<float> dB(outputSize, 0.0f);
         vector<float> d_input(inputSize, 0.0f); // dL/dX (gradient w.r.t. input)

         for (int i = 0; i < outputSize; i++) { // Loop through outputs (neurons in this layer)
             // Bias gradient: dB[i] = dZ[i] * 1
             dB[i] = dZ[i];

             for (int j = 0; j < inputSize; j++) { // Loop through inputs
                 // Weight gradient: dW[i][j] = dZ[i] * X[j]
                 dW[i][j] = dZ[i] * last_input[j];

                 // Input gradient: dX[j] += dZ[i] * W[i][j]
                 d_input[j] += dZ[i] * weights[i][j];
             }
         }

         // --- Update weights and biases ---
         // Average gradients if using batching before this step
         for (int i = 0; i < outputSize; i++) {
             biases[i] -= LEARNING_RATE * dB[i];
             for (int j = 0; j < inputSize; j++) {
                 weights[i][j] -= LEARNING_RATE * dW[i][j];
             }
         }

         return d_input; // Return gradient w.r.t. the input of this layer
     }
};

// --- CNN Model Definition ---
// --- CNN Model Definition ---
class CNN {
public:
    // --- Configuration derived from architecture (Compile-time calculation) ---
    // Calculate and store the flattened size after the last pool layer *first*
    static constexpr int CONV1_OUTC = 32;
    static constexpr int CONV2_OUTC = 32;
    static constexpr int CONV3_OUTC = 64;
    static constexpr int CONV4_OUTC = 64; // Output channels of conv4/last conv before flatten
    static constexpr int FC1_HIDDEN = 128;

    static constexpr int POOL1_FACTOR = 2;
    static constexpr int POOL2_FACTOR = 2;
    // Calculate spatial dimensions after all pooling
    static constexpr int FINAL_H = IMAGE_HEIGHT / (POOL1_FACTOR * POOL2_FACTOR); // e.g., 32 / 4 = 8
    static constexpr int FINAL_W = IMAGE_WIDTH / (POOL1_FACTOR * POOL2_FACTOR);  // e.g., 32 / 4 = 8
    // Calculate flattened size for the first FC layer
    static constexpr int FC1_INPUT_SIZE = CONV4_OUTC * FINAL_H * FINAL_W; // e.g., 64 * 8 * 8 = 4096

    // --- Declare Layers (Now fc1 can use the calculated size) ---
    ConvLayer conv1;
    ConvLayer conv2;
    MaxPoolLayer pool1;
    ConvLayer conv3;
    ConvLayer conv4;
    MaxPoolLayer pool2;
    FullyConnectedLayer fc1; // Declaration order is now correct relative to size usage
    FullyConnectedLayer fc2; // Output layer

    // Constructor using the constexpr values
    CNN()
        // Layer Definitions: (inC, outC, kernelSize, stride, padding)
        : conv1(IMAGE_CHANNELS, CONV1_OUTC, 3, 1, -1),         // 3 -> 32
          conv2(CONV1_OUTC, CONV2_OUTC, 3, 1, -1),             // 32 -> 32
          pool1(POOL1_FACTOR, POOL1_FACTOR),                   // Pool by 2
          conv3(CONV2_OUTC, CONV3_OUTC, 3, 1, -1),             // 32 -> 64
          conv4(CONV3_OUTC, CONV4_OUTC, 3, 1, -1),             // 64 -> 64
          pool2(POOL2_FACTOR, POOL2_FACTOR),                   // Pool by 2
          // FC layers initialization using the pre-calculated static constexpr size
          fc1(FC1_INPUT_SIZE, FC1_HIDDEN, true),           // 4096 -> 128 (with ReLU)
          fc2(FC1_HIDDEN, NUM_CLASSES, false)              // 128 -> 10 (no ReLU)
    {
        cout << "CNN initialized." << endl;
        cout << "FC1 input size calculated as: " << FC1_INPUT_SIZE << endl; // Use constant
        // You can still assert here if desired, though constexpr guarantees it if inputs are right
        assert(FC1_INPUT_SIZE == 64 * 8 * 8); // Verify calculation based on architecture numbers used
         if (FC1_INPUT_SIZE <= 0) { // Add runtime check just in case logic changes
             cerr << "Error: Calculated FC1_INPUT_SIZE is not positive: " << FC1_INPUT_SIZE << endl;
             // Consider throwing an exception or exiting
             exit(1);
         }
    }

    // Utility: Convert flat image vector (R..R G..G B..B) to 3D volume [C][H][W]
    // (imageToVolume function remains the same)
     Volume imageToVolume(const Image& img) {
        Volume volume(IMAGE_CHANNELS, vector<vector<float>>(IMAGE_HEIGHT, vector<float>(IMAGE_WIDTH)));
        int channelSize = IMAGE_HEIGHT * IMAGE_WIDTH;
        assert((int)img.pixels.size() == IMAGE_SIZE);

        for (int c = 0; c < IMAGE_CHANNELS; ++c) {
            for (int i = 0; i < IMAGE_HEIGHT; ++i) {
                for (int j = 0; j < IMAGE_WIDTH; ++j) {
                    // Indexing assumes RRR... GGG... BBB... format
                    volume[c][i][j] = img.pixels[c * channelSize + i * IMAGE_WIDTH + j];
                }
            }
        }
        return volume;
    }


    // Flatten a Volume into a 1D vector
    // (flattenVolume function remains the same)
     vector<float> flattenVolume(const Volume& vol) {
        vector<float> flattened;
         if (vol.empty() || vol[0].empty() || vol[0][0].empty()) return flattened;

        int channels = vol.size();
        int height = vol[0].size();
        int width = vol[0][0].size();
        flattened.reserve(channels * height * width); // Pre-allocate memory

        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < height; ++i) {
                for (int j = 0; j < width; ++j) {
                    flattened.push_back(vol[c][i][j]);
                }
            }
        }
        return flattened;
    }


    // Forward pass through the entire network
    // (Forward pass logic remains the same, uses the initialized layers)
     vector<float> forward(const Image& img) {
        Volume current_vol = imageToVolume(img);

        // Block 1
        current_vol = conv1.forward(current_vol);
        current_vol = conv2.forward(current_vol);
        current_vol = pool1.forward(current_vol);

        // Block 2
        current_vol = conv3.forward(current_vol);
        current_vol = conv4.forward(current_vol);
        current_vol = pool2.forward(current_vol);

        // Flatten
        vector<float> flattened = flattenVolume(current_vol);
         // Use the constexpr value for assertion
         assert(static_cast<int>(flattened.size()) == FC1_INPUT_SIZE);

        // Fully Connected Layers
        vector<float> fc1_out = fc1.forward(flattened); // Includes ReLU if configured
        vector<float> fc2_out = fc2.forward(fc1_out);   // Raw scores (logits), no ReLU

        // Output Activation
        vector<float> probs = softmax(fc2_out);
        return probs;
    }


    // Backward pass through the network and update weights
    // (Backward pass needs update to use constexpr for unflattening dimensions)
    void backward(const Image& img, const vector<float>& target) {
        // --- Re-run forward pass to cache intermediate values ---
        Volume vol0 = imageToVolume(img);
        Volume vol1 = conv1.forward(vol0);
        Volume vol2 = conv2.forward(vol1);
        Volume vol3 = pool1.forward(vol2);
        Volume vol4 = conv3.forward(vol3);
        Volume vol5 = conv4.forward(vol4);
        Volume vol6 = pool2.forward(vol5); // Output of last pooling layer

        vector<float> flattened = flattenVolume(vol6);
        vector<float> fc1_out_activated = fc1.forward(flattened); // Includes caching within fc1
        vector<float> fc2_out_logits = fc2.forward(fc1_out_activated); // Includes caching within fc2
        vector<float> probs = softmax(fc2_out_logits);

        // --- Start Backpropagation ---
        vector<float> d_logits = probs;
        assert((int)target.size() == NUM_CLASSES);
        for (int i = 0; i < NUM_CLASSES; i++) {
            d_logits[i] -= target[i];
        }

        // Backprop through fc2
        vector<float> d_fc1_activated = fc2.backward(d_logits);
        // Backprop through fc1
        vector<float> d_flattened = fc1.backward(d_fc1_activated);

        // --- Unflatten the gradient ---
        // Reshape d_flattened back into the shape of pool2's output (vol6)
        // Use the constexpr values for dimensions!
        assert(static_cast<int>(d_flattened.size()) == FC1_INPUT_SIZE);

        // These dimensions MUST match the output of pool2 / input of fc1
        const int C_pool2 = CONV4_OUTC;
        const int H_pool2 = FINAL_H;
        const int W_pool2 = FINAL_W;

        // Check consistency with the actual forward pass output shape
         assert(vol6.size() == C_pool2);
         assert(vol6.empty() || vol6[0].size() == H_pool2);
         assert(vol6.empty() || vol6[0].empty() || vol6[0][0].size() == W_pool2);


        Volume d_pool2_out(C_pool2, vector<vector<float>>(H_pool2, vector<float>(W_pool2)));
        int idx = 0;
        for (int c = 0; c < C_pool2; ++c) {
            for (int i = 0; i < H_pool2; ++i) {
                for (int j = 0; j < W_pool2; ++j) {
                     // Bounds check for safety, though idx should equal FC1_INPUT_SIZE at the end
                     if (idx < d_flattened.size()) {
                        d_pool2_out[c][i][j] = d_flattened[idx++];
                     } else {
                         cerr << "Error: Unflattening index out of bounds!" << endl;
                         // Handle error appropriately, maybe fill with 0 or exit
                         d_pool2_out[c][i][j] = 0.0f;
                     }
                }
            }
        }
         assert(idx == FC1_INPUT_SIZE); // Verify all gradients were used

        // Backprop through Block 2
        Volume d_conv4_out = pool2.backward(d_pool2_out);
        Volume d_conv3_out = conv4.backward(d_conv4_out);
        Volume d_pool1_out = conv3.backward(d_conv3_out);

        // Backprop through Block 1
        Volume d_conv2_out = pool1.backward(d_pool1_out);
        Volume d_conv1_out = conv2.backward(d_conv2_out);
        /* Volume d_input_img = */ conv1.backward(d_conv1_out);
    }


    // Train on one example (SGD)
    // (trainOnExample function remains the same)
    void trainOnExample(const Image& img) {
        vector<float> target = one_hot(img.label);
        // Forward pass is re-run inside backward to set caches correctly
        backward(img, target);
    }
}; // End of CNN Class definition

// --- Testing Function ---
float testAccuracy(CNN& model, const vector<Image>& test_images) {
    int correct = 0;
     if (test_images.empty()) return 0.0f;

    for (const auto &img : test_images) {
        vector<float> probs = model.forward(img);
         if (probs.empty()) continue; // Skip if forward pass failed
        int predicted = distance(probs.begin(), max_element(probs.begin(), probs.end()));
        if (predicted == img.label)
            correct++;
    }
    return static_cast<float>(correct) / test_images.size();
}

// --- Main Function ---
int main() {
    string folder = "cifar-10-batches-bin"; // CHANGE THIS PATH if needed
    vector<Image> train_images, test_images;

    cout << "Loading CIFAR-10 data..." << endl;
    loadCIFAR10Data(folder, train_images, test_images);

     if (train_images.empty() || test_images.empty()) {
         cerr << "Error: Failed to load data. Check the path: " << folder << endl;
         cerr << "Ensure the .bin files (data_batch_1.bin, ..., test_batch.bin) are present." << endl;
         return 1;
     }

    cout << "Normalizing images..." << endl;
    normalizeImages(train_images);
    normalizeImages(test_images);

    cout << "Initializing CNN model..." << endl;
    CNN model;
    vector<float> epochAccuracies;

    cout << "Starting training..." << endl;
    for (int epoch = 0; epoch < EPOCHS; ++epoch) {
        cout << "--- Epoch " << epoch + 1 << "/" << EPOCHS << " ---" << endl;

        // Optional: Shuffle training data each epoch
        // random_shuffle(train_images.begin(), train_images.end());

        int count = 0;
        for (const auto& img : train_images) {
            model.trainOnExample(img);
            count++;
            if (count % 1000 == 0) { // Print progress
                cout << "\rProcessed " << count << "/" << train_images.size() << " images..." << flush;
            }
        }
        cout << "\rProcessed " << count << "/" << train_images.size() << " images." << endl; // Final count for the epoch
        // Accuracy on train set after epoch
        float acc_train = testAccuracy(model, train_images);
        cout << "Train accuracy" << acc_train * 100.0f << "%" << endl;

        // Evaluate on test set
        cout << "Evaluating on test set..." << endl;
        float acc = testAccuracy(model, test_images);
        epochAccuracies.push_back(acc);
        cout << "Test Accuracy after epoch " << epoch + 1 << ": " << acc * 100.0f << "%" << endl;

        //  Optional: Simple Learning Rate Decay
         if ((epoch + 1) % 5 == 0) { // Example: Decay every 5 epochs
            LEARNING_RATE *= 0.5f;
            cout << "Learning rate decayed to: " << LEARNING_RATE << endl;
         }
    }

    cout << "Training finished." << endl;

    // Save results
    ofstream file("accuracy_deeper_cnn.csv");
    if (file.is_open()) {
        file << "Epoch,Accuracy\n";
        for (size_t i = 0; i < epochAccuracies.size(); i++) {
            file << i + 1 << "," << epochAccuracies[i] << "\n";
        }
        file.close();
        cout << "Accuracy data saved to accuracy_deeper_cnn.csv" << endl;
    } else {
        cerr << "Error writing accuracy data to file." << endl;
    }

    return 0;
}