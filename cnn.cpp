#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <cassert>
// #include <opencv2/opencv.hpp>

using namespace std;
// using namespace cv;

const int IMAGE_SIZE = 32 * 32 * 3;
const int NUM_IMAGES = 1000;
const int IMAGE_WIDTH = 32;
const int IMAGE_HEIGHT = 32;
const int IMAGE_CHANNELS = 3;
const int NUM_CLASSES = 10;
const float LEARNING_RATE = 0.001f;
const int EPOCHS = 2;

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
            
            // Convert buffer to float (and optionally normalize here or later)
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
            pixel = static_cast<float>(pixel) / 255.0f;
        }
    }
}

// void displayImage(const Image& img) {
//     // Create an empty Mat of 32x32 with 3 channels (CV_32FC3)
//     Mat image(32, 32, CV_32FC3);
    
//     const int channelSize = 1024; // 32*32
//     // Loop over each pixel position
//     for (int i = 0; i < 32; ++i) {
//         for (int j = 0; j < 32; ++j) {
//             int index = i * 32 + j;
//             // Get channels from the CIFAR-10 format:
//             float red   = img.pixels[index];          // First 1024 values
//             float green = img.pixels[index + channelSize];  // Next 1024 values
//             float blue  = img.pixels[index + 2 * channelSize];  // Last 1024 values
            
//             // OpenCV uses BGR ordering
//             Vec3f& color = image.at<Vec3f>(i, j);
//             color[0] = blue;
//             color[1] = green;
//             color[2] = red;
//         }
//     }
//     imshow("CIFAR-10 Image", image);
//     waitKey(0);
// }

float relu(float x) {
    return max(0.0f, x);
}

float relu_derivative(float x) {
    return x>0?1:0;
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

class ConvLayer {
public:
    int inChannels, outChannels;
    int kernelSize;
    vector<vector<vector<vector<float>>>> weights;
    vector<float> biases;

    ConvLayer(int inChannels, int outChannels, int kernelSize) : inChannels(inChannels), outChannels(outChannels), kernelSize(kernelSize){
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

    vector<vector<vector<float>>> forward(const vector<vector<vector<float>>>& input) {
        int H = input[0].size();
        int W = input[0][0].size();
        int pad = kernelSize / 2;
        // Initialize output volume [outChannels][H][W] with zeros.
        vector<vector<vector<float>>> output(outChannels,
            vector<vector<float>>(H, vector<float>(W, 0.0f)));

        // Convolution operation:
        for (int oc = 0; oc < outChannels; ++oc) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    float sum = biases[oc];
                    // Iterate over kernel window:
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
                    // Apply ReLU activation here
                    output[oc][i][j] = relu(sum);
                }
            }
        }
        return output;
    }
};

class MaxPoolLayer {
public:
    int poolSize;

    MaxPoolLayer(int poolSize) : poolSize(poolSize) {}

    // Forward pass for max pooling. Input shape: [channels][H][W]
    // Output shape will be [channels][H/poolSize][W/poolSize]
    vector<vector<vector<float>>> forward(const vector<vector<vector<float>>>& input) {
        int channels = input.size();
        int H = input[0].size();
        int W = input[0][0].size();
        int outH = H / poolSize;
        int outW = W / poolSize;
        vector<vector<vector<float>>> output(channels,
            vector<vector<float>>(outH, vector<float>(outW, 0.0f)));
        for (int c = 0; c < channels; ++c) {
            for (int i = 0; i < outH; ++i) {
                for (int j = 0; j < outW; ++j) {
                    float maxVal = -1e9;
                    for (int m = 0; m < poolSize; ++m) {
                        for (int n = 0; n < poolSize; ++n) {
                            int ii = i * poolSize + m;
                            int jj = j * poolSize + n;
                            if (input[c][ii][jj] > maxVal) {
                                maxVal = input[c][ii][jj];
                            }
                        }
                    }
                    output[c][i][j] = maxVal;
                }
            }
        }
        return output;
    }
    // Again, no backward pass is implemented for brevity.
};

class FullyConnectedLayer {
public:
    int inputSize, outputSize;
    vector<vector<float>> weights; // dimensions: [outputSize][inputSize]
    vector<float> biases; // dimensions: [outputSize]

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
            // No activation here; softmax will be applied later.
            output[i] = sum;
        }
        return output;
    }
    // Backward pass and update routines are omitted.
};

class CNN {
public:
    ConvLayer conv;
    MaxPoolLayer pool;
    FullyConnectedLayer fc;

    CNN() 
        // Example architecture: conv layer with 8 filters, 3x3 kernel; pool with 2x2; fc from flattened pooled features to 10 classes.
        : conv(IMAGE_CHANNELS, 8, 3), pool(2), fc(8 * (IMAGE_HEIGHT/2) * (IMAGE_WIDTH/2), NUM_CLASSES)
    {}

    // Utility: Convert the flat image vector (length 3072) into a 3D volume [channels][height][width]
    vector<vector<vector<float>>> imageToVolume(const Image& img) {
        vector<vector<vector<float>>> volume(IMAGE_CHANNELS, vector<vector<float>>(IMAGE_HEIGHT, vector<float>(IMAGE_WIDTH, 0.0f)));
        // CIFAR-10 stores channels sequentially
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
        // Convert image to volume shape.
        auto inputVolume = imageToVolume(img);
        // Conv layer forward pass.
        auto convOut = conv.forward(inputVolume);
        // Pool layer forward pass.
        auto poolOut = pool.forward(convOut);
        // Flatten the pooled output.
        vector<float> flattened;
        for (auto& channel : poolOut) {
            for (auto& row : channel) {
                for (float val : row) {
                    flattened.push_back(val);
                }
            }
        }
        // Fully connected layer forward pass.
        auto fcOut = fc.forward(flattened);
        // Apply softmax to get probabilities.
        auto probs = softmax(fcOut);
        return probs;
    }

    // Dummy backward pass (full backpropagation implementation would require computing gradients
    // for conv, pool, and fc layers). Here we only provide the forward pass and a placeholder.
    void backward(const Image& img, const vector<float>& target) {
        // Compute loss and gradients here.
        // Then update weights in conv.weights, conv.biases, fc.weights, fc.biases.
        // For brevity, the detailed derivation is omitted.
    }

    // Train on one example.
    void trainOnExample(const Image& img) {
        // Forward pass
        vector<float> output = forward(img);
        // Compute cross-entropy loss gradient for output layer.
        vector<float> target = one_hot(img.label);
        // In a real implementation, you would compute the gradient of the loss with respect to each parameter.
        // Here we simply call backward() as a placeholder.
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

    // Test on one image and display the prediction
    if (!test_images.empty()) {
        auto probs = model.forward(test_images[0]);
        cout << "Predicted probabilities: ";
        for (auto p : probs) {
            cout << p << " ";
        }
        cout << "\nActual label: " << static_cast<int>(test_images[0].label) << endl;

        // Optionally display the image using OpenCV
        // Mat image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC3);
        // // Reconstruct interleaved image for display (note the CIFAR-10 order is channelwise)
        // int channelSize = IMAGE_HEIGHT * IMAGE_WIDTH;
        // for (int i = 0; i < IMAGE_HEIGHT; ++i) {
        //     for (int j = 0; j < IMAGE_WIDTH; ++j) {
        //         Vec3f color;
        //         color[2] = test_images[0].pixels[0 * channelSize + i * IMAGE_WIDTH + j]; // Red
        //         color[1] = test_images[0].pixels[1 * channelSize + i * IMAGE_WIDTH + j]; // Green
        //         color[0] = test_images[0].pixels[2 * channelSize + i * IMAGE_WIDTH + j]; // Blue
        //         image.at<Vec3f>(i, j) = color;
        //     }
        // }
        // imshow("Test Image", image);
        // waitKey(0);
    }

    return 0;
}