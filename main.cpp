#include <iostream>
#include <vector>
#include "cnn_new.hpp"
#include <fstream>
// #include <opencv2/opencv.hpp>

using namespace std;
// using namespace cv;
// const int IMAGE_SIZE = 32 * 32 * 3;
const int NUM_IMAGES = 1000;
// const int IMAGE_WIDTH = 32;
// const int IMAGE_HEIGHT = 32;
// const int IMAGE_CHANNELS = 3;
// const int NUM_CLASSES = 10;
// const float LEARNING_RATE = 0.001f;
// const int EPOCHS = 2;

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

int main() {
    string folder = "cifar-10-batches-bin"; // Path to CIFAR-10 data
    vector<Image> train_images, test_images;
    loadCIFAR10Data(folder, train_images, test_images);

    CNN model;
    const int EPOCHS = 5;

    // Training loop: process one image at a time.
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
        vector<vector<vector<float>>> convOut, poolOut;
        vector<float> flat;
        auto probs = model.forward(test_images[0], convOut, poolOut, flat);
        cout << "Predicted probabilities: ";
        for (auto p : probs) {
            cout << p << " ";
        }
        cout << "\nActual label: " << static_cast<int>(test_images[0].label) << endl;

        // Reconstruct and display the image (channels are stored sequentially).
        // Mat image(IMAGE_HEIGHT, IMAGE_WIDTH, CV_32FC3);
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
