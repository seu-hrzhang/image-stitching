//
// Created by Starry Night on 2021/3/20.
//

#include "filter.h"

using namespace std;
using namespace cv;

// Get 2D convolution of input image and kernel
// Notes: Borders of the source image are extended using mirror symmetry
void Convolve2D(Mat src, Mat *dst, Mat kernel) {
    int size = kernel.rows;
    int center = (size - 1) / 2;

    // Extend image borders
    Mat srcExt;
    copyMakeBorder(src, srcExt, center, center, center, center,
                   BORDER_REFLECT_101);

    Mat temp(src.size(), dst->type());

    // Convolution
    for (int i = center; i < srcExt.rows - center; i++) {
        for (int j = center; j < srcExt.cols - center; j++) {
            double sum = 0.0;
            for (int x = 0; x < size; x++) {
                for (int y = 0; y < size; y++) {
                    sum += srcExt.at<uchar>(i - x + center, j - y + center) *
                           kernel.at<double>(x, y);
                }
            }
            switch (temp.type()) {
            case 0: {
                if (sum < 0)
                    sum = -sum;
                temp.at<uchar>(i - center, j - center) = sum;
                break;
            }
            case 6:
                temp.at<double>(i - center, j - center) = sum;
                break;
            default:
                break;
            }
        }
    }
    temp.copyTo(*dst);
}

// Generate a Gaussian kernel matrix using input params 'size' and 'sigma'
Mat genGaussianKernel(int size, double sigma) {
    int center = (int)(size - 1) / 2;
    double sum = 0.0;

    // Initialize kernel matrix
    Mat kernel(Size(size, size), CV_64FC1);

    // Calculate power values using Gaussian function
    // Note: Coefficients in Gaussian function are omitted for normalization
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel.at<double>(i, j) = exp(
                -((i - center) * (i - center) + (j - center) * (j - center)) /
                (2 * sigma * sigma));
            sum += kernel.at<double>(i, j);
        }
    }

    // Normalization
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            kernel.at<double>(i, j) /= sum;
        }
    }

    return kernel;
}

// Gaussian filter for color images (RGB)
void GaussianFilterRGB(Mat src, Mat *dst, int size, double sigma) {
    int center = (size - 1) / 2;

    // Extend image borders
    Mat srcExt;
    copyMakeBorder(src, srcExt, center, center, center, center,
                   BORDER_REFLECT_101);

    // Split channels of color image
    vector<Mat> channels;
    split(srcExt, channels);

    Mat kernel = genGaussianKernel(size, sigma);

    // All 3 channels of color image are processed using loop 'k'
    for (int k = 0; k < 3; k++) {
        Mat temp(channels[k].size(), channels[k].type());
        Convolve2D(channels[k], &temp, kernel);
        channels[k] = temp.clone();
    }

    merge(channels, *dst);
}

// Gaussian filter for gray images
void GaussianFilterGray(Mat src, Mat *dst, int size, double sigma) {
    int center = (size - 1) / 2;

    // Extend image borders
    Mat srcExt;
    copyMakeBorder(src, srcExt, center, center, center, center,
                   BORDER_REFLECT_101);

    Mat kernel = genGaussianKernel(size, sigma);

    Mat temp(src.size(), src.type());

    Convolve2D(src, &temp, kernel);

    temp.copyTo(*dst);
}