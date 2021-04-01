//
// Created by Starry Night on 2021/3/20.
//

#ifndef IMAGE_STITCHING_FILTER_H
#define IMAGE_STITCHING_FILTER_H

#include "base.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

Mat RGB2Gray(Mat src);

void Convolve2D(Mat src, Mat *dst, Mat kernel);

Mat genGaussianKernel(int size, double sigma);

void GaussianFilterRGB(Mat src, Mat *dst, int size, double sigma);

void GaussianFilterGray(Mat src, Mat *dst, int size, double sigma);

#endif // IMAGE_STITCHING_FILTER_H
