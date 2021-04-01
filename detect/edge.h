//
// Created by Starry Night on 2021/3/20.
//

#ifndef IMAGE_STITCHING_EDGE_H
#define IMAGE_STITCHING_EDGE_H

#include "../util/base.h"
#include "../util/filter.h"
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

extern Mat sobel_x, sobel_y;

void findSobelEdge(Mat src, Mat *dst, Mat *theta, int size, double sigma);

void suppressNonMax(Mat src, Mat theta, Mat *dst);

void doubleThreshTrace(Mat &src, Mat *dst, double lowThresh, double highThresh);

void findCannyEdge(Mat src, Mat *dst, int size, double sigma);

#endif // IMAGE_STITCHING_EDGE_H
