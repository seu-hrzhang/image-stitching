//
// Created by Starry Night on 2021/3/20.
//

#ifndef IMAGE_STITCHING_BASE_H
#define IMAGE_STITCHING_BASE_H

#define PI 3.1415926535898
#define cot(x) 1 / tan(x)
#define BLACK 0
#define WHITE 255

#include <algorithm>
#include <iostream>
#include <math.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void printMatrix(Mat mat);

Mat RGB2gray(Mat src);
Mat getUnitMatrix(int size);
Mat getTranslateMatrix(int tx, int ty);
Mat getTranslateMatrix3D(int tx, int ty);

vector<Point2f> translateCorners(vector<Point2f> corners, double tx, double ty);

void imcast(Mat src_1, Mat src_2, Mat &dst);

vector<KeyPoint> Mat2KeyPointVec(Mat src, int rows, int cols, int thresh = 0);

#endif // IMAGE_STITCHING_BASE_H
