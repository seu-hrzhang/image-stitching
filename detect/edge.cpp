//
// Created by Starry Night on 2021/3/20.
//

#include "edge.h"

using namespace std;
using namespace cv;

extern Mat sobel_x = (Mat_<double>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
extern Mat sobel_y = (Mat_<double>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);

void findSobelEdge(Mat src, Mat *dst, Mat *theta, int size, double sigma) {
    if (src.channels() == 3)
        src = RGB2gray(src);

    Mat blur;
    GaussianFilterGray(src, &blur, size, sigma);

    Mat grad_x(src.size(), CV_64FC1);
    Mat grad_y(src.size(), CV_64FC1);

    Convolve2D(blur, &grad_x, sobel_x);
    Convolve2D(blur, &grad_y, sobel_y);

    // Compute gradient angles per pixel
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            // Avoid division by zero
            if (grad_x.at<double>(i, j) == 0)
                theta->at<double>(i, j) = atan(grad_y.at<double>(i, j) / 1e-5);
            else
                theta->at<double>(i, j) =
                    atan(grad_y.at<double>(i, j) / grad_x.at<double>(i, j));
        }
    }

    // imshow("Gradient X", grad_x);
    // imshow("Gradient Y", grad_y);

    // Merge x and y convolutions
    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst->at<uchar>(i, j) =
                sqrt(0.5 * grad_x.at<double>(i, j) * grad_x.at<double>(i, j) +
                     0.5 * grad_y.at<double>(i, j) * grad_y.at<double>(i, j));
        }
    }
}

// Adjoining pixels of (i, j)
/*

 (i - 1 , j - 1)      (i - 1 ,   j  )      (i - 1 , j + 1)

 (  i   , j - 1)      (  i   ,   j  )      (  i   , j + 1)

 (i + 1 , j - 1)      (i + 1 ,   j  )      (i + 1 , j + 1)

 */

// Suppress non-maximum points
void suppressNonMax(Mat src, Mat theta, Mat *dst) {
    src.copyTo(*dst);

    // Extend image borders
    Mat srcExt;
    copyMakeBorder(src, srcExt, 1, 1, 1, 1, BORDER_CONSTANT);

    double angle; // Gradient angle at (i, j)
    // Intersection points of gradient line and adjoining pixels
    double pix_1 = 0, pix_2 = 0;
    // Adjoining pixels on the corners
    double corner_1 = 0, corner_2 = 0, corner_3 = 0, corner_4 = 0;
    double weight = 0; // Weight of average (tan/cot)

    // Compute values of intersection points using weighted average
    for (int i = 1; i < srcExt.rows - 1; i++) {
        for (int j = 1; j < srcExt.cols - 1; j++) {
            angle = theta.at<double>(i - 1, j - 1);
            if (angle > 0 && angle <= PI / 4) {
                weight = tan(angle);
                corner_1 = srcExt.at<uchar>(i, j + 1);
                corner_2 = srcExt.at<uchar>(i - 1, j + 1);
                corner_3 = srcExt.at<uchar>(i, j - 1);
                corner_4 = srcExt.at<uchar>(i + 1, j - 1);
            } else if (angle > PI / 4 && angle < PI / 2) {
                weight = cot(angle);
                corner_1 = srcExt.at<uchar>(i - 1, j);
                corner_2 = srcExt.at<uchar>(i - 1, j + 1);
                corner_3 = srcExt.at<uchar>(i + 1, j);
                corner_4 = srcExt.at<uchar>(i + 1, j - 1);
            } else if (angle < 0 && angle >= -PI / 4) {
                weight = fabs(tan(angle));
                corner_1 = srcExt.at<uchar>(i, j + 1);
                corner_2 = srcExt.at<uchar>(i + 1, j + 1);
                corner_3 = srcExt.at<uchar>(i, j - 1);
                corner_4 = srcExt.at<uchar>(i - 1, j - 1);
            } else if (angle < -PI / 4 && angle >= -PI / 2) {
                weight = fabs(cot(angle));
                corner_1 = srcExt.at<uchar>(i + 1, j);
                corner_2 = srcExt.at<uchar>(i + 1, j + 1);
                corner_3 = srcExt.at<uchar>(i - 1, j);
                corner_4 = srcExt.at<uchar>(i - 1, j - 1);
            } else if (angle == PI / 2 || angle == -PI / 2) {
                corner_1 = corner_2 = srcExt.at<uchar>(i - 1, j);
                corner_3 = corner_4 = srcExt.at<uchar>(i + 1, j);
            } else if (angle == 0) {
                corner_1 = corner_2 = srcExt.at<uchar>(i, j - 1);
                corner_3 = corner_4 = srcExt.at<uchar>(i, j + 1);
            } else
                cout << "error" << endl;
            // Weighted average
            pix_1 = corner_1 * weight + corner_2 * (1 - weight);
            pix_2 = corner_3 * weight + corner_4 + (1 - weight);

            if (srcExt.at<uchar>(i, j) <= MAX(pix_1, pix_2))
                srcExt.at<uchar>(i, j) = BLACK;
            // else
            // dst->at<uchar>(i - 1, j - 1) = WHITE;
        }
    }
    *dst = srcExt(Range(1, srcExt.rows - 1), Range(1, srcExt.cols - 1)).clone();
}

void doubleThreshTrace(Mat &src, Mat *dst, int lowThresh, int highThresh) {
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            if (src.at<uchar>(i, j) > highThresh)
                src.at<uchar>(i, j) = 255;
            else if (src.at<uchar>(i, j) < lowThresh)
                src.at<uchar>(i, j) = 0;
            else {
                for (int x = 0; x < 3; x++) {
                    for (int y = 0; y < 3; y++) {
                        if (x == 1 && y == 1)
                            continue;
                        if (src.at<uchar>(i + x - 1, j + y - 1) == 255) {
                            src.at<uchar>(i, j) = 255;
                            doubleThreshTrace(src, dst, lowThresh, highThresh);
                        } else
                            src.at<uchar>(i, j) = 0;
                    }
                }
            }
        }
    }
    src.copyTo(*dst);
}

void findCannyEdge(Mat src, Mat *dst, int size, double sigma) {
    int lowThresh = 40;
    int highThresh = 80;

    Mat theta(src.size(), CV_64FC1);    // Gradient angles
    Mat sobelEdge(src.size(), CV_8UC1); // Sobel edge
    Mat nonMaxEdge; // Edge with non-maximum values suppressed

    findSobelEdge(src, &sobelEdge, &theta, size, sigma);
    imshow("Sobel Edge", sobelEdge);

    suppressNonMax(sobelEdge, theta, &nonMaxEdge);
    imshow("Non-maximum Edge", nonMaxEdge);

    doubleThreshTrace(nonMaxEdge, dst, lowThresh, highThresh);
    imshow("Canny Edge", *dst);
}
