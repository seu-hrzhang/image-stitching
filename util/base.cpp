//
// Created by Starry Night on 2021/3/20.
//

#include "base.h"

using namespace std;
using namespace cv;

// Print matrix
void printMatrix(Mat mat) {
    switch (mat.type()) {
    case 0: {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                cout << (int)mat.at<uchar>(i, j) << " ";
            }
            cout << endl;
        }
        break;
    }
    case 6: {
        for (int i = 0; i < mat.rows; i++) {
            for (int j = 0; j < mat.cols; j++) {
                cout << mat.at<double>(i, j) << " ";
            }
            cout << endl;
        }
        break;
    }
    default:
        break;
    }
}

// Convert RGB image to gray image
Mat RGB2gray(Mat src) {
    Mat dst(src.size(), CV_8UC1);

    for (int i = 0; i < src.rows; i++) {
        for (int j = 0; j < src.cols; j++) {
            dst.at<uchar>(i, j) = 0.299 * src.at<Vec3b>(i, j)[0] +
                                  0.587 * src.at<Vec3b>(i, j)[1] +
                                  0.114 * src.at<Vec3b>(i, j)[2];
        }
    }
    return dst;
}

// Get size * size unit matrix
Mat getUnitMatrix(int size) {
    if (size <= 0)
        size = 1;
    Mat unit = Mat::zeros(size, size, CV_64FC1);
    for (int i = 0; i < size; ++i)
        unit.at<double>(i, i) = 1;
    return unit;
}

// Get translation matrix (x translation 'tx', y translation 'ty')
Mat getTranslateMatrix(int tx, int ty) {
    Mat translate = (Mat_<double>(2, 3) << 1, 0, tx, 0, 1, ty);
    return translate;
}

// Get 3D translation matrix (x translation 'tx', y translation 'ty')
Mat getTranslateMatrix3D(int tx, int ty) {
    Mat translate = (Mat_<double>(3, 3) << 1, 0, tx, 0, 1, ty, 0, 0, 1);
    return translate;
}

// Translate coordinates of corner points
vector<Point2f> translateCorners(vector<Point2f> corners, double tx,
                                 double ty) {
    vector<Point2f> corners_tl = corners;

    for (int i = 0; i < corners.size(); ++i) {
        corners_tl[i].x += tx;
        corners_tl[i].y += ty;
    }

    return corners_tl;
}

// Cast two input images into one (L-R)
void imcast(Mat src_1, Mat src_2, Mat &dst) {
    int rows = MAX(src_1.rows, src_2.rows);
    int cols = src_1.cols + src_2.cols;

    dst.create(rows, cols, src_1.type());
    src_1.copyTo(dst(Range(0, src_1.rows), Range(0, src_1.cols)));
    src_2.copyTo(dst(Range(0, src_2.rows), Range(src_1.cols, cols)));
}

vector<KeyPoint> Mat2KeyPointVec(Mat src, int rows, int cols, int thresh) {
    vector<KeyPoint> kpts;

    double rate_row, rate_col; // Ratio of scaling
    if (rows != src.rows || cols != src.cols) {
        rate_row = rows / src.rows;
        rate_col = cols / src.cols;
    } else
        rate_row = rate_col = 1;

    for (int i = 0; i < src.rows; ++i) {
        for (int j = 0; j < src.cols; ++j) {
            if (src.at<uchar>(i, j) > thresh) {
                KeyPoint pnt(i * rate_row, j * rate_col, 1);
                kpts.push_back(pnt);
            }
        }
    }
    return kpts;
}
