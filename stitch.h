//
// Created by Starry Night on 2021/3/22.
//

#ifndef IMAGE_STITCHING_STITCH_H
#define IMAGE_STITCHING_STITCH_H

#define STITCH_TO_RIGHT 1
#define STITCH_TO_LEFT 2

#include "detect/sift.h"

using namespace std;
using namespace cv;

vector<DMatch> cvBFMatch(cvSiftOperator src_1, cvSiftOperator src_2);

vector<DMatch> cvKnnMatch(cvSiftOperator src_1, cvSiftOperator src_2,
                          double thresh = 0.4);

Mat getHomography(cvSiftOperator src_1, cvSiftOperator src_2,
                  vector<DMatch> &match);

vector<Point2f> getCorners(const Mat &src, const Mat &matrix);

void imstitch(Mat src_1, Mat src_2, Mat &dst, Mat homo_matrix,
              vector<DMatch> pairs, int dir_opt = STITCH_TO_RIGHT);

void smoothenBorder(Mat img_l, Mat img_r, Mat stitch, Mat &dst,
                    vector<Point2f> edge_pts, int tx, int ty);

void smoothenBorders(Mat src_1, Mat warp, Mat stitch, Mat &dst,
                     vector<Point2f> corners);

#endif // IMAGE_STITCHING_STITCH_H