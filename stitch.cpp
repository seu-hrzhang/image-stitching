//
// Created by Starry Night on 2021/3/22.
//

#include "stitch.h"

using namespace std;
using namespace cv;

vector<DMatch> cvBFMatch(cvSiftOperator src_1, cvSiftOperator src_2) {
    Ptr<BFMatcher> matcher = BFMatcher::create();
    vector<DMatch> pairs;
    matcher->match(src_1.desc, src_2.desc, pairs);

    return pairs;
}

vector<DMatch> cvKnnMatch(cvSiftOperator src_1, cvSiftOperator src_2,
                          double thresh) {
    Ptr<FlannBasedMatcher> matcher = FlannBasedMatcher::create();
    vector<vector<DMatch>> pairs;
    vector<DMatch> filtered_pairs; // Matches filtered using Lowe's algorithm

    vector<Mat> train_desc(1, src_1.desc);
    matcher->add(train_desc);
    matcher->train();
    matcher->knnMatch(src_2.desc, pairs, 2);

    for (int i = 0; i < pairs.size(); ++i) {
        if (pairs[i][0].distance < thresh * pairs[i][1].distance)
            filtered_pairs.push_back(pairs[i][0]);
    }

    return filtered_pairs;
}

// Get homography transformation matrix from 'src_2' to 'src_1'
Mat getHomography(cvSiftOperator src_1, cvSiftOperator src_2,
                  vector<DMatch> &match) {
    vector<Point2f> pts_1, pts_2;
    vector<KeyPoint> temp_1, temp_2;

    // Find matched points coordinates
    if (match.size() == 0)
        match = cvBFMatch(src_1, src_2);
    for (int i = 0; i < match.size(); ++i) {
        temp_1.push_back(src_1.kpts[match[i].queryIdx]);
        temp_2.push_back(src_2.kpts[match[i].trainIdx]);
    }
    KeyPoint::convert(temp_1, pts_1);
    KeyPoint::convert(temp_2, pts_2);

    // Find optimum homography transformation matrix
    Mat homography = findHomography(pts_2, pts_1, RANSAC);

    // debug
    // cout << "Homography Matrix" << endl;
    // printMatrix(homography);

    return homography;
}

// Get corner coordinates after homography transformation
// Note: points organized in order of left-upper, right-upper,
// left-bottom, right-bottom
vector<Point2f> getCorners(const Mat &src, const Mat &matrix) {
    vector<Point2f> corners(4);

    // Find left-upper corner (0, 0, 1)
    double vec_2[] = {0, 0, 1};
    double vec_1[3];
    Mat src_vec = Mat(3, 1, CV_64FC1, vec_2);
    Mat dst_vec = Mat(3, 1, CV_64FC1, vec_1);
    dst_vec = matrix * src_vec;
    corners[0].x = vec_1[0] / vec_1[2];
    corners[0].y = vec_1[1] / vec_1[2];

    // Find right_upper corner (src.cols, 0, 1)
    vec_2[0] = src.cols;
    vec_2[1] = 0;
    vec_2[2] = 1;
    src_vec = Mat(3, 1, CV_64FC1, vec_2);
    dst_vec = Mat(3, 1, CV_64FC1, vec_1);
    dst_vec = matrix * src_vec;
    corners[1].x = vec_1[0] / vec_1[2];
    corners[1].y = vec_1[1] / vec_1[2];

    // Find left_bottom corner (0, src.rows, 1)
    vec_2[0] = 0;
    vec_2[1] = src.rows;
    vec_2[2] = 1;
    src_vec = Mat(3, 1, CV_64FC1, vec_2);
    dst_vec = Mat(3, 1, CV_64FC1, vec_1);
    dst_vec = matrix * src_vec;
    corners[2].x = vec_1[0] / vec_1[2];
    corners[2].y = vec_1[1] / vec_1[2];

    // Find right_bottom corner (src.cols, src.rows, 1)
    vec_2[0] = src.cols;
    vec_2[1] = src.rows;
    vec_2[2] = 1;
    src_vec = Mat(3, 1, CV_64FC1, vec_2);
    dst_vec = Mat(3, 1, CV_64FC1, vec_1);
    dst_vec = matrix * src_vec;
    corners[3].x = vec_1[0] / vec_1[2];
    corners[3].y = vec_1[1] / vec_1[2];

    // debug
    // for (int i = 0; i < 4; ++i)
    // cout << "corners[" << i << "] = " << corners[i].x << ", "
    // << corners[i].y << endl;

    return corners;
}

void imstitch(Mat src_1, Mat src_2, Mat &dst, Mat homography,
              vector<DMatch> pairs, int dir_opt) {

    // Get corners after transformation
    vector<Point2f> corners = getCorners(src_2, homography);
    int width = MAX(corners[1].x, corners[3].x);
    int height =
        MAX(corners[2].y, corners[3].y) - MIN(corners[0].y, corners[1].y);

    // Warp image with homography matrix
    // Note: image 'src_1' is to be warped
    Mat warp; // Image after warping with 'homography'
    warpPerspective(src_2, warp, homography, Size(width, src_1.rows));

    // debug
    imshow("Warped", warp);

    // Initialize stitched image
    Mat stitch = Mat::zeros(src_1.rows, warp.cols, CV_8UC3);

    warp.copyTo(stitch(Rect(0, 0, warp.cols, warp.rows)));
    src_1.copyTo(stitch(Rect(0, 0, src_1.cols, src_1.rows)));

    // debug
    imshow("Stitched", stitch);

    smoothenBorders(src_1, warp, stitch, dst, corners);

    // debug
    // imshow("Smoothened", dst);
}

// Points on edge should be given in format:
// (left_upper, left_lower, right_upper, right_lower)
void smoothenBorder(Mat img_l, Mat img_r, Mat stitch, Mat &dst,
                    vector<Point2f> edge_pts, int tx, int ty) {
    // debug
    // imshow("Left Image", img_l);
    // imshow("Right Image", img_r);
    // imshow("Stitched Image", stitch);

    // Starting x-coordinate of occlusion area
    int start = MIN(MIN(edge_pts[0].x, edge_pts[1].x),
                    MIN(edge_pts[2].x, edge_pts[3].x));
    // Destination x-coordinate of occlusion area
    int end = MAX(MAX(edge_pts[0].x, edge_pts[1].x),
                  MAX(edge_pts[2].x, edge_pts[3].x));
    // Width of occlusion area
    double width = end - start;

    // Size of area to be processed
    int rows = stitch.rows;
    int cols = end;

    // debug
    // cout << "rows = " << rows << endl;
    // cout << "cols = " << cols << endl << endl;

    // Weight of pixels in 'img_l'
    double weight = 1.0;

    for (int i = 0; i < rows; ++i) {
        for (int j = start; j < cols; ++j) {
            // 3 channels
            for (int k = 0; k < 3; ++k) {
                // Left image empty
                if (img_l.at<Vec3b>(i, j)[k] == 0)
                    weight = 0.0;
                // Weight changes adaptively with pixel position
                else {
                    if (img_r.at<Vec3b>(i, j)[k] == 0)
                        weight = 1.0;
                    else
                        weight = 1.0 - (j - start) / width;
                }

                for (int l = 0; l < 3; ++l)
                    dst.at<Vec3b>(i, j + l)[k] =
                        img_l.at<Vec3b>(i, j + l)[k] * weight +
                        img_r.at<Vec3b>(i, j + l)[k] * (1 - weight);
            }
        }
    }
}

void smoothenBorders(Mat src_1, Mat warp, Mat stitch, Mat &dst,
                     vector<Point2f> corners) {
    stitch.copyTo(dst);

    int start =
        MIN(corners[0].x, corners[2].x); // Starting point of occlusion area
    double width = src_1.cols - start;   // Width of occlusion area

    // Size of area to be processed
    int rows = stitch.rows;
    int cols = src_1.cols;

    double weight = 1.0; // Weight of pixels in 'src_1'

    for (int i = 0; i < rows; ++i) {
        for (int j = start; j < cols; ++j) {
            for (int k = 0; k < 3; ++k) {
                // Image 'warp' empty
                if (warp.at<Vec3b>(i, j)[k] == 0 &&
                    warp.at<Vec3b>(i, j + 1)[k] == 0 &&
                    warp.at<Vec3b>(i, j + 2)[k] == 0)
                    weight = 1.0;
                // Weight changes adaptively with pixel position
                else
                    weight = 1.0 - (j - start) / width;

                for (int l = 0; l < 3; ++l)
                    dst.at<Vec3b>(i, j + l)[k] =
                        src_1.at<Vec3b>(i, j + l)[k] * weight +
                        warp.at<Vec3b>(i, j + l)[k] * (1 - weight);
            }
        }
    }
}