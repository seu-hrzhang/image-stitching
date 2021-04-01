//
// Created by Starry Night on 2021/3/20.
//

#include "sift.h"

using namespace std;
using namespace cv;

SiftOperator::SiftOperator(Mat src, int nOctaves, int nScales)
    : src(src), nOctaves(nOctaves), nScales(nScales) {
    for (int i = 0; i < nOctaves; ++i) {
        scale_space.push_back(vector<Mat>(
            nScales +
            3)); // Scale space should contain nScales + 3 images in an octave
        diff_pyramid.push_back(vector<Mat>(
            nScales +
            2)); // DoG pyramid should contain nScales + 2 images in an octave
        kpts.push_back(vector<Mat>(nScales));
        magnitude.push_back(vector<Mat>(vector<Mat>(nScales)));
        orientation.push_back(vector<Mat>(nScales));
    }
}

void SiftOperator::run() {
    createScaleSpace();
    findExtrema();
    filterExtrema();
    // assignOrientations();
}

void SiftOperator::createScaleSpace() {
    Size ksize = Size(5, 5);
    double sigma_init = 1.0, sigma;

    pyrUp(src, src); // Enlarge scale of source image by 2 times
    src.copyTo(scale_space[0][0]);

    for (int i = 0; i < nOctaves; ++i) {
        sigma = sigma_init * pow(2, i);
        if (i > 0)
            pyrDown(scale_space[i - 1][0], scale_space[i][0]);

        for (int j = 0; j < nScales + 2; ++j) {
            sigma *= pow(2, 1 / nScales);
            GaussianBlur(scale_space[i][j], scale_space[i][j + 1], ksize,
                         sigma);

            diff_pyramid[i][j] = scale_space[i][j] - scale_space[i][j + 1];

            // debug
            // imshow("Scale Space Octave " + to_string(i) + " Scale " +
            // to_string(j), scale_space[i][j]);
        }
    }
}

void SiftOperator::findExtrema() {
    int rows, cols; // Row and column number of current octave

    Mat extrema;
    Mat upper_scale, current_scale, lower_scale;
    Mat local_max, local_min;

    for (int i = 0; i < nOctaves; ++i) {
        for (int j = 0; j < nScales; ++j) {
            upper_scale = diff_pyramid[i][j];
            current_scale = diff_pyramid[i][j + 1];
            lower_scale = diff_pyramid[i][j + 2];

            rows = current_scale.rows;
            cols = current_scale.cols;

            // Reset local maximum and minimum signs
            local_max = Mat::ones(Size(rows - 2, cols - 2), CV_8UC1);
            local_min = Mat::ones(Size(rows - 2, cols - 2), CV_8UC1);

            // Compare each pixel with its 26 neighbors to find extrema
            for (int k = -1; k <= 1; ++k) {
                for (int l = -1; l <= 1; ++l) {
                    if (k != 0 || l != 0) {
                        local_max &= current_scale(Range(1, rows - 1),
                                                   Range(1, cols - 1)) >
                                     current_scale(Range(1 + k, rows - 1 + k),
                                                   Range(1 + l, cols - 1 + l));

                        local_min &= current_scale(Range(1, rows - 1),
                                                   Range(1, cols - 1)) <
                                     current_scale(Range(1 + k, rows - 1 + k),
                                                   Range(1 + l, cols - 1 + l));
                    }

                    local_max &=
                        current_scale(Range(1, rows - 1), Range(1, cols - 1)) >
                        upper_scale(Range(1 + k, rows - 1 + k),
                                    Range(1 + l, cols - 1 + l));

                    local_max &=
                        current_scale(Range(1, rows - 1), Range(1, cols - 1)) >
                        lower_scale(Range(1 + k, rows - 1 + k),
                                    Range(1 + l, cols - 1 + l));

                    // printMatrix(local_max);

                    local_min &=
                        current_scale(Range(1, rows - 1), Range(1, cols - 1)) <
                        upper_scale(Range(1 + k, rows - 1 + k),
                                    Range(1 + l, cols - 1 + l));

                    local_min &=
                        current_scale(Range(1, rows - 1), Range(1, cols - 1)) <
                        lower_scale(Range(1 + k, rows - 1 + k),
                                    Range(1 + l, cols - 1 + l));
                }
            }

            extrema = local_max | local_min;
            copyMakeBorder(extrema, extrema, 1, 1, 1, 1,
                           BORDER_CONSTANT); // Restore image size
            threshold(extrema, extrema, 0, 255,
                      THRESH_BINARY); // Binarize extrema matrix to 0/255
            extrema.copyTo(kpts[i][j]);
        }
    }
}

void SiftOperator::filterExtrema() {
    vector<Point> locations; // Locations of extrema

    double contrastThresh = 0.03; // Threshold to filter low contrast points
    double edgeThresh;            // Threshold to filter points on edges

    int pnt_x, pnt_y;       // x, y coordinates of current pixel
    int nExtrema;           // Number of preliminary extrema
    int low_contrast, edge; // To count number of filtered extrema

    double Dxx, Dxy, Dyy; // Elements of Hessian matrix
    double tr, det;       // Trace and determinant of Hessian matrix

    Mat current_scale; // Current scale of image (range: 1 ~ nScales - 1)

    for (int i = 0; i < nOctaves; ++i) {
        for (int j = 0; j < nScales; ++j) {
            low_contrast = 0;
            edge = 0;

            current_scale = diff_pyramid[i][j + 1];

            findNonZero(kpts[i][j], locations);

            nExtrema = locations.size();

            for (int k = 0; k < nExtrema; ++k) {
                pnt_x = locations[k].x;
                pnt_y = locations[k].y;

                // debug
                cout << "x = " << pnt_x << " y = " << pnt_y << endl;

                if (abs(current_scale.at<uchar>(pnt_x, pnt_y + 1)) <
                    contrastThresh) {
                    kpts[i][j].at<uchar>(pnt_x, pnt_y) = 0;
                    low_contrast++;
                } else {
                    // Get Hessian matrix elements
                    Dxx = current_scale.at<uchar>(pnt_x, pnt_y + 1) +
                          current_scale.at<uchar>(pnt_x + 2, pnt_y + 1) -
                          2 * current_scale.at<uchar>(pnt_x + 1, pnt_y + 1);
                    Dyy = current_scale.at<uchar>(pnt_x + 1, pnt_y) +
                          current_scale.at<uchar>(pnt_x + 1, pnt_y + 2) -
                          2 * current_scale.at<uchar>(pnt_x + 1, pnt_y + 1);
                    Dxy = current_scale.at<uchar>(pnt_x, pnt_y) +
                          current_scale.at<uchar>(pnt_x + 2, pnt_y + 2) -
                          current_scale.at<uchar>(pnt_x + 2, pnt_y) -
                          current_scale.at<uchar>(pnt_x, pnt_y + 2);

                    // Find trace and determinant of Hessian matrix
                    tr = Dxx + Dyy;
                    det = Dxx * Dyy - Dxy * Dxy;

                    if (det < 0 || tr * tr / det < edgeThresh) {
                        kpts[i][j].at<uchar>(pnt_x, pnt_y) = 0;
                        edge++;
                    }
                }
            }

            // debug
            cout << "Octave: " << i << " Scale: " << j
                 << " Key Points: " << nExtrema
                 << " Filtered: " << low_contrast + edge << " (" << low_contrast
                 << " + " << edge << ")" << endl;
        }
    }
}

void SiftOperator::assignOrientations() {
    double dx, dy; // x, y differentiations

    Mat_<uchar> current_scale;

    for (int i = 0; i < nOctaves; ++i) {
        for (int j = 0; j < nScales; ++j) {
            current_scale = scale_space[i][j + 1];

            magnitude[i][j] = Mat::zeros(current_scale.size(), CV_64FC1);
            orientation[i][j] = Mat::zeros(current_scale.size(), CV_64FC1);

            for (int k = 0; k < current_scale.rows; ++k) {
                for (int l = 0; l < current_scale.cols; ++l) {
                    dx = current_scale(k + 1, l) - current_scale(k - 1, l);
                    dy = current_scale(k, l + 1) - current_scale(k, l - 1);

                    magnitude[i][j].at<double>(k, l) = sqrt(dx * dx + dy * dy);
                    orientation[i][j].at<double>(k, l) =
                        (abs(atan2(dy, dx) - PI) < 1e-3) ? -PI : atan2(dy, dx);
                }
            }
        }
    }
}

void SiftOperator::disp() {
    Mat dst;
    sythsKeyPoints();
    drawKeypoints(src, kpts_vector, dst);
    imshow("SIFT Detection Result", dst);
}

void SiftOperator::sythsKeyPoints() {
    vector<KeyPoint> temp;
    for (int i = 0; i < nOctaves; ++i) {
        for (int j = 0; j < nScales; ++j) {
            threshold(kpts[i][j], kpts[i][j], 0, 255, CV_8UC1); // Binarization
            temp = Mat2KeyPointVec(kpts[i][j], src.rows, src.cols);
            kpts_vector.insert(kpts_vector.end(), temp.begin(), temp.end());
        }
    }
}

cvSiftOperator::cvSiftOperator(Mat src, String name, int nFeatures, int nScales)
    : src(src), name(name), nFeatures(nFeatures), nScales(nScales) {
    detector = SiftFeatureDetector::create(nFeatures, nScales);
    extractor = SiftDescriptorExtractor::create(nFeatures, nScales);
}

void cvSiftOperator::run() {
    Mat gray; // Gray image of source for detection

    if (src.type() != CV_8UC1)
        cvtColor(src, gray, COLOR_RGB2GRAY);

    detector->detect(gray, kpts);

    extractor->compute(gray, kpts, desc);
}

void cvSiftOperator::disp() {
    drawKeypoints(src, kpts, dst);
    imshow(name, dst);
}