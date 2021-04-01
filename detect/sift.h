//
// Created by Starry Night on 2021/3/20.
//

#ifndef IMAGE_STITCHING_SIFT_H
#define IMAGE_STITCHING_SIFT_H

#include "../util/base.h"
#include "../util/filter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace std;
using namespace cv;

class SiftOperator {
private:
    Mat src;      // Source image
    int nOctaves; // Number of octaves in Gaussian pyramid
    int nScales;  // Number of scales in an octave

    vector<vector<Mat>> scale_space, // Scale space
    diff_pyramid,                // Gaussian difference pyramid
    kpts,                        // Key points detected
    magnitude,                   // Gradient magnitude of key points
    orientation;                 // Gradient orientation of key points

    vector<KeyPoint> kpts_vector; // Key points detected (in form of 'KeyPoint')

    void createScaleSpace();

    void findExtrema();

    void filterExtrema();

    void assignOrientations();

    void sythsKeyPoints();

public:
    SiftOperator(Mat src, int nOctaves, int nScales);

    void run();

    void disp();
};

// Implementation of SIFT using OpenCV libs
class cvSiftOperator {
public:
    Mat src;  // Source image
    Mat dst;  // Image drawn with key points
    Mat desc; // Descriptors of source image

    int nFeatures; // Expected number of feature points
    int nScales;   // Number of scales in key point pyramid

    String name; // Name of window when displayed

    vector<KeyPoint> kpts; // Key points detected

    Ptr<SiftFeatureDetector> detector;
    Ptr<SiftDescriptorExtractor> extractor;

    cvSiftOperator(Mat src, String name = "SIFT Detection Result",
                   int nFeatures = 0, int nScales = 3);

    void run();

    void disp();
};

#endif // IMAGE_STITCHING_SIFT_H
