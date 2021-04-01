//
// Created by Starry Night on 2021/3/24.
//

#ifndef IMAGE_STITCHING_GROUP_H
#define IMAGE_STITCHING_GROUP_H

#include "stitch.h"

using namespace std;
using namespace cv;

class Image : public cvSiftOperator {
public:
    int group;           // Group image belongs to
    int index;           // Image position within the group
    int key_index;       // Index of key point in 'KeyPoint' vectors
    double key_position; // Position of key point in image

    Image(Mat src, int group = -1, int index = -1, int key_index = -1,
          double key_position = 0);

    void resetParam(String name = "SIFT Detection Result", int nFeatures = 0,
                    int nScales = 3);

    void get_key_position();
};

class Sequence {
public:
    vector<Image> images; // Input images read from disk

    Sequence() {}

    Sequence(vector<Image> images) : images(images) {}

    void init();              // Run SIFT feature extraction for sources images
    void sort_seq();          // Sort the sequence by key point of images
    void addImage(Image img); // Add new image to sequence
    void addImage(Mat img,
                  bool run_sift =
                  1); // Add new image to sequence (default group/index: -1)
};

// Compare function for quick sort (descending)
bool Compare(const Image &src_1, const Image &src_2);

// Get key point index in train image
int get_pair_coordinates(vector<DMatch> pairs, int query_index);

// Read all files and store to groups
void read(String path, Sequence &seq);

// Regroup frames to segments, then to image sequences
void regroup(Sequence raw, vector<Sequence> &groups, double thresh = 8);

// Piece together segments to sequences using cross examine
void cross_examine(vector<Sequence> &groups, int thresh = 8, int nGroups = 0);

// Show all images in sequence groups
void showGroups(vector<Sequence> groups);

// Operate stitching of an entire sequence from center to sides
void bilateralStitch(Sequence seq, Mat &dst, String name = "Stitched Image");

#endif // IMAGE_STITCHING_GROUP_H
