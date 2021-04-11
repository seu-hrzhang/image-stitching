//
// Created by Starry Night on 2021/3/24.
//

#include "group.h"

using namespace std;
using namespace cv;

Image::Image(Mat src, int group, int index, int key_index, double key_position)
        : cvSiftOperator(src) {
    this->group = group;
    this->index = index;
    this->key_index = key_index;
    this->key_position = key_position;
}

void Image::resetParam(String name, int nFeatures, int nScales) {
    this->name = name;
    this->nFeatures = nFeatures;
    this->nScales = nScales;
}

void Image::get_key_position() {
    if (key_index != -1)
        key_position =
                cvSiftOperator::kpts[key_index].pt.x / cvSiftOperator::src.cols;
}

void Sequence::init() {
    for (int i = 0; i < images.size(); ++i)
        images[i].cvSiftOperator::run();
}

void Sequence::sort_seq() { sort(images.begin(), images.end(), Compare); }

void Sequence::addImage(Image img) { images.push_back(img); }

void Sequence::addImage(Mat img, bool run_sift) {
    Image new_image(img);

    if (run_sift)
        new_image.cvSiftOperator::run();

    images.push_back(new_image);
}

bool Compare(const Image &src_1, const Image &src_2) {
    return (src_1.key_position > src_2.key_position);
}

int get_pair_coordinates(vector<DMatch> pairs, int query_index) {
    for (int i = 0; i < pairs.size(); ++i)
        if (pairs[i].queryIdx == query_index)
            return pairs[i].trainIdx;

    return -1;
}

void read(String path, Sequence &seq) {
    seq.images.clear(); // Delete existing images in sequence

    vector<String> file_names;
    glob(path, file_names, false); // Get names of all files under 'path'

    // debug
    for (int i = 0; i < file_names.size(); ++i)
        cout << file_names[i] << endl;

    // Warning: non-image files may cause errors
    for (int i = 0; i < file_names.size(); ++i) {
        // Skippint .DS_Store file
        if (file_names[i] == path + "/.DS_Store")
            continue;

        seq.addImage(imread(file_names[i]), 0);
    }

    // debug
    cout << "Length of raw sequence: " << seq.images.size() << endl;

    seq.init();
}

// Split source images in 'seq' into multiple sequences in 'groups'
void regroup(Sequence raw, vector<Sequence> &groups, double thresh) {
    // Set groups empty
    if (groups.size() > 0)
        groups.clear();

    bool isMatched = false;

    Sequence temp;
    // temp.addImage(raw.images[0]);
    // groups.push_back(temp);
    //
    // cout << "Image 0 put to new group 0" << endl;

    for (int i = 0; i < raw.images.size(); ++i) {
        isMatched = false;
        for (int j = 0; j < groups.size(); ++j) {
            if (isMatched)
                break;
            for (int k = 0; k < groups[j].images.size(); ++k) {
                vector<DMatch> pairs =
                        cvKnnMatch(raw.images[i], groups[j].images[k]);

                if (pairs.size() >= thresh) {
                    // Train image's key point found, get key point for query
                    // image
                    // Train image's key point not found, get key point for both
                    if (groups[j].images[k].key_index == -1) {
                        // Use median position key point
                        groups[j].images[k].key_index =
                                (int) (groups[j].images[k].kpts.size() / 2);
                        groups[j].images[k].key_index =
                                (pairs[pairs.size() / 2].queryIdx);
                        // groups[j].images[k].key_index = (pairs[0].queryIdx);
                        groups[j].images[k].get_key_position();

                        // debug
                        // cout << groups[j].images[k].key_index << "\t"<<
                        // groups[j].images[k].key_position << endl;
                    }
                    raw.images[i].key_index = get_pair_coordinates(
                            pairs, groups[j].images[k].key_index);
                    raw.images[i].get_key_position();

                    // debug
                    // cout << raw.images[i].key_index << "\t"<<
                    // raw.images[i].key_position << endl;

                    // New image matched, but key point not found
                    if (raw.images[i].key_index < 0) {
                        temp.images.clear();
                        temp.addImage(raw.images[i]);
                        groups.push_back(temp);

                        // debug
                        cout << "Image " << i << "\tput to new group "
                             << groups.size() - 1 << " (key point not found)"
                             << endl;
                    } else {
                        groups[j].addImage(raw.images[i]);

                        // debug
                        cout << "Image " << i << "\tadded to group " << j
                             << endl;
                    }
                    isMatched = true;
                    break;
                }
            }
        }
        // Not match existing groups, create new group
        if (!isMatched) {
            temp.images.clear();
            temp.addImage(raw.images[i]);
            groups.push_back(temp);
            // debug
            cout << "Image " << i << "\tput to new group " << groups.size() - 1
                 << endl;
        }
    }

    for (int i = 0; i < groups.size(); ++i) {
        groups[i].sort_seq();

        // debug
        // cout << endl << "Segmented groups" << endl;
        // for (int j = 0; j < groups[i].images.size(); ++j) {
        // cout << "Group " << i << "\tImage " << j << ": "<<
        // groups[i].images[j].key_position << endl; imshow(to_string(i) + ", "
        // + to_string(j), groups[i].images[j].src);
        // }
    }

    cross_examine(groups, 30);

    // debug
    // showGroups(groups);
}

void cross_examine(vector<Sequence> &groups, int thresh, int nGroups) {
    for (int i = 0; i < groups.size(); ++i) {
        for (int j = i + 1; j < groups.size(); ++j) {
            // Match to the left
            vector<DMatch> pairs =
                    cvKnnMatch(groups[i].images[groups[i].images.size() - 1],
                               groups[j].images[0]);
            if (pairs.size() >= thresh) {
                groups[i].images.insert(groups[i].images.end(),
                                        groups[j].images.begin(),
                                        groups[j].images.end());
                groups.erase(groups.begin() + j);
                cross_examine(groups, thresh, nGroups);
                return;
            }
            // Match to the right
            pairs = cvKnnMatch(groups[i].images[0],
                               groups[j].images[groups[j].images.size() - 1]);
            if (pairs.size() >= thresh) {
                groups[j].images.insert(groups[j].images.end(),
                                        groups[i].images.begin(),
                                        groups[i].images.end());
                groups.erase(groups.begin() + i);
                cross_examine(groups, thresh, nGroups);
                return;
            }
        }
    }
}

void showGroups(vector<Sequence> groups) {
    for (int i = 0; i < groups.size(); ++i)
        for (int j = 0; j < groups[i].images.size(); ++j)
            imshow(to_string(i) + " - " + to_string(j),
                   groups[i].images[j].src);
}

void bilateralStitch(Sequence seq, Mat &dst, String name) {
    vector<Mat> homographies;
    int mid = seq.images.size() / 2;
    vector<DMatch> pairs;

    // debug
    // cout << "Midpoint: " << mid << endl;

    // Get recurrence homography matrix
    for (int i = 0; i < mid; ++i) {
        pairs = cvKnnMatch(seq.images[i], seq.images[i + 1]);
        homographies.push_back(
                getHomography(seq.images[i + 1], seq.images[i], pairs));
    }
    homographies.push_back(
            getUnitMatrix(3)); // Middle image to self, unit matrix
    for (int i = mid + 1; i < seq.images.size(); ++i) {
        pairs = cvKnnMatch(seq.images[i], seq.images[i - 1]);
        homographies.push_back(
                getHomography(seq.images[i - 1], seq.images[i], pairs));
    }

    // Get final homography matrix by recursion
    // Notes: multiply to the left !
    for (int i = mid - 1; i > 0; --i)
        homographies[i - 1] = homographies[i] * homographies[i - 1];
    for (int i = mid + 1; i < seq.images.size() - 1; ++i)
        homographies[i + 1] = homographies[i] * homographies[i + 1];

    // Find size of canvas
    vector<Point2f> corners_l = getCorners(seq.images[0].src, homographies[0]);
    vector<Point2f> corners_r =
            getCorners(seq.images[seq.images.size() - 1].src,
                       homographies[seq.images.size() - 1]);
    double width_l = 0 - MIN(corners_l[0].x, corners_l[2].x);
    double width_r = MAX(corners_r[1].x, corners_r[3].x);
    double height_u = 0 - MIN(MIN(corners_l[0].y, corners_l[1].y),
                              MIN(corners_r[0].y, corners_r[1].y));
    double height_d = MAX(MAX(corners_l[2].y, corners_l[3].y),
                          MAX(corners_r[2].y, corners_r[3].y));

    // debug
    cout << "Left maximum width = " << width_l << endl;
    cout << "Right maximum width = " << width_r << endl;
    cout << "Upper maximum height = " << height_u << endl;
    cout << "Lower maximum height = " << height_d << endl << endl;

    // Set canvas size to put in all images
    Mat canvas = Mat::zeros(height_u + height_d, width_l + width_r, CV_8UC3);

    vector<Point2f> corners_last; // Record corners in last image in recursion
    Mat image_last;               // Record last image in recursion

    // Warp images from right to left, copy to canvas
    for (int i = seq.images.size() - 1; i >= 0; --i) {
        vector<Point2f> corners =
                getCorners(seq.images[i].src, homographies[i]);
        int warp_width = MAX(corners[1].x, corners[3].x) + width_l;
        // int warp_height = MAX(corners[2].y, corners[3].y) + height_u;
        int warp_height = height_d + height_u;

        // debug
        cout << "Warping" << endl;
        cout << "warp_width = " << warp_width << endl;
        cout << "warp_height = " << warp_height << endl << endl;

        // Warp image with homography matrix
        Mat warp;
        warpPerspective(seq.images[i].src, warp,
                        getTranslateMatrix3D(width_l, height_u) *
                        homographies[i],
                        Size(warp_width, warp_height));

        // debug
        // imshow("Image " + to_string(i) + " Warped", warp);

        warp.copyTo(canvas(Rect(0, 0, warp.cols, warp.rows)));

        // Smoothen Stitching Borders
        vector<Point2f> edge_pts;
        // Starting smoothening from the second last image
        if (i < seq.images.size() - 1) {
            vector<Point2f> corners_tl =
                    translateCorners(corners, width_l, height_u);
            vector<Point2f> corners_last_tl =
                    translateCorners(corners_last, width_l, height_u);

            // debug
            cout << "Executing smoothening" << endl;
            cout << "Left-upper corner: " << corners_last_tl[0].x << endl;
            cout << "Left-lower corner: " << corners_last_tl[2].x << endl;
            cout << "Right-upper corner: " << corners_tl[1].x << endl;
            cout << "Right-lower corner: " << corners_tl[3].x << endl << endl;

            edge_pts.push_back(corners_last_tl[0]);
            edge_pts.push_back(corners_last_tl[2]);
            edge_pts.push_back(corners_tl[1]);
            edge_pts.push_back(corners_tl[3]);

            smoothenBorder(warp, image_last, canvas, canvas, edge_pts, width_l,
                           height_u);
        }
        corners_last = corners;
        image_last = warp;
    }
    canvas.copyTo(dst);

    imshow(name, dst);
}