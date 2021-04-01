#include "detect/sift.h"
#include "group.h"
#include "stitch.h"
#include "util/base.h"
#include "util/filter.h"
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

int main() {
    String path =
            "/Users/starrynight/Developer/CLionProjects/image_stitching/src";
    Sequence seq;
    vector<Sequence> groups;

    read(path, seq);
    regroup(seq, groups);

    vector<Mat> dst(groups.size());
    for (int i = 0; i < groups.size(); ++i)
        bilateralStitch(groups[i].images, dst[i],
                        "Stitched Image " + to_string(i));

    for (int i = 0; i < dst.size(); ++i)
        imwrite(path + "/Output Image " + to_string(i) + ".JPG", dst[i]);

    waitKey();

    return 0;
}