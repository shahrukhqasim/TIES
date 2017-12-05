#include <iostream>
#include <docproc/binarize/binarize.h>
#include "opencv2/core.hpp"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <memory>
#include <docproc/utility/timer.h>
#include <docproc/transform/transform.h>
#include <docproc/clean/clean.h>
#include <docproc/segment/segment.h>
#include <docproc/post_process/post_process.h>
#include <docproc/core/core.h>
#include <leptonica/allheaders.h>
#include <json/json/json.h>

using namespace std;
using namespace cv;

int main(int argc, char **argv) {
    if (argc != 2) {
        cout << "Oops! Check arguments!" << endl;
        return -1;
    }
    string imagePath = argv[1];

    Mat binarizedImage = imread(imagePath, 0);

    binarizedImage = docproc::transform::fixSkewAngle(binarizedImage);

    imwrite(imagePath, binarizedImage);

    return 0;
}