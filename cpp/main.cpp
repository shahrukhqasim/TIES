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
    if (argc != 3) {
        cout << "Oops! Check arguments!" << endl;
        return -1;
    }
    string imageFilePath = argv[1];
    string jsonFilePath = argv[2];
    cout<<jsonFilePath<<endl;

    Mat originalImage = imread(imageFilePath, 1);

    Mat grayScaleImage;
    cvtColor(originalImage, grayScaleImage, CV_BGR2GRAY);

    shared_ptr <tesseract::TessBaseAPI> api = shared_ptr<tesseract::TessBaseAPI>(new tesseract::TessBaseAPI());

    if (api->Init(NULL, "eng", tesseract::OEM_TESSERACT_ONLY)) {
        throw 1; // TODO: Convert to proper exception
    }
    api->SetVariable("tessedit_char_whitelist", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789,.-");

    api->SetImage(docproc::core::mat2PixGray(grayScaleImage)); // Gray scale is actually binary as per the UNLV dataset
    api->Recognize(0);
    tesseract::ResultIterator *itr = api->GetIterator();

    vector<string>words;
    vector<Rect>boundingBoxes;

    {
        tesseract::ResultIterator *ri = api->GetIterator();
        if (ri != 0) {
            do {
                int left, top, right, bottom;
                ri->BoundingBox(tesseract::RIL_WORD, &left, &top, &right, &bottom);
                Rect rect = Rect(left, top, right - left, bottom - top);
                words.push_back(ri->GetUTF8Text(tesseract::RIL_WORD));
                boundingBoxes.push_back(rect);

            } while (ri->Next(tesseract::RIL_WORD));
        }
    }

    Json::Value ocrJson;
    for (int i = 0; i < words.size(); i++) {
        string word = words[i];
        Rect boundingBox = boundingBoxes[i];
        Json::Value oneWord;
        oneWord["word"] = word;
        Json::Value bounds;
        bounds["x"] = boundingBox.x;
        bounds["y"] = boundingBox.y;
        bounds["width"] = boundingBox.width;
        bounds["height"] = boundingBox.height;
        oneWord["rect"] = bounds;

        ocrJson[i]  = oneWord;
    }
    ofstream outputFileStream(jsonFilePath);
    outputFileStream << ocrJson;
    outputFileStream.close();

    return 0;
}