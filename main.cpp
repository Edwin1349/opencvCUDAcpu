#include "main.h"

void extract_frames(const std::string& videoFilePath, std::vector<cv::Mat>& frames) {
    try {
        cv::VideoCapture cap(videoFilePath);
        //cap.get(CV_CAP_PROP_FRAME_COUNT) contains the number of frames in the video;
        for (int frameNum = 0; frameNum < cap.get(cv::CAP_PROP_FRAME_COUNT); frameNum++)
        {
            cv::Mat frame;
            cap >> frame;
            frames.push_back(frame);
        }
    }
    catch (cv::Exception& e) {
        std::cerr << e.msg << std::endl;
        exit(1);
    }

}

int main(void) {

    bool blnKNNTrainingSuccessful = loadKNNDataAndTrainKNN();

    std::vector<cv::Mat> imgsOriginalScene;

#ifdef IMG
    std::vector<cv::String> fn;
    cv::glob("Cars\\images\\*.png", fn, false);
    size_t count = fn.size(); {
        for (size_t i = 0; i < count; i++)
            imgsOriginalScene.push_back(cv::imread(fn[i]));
    }
#endif

#ifdef VIDEO
    extract_frames("Cars\\car1.avi", imgsOriginalScene);
#endif


    //imgsOriginalScene.clear();
    //imgsOriginalScene.push_back(cv::imread("Cars\\Cars1.png"));

    std::vector<std::vector<PossiblePlate>> vectorOfPossiblePlates = detectPlatesInScene(imgsOriginalScene);


    vectorOfPossiblePlates = detectCharsInPlates(vectorOfPossiblePlates);
    for (int img = 0; img < imgsOriginalScene.size(); img++) {
        if (vectorOfPossiblePlates[img].empty()) {
            //std::cout << std::endl << "no license plates were detected" << std::endl;
            continue;
        }

        else {
            std::sort(vectorOfPossiblePlates[img].begin(), vectorOfPossiblePlates[img].end(), PossiblePlate::sortDescendingByNumberOfChars);
             PossiblePlate a = vectorOfPossiblePlates[img].front();

            if (a.imgGrayscale.empty() || a.strChars.length() == 0) {
                continue;
            }
            cv::imshow("imgPlate", a.imgPlate);
            cv::imshow("imgThresh", a.imgThresh);
            cv::imshow("imgOriginalScene", imgsOriginalScene[img]);

            std::cout << std::endl << "license plate read from image = " << a.strChars << std::endl;
            std::cout << std::endl << "-----------------------------------------" << std::endl;

            cv::waitKey(0);


        }
    }
    cv::waitKey(0);
    return(0);
}
