#include "Preprocess.h"

double max_time;

void preprocess(std::vector<cv::Mat>& imgOriginal, std::vector<cv::Mat>& imgGrayscale, std::vector<cv::Mat>& imgThresh) {
    imgGrayscale = extractValue(imgOriginal);
    std::vector<cv::Mat> imgMaxContrastGrayscale = maximizeContrast(imgGrayscale);

    std::vector<cv::Mat> imgBlurred(imgOriginal.size());

    for (int i = 0; i < imgMaxContrastGrayscale.size(); i++) {
        cv::GaussianBlur(imgMaxContrastGrayscale[i], imgBlurred[i], GAUSSIAN_SMOOTH_FILTER_SIZE, 0);
        cv::adaptiveThreshold(imgBlurred[i], imgThresh[i], 255.0, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, ADAPTIVE_THRESH_BLOCK_SIZE, ADAPTIVE_THRESH_WEIGHT);
    }
}

std::vector<cv::Mat> extractValue(std::vector<cv::Mat>& imgOriginal) {
    std::vector<cv::Mat> imgHSV(imgOriginal.size());
    std::vector<std::vector<cv::Mat>> vectorOfHSVImages(imgOriginal.size());
    std::vector<cv::Mat> imgValue(imgOriginal.size());

    for (int i = 0; i < imgOriginal.size(); i++) {
        cv::cvtColor(imgOriginal[i], imgHSV[i], cv::COLOR_BGR2HSV);
        cv::split(imgHSV[i], vectorOfHSVImages[i]);
        imgValue[i] = vectorOfHSVImages[i][2];
    }
    return(imgValue);
}

std::vector<cv::Mat> maximizeContrast(std::vector<cv::Mat>& imgGrayscale) {
    std::vector<cv::Mat> imgTopHat(imgGrayscale.size());
    std::vector<cv::Mat> imgBlackHat(imgGrayscale.size());
    std::vector<cv::Mat> imgGrayscalePlusTopHat(imgGrayscale.size());
    std::vector<cv::Mat> imgGrayscalePlusTopHatMinusBlackHat(imgGrayscale.size());

    cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    for (int i = 0; i < imgGrayscale.size(); i++) {
        cv::morphologyEx(imgGrayscale[i], imgTopHat[i], cv::MORPH_TOPHAT, structuringElement);
        cv::morphologyEx(imgGrayscale[i], imgBlackHat[i], cv::MORPH_BLACKHAT, structuringElement);

        imgGrayscalePlusTopHat[i] = imgGrayscale[i] + imgTopHat[i];
        imgGrayscalePlusTopHatMinusBlackHat[i] = imgGrayscalePlusTopHat[i] - imgBlackHat[i];
    }
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    max_time += std::chrono::duration_cast<std::chrono::milliseconds> (end - begin).count();
    std::cout << "Max time: " << max_time << std::endl;

    return(imgGrayscalePlusTopHatMinusBlackHat);
}