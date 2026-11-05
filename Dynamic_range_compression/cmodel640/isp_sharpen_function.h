#pragma once

#include <opencv2/opencv.hpp>


cv::Mat sharpenImage_int8(const cv::Mat& inputImage, cv::Mat outputImage, int en, int kernel[][3], int temp_akt[]);
