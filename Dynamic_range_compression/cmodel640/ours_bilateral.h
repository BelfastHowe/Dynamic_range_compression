#pragma once

#include <opencv2/opencv.hpp>


void ours_bilateral(cv::Mat img_in, cv::Mat img_dn, bool dn_en, int sigma_color, int sigma_space);
