#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>

void ours_dpc(cv::Mat img, bool dpc_en, cv::Mat img_dpc, double dp_slope, double dp_thresh, int diff_th, int DEAD[][2], int dead_count = 0);

