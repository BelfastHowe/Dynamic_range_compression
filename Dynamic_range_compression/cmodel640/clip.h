#pragma once

#include <opencv2/opencv.hpp>


// 像素数据结构体（存储14bit值）
struct Pixel14b {
    uint16_t value;  // 存储14bit值（0-16383）
    int x;
    int y;
    Pixel14b(uint16_t v, int x_, int y_) : value(v), x(x_), y(y_) {}
};

// 每 8 个灰度值统计一次的裁剪版本。
// 14bit 像素范围为 0~16383，因此使用 2048 个直方图桶。
cv::Mat histogram_equalization_clipped_fixed2(const cv::Mat& image_14bit,
    std::vector<cv::Point>& min_positions,
    std::vector<uint16_t>& min_values, bool clip_en, int low_number, int high_number,
    int& autothreshold_low, int& autothreshold_high);









