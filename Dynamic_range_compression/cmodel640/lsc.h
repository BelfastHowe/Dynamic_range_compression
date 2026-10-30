#pragma once

#include <opencv2/opencv.hpp>
#include <fstream>
#include <stdexcept>



static constexpr int CENTER_RATIO = 10;          // 中心 10 %


// 对指定文件夹下的一组图像求平均值
cv::Mat avg(std::string folder_path, int nums);

// 基于 raw = g·I + d 的物理模型，用两帧同场景不同亮度图像的中心均值差/逐像素差估计 K=1/g 与 B=-d/g
void calibrate(cv::Mat img1, cv::Mat img2, int16_t** K, int16_t** B);

// 把 K/B 逐像素量化的 int16 结果，按"每行 KKKKBBBB"的 hex 文本格式导出为 ISP 系数表 .coe，服务于硬件 ROM 初始化。
void write_kb_coe(const std::string& filename, int16_t** K, int16_t** B, int height, int width);

// 用 y = ((K·x)>>14) + B 的定点式逐像素做镜头阴影校正
void lsc(cv::Mat img_in, int16_t** K, int16_t** B, cv::Mat img_out, bool en);

