#pragma once

#include "Dynamic_range_compression.h"

// 基准测试
int benchmark_main();

// 计算图像的信息熵，输入为单通道8位图像，输出为熵值（最大值为8，均匀分布时达到最大）
double calcEntropy(cv::InputArray src);

// 计算图像的平均梯度
double calcAverageGradient(cv::InputArray src);

// 计算图像的结构相似性指数（SSIM），输入为两张单通道8位图像，输出为SSIM值（范围[-1, 1]，越接近1越好）
double calcSSIM(cv::InputArray input, cv::InputArray in_ref);
