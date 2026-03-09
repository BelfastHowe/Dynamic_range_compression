// Dynamic_range_compression.h: 标准系统包含文件的包含文件
// 或项目特定的包含文件。

#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <filesystem>
#include <cmath>


inline
int imwrite_mdy_private(cv::InputArray input, const std::string file_name);

//处理Raw文件
int processRawFile(const std::string& inputPath, const std::string& outputPath);

// 百分位映射函数，将输入图像的像素值根据指定的低百分位和高百分位进行线性映射，输出8位图像
int percentile_mapping(cv::InputArray input, cv::OutputArray output, double lowPct, double highPct);

int benchmark_main();

// 计算图像的信息熵，输入为单通道8位图像，输出为熵值（最大值为8，均匀分布时达到最大）
double calcEntropy(cv::InputArray src);


// TODO: 在此处引用程序需要的其他标头。
