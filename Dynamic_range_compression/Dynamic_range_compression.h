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


// TODO: 在此处引用程序需要的其他标头。
