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

// 百分位映射函数，将输入图像的像素值根据指定的低百分位和高百分位进行线性映射，输出8位图像
int percentile_mapping(cv::InputArray input, cv::OutputArray output, double lowPct, double highPct);


// 多尺度Retinex算法实现，输入为单通道16位图像，输出为单通道16位图像，sigmas为高斯模糊的标准差列表
int multi_scale_retinex(cv::InputArray input, cv::OutputArray output, const std::vector<double>& sigmas);

// 线性映射函数，将输入图像的像素值线性映射到0-255范围，输入为单通道16位图像，输出为单通道8位图像
int linear_mapping(cv::InputArray input, cv::OutputArray output);

// CLAHE函数实现，输入为单通道16位图像，输出为单通道8位图像，clipLimit为对比度限制，tileSize为分块规则
int clahe_mapping(cv::InputArray input, cv::OutputArray output, double clipLimit, cv::Size tileSize);

int clahe_fixed_mapping(cv::InputArray input, cv::OutputArray output, int clipLimit, cv::Size tileSize);

// 全局局部自适应融合函数，根据局部图像的梯度计算权重图，将全局映射图和局部映射图进行加权融合，得到最终的增强图像
int global_local_adaptive_fusion(cv::InputArray input, cv::OutputArray output);

// 基于双边滤波的细节增强算法实现
int dde_enhance(cv::InputArray input, cv::OutputArray output);



// TODO: 在此处引用程序需要的其他标头。
