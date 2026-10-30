#pragma once


#include <opencv2/opencv.hpp>

// 定义图像结构
typedef struct {
    unsigned char* data;  // 图像数据指针
    int width;            // 图像宽度
    int height;           // 图像高度
    int channels;         // 通道数(1:灰度, 3:彩色)
} Image;

// 定义颜色映射类型
typedef enum {
    MAP_WhiteHot,
    MAP_BlackHot,
    MAP_IronRed,
    MAP_RedHot,

} MapTypes;


void ColorMap(cv::Mat originalImg, cv::Mat outImg, int type, int gamma_type);


