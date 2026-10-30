#pragma once

#include <opencv2/opencv.hpp>



typedef struct
{
    int width;
    int height;
    int channels;
    unsigned char* data;
} Image_GAMMA;


typedef enum
{
    GammaMAP_ori,
    GammaMAP_light,
    GammaMAP_dark
} MapTypes_GAMMA;


void gamma(cv::Mat input, cv::Mat output, int type);


