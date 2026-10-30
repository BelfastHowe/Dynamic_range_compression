#include "gamma.h"


void build_gamma_lut(unsigned char lut[256], float gamma)
{
    for (int i = 0; i < 256; ++i)
    {
        double normalized = (double)i / 255.0;
        double corrected = pow(normalized, gamma);
        lut[i] = (unsigned char)(corrected * 255.0 + 0.5);
        printf("lut[%d] <= 8'h%02x;\n", i, lut[i]);
    }
}

void GammaMAP(Image_GAMMA src, Image_GAMMA dst, MapTypes_GAMMA gammamap)
{
    if (src.channels != 1) {
        printf("Source image must be gray (1 channel)\n");
        return;
    }
    if (dst.channels != 1) {
        printf("Destination image must be gray (1 channels)\n");
        return;
    }
    if (src.width != dst.width || src.height != dst.height) {
        printf("Source and destination images must have same dimensions\n");
        return;
    }
    unsigned char lut_brighter[256];
    unsigned char lut_darker[256];
    build_gamma_lut(lut_brighter, 1 / 2.2f);
    build_gamma_lut(lut_darker, 2.2f);

    for (int i = 0; i < src.height; ++i)
        for (int j = 0; j < src.width; ++j)
            for (int k = 0; k < src.channels; ++k)
            {
                int idx = i * src.width * src.channels + j * src.channels + k;
                switch (gammamap)
                {
                case GammaMAP_ori:
                    dst.data[idx] = src.data[idx];
                    break;
                case GammaMAP_light:
                    dst.data[idx] = lut_brighter[src.data[idx]];
                    break;
                case GammaMAP_dark:
                    dst.data[idx] = lut_darker[src.data[idx]];
                    break;
                default:
                    dst.data[idx] = src.data[idx];
                    break;
                }
                //cout << j << ":" << int(src.data[idx]) << " " << int(dst.data[idx]) << endl;
            }
}

void gamma(cv::Mat input, cv::Mat output, int type)
{

    // 1. 读取图像信息
    int width = input.cols;
    int height = input.rows;
    int channels = input.channels();


    // 2. 构造Image结构体
    Image_GAMMA src = { width, height, channels, input.data };

    // 3. 创建输出图像
    cv::Mat img_ori(height, width, CV_8UC1);
    cv::Mat img_light(height, width, CV_8UC1);
    cv::Mat img_dark(height, width, CV_8UC1);

    Image_GAMMA dst_ori = { width, height, channels, img_ori.data };
    Image_GAMMA dst_light = { width, height, channels, img_light.data };
    Image_GAMMA dst_dark = { width, height, channels, img_dark.data };

    // 4. Gamma矫正
    GammaMAP(src, dst_ori, GammaMAP_ori);//type0
    GammaMAP(src, dst_light, GammaMAP_light);//type1
    GammaMAP(src, dst_dark, GammaMAP_dark);//type2

    // 5. 显示结果
    //cv::imshow("Original", input);
    //cv::imshow("Gamma Corrected ( ori)", img_ori);
    //cv::imshow("Gamma Corrected ( brighter)", img_light);
    //cv::imshow("Gamma Corrected ( darker)", img_dark);

    if (type == 1)
        img_light.copyTo(output);
    else if (type == 2)
        img_dark.copyTo(output);
    else
        img_ori.copyTo(output);

    //6. 释放内存
    input.release();
    img_ori.release();
    img_light.release();
    img_dark.release();

}



