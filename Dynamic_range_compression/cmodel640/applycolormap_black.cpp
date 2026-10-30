#include "colormap.h"



// 应用颜色映射
void applyColorMap1(Image src, Image dst, MapTypes colormap, int gamma_type)
{
    if (src.channels != 1) {
        printf("Source image must be grayscale (1 channel)\n");
        return;
    }

    if (dst.channels != 3) {
        printf("Destination image must be color (3 channels)\n");
        return;
    }

    if (src.width != dst.width || src.height != dst.height) {
        printf("Source and destination images must have same dimensions\n");
        return;
    }

    int size = src.width * src.height;

    for (int i = 0; i < size; i++) {
        unsigned char gray = src.data[i];
        unsigned char* color = &dst.data[i * 3];

        switch (colormap) {     //  白色热图
        case  MAP_WhiteHot: {
            color[0] = 255 - gray;
            color[1] = 255 - gray;
            color[2] = 255 - gray;
            break;
        }
        case MAP_BlackHot: {                             // 黑色热图
            color[0] = gray;
            color[1] = gray;
            color[2] = gray;
            break;
        }

        case MAP_IronRed: {
            if (255 - gray <= 15) {
                // 0-15: 黑蓝色渐变 (0,0,80) -> (0,80,255)
                int r0 = 0, g0 = 0, b0 = 80;
                int r1 = 0, g1 = 40, b1 = 120;
                int t = 255 - gray;
                color[0] = b0 + (b1 - b0) * t / 15;
                color[1] = g0 + (g1 - g0) * t / 15;
                color[2] = r0 + (r1 - r0) * t / 15;
            }
            else if (255 - gray <= 63) {
                // 16-63: 黑蓝色渐变紫色 (0,80,255) -> (128,0,255)
                int r0 = 0, g0 = 40, b0 = 120;
                int r1 = 128, g1 = 0, b1 = 220;
                int t = 255 - gray - 16;
                color[0] = b0 + (b1 - b0) * t / (63 - 16);
                color[1] = g0 + (g1 - g0) * t / (63 - 16);
                color[2] = r0 + (r1 - r0) * t / (63 - 16);
            }
            else if (255 - gray <= 127) {
                // 64-191: 紫色到橘黄色 (128,0,255) -> (255,180,0)
                int r0 = 128, g0 = 0, b0 = 220;
                int r1 = 255, g1 = 128, b1 = 0;
                int t = 255 - gray - 64;
                color[0] = b0 + (b1 - b0) * t / (127 - 64);
                color[1] = g0 + (g1 - g0) * t / (127 - 64);
                color[2] = r0 + (r1 - r0) * t / (127 - 64);
            }
            else if (255 - gray <= 223) {
                // 192-244: 橘黄到亮黄 (255,180,0) -> (255,255,0)
                int r0 = 255, g0 = 128, b0 = 0;
                int r1 = 255, g1 = 255, b1 = 0;
                int t = 255 - gray - 128;
                color[0] = b0;
                color[1] = g0 + (g1 - g0) * t / (223 - 128);
                color[2] = r0;
            }
            else {
                // 244-255: 亮黄到极亮黄 (255,255,0) -> (255,255,128)
                int r0 = 255, g0 = 255, b0 = 0;
                int r1 = 255, g1 = 255, b1 = 192;
                int t = 255 - gray - 224;
                color[0] = b0 + (b1 - b0) * t / (255 - 224);
                color[1] = g0;
                color[2] = r0;
            }
            break;
        }
        case MAP_RedHot: {
            int th;
            int s;
            if (gamma_type == 0) {
                th = 73;
                s = 0;
            }
            else if (gamma_type == 1) {
                th = 140;
                s = 90;
            }
            else {
                th = 170;
                s = 140;
            }
            if (gray < s) {
                color[2] = 255;
                color[1] = 0;
                color[0] = 0;
            }
            if (gray > th) {

                int t = 255 - gray;
                int len = 255 - th - 1;
                color[2] = (255 * t) / len;
                color[1] = (255 * t) / len;
                color[0] = (255 * t) / len;
            }
            else {
                // 205~255: 白色(255,255,255) -> 红色(255,0,0)
                int t = th - gray;
                int len = th - s; // 50
                color[2] = 255;
                color[1] = 255 - (255 * t) / len; // 255~0
                color[0] = 255 - (255 * t) / len; // 255~0
            }

            break;
        }

        default:
            // 默认使用JET映射
            color[0] = gray;
            color[1] = gray;
            color[2] = gray;
            break;
        }
    }
}

void ColorMap(cv::Mat originalImg, cv::Mat outImg, int type, int gamma_type)
{
    // 1. 加载原始图像（BMP格式）

    // 2. 转换为8位灰度图像
    cv::Mat grayImg;
    if (originalImg.channels() == 3) {
        cv::cvtColor(originalImg, grayImg, cv::COLOR_BGR2GRAY);
    }
    else if (originalImg.channels() == 4) {
        cv::cvtColor(originalImg, grayImg, cv::COLOR_BGRA2GRAY);
    }
    else {
        grayImg = originalImg.clone();
    }


    // 3. 配置要使用的颜色映射集合
    cv::Mat white_hot(grayImg.size(), CV_8UC3);
    cv::Mat black_hot(grayImg.size(), CV_8UC3);
    cv::Mat iron(grayImg.size(), CV_8UC3);
    cv::Mat red_hot(grayImg.size(), CV_8UC3);

    // Convert cv::Mat to Image structure
    Image srcImage = { grayImg.data, grayImg.cols, grayImg.rows, grayImg.channels() };
    Image dstWhiteHot = { white_hot.data, white_hot.cols, white_hot.rows, white_hot.channels() };
    Image dstBlackHot = { black_hot.data, black_hot.cols, black_hot.rows, black_hot.channels() };
    Image dstIron = { iron.data, iron.cols, iron.rows, iron.channels() };
    Image dstRedHot = { red_hot.data, red_hot.cols, red_hot.rows, red_hot.channels() };

    // 4. 应用伪彩色映射
    applyColorMap1(srcImage, dstWhiteHot, MAP_WhiteHot, gamma_type);
    applyColorMap1(srcImage, dstBlackHot, MAP_BlackHot, gamma_type);
    applyColorMap1(srcImage, dstIron, MAP_IronRed, gamma_type);
    applyColorMap1(srcImage, dstRedHot, MAP_RedHot, gamma_type);

    // 5. 显示图像
    //imshow("ori", originalImg);
    //imshow("gray", grayImg);

    //rgb565(white_hot);
    //rgb565(black_hot);
    //rgb565(red_hot);
    //rgb565(iron);

    if (type == 0) {
        imshow("White Hot", white_hot);
        imwrite("MAP_WhiteHot.png", white_hot);
        white_hot.copyTo(outImg);
    }
    else if (type == 1) {
        imshow("Black Hot", black_hot);
        imwrite("MAP_BlackHot.png", black_hot);
        black_hot.copyTo(outImg);
    }
    else if (type == 2) {
        imshow("Iron", iron);
        imwrite("MAP_IronRed.png", iron);
        iron.copyTo(outImg);
    }
    else if (type == 3) {
        imshow("Red Hot", red_hot);
        imwrite("MAP_RedHot.png", red_hot);
        red_hot.copyTo(outImg);
    }
    else {
        imshow("White Hot", white_hot);
        imwrite("MAP_WhiteHot.png", white_hot);
        white_hot.copyTo(outImg);
    }

    // 释放内存
    grayImg.release();
    white_hot.release();
    black_hot.release();
    iron.release();
    red_hot.release();

}





