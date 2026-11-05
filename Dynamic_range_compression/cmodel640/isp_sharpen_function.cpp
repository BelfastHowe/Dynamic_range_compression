#include "isp_sharpen_function.h"


cv::Mat sharpenImage_int8(const cv::Mat& inputImage, cv::Mat outputImage, int en, int kernel[][3], int akt[])
{
    // 确保输入是单通道灰度图像
    CV_Assert(inputImage.type() == CV_8UC1);

     // 遍历图像（考虑边缘）
    for (int y = 0; y < inputImage.rows; y++) {
        for (int x = 0; x < inputImage.cols; x++) {
            int sum = 0;
            int sum_t = 0;
            int sum_a = 0;

            //计算

            // 应用3x3核
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    // 计算对称位置处理边缘
                    int px = x + kx;
                    int py = y + ky;

                    // 对称处理边缘像素
                    if (px < 0) px = -px;
                    else if (px >= inputImage.cols) px = inputImage.cols - 2;

                    if (py < 0) py = -py;
                    else if (py >= inputImage.rows) py = inputImage.rows - 2;

                    // 累加核运算
                    sum += (inputImage.at<uchar>(py, px) * kernel[ky + 1][kx + 1]);   //需要右移4位
                }
            }

            sum = sum * akt[1];   //需要右移6位
            sum = sum >> 10;
            sum_a = inputImage.at<uchar>(y, x) * akt[0];   //需要右移6位
            sum_a = sum_a >> 6;
            sum_t = inputImage.at<uchar>(y, x) * akt[2];   //需要右移4位
            sum_t = sum_t >> 4;

            if (abs(sum) < abs(sum_t))
                sum = sum_a + sum;
            else {
                if (sum > 0)
                    sum = sum_a + sum_t;
                else
                    sum = sum_a - sum_t;
            }
            // 确保值在0-255范围内
           // outputImage.at<uchar>(y, x) = saturate_cast<uchar>(sum);

            //防溢出
            if (sum > 255)
                outputImage.at<uchar>(y, x) = 255;
            else if (sum < 0)
                outputImage.at<uchar>(y, x) = 0;
            else
                outputImage.at<uchar>(y, x) = sum;
        }
    }

    if (en == 0) {
        inputImage.copyTo(outputImage);
    }

    return outputImage;
}

