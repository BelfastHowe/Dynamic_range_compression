#include "lsc.h"

cv::Mat avg(std::string folder_path, int nums)
{
    cv::Mat accumulated_image;
    int valid_image_count = 0;

    for (int i = 1; i <= nums; i++) {
        // 构建图像文件路径
        std::stringstream ss;
        ss << folder_path << "/(" << i << ")" << ".png";
        std::string image_path = ss.str();

        // 读取图像
        cv::Mat image = cv::imread(image_path, cv::IMREAD_UNCHANGED);

        if (image.empty()) {
            std::cout << "警告: 无法读取图像: " << image_path << std::endl;
            continue;
        }

        // 如果是第一张有效图像，初始化累积矩阵
        if (accumulated_image.empty()) {
            image.convertTo(accumulated_image, CV_64FC1); // 转换为浮点型以保持精度
        }
        else {
            // 检查图像尺寸是否一致
            if (image.size() != accumulated_image.size() || image.type() != CV_16UC1) {
                std::cout << "警告: 图像 " << image_path << " 尺寸或类型不匹配，跳过" << std::endl;
                continue;
            }

            cv::Mat float_image;
            image.convertTo(float_image, CV_64FC1);
            accumulated_image += float_image;
        }

        valid_image_count++;
        std::cout << "已处理图像: " << image_path << std::endl;
    }

    if (valid_image_count == 0) {
        std::cerr << "错误: 没有找到有效的图像文件" << std::endl;
        return cv::Mat();
    }

    // 计算平均值
    cv::Mat average_image;
    accumulated_image /= valid_image_count;

    accumulated_image.convertTo(average_image, CV_16UC1);

    std::cout << "成功处理 " << valid_image_count << " 张图像" << std::endl;

    return average_image;

}


static float centerMean(cv::Mat m)
{
    int c = m.cols * CENTER_RATIO / 100;
    int r = m.rows * CENTER_RATIO / 100;
    int x0 = (m.cols - c) / 2;
    int y0 = (m.rows - r) / 2;
    cv::Mat roi = m(cv::Rect(x0, y0, c, r));
    //cout << roi << endl;
    cv::Scalar mu = cv::mean(roi);
    return static_cast<float>(mu[0]);
}

void calibrate(cv::Mat img1, cv::Mat img2, int16_t** K, int16_t** B)
{
    float y1 = centerMean(img1);
    float y2 = centerMean(img2);
    std::cout << "Center mean  y1=" << y1 << "  y2=" << y2 << '\n';
    int height = img2.rows;
    int width = img2.cols;
    float sub1 = y2 - y1;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            float sub2 = img2.at<uint16_t>(i, j) - img1.at<uint16_t>(i, j);
            if (fabs(sub2) < 1e-6f) {
                K[i][j] = K[i][j - 1];
                B[i][j] = B[i][j - 1];
            }
            else {
                float k = sub1 / sub2;
                float b = (y1 - k * img1.at<uint16_t>(i, j) + y2 - k * img2.at<uint16_t>(i, j)) / 2;
                K[i][j] = static_cast<int16_t>(round(k * (1 << 14))); // 左移14位四舍五入取S2.14
                B[i][j] = static_cast<int16_t>(round(b)); // 直接四舍五入取整S14.0

            }
        }
    }

    std::cout << "LSC calibrate done\n";
}

void write_kb_coe(const std::string& filename, int16_t** K, int16_t** B, int height, int width)
{
    std::ofstream coe(filename);
    if (!coe.is_open()) throw std::runtime_error("can not create " + filename);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            coe << std::hex << std::setw(4) << std::setfill('0') << K[i][j];
            coe << std::hex << std::setw(4) << std::setfill('0') << B[i][j];
            if (i != height - 1 || j != width - 1) coe << "\n";
        }
    }
    coe << "\n";
    coe.close();
}

void lsc(cv::Mat img_in, int16_t** K, int16_t** B, cv::Mat img_out, bool en)
{
    int height = img_in.rows;
    int width = img_in.cols;

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            int k = K[i][j];
            int b = B[i][j];
            uint16_t in = img_in.at<uint16_t>(i, j);

            int out = ((k * in) >> 14) + b;//+ (1 << 13)

            if (out < 0) {
                out = 0;
            }
            else if (out >= (1 << 14)) {
                out = (1 << 14) - 1;
            }
            if (i == 0 && j == 35)
                std::cout << "k:" << k << " b:" << b << " in:" << in << " out:" << out << std::endl;

            if (en) {
                img_out.at<uint16_t>(i, j) = uint16_t(out);
            }
            else {
                img_out.at<uint16_t>(i, j) = in;
            }

        }
    }
}



