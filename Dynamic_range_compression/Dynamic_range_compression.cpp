// Dynamic_range_compression.cpp: 定义应用程序的入口点。
//

#include "Dynamic_range_compression.h"

namespace fs = std::filesystem;

inline
int imwrite_mdy_private(cv::InputArray input, const std::string file_name)
{
    cv::Mat src = input.getMat().clone();

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);

    std::tm now_tm;
    localtime_s(&now_tm, &now_c);

    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y%m%d_%H%M%S_") << now.time_since_epoch().count() << std::string("_");

    std::string output_file_name = std::string("C:\\Users\\Belfast\\Desktop\\result\\") + oss.str() + file_name + std::string(".png");

    std::cout << output_file_name << std::endl;
    cv::imwrite(output_file_name, src);
    cv::waitKey(1);

    return 0;
}

int processRawFile(const std::string& inputPath, const std::string& outputPath)
{
    // 图像参数
    const int width = 384;
    const int height = 288;
    const int imageSize = width * height * 3;

    // 读取.raw文件
    std::ifstream file(inputPath, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << inputPath << std::endl;
        return -1;
    }

    // 读取文件内容
    std::vector<char> buffer(imageSize);
    file.read(buffer.data(), imageSize);
    file.close();

    // 转换为8UC3图像数据
    cv::Mat image(height, width, CV_8UC3, buffer.data());

    // 检查图像数据是否有效
    if (image.empty()) {
        std::cerr << "图像数据为空: " << inputPath << std::endl;
        return -1;
    }

    cv::Mat result(height, width, CV_16UC1);

    // 遍历每个像素
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 获取RGB值
            cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);
            uint8_t r = pixel[0];  // 注意：raw是RGB顺序，mat image是BGR顺序
            uint8_t g = pixel[1];
            uint8_t b = pixel[2];

            // 按RGB顺序从高位到低位拼接
            uint32_t combined = ((uint32_t)r << 16) | ((uint32_t)g << 8) | b;

            // 取低16位
            uint16_t value = combined & 0xFFFF;

            // 存储结果
            result.at<uint16_t>(y, x) = value;
        }
    }

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);

    std::cout << "最小值: " << minVal << " 位置: " << minLoc << std::endl;
    std::cout << "最大值: " << maxVal << " 位置: " << maxLoc << std::endl;

    // 保存为PNG文件
    if (!cv::imwrite(outputPath, result)) {
        std::cerr << "无法保存图像: " << outputPath << std::endl;
        return -1;
    }

    std::cout << "成功转换并保存: " << inputPath << " -> " << outputPath << std::endl;

    return 0;
}

cv::Mat retinex_enhance(const cv::Mat& src, double sigma)
{
    // 转换为float并加1避免log(0)
    cv::Mat src_float;
    src.convertTo(src_float, CV_32F, 1.0 / 255.0);
    src_float += 1.0f;

    // 对数域
    cv::Mat log_src;
    cv::log(src_float, log_src);

    // 估计光照分量（高斯模糊）
    cv::Mat log_illumination;
    cv::GaussianBlur(log_src, log_illumination, cv::Size(0, 0), sigma);

    // 计算反射分量
    cv::Mat log_reflectance = log_src - log_illumination;

    // 指数还原
    cv::Mat reflectance;
    cv::exp(log_reflectance, reflectance);

    // 归一化到0-255
    cv::Mat result;
    cv::normalize(reflectance, result, 0, 255, cv::NORM_MINMAX, CV_8U);

    return result;
}

int rgb2png()
{
    //std::string inputDir = "C:/Users/Belfast/Desktop/before_mapping/2026-03-02_15-46-24";
    std::string inputDir = "C:/Users/Belfast/Desktop/before_mapping/2026-03-02_16-04-05";
    std::string outputDir = "C:/Users/Belfast/Desktop/result";

    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    // 遍历输入目录中的所有.raw和.rgb文件
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if ((entry.path().extension() == ".raw") || (entry.path().extension() == ".rgb")) {
            // 构造输出文件路径
            std::string inputPath = entry.path().string();
            std::string fileName = entry.path().stem().string(); // 获取不带扩展名的文件名

            std::string outputPath = outputDir + "/" + fileName + ".png";

            // 处理单个文件
            processRawFile(inputPath, outputPath);
        }
    }

	return 0;
}

// 线性映射函数，将14位图像映射到8位图像
int linear_mapping(cv::InputArray input, cv::OutputArray output)
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    double minVal, maxVal;
    cv::minMaxLoc(img14bit, &minVal, &maxVal);

    cv::Mat img8bit;
    img14bit.convertTo(img8bit, CV_8U, 255.0 / (maxVal - minVal), -minVal * 255.0 / (maxVal - minVal));//饱和溢出保护

	output.assign(img8bit);

    return 0;
}

// 百分位映射函数，将14位图像映射到8位图像，使用指定的低百分位和高百分位进行线性映射
int percentile_mapping(cv::InputArray input, cv::OutputArray output, double lowPct = 0.5, double highPct = 99.5)
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    // 计算百分位
    cv::Mat flat = img14bit.reshape(1, 1);
    cv::Mat sorted;
    cv::sort(flat, sorted, cv::SORT_ASCENDING);

    int lowIdx = static_cast<int>(sorted.cols * lowPct / 100.0);
    int highIdx = static_cast<int>(sorted.cols * highPct / 100.0);
    double pLow = sorted.at<uint16_t>(0, lowIdx);
    double pHigh = sorted.at<uint16_t>(0, highIdx);

    cv::Mat clipped;
    cv::threshold(img14bit, clipped, pHigh, pHigh, cv::THRESH_TRUNC);
    cv::Mat clipped2;
    cv::max(clipped, pLow, clipped2);

    cv::Mat img8bit;
    clipped2.convertTo(img8bit, CV_8U, 255.0 / (pHigh - pLow), -pLow * 255.0 / (pHigh - pLow));

	output.assign(img8bit);
    return 0;
}

// CLAHE映射函数，将14位图像映射到8位图像，使用CLAHE进行局部对比度增强
int clahe_mapping(cv::InputArray input, cv::OutputArray output, double clipLimit = 2.0, cv::Size tileSize = { 8, 8 })
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    // 先线性压到16bit（CLAHE支持16bit）
    cv::Mat img16bit;
    img14bit.convertTo(img16bit, CV_16U, 65535.0 / 16383.0);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileSize);
    cv::Mat result;
    clahe->apply(img16bit, result);

    // 再压到8bit
    cv::Mat img8bit;
    result.convertTo(img8bit, CV_8U, 255.0 / 65535.0);

	output.assign(img8bit);
    return 0;
}

int gammaMapping(cv::InputArray input, cv::OutputArray output, double gamma = 0.5)
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    cv::Mat normalized;
    img14bit.convertTo(normalized, CV_32F, 1.0 / 16383.0);

    cv::Mat corrected;
    cv::pow(normalized, gamma, corrected);

    cv::Mat img8bit;
    corrected.convertTo(img8bit, CV_8U, 255.0);

	output.assign(img8bit);
    return 0;
}

int main()
{
    // rgb2png();
    // 测试retinex增强
    cv::Mat src = cv::imread("C:/Users/Belfast/Desktop/before_mapping/2026-03-02_16-04-05/20260302_160405_000000.raw.png");
    if (src.empty()) {
        std::cerr << "无法读取图像" << std::endl;
        return -1;
    }
    cv::Mat enhanced = retinex_enhance(src, 15.0);
    imwrite_mdy_private(enhanced, "retinex_enhanced");
    return 0;
}
