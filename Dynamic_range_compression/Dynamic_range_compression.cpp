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

// 直方图显示配置结构体
struct HistDisplayConfig
{
    int    histW = 1024;          // 显示窗口宽度
    int    histH = 576;          // 显示窗口高度
    int    binCount = 256;          // 直方图 bin 数量
    cv::Scalar barColor = cv::Scalar(200, 200, 200);  // 柱状颜色
    cv::Scalar bgColor = cv::Scalar(30, 30, 30);   // 背景颜色
    cv::Scalar axisColor = cv::Scalar(180, 180, 180);  // 坐标轴颜色
    bool   logScale = false;        // 是否对数纵轴
    std::string winName = "Histogram"; // 窗口名称
};

// 显示单通道图像的直方图，支持8位和16位图像
int showHistogram(cv::InputArray input, const HistDisplayConfig& cfg = {})
{
	cv::Mat img = input.getMat().clone();

    if (img.channels() != 1)
        return -1;
    if (img.depth() != CV_8U && img.depth() != CV_16U)
        return -1;

    // Step 1: 计算直方图
    float rangeMax = (img.depth() == CV_8U) ? 256.0f : 65536.0f;
    float ranges[] = { 0, rangeMax };
    const float* histRange = { ranges };
    int binCount = cfg.binCount;

    cv::Mat hist;
    cv::calcHist(&img, 1, nullptr, cv::Mat(), hist, 1, &binCount, &histRange);

    // Step 2: 对数纵轴（可选）
    if (cfg.logScale) {
        cv::log(hist + 1.0f, hist);
    }

    // Step 3: 归一化到画布高度
    cv::normalize(hist, hist, 0, cfg.histH - 20, cv::NORM_MINMAX);

    // Step 4: 绘制画布
    cv::Mat canvas(cfg.histH, cfg.histW, CV_8UC3, cfg.bgColor);

    int binW = cfg.histW / binCount;
    for (int i = 0; i < binCount; i++)
    {
        int barH = static_cast<int>(hist.at<float>(i));
        cv::rectangle(
            canvas,
            cv::Point(i * binW, cfg.histH - barH),
            cv::Point((i + 1) * binW - 1, cfg.histH - 1),
            cfg.barColor,
            cv::FILLED
        );
    }

    // Step 5: 坐标轴
    // x 轴
    cv::line(canvas,
        cv::Point(0, cfg.histH - 1),
        cv::Point(cfg.histW - 1, cfg.histH - 1),
        cfg.axisColor, 1);
    // y 轴
    cv::line(canvas,
        cv::Point(0, 0),
        cv::Point(0, cfg.histH - 1),
        cfg.axisColor, 1);

    // Step 6: 刻度标注
    int tickCount = 4;
    for (int i = 0; i <= tickCount; i++)
    {
        int x = i * (cfg.histW - 1) / tickCount;
        float val = i * rangeMax / tickCount;

        // 刻度线
        cv::line(canvas,
            cv::Point(x, cfg.histH - 1),
            cv::Point(x, cfg.histH - 5),
            cfg.axisColor, 1);

        // 刻度标签
        std::string label = (val >= 1000)
            ? std::to_string(static_cast<int>(val / 1000)) + "k"
            : std::to_string(static_cast<int>(val));
        cv::putText(canvas, label,
            cv::Point(x + 2, cfg.histH - 6),
            cv::FONT_HERSHEY_PLAIN, 0.7, cfg.axisColor, 1);
    }

    // 深度标注
    std::string depthLabel = (img.depth() == CV_8U) ? "8bit" : "16bit";
    if (cfg.logScale) depthLabel += " (log)";
    cv::putText(canvas, depthLabel,
        cv::Point(cfg.histW - 55, 15),
        cv::FONT_HERSHEY_PLAIN, 0.9, cfg.axisColor, 1);

    cv::imshow(cfg.winName, canvas);
    cv::waitKey();

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

// 16UC1图像的直方图均衡化函数
int equalize_hist_16UC1(cv::InputArray input, cv::OutputArray output, double maxVal = 65535.0)
{
    cv::Mat src = input.getMat().clone();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    int binCount = static_cast<int>(maxVal) + 1;
    int totalPixels = src.rows * src.cols;

    // Step 1: 统计直方图
    std::vector<int> hist(binCount, 0);
    src.forEach<uint16_t>([&](uint16_t val, const int*)
        {
        hist[val]++;
        });

    // Step 2: 计算 CDF
    std::vector<int> cdf(binCount, 0);
    cdf[0] = hist[0];
    for (int i = 1; i < binCount; i++)
    {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    int cdfMin = 0;
    for (int i = 0; i < binCount; i++)
    {
        if (cdf[i] > 0)
        { 
            cdfMin = cdf[i];
            break;
        }
    }

    // Step 3: 构建 LUT
    std::vector<uint16_t> lut(binCount, 0);
    for (int i = 0; i < binCount; i++)
    {
        if (cdf[i] > 0)
        {
            lut[i] = static_cast<uint16_t>(
                std::round(static_cast<double>(cdf[i] - cdfMin) / (totalPixels - cdfMin) * maxVal)
                );
        }
    }

    // Step 4: 应用映射
    cv::Mat dst = cv::Mat::zeros(src.size(), src.type());
    src.forEach<uint16_t>([&](uint16_t val, const int* pos)
        {
        dst.at<uint16_t>(pos[0], pos[1]) = lut[val];
        });

    output.assign(dst);
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
    //result.convertTo(img8bit, CV_8U, 255.0 / 65535.0);
	linear_mapping(result, img8bit);

	output.assign(img8bit);
    return 0;
}

int gamma_transform_16UC1(cv::InputArray input, cv::OutputArray output, double gamma = 0.5)
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    cv::Mat normalized;
    img14bit.convertTo(normalized, CV_32F, 1.0 / 16383.0);

    cv::Mat corrected;
    cv::pow(normalized, gamma, corrected);

    cv::Mat dst;
    corrected.convertTo(dst, CV_16U, 16383.0);

	output.assign(dst);
    return 0;
}

int main()
{
    std::string image_path = std::string(IMAGE_DIR) + "19700101_000143_533.png";

    cv::Mat src = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (src.empty())
    {
        std::cerr << "无法读取图像" << std::endl;
        return -1;
    }

    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    cv::imshow("原始图像", src);

    HistDisplayConfig histcfg_src;
	histcfg_src.winName = "原始图像直方图";
	histcfg_src.logScale = true;
    histcfg_src.binCount = 512;
	showHistogram(src, histcfg_src);

    cv::Mat dst_CLAHE;
    clahe_mapping(src, dst_CLAHE);

	HistDisplayConfig histcfg_clahe;
	histcfg_clahe.winName = "CLAHE图像直方图";
	histcfg_clahe.logScale = true;
	histcfg_clahe.binCount = 512;
	showHistogram(dst_CLAHE, histcfg_clahe);

    //cv::Mat enhanced = retinex_enhance(src, 15.0);
    imwrite_mdy_private(dst_CLAHE, "CLAHE");

    return 0;
}
