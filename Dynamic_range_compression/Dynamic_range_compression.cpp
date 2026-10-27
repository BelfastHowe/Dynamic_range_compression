// Dynamic_range_compression.cpp: 定义应用程序的入口点。
//

#include "Dynamic_range_compression.h"
#include "quantitative_assessment.h"
#include "Quantization.h"
#include "DDE_improvement.h"

namespace fs = std::filesystem;
using Ms = std::chrono::duration<double, std::milli>;

inline
int imwrite_mdy_private(cv::InputArray input, const std::string file_name)
{
    cv::Mat src = input.getMat();

    PWSTR desktop_path_w = nullptr;
    HRESULT hr = SHGetKnownFolderPath(FOLDERID_Desktop, 0, NULL, &desktop_path_w);

    if (FAILED(hr))
    {
        std::cerr << "获取桌面路径失败！" << std::endl;
        return -1;
    }

    std::filesystem::path result_dir = std::filesystem::path(desktop_path_w) / "result";

    CoTaskMemFree(desktop_path_w);
    desktop_path_w = nullptr;

    std::filesystem::create_directories(result_dir);

    auto now = std::chrono::system_clock::now();
    auto now_c = std::chrono::system_clock::to_time_t(now);

    std::tm now_tm;
    localtime_s(&now_tm, &now_c);

    std::ostringstream oss;
    oss << std::put_time(&now_tm, "%Y%m%d_%H%M%S_") << now.time_since_epoch().count() << std::string("_");

    std::filesystem::path output_path = result_dir / (oss.str() + file_name + ".png");
    std::string output_file_name = output_path.string();

    //std::cout << "已保存至: " << output_file_name << std::endl;
    cv::imwrite(output_file_name, src);
    cv::waitKey(1);

    return 0;
}

int imwrite_mdy_private_normalization_8u(cv::InputArray input, const std::string file_name)
{
    cv::Mat src = input.getMat();

    cv::Mat normalized;
    cv::normalize(src, normalized, 0, 255, cv::NORM_MINMAX, CV_8U);

    return imwrite_mdy_private(normalized, file_name);
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

int processYuvFile(const std::string& inputPath, const std::string& outputPath)
{
    // 图像参数
    const int width = 640;
    const int height = 512;
	const size_t imageSize = static_cast<size_t>(width) * height * sizeof(uint16_t);

    // 读取.yuv文件
    std::ifstream file(inputPath, std::ios::binary);
    if (!file) {
        std::cerr << "无法打开文件: " << inputPath << std::endl;
        return -1;
    }

    // 读取文件内容
    std::vector<char> buffer(imageSize);
    file.read(buffer.data(), imageSize);

    // 检查图像数据是否有效
	if (static_cast<size_t>(file.gcount()) != imageSize)
    {
		std::cerr << "读取的图像数据大小不正确: " << inputPath << std::endl;
		return -1;
	}

    file.close();

    // 转换为16uc1图像数据
    cv::Mat image(height, width, CV_8UC2, buffer.data());

    cv::Mat result(height, width, CV_16UC1);

    // 遍历每个像素
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // 获取RGB值
            cv::Vec2b pixel = image.at<cv::Vec2b>(y, x);
            uint8_t g = pixel[0];
            uint8_t b = pixel[1];

            // 按RGB顺序从高位到低位拼接
            uint32_t combined = ((uint32_t)g << 8) | b;

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

// 单尺度Retinex算法实现
int single_scale_retinex(cv::InputArray input, cv::OutputArray output, double sigma)
{
    cv::Mat src = input.getMat().clone();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    // 归一化到0-1
    cv::Mat src_normal;
    //cv::normalize(src, src_normal, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    src.convertTo(src_normal, CV_32F, 1.0 / 16384.0);
    src_normal += 1.0f;

    // 对数域
    cv::Mat log_src;
    cv::log(src_normal, log_src);

    // 估计光照分量（高斯模糊）
    cv::Mat gauss;
    cv::GaussianBlur(src_normal, gauss, cv::Size(0, 0), sigma);

    cv::Mat log_blur;
    cv::log(gauss, log_blur);

    // 计算反射分量
    cv::Mat log_reflectance = log_src - log_blur;

    // 指数还原
    /*cv::Mat reflectance;
    cv::exp(log_reflectance, reflectance);*/

    // 归一化到0-255
    cv::Mat result;
    cv::normalize(log_reflectance, result, 0, 255, cv::NORM_MINMAX, CV_8U);

    output.assign(result);
    return 0;
}

int multi_scale_retinex(cv::InputArray input, cv::OutputArray output, const std::vector<double>& sigmas)
{
    cv::Mat src = input.getMat().clone();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    auto ksize = 61;

    // 归一化到0-1
    cv::Mat src_normal;
    cv::normalize(src, src_normal, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    src_normal += 1.0f;

    // 对数域
    cv::Mat log_src;
    cv::log(src_normal, log_src);

    // 多尺度Retinex
    cv::Mat log_reflectance = cv::Mat::zeros(src.size(), CV_32F);
    for (const auto& sigma : sigmas)
    {
        // 估计光照分量（高斯模糊）
        cv::Mat gauss;
        cv::GaussianBlur(src_normal, gauss, cv::Size(ksize, ksize), sigma);

        cv::Mat log_blur;
        cv::log(gauss, log_blur);

        cv::Mat log_re = log_src - log_blur;

        // 累加反射分量
        log_reflectance += log_re;
    }

    log_reflectance /= static_cast<float>(sigmas.size());

    // 归一化到0-255
    cv::Mat result;
    /*cv::normalize(log_reflectance, result, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

    cv::Mat gamma;
    cv::pow(result, 2, gamma);
    cv::normalize(gamma, result, 0, 255, cv::NORM_MINMAX, CV_8U);*/

    cv::normalize(log_reflectance, result, 0, 255, cv::NORM_MINMAX, CV_8U);
    //percentile_mapping(result, result, 0.25, 99.75);

    output.assign(result);
    return 0;
}

int rgb2png()
{
    std::string inputDir = "C:/Users/Belfast/Desktop/before_mapping/2026-09-24_11-17-28";
    //std::string inputDir = "C:/Users/Belfast/Desktop/before_mapping/30度黑体2";
    //std::string inputDir = "C:/Users/Belfast/Desktop/before_mapping/人体/2026-09-04_17-46-55";
    std::string outputDir = "C:/Users/Belfast/Desktop/result";
    //std::string inputDir = "C:/Users/13012/RunboIR_Archive/2026-07-06_16-54-24";
    //std::string outputDir = "C:/Users/13012/Desktop/result";

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

int yuv2png()
{
    std::string inputDir = "C:\\Users\\13012\\Desktop\\640-1021\\uvc_红外摄像头 1_20261021_163105_927_9ee13f70";
    //std::string inputDir = "C:/Users/Belfast/Desktop/before_mapping/30度黑体2";
    //std::string inputDir = "C:/Users/Belfast/Desktop/before_mapping/人体/2026-09-04_17-46-55";
    //std::string outputDir = "C:/Users/Belfast/Desktop/result";
    //std::string inputDir = "C:/Users/13012/RunboIR_Archive/2026-07-06_16-54-24";
    std::string outputDir = "C:/Users/13012/Desktop/result";

    if (!fs::exists(outputDir)) {
        fs::create_directory(outputDir);
    }

    // 遍历输入目录中的所有.raw和.rgb文件
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if ((entry.path().extension() == ".raw") || (entry.path().extension() == ".yuv")) {
            // 构造输出文件路径
            std::string inputPath = entry.path().string();
            std::string fileName = entry.path().stem().string(); // 获取不带扩展名的文件名

            std::string outputPath = outputDir + "/" + fileName + ".png";

            // 处理单个文件
            processYuvFile(inputPath, outputPath);
        }
    }

    return 0;
}

// 显示单通道图像的直方图，支持8位和16位图像
int showHistogram(cv::InputArray input, const HistDisplayConfig& cfg)
{
    cv::Mat img = input.getMat();

    if (img.channels() != 1)
        return -1;
    if (img.depth() != CV_8U && img.depth() != CV_16U && img.depth() != CV_32F)
    {
		std::cerr << "不支持的图像深度！仅支持8位、16位和32位单通道图像。" << std::endl;
        return -1;
    }

    // Step 1: 计算直方图
    float rangeMin = 0.0f;
    float rangeMax = 256.0f;

    if (img.depth() == CV_8U)
    {
        rangeMax = 256.0f;
    }
    else if (img.depth() == CV_16U)
    {
        rangeMax = 65536.0f;
    }
    else if (img.depth() == CV_32F)
    {
        // 针对 32F 动态获取范围
        double dMin, dMax;
        cv::minMaxLoc(img, &dMin, &dMax);
        rangeMin = static_cast<float>(dMin);
        rangeMax = static_cast<float>(dMax);
        if (rangeMin == rangeMax) rangeMax += 1e-5f;
    }

    float ranges[] = { rangeMin, rangeMax };
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
        float val = rangeMin + i * (rangeMax - rangeMin) / tickCount;

        // 刻度线
        cv::line(canvas,
            cv::Point(x, cfg.histH - 1),
            cv::Point(x, cfg.histH - 5),
            cfg.axisColor, 1);

        // 刻度标签
        std::string label;
        if (img.depth() == CV_32F)
        {
            // 使用 C++ 标准库控制浮点数输出格式（保留2位小数）
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.2f", val);
            label = buf;
        }
        else
        {
            label = (val >= 1000) ? std::to_string(static_cast<int>(val / 1000)) + "k"
                : std::to_string(static_cast<int>(val));
        }

        cv::putText(canvas, label,
            cv::Point(x + 2, cfg.histH - 6),
            cv::FONT_HERSHEY_PLAIN, 0.7, cfg.axisColor, 1);
    }

    // 深度标注
    std::string depthLabel = (img.depth() == CV_8U) ? "8bit" : ((img.depth() == CV_16U) ? "16bit" : "32Float");
    if (cfg.logScale) depthLabel += " (log)";
    cv::putText(canvas, depthLabel,
        cv::Point(cfg.histW - 55, 15),
        cv::FONT_HERSHEY_PLAIN, 0.9, cfg.axisColor, 1);

	imwrite_mdy_private(canvas, "histogram");
    cv::imshow(cfg.winName, canvas);
    cv::waitKey(1);

    return 0;
}

int show_Histogram_from_vector(const std::vector<int>& histData,
    const HistDisplayConfig& cfg,
    HistCompressMode mode)
{
    if (histData.empty())
        return -1;

    const int srcBins = static_cast<int>(histData.size());
    const int dstBins = std::min(cfg.binCount, srcBins);

    std::vector<float> hist(dstBins, 0.f);

    //----------------------------------------
    // 压缩Histogram
    //----------------------------------------
    double step = static_cast<double>(srcBins) / dstBins;

    for (int i = 0; i < dstBins; i++)
    {
        int begin = static_cast<int>(i * step);
        int end = static_cast<int>((i + 1) * step);

        end = std::min(end, srcBins);

        if (begin >= end)
            continue;

        switch (mode)
        {
        case HistCompressMode::SUM:
        {
            double sum = 0;
            for (int j = begin; j < end; j++)
                sum += histData[j];
            hist[i] = static_cast<float>(sum);
            break;
        }

        case HistCompressMode::MAX:
        {
            int vmax = 0;
            for (int j = begin; j < end; j++)
                vmax = std::max(vmax, histData[j]);
            hist[i] = static_cast<float>(vmax);
            break;
        }

        case HistCompressMode::MEAN:
        {
            double sum = 0;
            for (int j = begin; j < end; j++)
                sum += histData[j];
            hist[i] = static_cast<float>(sum / (end - begin));
            break;
        }
        }
    }

    //----------------------------------------
    // Log Scale
    //----------------------------------------
    cv::Mat histMat(dstBins, 1, CV_32F);

    for (int i = 0; i < dstBins; i++)
        histMat.at<float>(i) = hist[i];

    double maxValue;
    cv::minMaxLoc(hist, nullptr, &maxValue);

    if (cfg.logScale)
        cv::log(histMat + 1.0f, histMat);

    //----------------------------------------
    // Normalize
    //----------------------------------------
    cv::normalize(histMat,
        histMat,
        0,
        cfg.histH - 20,
        cv::NORM_MINMAX);

    //----------------------------------------
    // Canvas
    //----------------------------------------
    cv::Mat canvas(cfg.histH,
        cfg.histW,
        CV_8UC3,
        cfg.bgColor);

    int binW = std::max(1, cfg.histW / dstBins);

    for (int i = 0; i < dstBins; i++)
    {
        int barH = static_cast<int>(histMat.at<float>(i));

        cv::rectangle(
            canvas,
            cv::Point(i * binW, cfg.histH - barH),
            cv::Point((i + 1) * binW - 1, cfg.histH - 1),
            cfg.barColor,
            cv::FILLED);
    }

    //----------------------------------------
    // Axis
    //----------------------------------------
    cv::line(canvas,
        cv::Point(0, cfg.histH - 1),
        cv::Point(cfg.histW - 1, cfg.histH - 1),
        cfg.axisColor);

    cv::line(canvas,
        cv::Point(0, 0),
        cv::Point(0, cfg.histH - 1),
        cfg.axisColor);

    //----------------------------------------
    // X Tick
    //----------------------------------------
    const int tickCount = 4;

    for (int i = 0; i <= tickCount; i++)
    {
        int x = i * (cfg.histW - 1) / tickCount;

        int value =
            static_cast<int>(
                i * (srcBins - 1) / static_cast<double>(tickCount));

        cv::line(canvas,
            cv::Point(x, cfg.histH - 1),
            cv::Point(x, cfg.histH - 5),
            cfg.axisColor);

        cv::putText(canvas,
            std::to_string(value),
            cv::Point(x + 2, cfg.histH - 6),
            cv::FONT_HERSHEY_PLAIN,
            0.8,
            cfg.axisColor);
    }

    //----------------------------------------
    // Info
    //----------------------------------------
    std::string title =
        "Histogram (" + std::to_string(srcBins) + " bins)";

    if (cfg.logScale)
        title += " Log";

	title += " Max: " + std::to_string(cv::saturate_cast<int>(maxValue));

    cv::putText(canvas,
        title,
        cv::Point(10, 15),
        cv::FONT_HERSHEY_PLAIN,
        1.0,
        cfg.axisColor);

    imwrite_mdy_private(canvas, "histogram");
    cv::imshow(cfg.winName, canvas);
    cv::waitKey(1);

    return 0;
}


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

int percentile_truncation_32F(cv::InputArray input, cv::OutputArray output, double low_percent, double high_percent)
{
    cv::Mat src = input.getMat();
	CV_CheckTypeEQ(src.type(), CV_32FC1, "");

    cv::Mat dn_16u;
    src.convertTo(dn_16u, CV_16U);

    const int H = dn_16u.rows, W = dn_16u.cols;
    const int total = H * W;

    std::vector<int> hist(16384, 0);
    for (int i = 0; i < H; ++i)
    {
        const uint16_t* row = dn_16u.ptr<uint16_t>(i);
        for (int j = 0; j < W; ++j)
            hist[row[j] & 0x3FFF]++;
    }

    std::vector<int> cdf(16384, 0);
    cdf[0] = hist[0];
    for (int i = 1; i < 16384; ++i)
        cdf[i] = cdf[i - 1] + hist[i];

    const int low_num = (int)(total * low_percent / 100.f);
    const int high_num = (int)(total * high_percent / 100.f);

    int thr_low = 0, thr_high = 16383;
    for (int i = 0; i < 16383; ++i)
    {
        if (cdf[i] <= low_num && cdf[i + 1] > low_num)  thr_low = i + 1;
        if (cdf[i] < high_num && cdf[i + 1] >= high_num) thr_high = i;
    }

    cv::Mat clip_32f;
    src.copyTo(clip_32f);
    clip_32f = cv::max(clip_32f, cv::saturate_cast<float>(thr_low));
    clip_32f = cv::min(clip_32f, cv::saturate_cast<float>(thr_high));

	output.assign(clip_32f);
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

int plateauHE_14bit_to_8bit(cv::InputArray input, cv::OutputArray output, double plateauRatio)
{
    const cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    const int HIST_SIZE = 16384;
    std::vector<int> hist(HIST_SIZE, 0);

    for (int r = 0; r < src.rows; ++r)
    {
        const ushort* ptr = src.ptr<ushort>(r);
        for (int c = 0; c < src.cols; ++c)
        {
            ushort val = ptr[c];
            if (val < HIST_SIZE)
            {
                hist[val]++;
            }
            else
            {
                hist[HIST_SIZE - 1]++;
            }
        }
    }

    show_Histogram_from_vector(hist);

    const int totalPixels = src.rows * src.cols;

    int Tp = static_cast<int>(plateauRatio * totalPixels / HIST_SIZE);
    Tp = std::max(Tp, 1);

	std::cout << "PHE Tp = " << Tp << std::endl;

    std::vector<int> clipped_hist(HIST_SIZE);
    for (int i = 0; i < HIST_SIZE; ++i)
    {
        clipped_hist[i] = std::min(hist[i], Tp);
    }

    std::vector<int> cdf(HIST_SIZE, 0);
    cdf[0] = clipped_hist[0];
    for (int i = 1; i < HIST_SIZE; ++i)
    {
        cdf[i] = cdf[i - 1] + clipped_hist[i];
    }

    int total_clipped_pixels = cdf[HIST_SIZE - 1];
    std::vector<uchar> lut(HIST_SIZE, 0);
    for (int i = 0; i < HIST_SIZE; ++i)
    {
        if (total_clipped_pixels > 0)
        {
            lut[i] = cv::saturate_cast<uchar>(cvRound(255.0 * cdf[i] / total_clipped_pixels));
        }
        else
        {
            lut[i] = 0;
        }
    }

    {
        std::vector<int>clip_test(clipped_hist.size(), 0);
        std::transform(clipped_hist.begin(), clipped_hist.end(), clip_test.begin(),
            [&](int val) { return val * totalPixels / total_clipped_pixels; });
		show_Histogram_from_vector(clip_test);
    }

	cv::Mat dst = cv::Mat::zeros(src.size(), CV_8UC1);
    for (int r = 0; r < src.rows; ++r)
    {
        const auto* src_ptr = src.ptr<ushort>(r);
        auto* dst_ptr = dst.ptr<uchar>(r);
        for (int c = 0; c < src.cols; ++c)
        {
            ushort val = src_ptr[c];
            if (val >= HIST_SIZE) val = HIST_SIZE - 1; // 溢出截断
            dst_ptr[c] = lut[val]; 
        }
    }

    output.assign(dst);
    return 0;
}

int improve_plateauHE_14bit_to_8bit(cv::InputArray input, cv::OutputArray output, double plateauRatio, int minDynamicRange)
{
    const cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    constexpr int HIST_SIZE = 16384;

    std::vector<int> hist(HIST_SIZE, 0);

    for (int r = 0; r < src.rows; r++)
    {
        const ushort* ptr = src.ptr<ushort>(r);
        for (int c = 0; c < src.cols; c++)
        {
            ushort v = ptr[c];
            if (v >= HIST_SIZE)
                v = HIST_SIZE - 1;

            hist[v]++;
        }
    }

    int min_index = 0;
	int max_index = HIST_SIZE - 1;
    for (int i = 0; i < HIST_SIZE; i++)
    {
        if (hist[i] != 0)
        {
            min_index = i;
            break;
        }
    }
    for (int i = HIST_SIZE - 1; i >= 0; i--)
    {
        if (hist[i] != 0)
        {
            max_index = i;
            break;
        }
    }
	int dynamic_range = max_index - min_index + 1;

    if (dynamic_range < minDynamicRange)
    {
        int center = (min_index + max_index) >> 1;
        int left = center - cv::saturate_cast<int>(((cv::saturate_cast<double>(center) + 3000.0) / (8192.0 + 3000.0)) * minDynamicRange);
        int right = left + minDynamicRange - 1;

        // 边界处理
        if (left < 0)
        {
            left = 0;
            right = minDynamicRange - 1;
        }

        if (right >= HIST_SIZE)
        {
            right = HIST_SIZE - 1;
            left = HIST_SIZE - minDynamicRange;
        }
		min_index = left;
		max_index = right;
        dynamic_range = minDynamicRange;
    }
	std::cout << "PHE test min_index = " << min_index << ", max_index = " << max_index << ", dynamic_range = " << dynamic_range << std::endl;

    const int totalPixels = src.rows * src.cols;

    int Tp = static_cast<int>(plateauRatio * totalPixels / HIST_SIZE);
    Tp = std::max(Tp, 1);

    std::cout << "PHE test Tp = " << Tp << std::endl;

    std::vector<int> clipHist = hist;

    int extraPixels = 0;

    for (int i = 0; i < HIST_SIZE; i++)
    {
        if (clipHist[i] > Tp)
        {
            extraPixels += clipHist[i] - Tp;
            clipHist[i] = Tp;
        }
    }

    int add = extraPixels / dynamic_range;
    int remain = extraPixels - dynamic_range * add;

    for (int i = min_index; i <= max_index; i++)
    {
        clipHist[i] += add;
    }

    /*if (remain != 0)
    {
		int residualStep = std::max(dynamic_range / remain, 1);
		for (int i = min_index; i <= max_index && remain > 0; i += residualStep, remain--)
		{
			clipHist[i]++;
		}
    }*/
    if (remain > 1)
    {
        int half = remain >> 1;

        // 左侧分配
        for (int i = 0; i < half; i++)
        {
            int index = min_index + i;

            if (index <= max_index)
                clipHist[index]++;
        }


        // 右侧分配
        for (int i = 0; i < half; i++)
        {
            int index = max_index - i;

            if (index >= min_index)
                clipHist[index]++;
        }
    }

	show_Histogram_from_vector(clipHist);

    std::vector<int> cdf(HIST_SIZE, 0);

    cdf[0] = clipHist[0];
    for (int i = 1; i < HIST_SIZE; i++)
    {
        cdf[i] = cdf[i - 1] + clipHist[i];
    }

    int cdf_min = 0;
    for (int i = 0; i < HIST_SIZE; i++)
    {
        if (cdf[i] != 0)
        {
            cdf_min = cdf[i];
            break;
        }
    }

    std::vector<uchar> lut(HIST_SIZE, 0);

    for (int i = 0; i < HIST_SIZE; i++)
    {
        if (cdf[i] <= cdf_min)
        {
            lut[i] = 0;
        }
        else
        {
            double value = double(cdf[i] - cdf_min) / double(totalPixels - cdf_min);

            lut[i] = cv::saturate_cast<uchar>(std::round(value * 255.0));
        }
    }

    cv::Mat dst(src.size(), CV_8UC1);

    for (int r = 0; r < src.rows; r++)
    {
        const ushort* sp = src.ptr<ushort>(r);
        uchar* dp = dst.ptr<uchar>(r);

        for (int c = 0; c < src.cols; c++)
        {
            ushort v = sp[c];

            if (v >= HIST_SIZE)
                v = HIST_SIZE - 1;

            dp[c] = lut[v];
        }
    }

    output.assign(dst);

    return 0;
}


int clahe_mapping(cv::InputArray input, cv::OutputArray output, double clipLimit = 128.0, cv::Size tileSize = { 8, 8 })
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    //img14bit /= 16;
    //cv::normalize(img14bit, img14bit, 0, 1023, cv::NORM_MINMAX);

    cv::Mat CLAHE;
    cv::Ptr<CLAHE_Float> clahe = createCLAHE_Float(clipLimit, tileSize, 16384);
    clahe->apply(img14bit, CLAHE);

    cv::Mat dst;
    linear_mapping(CLAHE, dst);

    output.assign(dst);
    return 0;
}

int clahe_fixed_mapping(cv::InputArray input, cv::OutputArray output, int clipLimit = 128, cv::Size tileSize = { 8, 8 })
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    //img14bit /= 16;
    //cv::normalize(img14bit, img14bit, 0, 1023, cv::NORM_MINMAX);

    cv::Mat CLAHE;
    cv::Ptr<CLAHE_Fixed> clahe = createCLAHE_Fixed(clipLimit, tileSize, 16384);
    clahe->apply(img14bit, CLAHE);

    auto global_min = clahe->getGlobalMin();
    auto global_max = clahe->getGlobalMax();

    auto h = CLAHE.rows;
    auto w = CLAHE.cols;
    {
        double dst_minf, dst_maxf;
        cv::minMaxLoc(CLAHE, &dst_minf, &dst_maxf);

        uint16_t in_min = CLAHE.ptr<uint16_t>()[0];
        uint16_t in_max = CLAHE.ptr<uint16_t>()[0];

        for (int i = 0; i < h; ++i)
        {
            const auto* psrc = CLAHE.ptr<uint16_t>(i);
            for (int j = 0; j < w; ++j)
            {
                if (psrc[j] < in_min) in_min = psrc[j];
                if (psrc[j] > in_max) in_max = psrc[j];
            }
        }

        if (in_min != global_min || in_max != global_max)
        {
            std::cerr << "Warning: CLAHE_Fixed global min/max mismatch! Computed: [" << in_min << ", " << in_max << "], Reported: [" 
                << global_min << ", " << global_max << "]" << std::endl;
            //global_min = in_min;
            //global_max = in_max;
        }
    }

    uint64_t range = cv::saturate_cast<uint64_t>(global_max - global_min);
    if (range == 0) range = 1;

    constexpr int      LQ = 16;
    constexpr uint64_t OUT_MAX = 255;
    constexpr uint64_t ROUND = 1 << (LQ - 1); // 四舍五入偏移
    uint64_t scale_fixed = ((OUT_MAX << (LQ + 1)) / range + 1) >> 1;

    // Step3: 逐像素映射
    cv::Mat  dst(CLAHE.size(), CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < h; ++i)
    {
        const auto* psrc = CLAHE.ptr<uint16_t>(i);
        auto* pdst = dst.ptr<uint8_t>(i);
        for (int j = 0; j < w; ++j)
        {
            uint64_t val = cv::saturate_cast<uint64_t>(psrc[j]);
            uint64_t diff = val - cv::saturate_cast<uint64_t>(global_min);
            uint64_t res = (diff * scale_fixed + ROUND) >> LQ;
            pdst[j] = cv::saturate_cast<uint8_t>(res);
        }
    }

    output.assign(dst);
    return 0;
}

int clahe_mapping_with_percentile(cv::InputArray input, cv::OutputArray output, double clipLimit = 2.0, cv::Size tileSize = { 8, 8 })
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    cv::Mat img8bit;
    percentile_mapping(img14bit, img8bit, 0.25, 99.75);

    cv::Mat dst;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileSize);
    clahe->apply(img8bit, dst);

    output.assign(dst);
    return 0;
}

int gamma_transform_16UC1(cv::InputArray input, cv::OutputArray output, double gamma = 0.5)
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    cv::Mat normalized;
    //img14bit.convertTo(normalized, CV_32F, 1.0 / 65535.0);
    cv::normalize(img14bit, normalized, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

    cv::Mat corrected;
    cv::pow(normalized, gamma, corrected);

    cv::Mat dst;
    corrected.convertTo(dst, CV_16U, 65535.0);

    output.assign(dst);
    return 0;
}

// 根据局部图像的梯度计算权重图，梯度越大权重越高，使用Sigmoid函数映射为0-1之间的权重值，并进行高斯平滑
int computeWeightByGradient(cv::InputArray localImg, cv::OutputArray weight_map, double k = 1.0)
{
    cv::Mat local = localImg.getMat().clone();
    CV_CheckTypeEQ(local.type(), CV_8UC1, "");

    // Step 1: 计算梯度幅值
    cv::Mat gradX, gradY;
    cv::Sobel(local, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(local, gradY, CV_32F, 0, 1, 3);

    cv::Mat gradMag;
    cv::magnitude(gradX, gradY, gradMag);

    // Step 2: 归一化到 [0, 1]
    cv::Mat gradMagNorm;
    cv::normalize(gradMag, gradMagNorm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

    // Step 3: Sigmoid 映射为权重
    // weight = 1 / (1 + exp(-k * (x - 0.5)))
    cv::Mat exponent = -k * (gradMagNorm - 0.5);
    cv::exp(exponent, exponent);
    cv::Mat weightMap = 1.0 / (1.0 + exponent);

    // Step 4: 高斯平滑
    cv::GaussianBlur(weightMap, weightMap, cv::Size(5, 5), 1);

    weight_map.assign(weightMap);
    return 0;
}


int global_local_adaptive_fusion(cv::InputArray input, cv::OutputArray output)
{
    cv::Mat src = input.getMat().clone();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    cv::Mat img_global;
    cv::Mat img_local;

    percentile_mapping(src, img_global, 0.25, 99.75);
    clahe_mapping_with_percentile(src, img_local, 3.0, cv::Size(8, 8));

    cv::Mat weightMap;
    computeWeightByGradient(img_local, weightMap, 10.0);

    cv::Mat globalFloat, localFloat;
    img_global.convertTo(globalFloat, CV_32F);
    img_local.convertTo(localFloat, CV_32F);

    cv::Mat fusedFloat = globalFloat.mul(1.0 - weightMap) + localFloat.mul(weightMap);

    cv::Mat dst;
    fusedFloat.convertTo(dst, CV_8U);

    output.assign(dst);
    return 0;
}

// 根据局部方差自适应调整细节增强增益，方差大时降低增益，方差小的平坦区域保持较高增益，防止过增强产生光晕
int DDE_adaptive_gain(cv::InputArray input, cv::OutputArray output, double baseGain, double sigma = 5.0)
{
    cv::Mat detailLayer = input.getMat();
    CV_CheckTypeEQ(detailLayer.type(), CV_32F, "");

    // 局部方差估计
    cv::Mat mu, mu2, variance;
    cv::GaussianBlur(detailLayer, mu, cv::Size(0, 0), sigma);
    cv::GaussianBlur(detailLayer.mul(detailLayer), mu2, cv::Size(0, 0), sigma);
    variance = mu2 - mu.mul(mu);
    cv::max(variance, 0.0f, variance);

    // 方差归一化到 [0, 1]
    cv::Mat varianceNorm;
    cv::normalize(variance, varianceNorm, 0.0, 1.0, cv::NORM_MINMAX);

    // gain = baseGain / (1 + k * variance)，方差大时自动降低增益
    double k = baseGain - 1.0;
    cv::Mat gainMap = baseGain / (1.0 + k * varianceNorm);

    output.assign(gainMap);
    return 0;
}


int dde_enhance(cv::InputArray input, cv::OutputArray output)
{
    struct DDEConfig
    {
        // 基础层分离（双边滤波）
        int    d = 3;               // 邻域直径（像素，奇数）
        double sigmaColor = 0.2;    // 值域 sigma
        double sigmaSpace = 3.0;    // 空间域 sigma

        // 细节增强
        double detailGain = 3.0;    // 细节增益系数
        double detailClip = 0.2;    // 细节层截断，防止过增强（归一化）

        // 基础层压缩
        double baseGamma = 0.5;    // gamma < 1 压缩亮区，提升暗区

        // 输出
        double lowPct = 0.25;
        double highPct = 99.75;
    };

    // 针对强热源场景：降低细节增益，加强基础层压缩
    DDEConfig cfg;
    //cfg.detailGain = 2.0;   // 热源周围细节不过增强
    cfg.baseGamma = 1.0;   // 更强的动态范围压缩
    //cfg.sigmaColor = 0.05;  // 更严格的边缘保护
    //cfg.sigmaColor = 146.1;
    //cfg.sigmaSpace = 15.0;
    //cfg.detailClip = 0.15;

    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    // Step 1:
    cv::Mat img_norm;
    double minVal, maxVal;

    if (1)
    {
        //cfg.sigmaSpace = cv::saturate_cast<double>(cfg.d) / 3.0;
        cv::normalize(src, img_norm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
        cv::minMaxLoc(img_norm, &minVal, &maxVal);
    }
    else
    {
        cfg.sigmaSpace = cv::saturate_cast<double>(cfg.d) / 3.0;
        src.convertTo(img_norm, CV_32F);
        cv::minMaxLoc(img_norm, &minVal, &maxVal);
        //auto dynamicRange = maxVal - minVal;
        double dynamicRange = 1461;
        cfg.sigmaColor = dynamicRange * cfg.sigmaColor;
        cfg.detailClip = dynamicRange * cfg.detailClip;
    }

    std::cout << "Min: " << minVal << ", Max: " << maxVal << std::endl;

    // Step 2: 双边滤波分离基础层（低频光照）和细节层（高频细节）
    //         sigmaColor 在归一化域下有意义，无需随深度调整
    cv::Mat baseLayer;
    cv::bilateralFilter(img_norm, baseLayer,
        cfg.d,
        cfg.sigmaColor,
        cfg.sigmaSpace);

    cv::Mat detailLayer = img_norm - baseLayer;

    cv::Mat detail_observe;
    cv::normalize(detailLayer, detail_observe, 0, 255, cv::NORM_MINMAX, CV_8U);
    //imwrite_mdy_private(detail_observe, "DDE_Detail");

    // Step 3: 基础层 Gamma 压缩，抑制大范围光照不均
    cv::Mat baseCompressed;
    cv::pow(baseLayer, cfg.baseGamma, baseCompressed);

    // Step 4: 自适应细节增强
    cv::Mat gainMap;
    DDE_adaptive_gain(detailLayer, gainMap, cfg.detailGain);
    cv::Mat detailEnhanced = gainMap.mul(detailLayer);

    // 细节层截断，防止过增强产生光晕
    cv::Mat detailClipped;

    cv::min(detailEnhanced, cfg.detailClip, detailClipped);
    cv::max(detailClipped, -cfg.detailClip, detailClipped);


    // Step 5: 重建
    cv::Mat reconstructed = baseCompressed + detailClipped;

    // Step 6: 百分位截断 + 线性拉伸到 [0, 255]
    cv::Mat dst;
    cv::normalize(reconstructed, dst, 0, 255, cv::NORM_MINMAX, CV_8U);
    //percentile_mapping(dst, dst, cfg.lowPct, cfg.highPct);

    output.assign(dst);

    return 0;
}


void order(ushort* nums, int len)
{
    ushort temp = 0;
    for (int i = 0; i < len - 1; i++) {
        for (int j = 0; j < len - 1 - i; j++) {
            if (nums[j] > nums[j + 1]) {
                temp = nums[j];
                nums[j] = nums[j + 1];
                nums[j + 1] = temp;
            }
        }
    }
}

ushort find_med_3(ushort x1, ushort x2, ushort x3) {
    ushort max = 0;
    ushort min = 0;
    ushort med = 0;

    if (x1 >= x2 && x2 >= x3) {
        max = x1;
        min = x3;
        med = x2;
    }
    else if (x3 >= x2 && x2 >= x1) {
        max = x3;
        min = x1;
        med = x2;
    }

    if (x3 >= x1 && x1 >= x2) {
        max = x3;
        min = x2;
        med = x1;
    }
    else if (x2 >= x1 && x1 >= x3) {
        max = x2;
        min = x3;
        med = x1;
    }

    if (x1 >= x3 && x3 >= x2) {
        max = x1;
        min = x2;
        med = x3;
    }
    else if (x2 >= x3 && x3 >= x1) {
        max = x2;
        min = x1;
        med = x3;
    }

    return med;
}

void ours_dpc(cv::Mat img, bool dpc_en, cv::Mat img_dpc, double dp_slope, double dp_thresh, int diff_th, int DEAD[][2], int dead_count) {

    // 获取图像尺寸
    int width = img.cols;
    int height = img.rows;

    // 创建校正后的图像
    cv::Mat img_correct = cv::Mat::zeros(height, width, CV_16UC1);
    int padding_size1 = 1;
    int padding_size2 = 2;
    // 图像边缘扩展
    int extended_height = height + 2 * padding_size1;
    int extended_width = width + 2 * padding_size2;
    cv::Mat imgn = cv::Mat::zeros(extended_height, extended_width, CV_16UC1);
    cv::copyMakeBorder(img, imgn, padding_size1, padding_size1, padding_size2, padding_size2, cv::BORDER_REFLECT_101);

    if (dpc_en) {
        for (int i = padding_size1; i < height + padding_size1; i++) {
            for (int j = padding_size2; j < width + padding_size2; j++) {

                std::vector<ushort> img_r_values;
                for (int m = -padding_size1; m <= padding_size1; m++) {
                    for (int n = -padding_size2; n <= padding_size2; n++) {
                        img_r_values.push_back(imgn.at<ushort>(i + m, j + n));

                    }
                }

                // 中心像素值
                ushort data_r_center = imgn.at<ushort>(i, j);

                // 计算与中心像素的差值绝对值
                std::vector<double> data_r_diff(img_r_values.size());
                for (size_t k = 0; k < img_r_values.size(); k++) {
                    data_r_diff[k] = abs(static_cast<double>(img_r_values[k]) - data_r_center);
                }
                // 判断所有 signed differences 是否同号
                bool all_same_sign = true;
                if (!data_r_diff.empty()) {
                    bool all_positive = true;
                    bool all_negative = true;
                    for (size_t k = 0; k < img_r_values.size(); k++) {
                        if (data_r_diff[k] < 0) all_positive = false;
                        if (data_r_diff[k] > 0) all_negative = false;
                    }
                    all_same_sign = all_positive || all_negative;
                }
                ushort ori1[5];
                ori1[0] = img_r_values[0];
                ori1[1] = img_r_values[1];
                ori1[2] = img_r_values[2];
                ori1[3] = img_r_values[3];
                ori1[4] = img_r_values[4];
                ushort ori2[5];
                ori2[0] = img_r_values[5];
                ori2[1] = img_r_values[6];
                ori2[2] = img_r_values[7];
                ori2[3] = img_r_values[8];
                ori2[4] = img_r_values[9];
                ushort ori3[5];
                ori3[0] = img_r_values[10];
                ori3[1] = img_r_values[11];
                ori3[2] = img_r_values[12];
                ori3[3] = img_r_values[13];
                ori3[4] = img_r_values[14];
                order(ori1, 5);
                ushort median1 = ori1[2];

                order(ori2, 5);
                ushort median2 = ori2[2];

                order(ori3, 5);
                ushort median3 = ori3[2];


                ushort data_r_median = find_med_3(median1, median2, median3);/*avg_on_min_gradient_3x3(img_r_values[1], img_r_values[2], img_r_values[3], img_r_values[6], img_r_values[8], img_r_values[11], img_r_values[12], img_r_values[13]);*/

                // 计算检测阈值
                std::vector<double> data_r_detect(img_r_values.size());
                int data_r_judge = 0;
                double detect_sum = 0.0;

                for (size_t k = 0; k < img_r_values.size(); k++) {
                    data_r_detect[k] = data_r_diff[k] * dp_slope - dp_thresh;
                    if (data_r_detect[k] < 0) data_r_detect[k] = 0;
                    if (data_r_detect[k] > 1024) data_r_detect[k] = 1024;
                    if (data_r_detect[k] > 0) {
                        data_r_judge++;
                        detect_sum += data_r_detect[k];
                    }
                }

                int data_r_weight;
                if (data_r_judge > 0) {
                    int ratio = detect_sum / data_r_judge;
                    data_r_weight = ratio;
                }
                else {
                    data_r_weight = 0;
                    /*for (size_t k = 0; k < img_r_values.size(); k++) {
                        cout << data_r_diff[k]<<" ";
                    }
                    cout << "\n";
                    for (size_t k = 0; k < img_r_values.size(); k++) {
                        cout << data_r_detect[k] << " ";
                    }*/
                }

                int data_r_correct;

                // 判断当前像素是否在 DEAD 列表中（如果 dead_count <=0 或 DEAD==nullptr 则认为没有坏点）
                bool is_dead_pixel = false;
                if (DEAD != nullptr && dead_count > 0) {
                    for (int k = 0; k < dead_count; ++k) {
                        if ((i - padding_size1) == DEAD[k][0] && (j - padding_size2) == DEAD[k][1]) {
                            is_dead_pixel = true;
                            std::cout << DEAD[k][0] << " " << DEAD[k][1] << std::endl;
                            break;
                        }
                    }
                }

                // 根据判断条件进行校正
                // 新增：如果邻域所有 signed differences 同号（全部正或全部负），也视为坏点（用中值修正）
                if (data_r_judge >= 7 && abs(data_r_center - data_r_median) > diff_th/* && all_same_sign*/) {
                    data_r_correct = data_r_median * data_r_weight + (1024 - data_r_weight) * data_r_center;
                    data_r_correct /= 1024;
                }
                else if (is_dead_pixel) {
                    data_r_correct = data_r_median;
                }
                else {
                    data_r_correct = data_r_center;
                }

                img_dpc.at<ushort>(i - padding_size1, j - padding_size2) = static_cast<ushort>(data_r_correct);
            }
        }
    }
    else {
        img.copyTo(img_dpc);
    }
}

int ir_pipeline_cv(cv::InputArray input,
    cv::OutputArray output,
    int   sigma_space = 10,
    int   sigma_color = 30,
    float low_percent = 0.5f,
    float high_percent = 2.0f,
    float sharpen_strength = 0.5f)
{
    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    //ours_dpc(src, 1, src);

    // ── 双边滤波 ─────────────────────────────
    cv::Mat src_32f;
    src.convertTo(src_32f, CV_32F);

    //估计噪声标准差
    //auto noise_SD = estimateNoiseWaveletMAD(src_32f);
	//std::cout << "Estimated noise SD: " << noise_SD << std::endl;

    cv::Mat dn_32f;
    cv::bilateralFilter(src_32f, dn_32f, 7, sigma_color, sigma_space);
    //dn_32f = src_32f.clone();

	double minVal, maxVal;
	cv::minMaxLoc(src_32f, &minVal, &maxVal);
	std::cout << "src_MinMax: " << minVal << "-" << maxVal << std::endl;

	imwrite_mdy_private_normalization_8u(src, "IR_src");
	imwrite_mdy_private_normalization_8u(dn_32f, "IR_base");
	cv::Mat detail = src_32f - dn_32f;
	imwrite_mdy_private_normalization_8u(detail, "IR_detail");

    // ── 百分位截断 ───────────────────────────
    cv::Mat dn_16u;
    dn_32f.convertTo(dn_16u, CV_16U);

    const int H = dn_16u.rows, W = dn_16u.cols;
    const int total = H * W;

    std::vector<int> hist(16384, 0);
    for (int i = 0; i < H; ++i)
    {
        const uint16_t* row = dn_16u.ptr<uint16_t>(i);
        for (int j = 0; j < W; ++j)
            hist[row[j] & 0x3FFF]++;
    }

    std::vector<int> cdf(16384, 0);
    cdf[0] = hist[0];
    for (int i = 1; i < 16384; ++i)
        cdf[i] = cdf[i - 1] + hist[i];

    const int low_num = (int)(total * low_percent / 100.f);
    const int high_num = (int)(total * (100.f - high_percent) / 100.f);

    int thr_low = 0, thr_high = 16383;
    for (int i = 0; i < 16383; ++i)
    {
        if (cdf[i] <= low_num && cdf[i + 1] > low_num)  thr_low = i + 1;
        if (cdf[i] < high_num && cdf[i + 1] >= high_num) thr_high = i;
    }

    cv::Mat clip_32f;
    dn_32f.copyTo(clip_32f);
    clip_32f = cv::max(clip_32f, cv::saturate_cast<float>(thr_low));
    clip_32f = cv::min(clip_32f, cv::saturate_cast<float>(thr_high));

    imwrite_mdy_private_normalization_8u(clip_32f, "IR_clip");

    // ── laplacian锐化 ───────────────
    cv::Mat laplacian;
    cv::Laplacian(clip_32f, laplacian, CV_32F);

	laplacian *= sharpen_strength;
    cv::Mat sharpen_clip = clip_32f * 0.25;

	cv::min(laplacian, sharpen_clip, laplacian);
    cv::max(laplacian, -sharpen_clip, laplacian);
    cv::Mat sharpen = clip_32f - laplacian;

    // ── 线性归一化到 [0, 255] ───────────────
    cv::Mat norm_8u;
    cv::normalize(sharpen, norm_8u, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat dst = norm_8u.clone();
    output.assign(dst);
    return 0;
}


int Test_single_method()
{
    //std::string self_png = "C:/Users/Belfast/Desktop/before_mapping/self_png";
    //std::string self_png = "C:/Users/13012/Desktop/640";
    //std::string self_png = "C:/Users/Belfast/Desktop/640";
    std::string self_png = "C:/Users/Belfast/Desktop/before_mapping/60度黑体";

    fs::path inputDir(self_png);

    //fs::path inputDir(IMAGE_DIR);
    //fs::path inputDir(NOISE_IMAGE_DIR);
    
    if (!fs::exists(inputDir))
    {
        std::cerr << "Input directory not found: " << NOISE_IMAGE_DIR << std::endl;
        return -1;
    }

    double entropy = 0.0;
    double ag = 0.0;
    double ssim = 0.0;
    int imageCount = 0;
    double range_sum = 0.0;

    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        // 过滤 png 文件
        if (!entry.is_regular_file() || entry.path().extension() != ".png")
            continue;

        cv::Mat src = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
        if (src.empty())
        {
            std::cerr << "Failed to read: " << entry.path() << std::endl;
            continue;
        }

        // 处理图像
        CV_CheckTypeEQ(src.type(), CV_16UC1, "");
        cv::subtract(16383, src, src); 

        /*cv::Mat dst_linear;
        linear_mapping(src, dst_linear);
        imwrite_mdy_private(dst_linear, "Linear");*/


        /*cv::Mat dst_CLAHE;
        clahe_mapping(src, dst_CLAHE, 3.0, cv::Size(8, 8));
        imwrite_mdy_private(dst_CLAHE, "CLAHE");*/


        /*cv::Mat dst_CLAHE_Fixed;
        clahe_fixed_mapping(src, dst_CLAHE_Fixed, 3, cv::Size(8, 8));
        imwrite_mdy_private(dst_CLAHE_Fixed, "CLAHE_Fixed");*/


        /*cv::Mat dst_GLAF;
        global_local_adaptive_fusion(src, dst_GLAF);
        imwrite_mdy_private(dst_GLAF, "GLAF");*/


        /*cv::Mat dst_MSR;
        multi_scale_retinex(src, dst_MSR, { 15.0, 80.0, 250.0 });
        imwrite_mdy_private(dst_MSR, "MSR");*/


        /*cv::Mat dst_DDE;
        dde_enhance(src, dst_DDE);
        imwrite_mdy_private(dst_DDE, "DDE");*/


       /* cv::Mat dst_SSR;
        single_scale_retinex(src, dst_SSR, 50);
        imwrite_mdy_private(dst_SSR, "SSR");*/

		cv::Mat dst_ir_pipeline;
		ir_pipeline_cv(src, dst_ir_pipeline, 10, 30, 0.5f, 2.0f, 0.5f);
        //ir_pipeline_cv(src, dst_ir_pipeline, 10, 10, 2.0f, 1.0f, 0.5f);
		imwrite_mdy_private(dst_ir_pipeline, "IR_Pipeline_result");


        entropy += calcEntropy(dst_ir_pipeline);
        ag += calcAverageGradient(dst_ir_pipeline);
        /*ssim += calcSSIM(dst_DDE, dst_linear);*/
        double minVal, maxVal;
        cv::minMaxLoc(src, &minVal, &maxVal);
        auto range = maxVal - minVal;
        range_sum += range;
        imageCount++;

        std::cout << "Range: " << range << std::endl;

        cv::waitKey(1);
    }

    if (imageCount > 0)
    {
        std::cout << "Average Entropy: " << entropy / imageCount << std::endl;
        std::cout << "Average Gradient: " << ag / imageCount << std::endl;
        std::cout << "Average SSIM: " << ssim / imageCount << std::endl;
        std::cout << "Average Range: " << range_sum / imageCount << std::endl;
    }

    return 0;
}



int Test_all_methods()
{
    //std::string image_path = std::string(IMAGE_DIR) + "/19700101_000143_533.png";

    fs::path inputDir(IMAGE_DIR);

    if (!fs::exists(inputDir))
    {
        std::cerr << "Input directory not found: " << IMAGE_DIR << std::endl;
        return -1;
    }

    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        // 过滤 png 文件
        if (!entry.is_regular_file() || entry.path().extension() != ".png")
            continue;

        cv::Mat src = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
        if (src.empty())
        {
            std::cerr << "Failed to read: " << entry.path() << std::endl;
            continue;
        }

        // 处理图像
        CV_CheckTypeEQ(src.type(), CV_16UC1, "");

        //cv::imshow("原始图像", src);

        HistDisplayConfig histcfg_src;
        histcfg_src.winName = "原始图像直方图";
        histcfg_src.logScale = true;
        histcfg_src.binCount = 512;
        showHistogram(src, histcfg_src);


        cv::Mat dst_linear;
        linear_mapping(src, dst_linear);

        cv::imshow("线性映射图像", dst_linear);

        HistDisplayConfig histcfg_linear;
        histcfg_linear.winName = "线性映射图像直方图";
        histcfg_linear.logScale = true;
        showHistogram(dst_linear, histcfg_linear);

        imwrite_mdy_private(dst_linear, "Linear");


        cv::Mat dst_CLAHE;
        clahe_mapping(src, dst_CLAHE, 2.0, cv::Size(8, 8));

        cv::imshow("CLAHE映射图像", dst_CLAHE);

        HistDisplayConfig histcfg_clahe;
        histcfg_clahe.winName = "CLAHE图像直方图";
        histcfg_clahe.logScale = true;
        showHistogram(dst_CLAHE, histcfg_clahe);

        //cv::Mat enhanced = retinex_enhance(src, 15.0);
        imwrite_mdy_private(dst_CLAHE, "CLAHE");


        cv::Mat dst_GLAF;
        global_local_adaptive_fusion(src, dst_GLAF);

        cv::imshow("全局局部自适应融合图像", dst_GLAF);
        imwrite_mdy_private(dst_GLAF, "GLAF");


        std::cout << "Processed: " << entry.path().filename() << std::endl;
        cv::waitKey(0);
    }

    return 0;
}

int mean_2sigma_from_folder()
{
    struct SigmaReport
    {
        // 时域统计（逐像素，反映噪声）
        double temporal_sigma_mean; // 所有像素时域 σ 的均值
        double temporal_sigma_max;  // 最大时域 σ（最不稳定的像素）
        double temporal_sigma_min;  // 最小时域 σ

        // 空域统计（逐帧，反映对比度）
        double spatial_sigma_mean;  // 所有帧空域 σ 的均值
        double spatial_sigma_max;   // 最大空域 σ（对比度最高的帧）
        double spatial_sigma_min;   // 最小空域 σ

        int    frame_count;
    };

    fs::path inputDir("C:/Users/13012/Desktop/result");

    if (!fs::exists(inputDir))
    {
        std::cerr << "Input directory not found: " << "C:/Users/13012/Desktop/result" << std::endl;
        return -1;
    }

    std::vector<cv::Mat> images;
    for (const auto& entry : fs::directory_iterator(inputDir))
    {
        // 过滤 png 文件
        if (!entry.is_regular_file() || entry.path().extension() != ".png")
            continue;

        cv::Mat src = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
        if (src.empty())
        {
            std::cerr << "Failed to read: " << entry.path() << std::endl;
            continue;
        }

        // 处理图像
        CV_CheckTypeEQ(src.type(), CV_16UC1, "");

        images.push_back(src);
    }

    // 读取第一帧获取尺寸
    cv::Mat first = images[0];
    CV_Assert(!first.empty() && first.type() == CV_16UC1);
    const int rows = first.rows;
    const int cols = first.cols;
    const int N = static_cast<int>(images.size());

    // 时域统计用的累加器
    cv::Mat sum(rows, cols, CV_64FC1, cv::Scalar(0));
    cv::Mat sum2(rows, cols, CV_64FC1, cv::Scalar(0));

    // 空域统计
    double spatial_sum = 0;
    double spatial_max = 0;
    double spatial_min = std::numeric_limits<double>::max();
    int    count = 0;

    for (const auto& frame : images)
    {
        // 空域统计：整帧 mean/stddev
        cv::Scalar mean, stddev;
        cv::meanStdDev(frame, mean, stddev);
        spatial_sum += stddev[0];
        spatial_max = std::max(spatial_max, stddev[0]);
        spatial_min = std::min(spatial_min, stddev[0]);

        // 时域统计：逐像素累加
        for (int r = 0; r < rows; ++r)
        {
            const uint16_t* pf = frame.ptr<uint16_t>(r);
            double* ps = sum.ptr<double>(r);
            double* ps2 = sum2.ptr<double>(r);
            for (int c = 0; c < cols; ++c)
            {
                double v = pf[c];
                ps[c] += v;
                ps2[c] += v * v;
            }
        }

        count++;
    }

    CV_Assert(count > 0);

    // 计算时域 σ
    double temporal_sum = 0;
    double temporal_max = 0;
    double temporal_min = std::numeric_limits<double>::max();

    for (int r = 0; r < rows; ++r)
    {
        const double* ps = sum.ptr<double>(r);
        const double* ps2 = sum2.ptr<double>(r);
        for (int c = 0; c < cols; ++c)
        {
            double mean_v = ps[c] / count;
            double var = ps2[c] / count - mean_v * mean_v;
            double sigma = std::sqrt(std::max(var, 0.0));
            temporal_sum += sigma;
            temporal_max = std::max(temporal_max, sigma);
            temporal_min = std::min(temporal_min, sigma);
        }
    }

    SigmaReport rpt{};
    rpt.frame_count = count;
    rpt.temporal_sigma_mean = temporal_sum / (rows * cols);
    rpt.temporal_sigma_max = temporal_max;
    rpt.temporal_sigma_min = temporal_min;
    rpt.spatial_sigma_mean = spatial_sum / count;
    rpt.spatial_sigma_max = spatial_max;
    rpt.spatial_sigma_min = spatial_min;

    printf("=== Sigma 分析报告 ===\n");
    printf("有效帧数              : %d\n", rpt.frame_count);
    printf("--- 时域（噪声）---\n");
    printf("  均值 σ              : %.4f DN\n", rpt.temporal_sigma_mean);
    printf("  最大 σ              : %.4f DN\n", rpt.temporal_sigma_max);
    printf("  最小 σ              : %.4f DN\n", rpt.temporal_sigma_min);
    printf("--- 空域（对比度）---\n");
    printf("  均值 σ              : %.4f DN\n", rpt.spatial_sigma_mean);
    printf("  最大 σ              : %.4f DN\n", rpt.spatial_sigma_max);
    printf("  最小 σ              : %.4f DN\n", rpt.spatial_sigma_min);

    return 0;
}

int main()
{
    //Test_all_methods();
    Test_single_method();

    //benchmark_main();

    //test_precision_batch_14to8();
    //rgb2png();
	//yuv2png();
    //mean_2sigma_from_folder();

    //DDE_improvement();

    return 0;
}



