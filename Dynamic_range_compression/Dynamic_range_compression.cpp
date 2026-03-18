// Dynamic_range_compression.cpp: 定义应用程序的入口点。
//

#include "Dynamic_range_compression.h"
#include "quantitative_assessment.h"
#include "Quantization.h"

namespace fs = std::filesystem;
using Ms = std::chrono::duration<double, std::milli>;

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

// 单尺度Retinex算法实现
int single_scale_retinex(cv::InputArray input, cv::OutputArray output, double sigma)
{
    cv::Mat src = input.getMat().clone();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    // 归一化到0-1
    cv::Mat src_normal;
    cv::normalize(src, src_normal, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
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
        cv::GaussianBlur(src_normal, gauss, cv::Size(0, 0), sigma);

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

    cv::normalize(log_reflectance, result, 0, 65535, cv::NORM_MINMAX, CV_16U);
    percentile_mapping(result, result, 0.25, 99.75);

    output.assign(result);
    return 0;
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


int clahe_mapping(cv::InputArray input, cv::OutputArray output, double clipLimit = 2.0, cv::Size tileSize = { 8, 8 })
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    cv::Mat img8bit;
    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit, tileSize);
    clahe->apply(img14bit, img8bit);

    cv::Mat dst;
    linear_mapping(img8bit, dst);

    output.assign(dst);
    return 0;
}

int clahe_fixed_mapping(cv::InputArray input, cv::OutputArray output, double clipLimit = 2.0, cv::Size tileSize = { 8, 8 })
{
    cv::Mat img14bit = input.getMat().clone();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    cv::Mat img8bit;
    cv::Ptr<CLAHE_Fixed> clahe = createCLAHE_Fixed(clipLimit, tileSize);
    clahe->apply(img14bit, img8bit);

    cv::Mat dst;
    linear_mapping(img8bit, dst);

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
        int    spatialSigma = 9;      // 空间域 sigma（像素，奇数）
        double rangeSigma = 0.1;    // 值域 sigma（归一化后）

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
    //cfg.rangeSigma = 0.05;  // 更严格的边缘保护
    //cfg.detailClip = 0.15;

    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    // Step 1: 归一化到 [0, 1]
    cv::Mat img_norm;
    cv::normalize(src, img_norm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

    // Step 2: 双边滤波分离基础层（低频光照）和细节层（高频细节）
    //         rangeSigma 在归一化域下有意义，无需随深度调整
    cv::Mat baseLayer;
    cv::bilateralFilter(img_norm, baseLayer,
        cfg.spatialSigma,
        cfg.rangeSigma,
        cfg.rangeSigma);

    cv::Mat detailLayer = img_norm - baseLayer;

    // Step 3: 基础层 Gamma 压缩，抑制大范围光照不均
    cv::Mat baseCompressed;
    cv::pow(baseLayer, cfg.baseGamma, baseCompressed);

    // Step 4: 自适应细节增强
    cv::Mat gainMap;
    DDE_adaptive_gain(detailLayer, gainMap, cfg.detailGain);
    cv::Mat detailEnhanced = gainMap.mul(detailLayer);

    // 细节层截断，防止过增强产生光晕
    cv::Mat detailClipped;
    cv::threshold(detailEnhanced, detailClipped, cfg.detailClip, cfg.detailClip, cv::THRESH_TRUNC);
    cv::threshold(-detailClipped, detailClipped, cfg.detailClip, cfg.detailClip, cv::THRESH_TRUNC);
    detailClipped = -detailClipped;  // 还原符号

    // Step 5: 重建
    cv::Mat reconstructed = baseCompressed + detailClipped;

    // Step 6: 百分位截断 + 线性拉伸到 [0, 255]
    cv::Mat dst;
    cv::normalize(reconstructed, dst, 0, 65535, cv::NORM_MINMAX, CV_16U);
    percentile_mapping(dst, dst, cfg.lowPct, cfg.highPct);

    output.assign(dst);

    return 0;
}

int Test_single_method()
{
    fs::path inputDir(IMAGE_DIR);

    if (!fs::exists(inputDir))
    {
        std::cerr << "Input directory not found: " << IMAGE_DIR << std::endl;
        return -1;
    }

    double entropy = 0.0;
    double ag = 0.0;
    double ssim = 0.0;
    int imageCount = 0;

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


        cv::Mat dst_CLAHE_Fixed;
        clahe_fixed_mapping(src, dst_CLAHE_Fixed, 3.0, cv::Size(8, 8));
        imwrite_mdy_private(dst_CLAHE_Fixed, "CLAHE_Fixed");


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


        /*entropy += calcEntropy(dst_DDE);
        ag += calcAverageGradient(dst_DDE);
        ssim += calcSSIM(dst_DDE, dst_linear);*/
        imageCount++;

        cv::waitKey(1);
    }

    if (imageCount > 0)
    {
        std::cout << "Average Entropy: " << entropy / imageCount << std::endl;
        std::cout << "Average Gradient: " << ag / imageCount << std::endl;
        std::cout << "Average SSIM: " << ssim / imageCount << std::endl;
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

int main()
{
    //Test_all_methods();
    Test_single_method();

    //benchmark_main();

    return 0;
}



