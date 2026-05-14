#include "DDE_improvement.h"


namespace fs = std::filesystem;

int DDE_origin_adaptive_gain(cv::InputArray input, cv::OutputArray output, double baseGain, double sigma = 5.0)
{
    cv::Mat detailLayer = input.getMat();
    CV_CheckTypeEQ(detailLayer.type(), CV_32F, "");

    // 局部方差估计
    cv::Mat mu, mu2, variance;
	cv::Mat detailLayerSquared = detailLayer.mul(detailLayer);
    cv::GaussianBlur(detailLayer, mu, cv::Size(0, 0), sigma);
    cv::GaussianBlur(detailLayerSquared, mu2, cv::Size(0, 0), sigma);
	cv::Mat muSquared = mu.mul(mu);
	variance = mu2 - muSquared;
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


int dde_origin(cv::InputArray input, cv::OutputArray output)
{
    struct DDEConfig
    {
        // 基础层分离（双边滤波）
        int    d = 9;               // 邻域直径（像素，奇数）
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

    DDEConfig cfg;
    cfg.baseGamma = 1.0;   // 更强的动态范围压缩

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
        auto dynamicRange = maxVal - minVal;
        //double dynamicRange = 1461;
        cfg.sigmaColor = dynamicRange * cfg.sigmaColor;
        cfg.detailClip = dynamicRange * cfg.detailClip;
    }

    std::cout << "Min: " << minVal << ", Max: " << maxVal << std::endl;

    cv::Mat baseLayer;
    cv::bilateralFilter(img_norm, baseLayer,
        cfg.d,
        cfg.sigmaColor,
        cfg.sigmaSpace);

    cv::Mat detailLayer = img_norm - baseLayer;

    cv::Mat detail_observe;
    cv::normalize(detailLayer, detail_observe, 0, 255, cv::NORM_MINMAX, CV_8U);
    //imwrite_mdy_private(detail_observe, "DDE_Detail");

    cv::Mat baseCompressed;
    cv::pow(baseLayer, cfg.baseGamma, baseCompressed);

    cv::Mat gainMap;
    DDE_origin_adaptive_gain(detailLayer, gainMap, cfg.detailGain);
    cv::Mat detailEnhanced = gainMap.mul(detailLayer);

    cv::Mat detailClipped;

    cv::min(detailEnhanced, cfg.detailClip, detailClipped);
    cv::max(detailClipped, -cfg.detailClip, detailClipped);

    cv::Mat reconstructed = baseCompressed + detailClipped;

    cv::Mat dst;
    cv::normalize(reconstructed, dst, 0, 255, cv::NORM_MINMAX, CV_8U);
    //cv::normalize(reconstructed, dst, 0, 65535, cv::NORM_MINMAX, CV_16U);
    //percentile_mapping(dst, dst, cfg.lowPct, cfg.highPct);

    output.assign(dst);

    return 0;
}



int test_DDE_improve()
{
    fs::path inputDir(NOISE_IMAGE_DIR);
    //fs::path inputDir(IMAGE_DIR);

    if (!fs::exists(inputDir))
    {
        std::cerr << "Input directory not found: " << NOISE_IMAGE_DIR << std::endl;
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
        cv::subtract(16383, src, src);

        images.push_back(src);
    }

    for (int i = 0; i < images.size(); ++i)
    {
        cv::Mat dst_DDE;
        dde_origin(images[i], dst_DDE);
        imwrite_mdy_private(dst_DDE, "DDE_origin");
    }

    return 0;
}

int DDE_improvement()
{
    test_DDE_improve();

    return 0;
}

