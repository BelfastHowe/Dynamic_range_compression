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

    cv::Mat canny;

    cv::Mat detailLayer_8U;
	cv::normalize(detailLayer, detailLayer_8U, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::Canny(detailLayer_8U, detailLayer_8U, 100, 200);
	detailLayer_8U.convertTo(canny, CV_32F, 1.0 / 2550.0);


    cv::Mat detailEnhanced = gainMap.mul(canny);

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


int correctFPN_ColMedian(cv::InputArray input, cv::OutputArray output, int smoothWin = 21)
{
	auto src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_32F, "");

    int rows = src.rows;
    int cols = src.cols;

    // 1. 计算每一列的特征值（使用中值比均值更鲁棒，能避开孤立热源的影响）
    cv::Mat colFeatures = cv::Mat::zeros(1, cols, CV_32F);

    for (int j = 0; j < cols; ++j)
    {
        std::vector<float> colPixels;
        colPixels.reserve(rows);
        for (int i = 0; i < rows; ++i)
        {
            colPixels.push_back(src.at<float>(i, j));
        }

        std::nth_element(colPixels.begin(), colPixels.begin() + rows / 2, colPixels.end());
        colFeatures.at<float>(0, j) = colPixels[rows / 2];
    }

    // 2. 对列特征序列进行横向高斯模糊，提取“低频趋势”
    // 这个趋势代表了场景真实的横向亮度变化，而偏离这个趋势的就是条纹
    cv::Mat colTrend;
    cv::GaussianBlur(colFeatures, colTrend, cv::Size(smoothWin, 1), 0);

    // 3. 计算修正增益 (Correction = 趋势 - 实际)
    cv::Mat correction = colTrend - colFeatures;

    // 4. 将修正应用到每一行
    auto dst = src.clone();
    for (int i = 0; i < rows; ++i)
    {
        float* rowPtr = dst.ptr<float>(i);
        float* corrPtr = correction.ptr<float>(0);
        for (int j = 0; j < cols; ++j)
        {
            rowPtr[j] += corrPtr[j]; // 补偿每一列的偏置
        }
    }

    output.assign(dst);
	return 0;
}

int estimate_NUC_Temporal(const std::vector<cv::Mat>& frames, cv::OutputArray output)
{
    CV_Assert(!frames.empty());

    // 统一转为 CV_32F
    auto toFloat = [](const cv::Mat& m) -> cv::Mat
        {
            cv::Mat f;
            if (m.type() == CV_32F) f = m.clone();
            else m.convertTo(f, CV_32F);
            return f;
        };

    const cv::Size sz   = frames[0].size();
    const int      type = frames[0].type();
    for (const auto& f : frames)
        CV_Assert(f.size() == sz && (f.type() == type));

    cv::Mat mean = cv::Mat::zeros(sz, CV_32F);
    int     n = frames.size();

    for (const auto& raw : frames)
    {
        cv::Mat f = toFloat(raw);
        mean += f / static_cast<float>(n);
    }

    cv::Scalar globalMean = cv::mean(mean);
    cv::Mat fpn = mean - globalMean[0];

	output.assign(fpn);

    return 0;
}

double estimateNoiseWaveletMAD(cv::InputArray input)
{
    cv::Mat gray = input.getMat();
	CV_CheckTypeEQ(gray.type(), CV_32FC1, "Input must be a single-channel 32F image.");

    // 确保图像尺寸为偶数，方便 2x2 拆分
    int hh_rows = gray.rows / 2;
    int hh_cols = gray.cols / 2;

    std::vector<float> hh_coefficients;
    hh_coefficients.reserve(hh_rows * hh_cols);

    // 1. 手动计算 1 层 Haar 小波的 HH1 分量（对角线高频系数）
    // Haar 小波基的 HH 滤波器系数矩阵为: [ 1, -1; -1,  1 ] / 2
    for (int i = 0; i < hh_rows; ++i)
    {
        const float* row0 = gray.ptr<float>(2 * i);
        const float* row1 = gray.ptr<float>(2 * i + 1);

        for (int j = 0; j < hh_cols; ++j)
        {
            float p00 = row0[2 * j];
            float p01 = row0[2 * j + 1];
            float p10 = row1[2 * j];
            float p11 = row1[2 * j + 1];

            // 计算 HH1 系数: (p00 - p01 - p10 + p11) * 0.5
            float hh = (p00 - p01 - p10 + p11) * 0.5f;

            // 取绝对值存入数组
            hh_coefficients.push_back(std::abs(hh));
        }
    }

    if (hh_coefficients.empty()) return 0.0;

    // 2. 寻找绝对值的中位数 (Median)
    size_t mid_index = hh_coefficients.size() / 2;
    std::nth_element(hh_coefficients.begin(), hh_coefficients.begin() + mid_index, hh_coefficients.end());
    float median_abs = hh_coefficients[mid_index];

    // 3. 根据 MAD 公式计算噪声标准差 sigma
    double sigma = median_abs / 0.6745;

    return sigma * sigma;
}

int DDE_noise_adaptive_gain(cv::InputArray input, cv::OutputArray output, double baseGain, double noise_Variance, double sigma = 5.0)
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

    double edgeSaturationThresh = 50.0 * noise_Variance;
    if (edgeSaturationThresh < 1e-6) edgeSaturationThresh = 1e-6; // 防止除零

    cv::Mat varianceNorm = variance / edgeSaturationThresh;
    cv::threshold(varianceNorm, varianceNorm, 1.0, 1.0, cv::THRESH_TRUNC);

    double k = baseGain - 1.0;
    cv::Mat gainMap = baseGain / (1.0 + k * varianceNorm);

    double alpha = 2.0; // 噪声控制敏感系数，通常取 2.0 - 5.0
    cv::Mat noiseSuppression;
    cv::add(variance, cv::Scalar(alpha * noise_Variance + 1e-6), noiseSuppression);
    cv::divide(variance, noiseSuppression, noiseSuppression);

    cv::Mat finalGainMap = gainMap.mul(noiseSuppression);

    output.assign(finalGainMap);
    return 0;
}

int dde_denoise(cv::InputArray input, cv::OutputArray output, cv::InputArray nuc_in)
{
    struct DDEConfig
    {
        int    d = 9;               // 邻域直径（像素，奇数）
        double sigmaColor = 0.2;    // 值域 sigma
        double sigmaSpace = 3.0;    // 空间域 sigma

        double detailGain = 3.0;    // 细节增益系数
        double detailClip = 0.2;    // 细节层截断，防止过增强（归一化）

        double baseGamma = 0.5;    // gamma < 1 压缩亮区，提升暗区

        double lowPct = 0.25;
        double highPct = 99.75;
    };

    DDEConfig cfg;
    cfg.baseGamma = 1.0;

    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");
    cv::Mat nuc = nuc_in.getMat();
    CV_CheckTypeEQ(nuc.type(), CV_32FC1, "");

    /*cv::Mat src_FPN;
	src.convertTo(src_FPN, CV_32F);
	correctFPN_ColMedian(src_FPN, src_FPN, 21);*/

    cv::Mat img_norm;
    double minVal, maxVal;

    if (1)
    {
        //cfg.sigmaSpace = cv::saturate_cast<double>(cfg.d) / 3.0;
        cv::normalize(src, img_norm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

        /*src.convertTo(img_norm, CV_32F);
        cv::subtract(img_norm, nuc, img_norm);
        cv::normalize(img_norm, img_norm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);*/

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

	auto noise_Variance = estimateNoiseWaveletMAD(img_norm);

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
    DDE_noise_adaptive_gain(detailLayer, gainMap, cfg.detailGain, noise_Variance);
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

    cv::Mat nuc;
	estimate_NUC_Temporal(images, nuc);

    for (int i = 0; i < images.size(); ++i)
    {
        cv::Mat dst_DDE;
        //dde_denoise(images[i], dst_DDE, nuc);
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

