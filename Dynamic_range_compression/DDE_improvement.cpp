#include "DDE_improvement.h"


namespace fs = std::filesystem;

int local_variance_32F(cv::InputArray input, cv::OutputArray output, double sigma = 5.0)
{
    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_32FC1, "");

    cv::Mat mu, mu2, variance;
    cv::GaussianBlur(src, mu, cv::Size(0, 0), sigma);
    cv::GaussianBlur(src.mul(src), mu2, cv::Size(0, 0), sigma);
    variance = mu2 - mu.mul(mu);
    cv::max(variance, 0.0f, variance);

    output.assign(variance);
    return 0;
}

int local_variance_32F_box(cv::InputArray input, cv::OutputArray output, int radius = 8)
{
    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_32FC1, "");

    cv::Size ksize(2 * radius + 1, 2 * radius + 1);

    cv::Mat mu, mu2, variance;
    cv::boxFilter(src, mu, CV_32F, ksize);
    cv::boxFilter(src.mul(src), mu2, CV_32F, ksize);
    variance = mu2 - mu.mul(mu);
    cv::max(variance, 0.0f, variance);

    output.assign(variance);
    return 0;
}

int calc_Gradient_32F(cv::InputArray src, cv::OutputArray output)
{
    cv::Mat img = src.getMat();
    CV_CheckTypeEQ(img.type(), CV_32FC1, "");

    cv::Mat gradX, gradY, gradMag;
    cv::Scharr(img, gradX, CV_32F, 1, 0);
    cv::Scharr(img, gradY, CV_32F, 0, 1);
    //cv::Sobel(img, gradX, CV_32F, 1, 0, 3);
    //cv::Sobel(img, gradY, CV_32F, 0, 1, 3);
    cv::magnitude(gradX, gradY, gradMag);

    output.assign(gradMag);
    return 0;
}

int residual_auto_canny(cv::InputArray input, cv::OutputArray output, double sigma = 2.0)
{
    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_32FC1, "");
    
    cv::Mat posPart, negPart;
    cv::max(src, 0.0f, posPart);
    cv::max(-src, 0.0f, negPart);

    double posminVal, posmaxVal, negminVal, negmaxVal;
    cv::minMaxLoc(posPart, &posminVal, &posmaxVal);
    cv::minMaxLoc(negPart, &negminVal, &negmaxVal);

    double scale;
    if (posmaxVal >= negmaxVal)
        scale = (posmaxVal > 1e-6) ? (255.0 / posmaxVal) : 1.0;
    else
        scale = (negmaxVal > 1e-6) ? (255.0 / negmaxVal) : 1.0;

    cv::Mat pos8U, neg8U;
    posPart.convertTo(pos8U, CV_8U, scale);
    negPart.convertTo(neg8U, CV_8U, scale);

    cv::Mat mean_benchmark;
    if (posmaxVal >= negmaxVal)
        mean_benchmark = pos8U.clone();
    else
        mean_benchmark = neg8U.clone();

    //auto mean = cv::mean(mean_benchmark)[0];
    int sum = 0;
    int n = 0;
    for (int i = 0; i < mean_benchmark.rows; i++)
    {
        const uchar* rowPtr = mean_benchmark.ptr<uchar>(i);
        for (int j = 0; j < mean_benchmark.cols; j++)
        {
            auto val = rowPtr[j];
            if (val == 0) continue;
            sum += val;
            n++;
        }
    }
    double mean = (n > 0) ? (static_cast<double>(sum) / n) : 0.0;

    int lower = static_cast<int>(std::max(0.0, 3.0 * mean));
    int upper = static_cast<int>(std::min(255.0, (1.0 + sigma) * 3.0 * mean));

    cv::Mat dst, posCanny, negCanny;
    cv::Canny(pos8U, posCanny, lower, upper);
    cv::Canny(neg8U, negCanny, lower, upper);
    cv::add(posCanny, negCanny, dst);

    output.assign(dst);
    return 0;
}

int auto_canny_32F(cv::InputArray input, cv::OutputArray output, double sigma = 0.33)
{
    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_32FC1, "");

    cv::Mat src_gauss;
    //cv::GaussianBlur(src, src_gauss, cv::Size(0, 0), 5);

    cv::Mat src_norm;
    cv::normalize(src, src_norm, 0.0, 255.0, cv::NORM_MINMAX, CV_8U);

    cv::Mat mask = (src_norm != 0);
    double mean = 0.0;
    auto nonZeroCount = cv::countNonZero(mask);
    if (nonZeroCount == 0)
    {
        mean = 0.0;
    }
    mean = cv::mean(src_norm)[0];

    cv::Mat flat = src_norm.reshape(1, 1);

    std::vector<uchar> vec;
    flat.copyTo(vec);

    std::nth_element(
        vec.begin(),
        vec.begin() + vec.size() / 2,
        vec.end());

    double med = vec[vec.size() / 2];
    std::cout << "auto_canny_16F Median: " << med << ", auto_canny_16F Mean: " << mean << std::endl;

    int lower = static_cast<int>(std::max(0.0, (1-sigma) * med));
    int upper = static_cast<int>(std::min(255.0, (1.0 + sigma) * med));
    cv::Mat dst;
    cv::Canny(src_norm, dst, lower, upper);

    output.assign(dst);
    return 0;
}

// 梯度低分位数估计噪声水平
static double estimateNoiseFromGradient(cv::InputArray input, double percentile = 0.50)
{
    cv::Mat gradMag = input.getMat();
    CV_CheckTypeEQ(gradMag.type(), CV_32FC1, "");

    std::vector<float> vals;
    gradMag.reshape(1, 1).copyTo(vals);

    size_t idx = static_cast<size_t>(percentile * vals.size());
    idx = std::min(idx, vals.size() - 1);
    std::nth_element(vals.begin(), vals.begin() + idx, vals.end());

    return static_cast<double>(vals[idx]);
}

// 双端抑制权重图
// lo_thresh: 噪声抑制下界（低于此全压制）
// hi_thresh: 边缘饱和上界（高于此全压制）
// steepness: sigmoid 斜率，越大过渡越硬
static int dualSidedSuppressionWeight(
    cv::InputArray input,
    cv::OutputArray output,
    double lo_thresh,
    double hi_thresh,
    double steepness = 6.0)
{
    cv::Mat gradMag = input.getMat();
    CV_CheckTypeEQ(gradMag.type(), CV_32FC1, "");
    output.create(gradMag.size(), CV_32FC1);
    cv::Mat weightMap = output.getMat();

    // 归一化到 [lo, hi] 区间，使 sigmoid 参数与绝对幅度解耦
    double inv_lo = 1.0 / (lo_thresh + 1e-9);
    double inv_hi = 1.0 / (hi_thresh + 1e-9);

    for (int i = 0; i < gradMag.rows; ++i)
    {
        const float* gp = gradMag.ptr<float>(i);
        float* wp = weightMap.ptr<float>(i);

        for (int j = 0; j < gradMag.cols; ++j)
        {
            double g = gp[j];

            // 上升沿：g > lo_thresh 时权重升起
            double rise = 1.0 / (1.0 + std::exp(-steepness * (g * inv_lo - 1.0)));

            // 下降沿：g < hi_thresh 时权重保持，超过后下降
            double fall = 1.0 / (1.0 + std::exp(steepness * (g * inv_hi - 1.0)));

            wp[j] = static_cast<float>(rise * fall);
        }
    }

    return 0;
}

static void compute_gradient_gainMap(
    cv::InputArray input,
    cv::OutputArray output,
    double         baseGain,
    double         lo_thresh,
    double         hi_thresh,
    double         steepness = 6.0)
{
    cv::Mat gradMag = input.getMat();
    CV_CheckTypeEQ(gradMag.type(), CV_32FC1, "");
    output.create(gradMag.size(), CV_32FC1);
    cv::Mat gainMap = output.getMat();

    double inv_lo = 1.0 / (lo_thresh + 1e-9);
    double inv_hi = 1.0 / (hi_thresh + 1e-9);
    double extra = baseGain - 1.0;  // 超出1的增益部分

    for (int i = 0; i < gradMag.rows; ++i)
    {
        const float* gp = gradMag.ptr<float>(i);
        float* wp = gainMap.ptr<float>(i);

        for (int j = 0; j < gradMag.cols; ++j)
        {
            double g = gp[j];

            // 左侧：0 → 1，过lo_thresh时上升
            double rise = 1.0 / (1.0 + std::exp(-steepness * (g * inv_lo - 1.0)));

            // 右侧：1 → 0，过hi_thresh时下降
            double fall = 1.0 / (1.0 + std::exp(steepness * (g * inv_hi - 1.0)));

            // gain: 0 → baseGain → 1
            wp[j] = static_cast<float>(rise * (1.0 + extra * fall));
        }
    }
}

int DDE_canny_adaptive_gain(cv::InputArray input, cv::OutputArray output, cv::InputArray in_canny, double baseGain, double sigma = 5.0)
{
    cv::Mat detailLayer = input.getMat();
    CV_CheckTypeEQ(detailLayer.type(), CV_32FC1, "");
    cv::Mat cannyInput = in_canny.getMat();
    CV_CheckTypeEQ(cannyInput.type(), CV_8UC1, "");

    cv::Mat detailLayer_residual_canny;
    residual_auto_canny(detailLayer, detailLayer_residual_canny);
    imwrite_mdy_private(detailLayer_residual_canny, "DDE_detailLayer_residual_canny");

    cv::Mat detailLayer_norm_canny;
    auto_canny_32F(detailLayer, detailLayer_norm_canny);
    imwrite_mdy_private(detailLayer_norm_canny, "DDE_detailLayer_norm_canny");

    cv::Mat detailLayer_laplacian;
    cv::Laplacian(detailLayer, detailLayer_laplacian, CV_32F, 3);
    imwrite_mdy_private_normalization_8u(detailLayer_laplacian, "DDE_DetailLayer_Laplacian");

    cv::Mat detailLayer_gradient;
    calc_Gradient_32F(detailLayer, detailLayer_gradient);
    imwrite_mdy_private_normalization_8u(detailLayer_gradient, "DDE_DetailLayer_Gradient");

    /*cv::Mat detailLayer_canny_diff;
    cv::absdiff(detailLayer_residual_canny, detailLayer_norm_canny, detailLayer_canny_diff);
    imwrite_mdy_private(detailLayer_canny_diff, "DDE_detailLayer_canny_diff");*/

    //auto element = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    //cv::morphologyEx(canny, canny, cv::MORPH_DILATE, element);

    cv::Mat canny_weight;
    cv::GaussianBlur(detailLayer_gradient, canny_weight, cv::Size(0, 0), 5);
    cv::normalize(canny_weight, canny_weight, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

    cv::Mat variance;
    local_variance_32F(detailLayer, variance, sigma);

    cv::Mat varianceNorm;
    cv::normalize(variance, varianceNorm, 0.0, 1.0, cv::NORM_MINMAX);

    // gain = baseGain / (1 + k * variance)
    double k = baseGain - 1.0;
    cv::Mat gainMap = baseGain / (1.0 + k * varianceNorm);

    cv::Mat final_gainMap;
    //gainMap.copyTo(final_gainMap, canny);
    final_gainMap = gainMap.mul(canny_weight);

    output.assign(final_gainMap);
    return 0;
}


int dde_canny(cv::InputArray input, cv::OutputArray output)
{
    struct DDEConfig
    {
        // 基础层分离（双边滤波）
        int    d = 9;               // 邻域直径（像素，奇数）
        double sigmaColor = 0.2;    // 值域 sigma
        double sigmaSpace = 9.0;    // 空间域 sigma

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
    cfg.d = 15;
    cfg.sigmaSpace = 15.0;

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
    cv::Mat src_canny;
    auto_canny_32F(img_norm, src_canny);
    imwrite_mdy_private(src_canny, "DDE_src_Canny");

    cv::Mat baseLayer;
    cv::bilateralFilter(img_norm, baseLayer,
        cfg.d,
        cfg.sigmaColor,
        cfg.sigmaSpace);

    cv::Mat detailLayer = img_norm - baseLayer;

    cv::Mat baseLayer_canny;
    auto_canny_32F(baseLayer, baseLayer_canny);
    imwrite_mdy_private(baseLayer_canny, "DDE_BaseLayer_Canny");

    imwrite_mdy_private_normalization_8u(baseLayer, "DDE_BaseLayer");
    imwrite_mdy_private_normalization_8u(detailLayer, "DDE_DetailLayer");

    cv::Mat baseCompressed;
    cv::pow(baseLayer, cfg.baseGamma, baseCompressed);

    cv::Mat gainMap;
    DDE_canny_adaptive_gain(detailLayer, gainMap, src_canny, cfg.detailGain);

    cv::Mat detailEnhanced = gainMap.mul(detailLayer);

    cv::Mat detailClipped;

    cv::min(detailEnhanced, cfg.detailClip, detailClipped);
    cv::max(detailClipped, -cfg.detailClip, detailClipped);

    imwrite_mdy_private_normalization_8u(detailClipped, "Enhancement_layer");

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

    int hh_rows = gray.rows / 2;
    int hh_cols = gray.cols / 2;

    std::vector<float> hh_coefficients;
    hh_coefficients.reserve(hh_rows * hh_cols);

    // Haar 小波的 HH 滤波器系数矩阵为: [ 1, -1; -1,  1 ] / 2
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

            float hh = (p00 - p01 - p10 + p11) * 0.5f;

            hh_coefficients.push_back(std::abs(hh));
        }
    }

    if (hh_coefficients.empty()) return 0.0;

    size_t mid_index = hh_coefficients.size() / 2;
    std::nth_element(hh_coefficients.begin(), hh_coefficients.begin() + mid_index, hh_coefficients.end());
    float median_abs = hh_coefficients[mid_index];

    double sigma = median_abs / 0.6745;

    return sigma;
}

int DDE_noise_adaptive_gain(cv::InputArray input, cv::OutputArray output, double baseGain, double noise_Wavelet, double sigma = 5.0)
{
    cv::Mat detailLayer = input.getMat();
    CV_CheckTypeEQ(detailLayer.type(), CV_32F, "");

    cv::Mat variance;
    //local_variance_32F(detailLayer, variance, sigma);
    local_variance_32F_box(detailLayer, variance);

    //double noise_detail = estimateNoiseWaveletMAD(detailLayer);
    double noise_Variance = noise_Wavelet * noise_Wavelet;
    double edgeSaturationThresh = 50.0 * noise_Variance;
    if (edgeSaturationThresh < 1e-6) edgeSaturationThresh = 1e-6; // 防止除零

    cv::Mat varianceNorm = variance / edgeSaturationThresh;
    cv::threshold(varianceNorm, varianceNorm, 1.0, 1.0, cv::THRESH_TRUNC);

    double k = baseGain - 1.0;
    cv::Mat gainMap = baseGain / (1.0 + k * varianceNorm);

    double alpha = 5.0; // 噪声控制敏感系数，通常取 2.0 - 5.0
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
    cfg.d = 15;
    cfg.sigmaSpace = 15.0;

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

    auto noise_Wavelet = estimateNoiseWaveletMAD(img_norm);

    cv::Mat cache, baseLayer;
    cv::bilateralFilter(img_norm, baseLayer,
        cfg.d,
        cfg.sigmaColor,
        cfg.sigmaSpace);

    cv::Mat detailLayer = img_norm - baseLayer;

    cv::Mat baseCompressed;
    cv::pow(baseLayer, cfg.baseGamma, baseCompressed);

    cv::Mat gainMap;
    DDE_noise_adaptive_gain(detailLayer, gainMap, cfg.detailGain, noise_Wavelet);
    cv::Mat detailEnhanced = gainMap.mul(detailLayer);

    cv::Mat detailClipped;

    cv::min(detailEnhanced, cfg.detailClip, detailClipped);
    cv::max(detailClipped, -cfg.detailClip, detailClipped);

    cv::Mat reconstructed = baseCompressed + detailClipped;

    cv::Mat dst;
    cv::normalize(reconstructed, dst, 0, 255, cv::NORM_MINMAX, CV_8U);

    output.assign(dst);

    return 0;
}

int DDE_gradient_adaptive_gain(
    cv::InputArray  input,
    cv::OutputArray output,
    double          baseGain,
    double          k_lo = 2.0,   // 噪声下界倍数
    double          k_hi = 15.0,  // 边缘上界倍数
    double          steepness = 6.0)
{
    cv::Mat detailLayer = input.getMat();
    CV_CheckTypeEQ(detailLayer.type(), CV_32F, "");

    cv::Mat gradMag;
    calc_Gradient_32F(detailLayer, gradMag);
    imwrite_mdy_private_normalization_8u(gradMag, "DDE_DetailLayer_Gradient");

    cv::GaussianBlur(gradMag, gradMag, cv::Size(0, 0), 5);

    double noise_grad_5 = estimateNoiseFromGradient(gradMag, 0.5);
    double noise_grad_7 = estimateNoiseFromGradient(gradMag, 0.7);
    double noise_grad_9 = estimateNoiseFromGradient(gradMag, 0.9);
    double noise_grad_95 = estimateNoiseFromGradient(gradMag, 0.95);
    double noise_grad_99 = estimateNoiseFromGradient(gradMag, 0.99);

    std::cout << "Gradient Noise Estimate - 50th Percentile: " << noise_grad_5 << ", 90th Percentile: " << noise_grad_9 << ", 95th Percentile: " << noise_grad_95 << std::endl;

    cv::Scalar grad_mean, grad_stdv;
    cv::meanStdDev(gradMag, grad_mean, grad_stdv);
    std::cout << "Gradient Mean: " << grad_mean[0] << ", Gradient StdDev: " << grad_stdv[0] << std::endl;

    double lo_thresh = 0.0;
    double hi_thresh = 0.0;

    if (0)
    {
        lo_thresh = noise_grad_9;
        hi_thresh = noise_grad_99;
    }
    else
    {
        lo_thresh = noise_grad_5;
        hi_thresh = noise_grad_9;
    }

    lo_thresh = std::max(lo_thresh, 1e-6);
    hi_thresh = std::max(hi_thresh, 1e-5);

    /*cv::Mat weightMap;
    dualSidedSuppressionWeight(gradMag, weightMap, lo_thresh, hi_thresh, steepness);
    imwrite_mdy_private_normalization_8u(weightMap, "weightMap");*/

    //cv::Mat gainMap = baseGain * weightMap;

    cv::Mat gainMap;
    compute_gradient_gainMap(gradMag, gainMap, baseGain, lo_thresh, hi_thresh, steepness);
    //cv::GaussianBlur(gainMap, gainMap, cv::Size(0, 0), 5);
    imwrite_mdy_private_normalization_8u(gainMap, "Gradient_GainMap");

    output.assign(gainMap);
    return 0;
}

int dde_gradient(cv::InputArray input, cv::OutputArray output)
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

        int radius = 10;              // 引导滤波r
        double eps = 0.04;            // 引导滤波eps
    };

    DDEConfig cfg;
    cfg.baseGamma = 1.0;
    cfg.d = 15;
    cfg.sigmaSpace = 15.0;
    cfg.eps = 0.003;

    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    cv::Mat img_norm;
    double minVal, maxVal;

    //cfg.sigmaSpace = cv::saturate_cast<double>(cfg.d) / 3.0;
    cv::normalize(src, img_norm, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
    cv::minMaxLoc(img_norm, &minVal, &maxVal);

    std::cout << "Min: " << minVal << ", Max: " << maxVal << std::endl;

    cv::Mat baseLayer;
    if (0)
    {
        cv::bilateralFilter(img_norm, baseLayer,
            cfg.d,
            cfg.sigmaColor,
            cfg.sigmaSpace);
    }
    else
    {
        cv::ximgproc::guidedFilter(img_norm, img_norm, baseLayer, cfg.radius, cfg.eps);
    }

    cv::Mat detailLayer = img_norm - baseLayer;

    cv::Mat baseCompressed;
    cv::pow(baseLayer, cfg.baseGamma, baseCompressed);

    cv::Mat gainMap;
    DDE_gradient_adaptive_gain(detailLayer, gainMap, cfg.detailGain);
    cv::Mat detailEnhanced = gainMap.mul(detailLayer);

    cv::Mat detailClipped;

    cv::min(detailEnhanced, cfg.detailClip, detailClipped);
    cv::max(detailClipped, -cfg.detailClip, detailClipped);

    cv::Mat reconstructed = baseCompressed + detailClipped;

    cv::Mat dst;
    cv::normalize(reconstructed, dst, 0, 255, cv::NORM_MINMAX, CV_8U);

    output.assign(dst);

    return 0;
}

/**
 * @brief 计算梯度局部一致性分数 [0,1]
 *        一致性 = localMean / (center + eps)，归一化后饱和到1
 *        真边缘：邻域均值接近中心 → 分数高
 *        孤立噪声点：邻域均值远低于中心 → 分数低
 */
static int calc_GradientCoherence(cv::InputArray input, cv::OutputArray output, int coherenceRadius = 3)
{
    cv::Mat gradMag = input.getMat();
    CV_CheckTypeEQ(gradMag.type(), CV_32FC1, "");

    cv::Mat localMean;
    int ksize = 2 * coherenceRadius + 1;
    cv::boxFilter(gradMag, localMean, CV_32F, cv::Size(ksize, ksize));

    // coherence = localMean / (gradMag + eps)，再clamp到[0,1]
    cv::Mat coherence;
    cv::divide(localMean, gradMag + 1e-6f, coherence);
    cv::min(coherence, 1.0f, coherence);
    output.assign(coherence);  // [0,1]，越大越像真实边缘

    return 0;
}

/**
 * @brief 非对称双侧sigmoid增益曲线
 *        gain(x) = rise(x) * (1 + (baseGain - 1) * fall(x))
 *        其中 x 是融合后的"边缘得分" [0,1]
 *        x→0 : gain→0（噪声抑制）
 *        x→1 : gain→baseGain（边缘增强）
 */
static int apply_AsymmetricSigmoidGain(cv::InputArray input,
    cv::OutputArray output,
    double baseGain,
    double k_lo = 6.0,     // rise陡峭度
    double k_hi = 6.0,    // fall陡峭度
    double mid_lo = 0.30,  // rise中心：噪声抑制截止点
    double mid_hi = 0.70)  // fall中心：边缘增益起始点
{
    cv::Mat score = input.getMat();
    CV_CheckTypeEQ(score.type(), CV_32FC1, "");

    // rise(x) = sigmoid(k_lo * (x - mid_lo))，控制噪声侧抑制斜率
    // fall(x) = sigmoid(k_hi * (x - mid_hi))，控制边缘侧增益斜率
    // 两者组合：低分→0，高分→baseGain

    cv::Mat rise, fall;

    // rise: x < threshold → 接近0，x > threshold → 接近1
    cv::exp(-k_lo * (score - mid_lo), rise);
    rise = 1.0 / (1.0 + rise);  // sigmoid

    // fall: x > threshold → 接近1，驱动增益到baseGain
    cv::exp(-k_hi * (score - mid_hi), fall);
    fall = 1.0 / (1.0 + fall);  // sigmoid

    // gain = rise * (1 + (baseGain - 1) * fall)
    cv::Mat gainMap = rise.mul(1.0 + (baseGain - 1.0) * fall);
    output.assign(gainMap);

    return 0;
}

int DDE_adaptive_gain_guided_grad(cv::InputArray input, cv::OutputArray output, double baseGain, double sigma = 5.0)
{
    cv::Mat detailLayer = input.getMat();
    CV_CheckTypeEQ(detailLayer.type(), CV_32FC1, "");

    cv::Mat gradMag;
    calc_Gradient_32F(detailLayer, gradMag);
    imwrite_mdy_private_normalization_8u(gradMag, "detailLayer_Gradient");

	cv::GaussianBlur(gradMag, gradMag, cv::Size(3, 3), 0.5);

    const double pct_lo = 95.0;   // 低于此百分位 → 噪声
    const double pct_hi = 98.0;   // 高于此百分位 → 边缘

    auto thresh_lo = estimateNoiseFromGradient(gradMag, pct_lo / 100.0);
    auto thresh_hi = estimateNoiseFromGradient(gradMag, pct_hi / 100.0);

    std::cout << "Gradient Magnitude Percentile: - " << thresh_lo << ", " << thresh_hi << std::endl;

    cv::Mat ampScore;
    gradMag.convertTo(ampScore, CV_32F);
    ampScore = (ampScore - thresh_lo) / (thresh_hi - thresh_lo + 1e-6f);
    cv::min(ampScore, 1.0f, ampScore);
    cv::max(ampScore, 0.0f, ampScore);
	imwrite_mdy_private_normalization_8u(ampScore, "Gradient_Amplitude_Score");

    //cv::Mat coherenceScore;
    //calc_GradientCoherence(gradMag, coherenceScore, 1);
    //const double w_amp = 0.7, w_coh = 0.3;
    //cv::Mat edgeScore = w_amp * ampScore + w_coh * coherenceScore;  // [0,1]

    cv::Mat gainMap;
    //apply_AsymmetricSigmoidGain(ampScore, gainMap, baseGain);
	cv::GaussianBlur(ampScore, ampScore, cv::Size(0, 0), 5);
	cv::normalize(ampScore, gainMap, 0.0, 1.0, cv::NORM_MINMAX);
    imwrite_mdy_private_normalization_8u(gainMap, "Gradient_Adaptive_GainMap");

    output.assign(gainMap);
    return 0;
}


int dde_guided(cv::InputArray input, cv::OutputArray output)
{
    struct DDEConfig
    {
        int radius = 5;              // 引导滤波r
        double eps = 0.04;            // 引导滤波eps

        // 细节增强
        double detailGain = 3.0;    // 细节增益系数
        double detailClip = 0.2;    // 细节层截断，防止过增强（归一化）

        // 基础层压缩
        double baseGamma = 0.5;    // gamma < 1 压缩亮区，提升暗区

        // 输出
        double lowPct = 0.5;
        double highPct = 98.0;
    };

    DDEConfig cfg;
    cfg.baseGamma = 1.0;//先不做基础层gamma
    cfg.eps = 500;

    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    // Step 1:
    cv::Mat img_norm;
    double minVal, maxVal;

    src.convertTo(img_norm, CV_32F);
    cv::minMaxLoc(img_norm, &minVal, &maxVal);
    auto dynamicRange = maxVal - minVal;
    cfg.detailClip = dynamicRange * cfg.detailClip;

    std::cout << "Min: " << minVal << ", Max: " << maxVal << std::endl;

	percentile_truncation_32F(img_norm, img_norm, cfg.lowPct, cfg.highPct);

    cv::Mat baseLayer;
    cv::ximgproc::guidedFilter(img_norm, img_norm, baseLayer, cfg.radius, cfg.eps);

    cv::Mat detailLayer = img_norm - baseLayer;
    imwrite_mdy_private_normalization_8u(detailLayer, "Guided_DetailLayer");
    imwrite_mdy_private_normalization_8u(baseLayer, "Guided_BaseLayer");

    cv::Mat baseCompressed;
    cv::pow(baseLayer, cfg.baseGamma, baseCompressed);

    cv::Mat gainMap;
    DDE_adaptive_gain_guided_grad(detailLayer, gainMap, cfg.detailGain);
    cv::Mat detailEnhanced = gainMap.mul(detailLayer);

    cv::Mat detailClipped;

    cv::min(detailEnhanced, cfg.detailClip, detailClipped);
    cv::max(detailClipped, -cfg.detailClip, detailClipped);

    cv::Mat reconstructed = baseCompressed + detailClipped;

    cv::Mat dst;
    cv::normalize(reconstructed, dst, 0, 255, cv::NORM_MINMAX, CV_8U);
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
    //estimate_NUC_Temporal(images, nuc);

    for (int i = 0; i < images.size(); ++i)
    {
        cv::Mat dst_DDE;
        //dde_denoise(images[i], dst_DDE, nuc);
        //dde_canny(images[i], dst_DDE);
        //dde_gradient(images[i], dst_DDE);
        dde_guided(images[i], dst_DDE);
        imwrite_mdy_private(dst_DDE, "DDE_origin");
    }

    return 0;
}

int DDE_improvement()
{
    test_DDE_improve();

    return 0;
}

