#include "quantitative_assessment.h"


namespace fs = std::filesystem;
using Ms = std::chrono::duration<double, std::milli>;

double benchmark(const std::vector<cv::Mat>& images,
    const std::function<int(cv::InputArray, cv::OutputArray)>& func)
{
    cv::Mat dst;
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto& img : images)
    {
        func(img, dst);
    }
    auto end = std::chrono::high_resolution_clock::now();
    return Ms(end - start).count();
}

int benchmark_main()
{
    // 加载图像
    std::vector<cv::Mat> images;
    for (const auto& entry : fs::directory_iterator(IMAGE_DIR)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".png")
            continue;
        cv::Mat img = cv::imread(entry.path().string(), cv::IMREAD_UNCHANGED);
        if (!img.empty() && img.type() == CV_16UC1)
            images.push_back(img);
    }
    std::cout << "加载图像数量：" << images.size() << std::endl;

    // 注册各算法
    std::map<std::string, std::function<int(cv::InputArray, cv::OutputArray)>> algorithms;

    algorithms["Linear"] = [](cv::InputArray input, cv::OutputArray output)->int {
        return linear_mapping(input, output);
        };

    algorithms["CLAHE"] = [](cv::InputArray input, cv::OutputArray output)->int {
        return clahe_mapping(input, output, 3.0, cv::Size(8, 8));
        };

    algorithms["GLAF"] = [](cv::InputArray input, cv::OutputArray output)->int {
        return global_local_adaptive_fusion(input, output);
        };

    algorithms["MSR"] = [](cv::InputArray input, cv::OutputArray output)->int {
        return multi_scale_retinex(input, output, { 15.0, 80.0, 250.0 });
        };

    algorithms["DDE"] = [](cv::InputArray input, cv::OutputArray output)->int {
        return dde_enhance(input, output);
        };

    // 执行 benchmark 并打印结果
    std::cout << "\n--- Benchmark Results ---" << std::endl;
    std::cout << std::left
        << std::setw(20) << "Algorithm"
        << std::setw(15) << "Total(ms)"
        << std::setw(15) << "Avg(ms)"
        << std::endl;
    std::cout << std::string(50, '-') << std::endl;

    for (const auto& [name, func] : algorithms)
    {
        double totalMs = benchmark(images, func);
        double avgMs = totalMs / images.size();
        std::cout << std::left << std::setw(20) << name
            << std::fixed << std::setprecision(2)
            << std::setw(15) << totalMs
            << std::setw(15) << avgMs
            << std::endl;
    }

    return 0;
}

double calcEntropy(cv::InputArray src)
{
    cv::Mat img = src.getMat();
    CV_CheckTypeEQ(img.type(), CV_8UC1, "");

    // 计算直方图
    int histSize = 256;
    float range[] = { 0, 256 };
    const float* histRange = range;
    cv::Mat hist;
    cv::calcHist(&img, 1, nullptr, cv::Mat(), hist, 1, &histSize, &histRange);

    // 归一化为概率
    hist /= img.total();

    // H = -sum(p * log2(p))
    double entropy = 0.0;
    for (int i = 0; i < histSize; i++)
    {
        float p = hist.at<float>(i);
        if (p > 0)
            entropy -= p * std::log2(p);
    }
    return entropy;  // 最大值为 8（均匀分布）
}

double calcAverageGradient(cv::InputArray src)
{
	cv::Mat img = src.getMat();
    CV_CheckTypeEQ(img.type(), CV_8UC1, "");

    cv::Mat gradX, gradY, gradMag;
    cv::Sobel(img, gradX, CV_32F, 1, 0, 3);
    cv::Sobel(img, gradY, CV_32F, 0, 1, 3);
    cv::magnitude(gradX, gradY, gradMag);

    return cv::mean(gradMag)[0];
}

double calcSSIM(cv::InputArray input, cv::InputArray in_ref)
{
	cv::Mat src = input.getMat();
	cv::Mat ref = in_ref.getMat();
	CV_CheckTypeEQ(src.type(), CV_8UC1, "");
    CV_CheckTypeEQ(ref.type(), CV_8UC1, "");

    cv::Mat img1, img2;
    src.convertTo(img1, CV_32F);
    ref.convertTo(img2, CV_32F);

    const double C1 = 6.5025;   // (0.01 * 255)²
    const double C2 = 58.5225;  // (0.03 * 255)²

    cv::Mat mu1, mu2, mu1_sq, mu2_sq, mu1_mu2;
    cv::GaussianBlur(img1, mu1, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2, mu2, cv::Size(11, 11), 1.5);

    mu1_sq = mu1.mul(mu1);
    mu2_sq = mu2.mul(mu2);
    mu1_mu2 = mu1.mul(mu2);

    cv::Mat sigma1_sq, sigma2_sq, sigma12;
    cv::GaussianBlur(img1.mul(img1), sigma1_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img2.mul(img2), sigma2_sq, cv::Size(11, 11), 1.5);
    cv::GaussianBlur(img1.mul(img2), sigma12, cv::Size(11, 11), 1.5);

    sigma1_sq -= mu1_sq;
    sigma2_sq -= mu2_sq;
    sigma12 -= mu1_mu2;

    cv::max(sigma1_sq, 0.0f, sigma1_sq);
    cv::max(sigma2_sq, 0.0f, sigma2_sq);

    cv::Mat ssim_map;
    cv::Mat numerator = (2.0 * mu1_mu2 + C1).mul(2.0 * sigma12 + C2);
    cv::Mat denominator = (mu1_sq + mu2_sq + C1).mul(sigma1_sq + sigma2_sq + C2);
    cv::divide(numerator, denominator, ssim_map);

    return cv::mean(ssim_map)[0];  // 范围 [-1, 1]，越接近 1 越好
}
