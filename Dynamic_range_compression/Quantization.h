#pragma once

#include "Dynamic_range_compression.h"


// 计算LUT时量化位宽为
constexpr int Q = 18;
constexpr uint64_t roundOffset = 1ULL << (Q - 1);


// 定义像素偏移和权重量化位宽
constexpr int W_BITS = 18;
constexpr uint64_t W_ROUND_OFFSET = 1ULL << (W_BITS - 1);
constexpr uint64_t W_ROUND = 1ULL << (2 * W_BITS - 1); // 用于最后双重乘法后的四舍五入


class CLAHE_Float : public cv::Algorithm
{
public:
    virtual void apply(cv::InputArray src, cv::OutputArray dst) = 0;

    virtual void setClipLimit(double clipLimit) = 0;

    virtual double getClipLimit() const = 0;

    virtual void setTilesGridSize(cv::Size tileGridSize) = 0;

    virtual cv::Size getTilesGridSize() const = 0;

    virtual void setBitShift(int bitShift) = 0;

    virtual int getBitShift() const = 0;

    virtual void collectGarbage() = 0;
};

cv::Ptr<CLAHE_Float> createCLAHE_Float(double clipLimit = 40.0, cv::Size tileGridSize = cv::Size(8, 8), int histSize = 0);

class CLAHE_Fixed : public cv::Algorithm
{
public:
    virtual void apply(cv::InputArray src, cv::OutputArray dst) = 0;

    virtual void setClipLimit(int clipLimit) = 0;

    virtual int getClipLimit() const = 0;

    virtual void setTilesGridSize(cv::Size tileGridSize) = 0;

    virtual cv::Size getTilesGridSize() const = 0;

    virtual void setBitShift(int bitShift) = 0;

    virtual int getBitShift() const = 0;

    virtual void collectGarbage() = 0;

    virtual uint16_t getGlobalMin() const = 0;

    virtual uint16_t getGlobalMax() const = 0;
};

cv::Ptr<CLAHE_Fixed> createCLAHE_Fixed(int clipLimit = 40, cv::Size tileGridSize = cv::Size(8, 8), int histSize = 0);

int test_precision_batch_14to8();

int linear_mapping_fixed(cv::InputArray input, cv::OutputArray output);

class Gaussian_Blur_Fixed : public cv::Algorithm
{
public:
    virtual int apply(cv::InputArray input, cv::OutputArray output) const = 0;
    virtual void set_sigma(int sigma) = 0;

    virtual int ksize() const = 0;
    virtual int sigma() const = 0;

    virtual const std::vector<int32_t>& kernel() const = 0;
    virtual const std::vector<int32_t>& cv_kernel() const = 0;
};

cv::Ptr<Gaussian_Blur_Fixed> createGaussianBlur_Fixed(int sigma = 50, int gaussQ = 16);

class SSR_Fixed : public cv::Algorithm
{
public:
    virtual int apply(cv::InputArray input, cv::OutputArray output) = 0;
};

cv::Ptr<SSR_Fixed> createSSR_Fixed(int logQ = 16, int linearQ = 16, cv::Ptr<Gaussian_Blur_Fixed> gauss_ptr = nullptr);

