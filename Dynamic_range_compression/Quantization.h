#pragma once

#include "Dynamic_range_compression.h"


constexpr int Q = 18;
constexpr uint64_t roundOffset = 1ULL << (Q - 1);


// 定义权重量化位宽为 12 bit
constexpr int W_BITS = 16;
constexpr uint64_t W_ROUND_OFFSET = 1ULL << (W_BITS - 1);
constexpr uint64_t W_ROUND = 1ULL << (2 * W_BITS - 1); // 用于最后双重乘法后的四舍五入


class CLAHE_Fixed : public cv::Algorithm
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

cv::Ptr<CLAHE_Fixed> createCLAHE_Fixed(double clipLimit = 40.0, cv::Size tileGridSize = cv::Size(8, 8));

