#include "Quantization.h"





template <class T>
class CLAHE_CalcLut_Body : public cv::ParallelLoopBody
{
public:
    CLAHE_CalcLut_Body(const cv::Mat& src, const cv::Mat& lut, const cv::Size& tileSize, const int& tilesX, const int& clipLimit, const float& lutScale, const int& histSize, const int& shift) :
        src_(src), lut_(lut), tileSize_(tileSize), tilesX_(tilesX), clipLimit_(clipLimit), lutScale_(lutScale), histSize_(histSize), shift_(shift)
    {
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE;

private:
    cv::Mat src_;
    mutable cv::Mat lut_;

    cv::Size tileSize_;
    int tilesX_;
    int clipLimit_;
    float lutScale_;
    int histSize_;
    int shift_;
};

template <class T>
void CLAHE_CalcLut_Body<T>::operator ()(const cv::Range& range) const
{
    T* tileLut = lut_.ptr<T>(range.start);
    const size_t lut_step = lut_.step / sizeof(T);

    for (int k = range.start; k < range.end; ++k, tileLut += lut_step)
    {
        const int ty = k / tilesX_;
        const int tx = k % tilesX_;

        // retrieve tile submatrix

        cv::Rect tileROI;
        tileROI.x = tx * tileSize_.width;
        tileROI.y = ty * tileSize_.height;
        tileROI.width = tileSize_.width;
        tileROI.height = tileSize_.height;

        const cv::Mat tile = src_(tileROI);

        // calc histogram

        cv::AutoBuffer<int> _tileHist(histSize_);
        int* tileHist = _tileHist.data();
        std::fill(tileHist, tileHist + histSize_, 0);

        int height = tileROI.height;
        const size_t sstep = src_.step / sizeof(T);
        for (const T* ptr = tile.ptr<T>(0); height--; ptr += sstep)
        {
            int x = 0;
            for (; x <= tileROI.width - 4; x += 4)
            {
                int t0 = ptr[x], t1 = ptr[x + 1];
                tileHist[t0 >> shift_]++; tileHist[t1 >> shift_]++;
                t0 = ptr[x + 2]; t1 = ptr[x + 3];
                tileHist[t0 >> shift_]++; tileHist[t1 >> shift_]++;
            }

            for (; x < tileROI.width; ++x)
                tileHist[ptr[x] >> shift_]++;
        }

        // clip histogram

        if (clipLimit_ > 0)
        {
            // how many pixels were clipped
            int clipped = 0;
            for (int i = 0; i < histSize_; ++i)
            {
                if (tileHist[i] > clipLimit_)
                {
                    clipped += tileHist[i] - clipLimit_;
                    tileHist[i] = clipLimit_;
                }
            }

            // redistribute clipped pixels
            int redistBatch = clipped / histSize_;
            int residual = clipped - redistBatch * histSize_;

            for (int i = 0; i < histSize_; ++i)
                tileHist[i] += redistBatch;

            if (residual != 0)
            {
                int residualStep = MAX(histSize_ / residual, 1);
                for (int i = 0; i < histSize_ && residual > 0; i += residualStep, residual--)
                    tileHist[i]++;
            }
        }

        // calc Lut

        int sum = 0;
        for (int i = 0; i < histSize_; ++i)
        {
            sum += tileHist[i];
            tileLut[i] = cv::saturate_cast<T>(sum * lutScale_);
        }
    }
}

template <class T>
class CLAHE_Interpolation_Body : public cv::ParallelLoopBody
{
public:
    CLAHE_Interpolation_Body(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& lut, const cv::Size& tileSize, const int& tilesX, const int& tilesY, const int& shift) :
        src_(src), dst_(dst), lut_(lut), tileSize_(tileSize), tilesX_(tilesX), tilesY_(tilesY), shift_(shift)
    {
        buf.allocate(src.cols << 2);
        ind1_p = buf.data();
        ind2_p = ind1_p + src.cols;
        xa_p = (float*)(ind2_p + src.cols);
        xa1_p = xa_p + src.cols;

        int lut_step = static_cast<int>(lut_.step / sizeof(T));
        float inv_tw = 1.0f / tileSize_.width;

        for (int x = 0; x < src.cols; ++x)
        {
            float txf = x * inv_tw - 0.5f;

            int tx1 = cvFloor(txf);
            int tx2 = tx1 + 1;

            xa_p[x] = txf - tx1;
            xa1_p[x] = 1.0f - xa_p[x];

            tx1 = std::max(tx1, 0);
            tx2 = std::min(tx2, tilesX_ - 1);

            ind1_p[x] = tx1 * lut_step;
            ind2_p[x] = tx2 * lut_step;
        }
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE;

private:
    cv::Mat src_;
    mutable cv::Mat dst_;
    cv::Mat lut_;

    cv::Size tileSize_;
    int tilesX_;
    int tilesY_;
    int shift_;

    cv::AutoBuffer<int> buf;
    int* ind1_p, * ind2_p;
    float* xa_p, * xa1_p;
};

template <class T>
void CLAHE_Interpolation_Body<T>::operator ()(const cv::Range& range) const
{
    float inv_th = 1.0f / tileSize_.height;

    for (int y = range.start; y < range.end; ++y)
    {
        const T* srcRow = src_.ptr<T>(y);
        T* dstRow = dst_.ptr<T>(y);

        float tyf = y * inv_th - 0.5f;

        int ty1 = cvFloor(tyf);
        int ty2 = ty1 + 1;

        float ya = tyf - ty1, ya1 = 1.0f - ya;

        ty1 = std::max(ty1, 0);
        ty2 = std::min(ty2, tilesY_ - 1);

        const T* lutPlane1 = lut_.ptr<T>(ty1 * tilesX_);
        const T* lutPlane2 = lut_.ptr<T>(ty2 * tilesX_);

        for (int x = 0; x < src_.cols; ++x)
        {
            int srcVal = srcRow[x] >> shift_;

            int ind1 = ind1_p[x] + srcVal;
            int ind2 = ind2_p[x] + srcVal;

            float res = (lutPlane1[ind1] * xa1_p[x] + lutPlane1[ind2] * xa_p[x]) * ya1 +
                (lutPlane2[ind1] * xa1_p[x] + lutPlane2[ind2] * xa_p[x]) * ya;

            dstRow[x] = cv::saturate_cast<T>(res) << shift_;
        }
    }
}


template <class T>
class CLAHE_CalcLut_Body_Fixed : public cv::ParallelLoopBody
{
public:
    CLAHE_CalcLut_Body_Fixed(
        const cv::Mat& src, const cv::Mat& lut,
        const cv::Size& tileSize, const int& tilesX,
        const int& clipLimit, const uint64_t& lutScaleFixed, const int& histSize, const int& shift)
        : src_(src), lut_(lut), tileSize_(tileSize), tilesX_(tilesX),
        clipLimit_(clipLimit), lutScaleFixed_(lutScaleFixed), histSize_(histSize), shift_(shift)
    {
    }

    void operator()(const cv::Range& range) const CV_OVERRIDE;

private:
    cv::Mat       src_;
    mutable cv::Mat lut_;

    cv::Size      tileSize_;
    int           tilesX_;
    int           clipLimit_;
    int           histSize_;
    int           shift_;
    uint64_t      lutScaleFixed_; // 替换 float lutScale_
};

template <class T>
void CLAHE_CalcLut_Body_Fixed<T>::operator()(const cv::Range& range) const
{
    T* tileLut = lut_.ptr<T>(range.start);
    const size_t lut_step = lut_.step / sizeof(T);

    for (int k = range.start; k < range.end; ++k, tileLut += lut_step)
    {
        const int ty = k / tilesX_;
        const int tx = k % tilesX_;

        // ---- 取 tile 区域（不变）----
        cv::Rect tileROI;
        tileROI.x = tx * tileSize_.width;
        tileROI.y = ty * tileSize_.height;
        tileROI.width = tileSize_.width;
        tileROI.height = tileSize_.height;

        const cv::Mat tile = src_(tileROI);

        // ---- 统计直方图（不变，已经是整数）----
        cv::AutoBuffer<int> _tileHist(histSize_);
        int* tileHist = _tileHist.data();
        std::fill(tileHist, tileHist + histSize_, 0);

        int height = tileROI.height;
        const size_t sstep = src_.step / sizeof(T);
        for (const T* ptr = tile.ptr<T>(0); height--; ptr += sstep)
        {
            int x = 0;
            for (; x <= tileROI.width - 4; x += 4)
            {
                int t0 = ptr[x], t1 = ptr[x + 1];
                tileHist[t0 >> shift_]++; tileHist[t1 >> shift_]++;
                t0 = ptr[x + 2];   t1 = ptr[x + 3];
                tileHist[t0 >> shift_]++; tileHist[t1 >> shift_]++;
            }

            for (; x < tileROI.width; ++x)
                tileHist[ptr[x] >> shift_]++;
        }

        // ---- 限幅 + 重分配（不变，已经是整数）----
        if (clipLimit_ > 0)
        {
            int clipped = 0;
            for (int i = 0; i < histSize_; ++i)
            {
                if (tileHist[i] > clipLimit_)
                {
                    clipped += tileHist[i] - clipLimit_;
                    tileHist[i] = clipLimit_;
                }
            }

            int redistBatch = clipped / histSize_;
            int residual = clipped - redistBatch * histSize_;

            for (int i = 0; i < histSize_; ++i)
                tileHist[i] += redistBatch;

            if (residual != 0)
            {
                int residualStep = std::max(histSize_ / residual, 1);
                for (int i = 0; i < histSize_ && residual > 0;
                    i += residualStep, residual--)
                    tileHist[i]++;
            }
        }

        // ---- LUT 生成：用定点乘法替换浮点 ----
        // 原始：tileLut[i] = saturate_cast<T>(sum * lutScale_)
        // 定点：tileLut[i] = (sum * lutScale_fixed_) >> Q
        int sum = 0;
        for (int i = 0; i < histSize_; ++i)
        {
            sum += tileHist[i];
            uint64_t val = (cv::saturate_cast<uint64_t>(sum) * lutScaleFixed_ + roundOffset) >> Q;
            tileLut[i] = std::min(cv::saturate_cast<T>(val), cv::saturate_cast<T>(histSize_ - 1));
        }
    }
}


template <class T>
class CLAHE_Interpolation_Body_Fixed : public cv::ParallelLoopBody
{
public:
    CLAHE_Interpolation_Body_Fixed(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& lut,
        const cv::Size& tileSize, const int& tilesX, const int& tilesY, const int& shift) :
        src_(src), dst_(dst), lut_(lut), tileSize_(tileSize), tilesX_(tilesX), tilesY_(tilesY), shift_(shift)
    {
        buf.allocate(src.cols << 1); // 2个int数组的空间
        ind1_p = buf.data();
        ind2_p = ind1_p + src.cols;

        xbuf.allocate(src.cols << 1); // 2个定点权重数组的空间
        xa_p = xbuf.data();  // 定点化权重数组
        xa1_p = xa_p + src.cols;

        int lut_step = static_cast<int>(lut_.step / sizeof(T));
        uint64_t inv_tw_fixed = (1ULL << W_BITS) / static_cast<uint64_t>(tileSize_.width);

        /*for (int x = 0; x < src.cols; ++x)
        {
            auto txf = cv::saturate_cast<uint64_t>(x) * inv_tw_fixed;
            if (txf < W_ROUND_OFFSET)
            {
                int tx1 = 0;
                int tx2 = 0;

                xa_p[x] = 0;
                xa1_p[x] = 1 << W_BITS;
                
                ind1_p[x] = tx1 * lut_step;
                ind2_p[x] = tx2 * lut_step;
                
                continue;
            }

            txf -= W_ROUND_OFFSET;
            int tx1 = static_cast<int>(txf >> W_BITS);
            int tx2 = tx1 + 1;

            // --- 量化核心：将权重转为 Q12 定点数 ---
            xa_p[x] = txf & ((1ULL << W_BITS) - 1);
            xa1_p[x] = (1ULL << W_BITS) - xa_p[x];

            tx1 = std::max(tx1, 0);
            tx2 = std::min(tx2, tilesX_ - 1);

            ind1_p[x] = tx1 * lut_step;
            ind2_p[x] = tx2 * lut_step;
        }*/
        for (int x = 0; x < src.cols; ++x)
        {
            int64_t txf = cv::saturate_cast<int64_t>(cv::saturate_cast<uint64_t>(x) * inv_tw_fixed) - cv::saturate_cast<int64_t>(W_ROUND_OFFSET);

            int tx1 = static_cast<int>(txf >> W_BITS);
            int tx2 = tx1 + 1;

            // --- 量化核心：将权重转为 Q 定点数 ---
            xa_p[x] = cv::saturate_cast<uint64_t>(txf - (cv::saturate_cast<int64_t>(tx1) << W_BITS));
            xa1_p[x] = (1ULL << W_BITS) - xa_p[x];

            tx1 = std::max(tx1, 0);
            tx2 = std::min(tx2, tilesX_ - 1);

            ind1_p[x] = tx1 * lut_step;
            ind2_p[x] = tx2 * lut_step;
        }
    }

    void operator ()(const cv::Range& range) const CV_OVERRIDE;

private:
    cv::Mat src_;
    mutable cv::Mat dst_;
    cv::Mat lut_;

    cv::Size tileSize_;
    int tilesX_, tilesY_, shift_;

    cv::AutoBuffer<int> buf;
    int * ind1_p, * ind2_p;

    cv::AutoBuffer<uint64_t> xbuf;
    uint64_t * xa_p, * xa1_p; // 定点权重指针
};

template <class T>
void CLAHE_Interpolation_Body_Fixed<T>::operator ()(const cv::Range& range) const
{
    uint64_t inv_th_fixed = (1ULL << W_BITS) / static_cast<uint64_t>(tileSize_.height);

    for (int y = range.start; y < range.end; ++y)
    {
        const T* srcRow = src_.ptr<T>(y);
        T* dstRow = dst_.ptr<T>(y);

        // --- 纵向坐标定点化计算 (模拟硬件累加器/坐标映射器) ---
        // tyf = y * inv_th - 0.5
        int64_t tyf = cv::saturate_cast<int64_t>(cv::saturate_cast<uint64_t>(y) * inv_th_fixed) - cv::saturate_cast<int64_t>(W_ROUND_OFFSET);

        // 算术右移提取 Tile 索引
        int ty1 = static_cast<int>(tyf >> W_BITS);
        int ty2 = ty1 + 1;

        // 提取低位并缩放到 Q 权重
        uint64_t ya = cv::saturate_cast<uint64_t>(tyf - (cv::saturate_cast<int64_t>(ty1) << W_BITS));
        uint64_t ya1 = (1ULL << W_BITS) - ya;

        // 边界限制
        ty1 = std::max(ty1, 0);
        ty2 = std::min(ty2, tilesY_ - 1);

        const T* lutPlane1 = lut_.ptr<T>(ty1 * tilesX_);
        const T* lutPlane2 = lut_.ptr<T>(ty2 * tilesX_);

        for (int x = 0; x < src_.cols; ++x)
        {
            // 输入 14-bit 降采样到 histSize
            int srcVal = srcRow[x] >> shift_;

            int ind1 = ind1_p[x] + srcVal;
            int ind2 = ind2_p[x] + srcVal;

            // --- 核心插值公式量化 (三级乘法累加) ---
            uint64_t r1_res = cv::saturate_cast<uint64_t>(lutPlane1[ind1]) * xa1_p[x] + cv::saturate_cast<uint64_t>(lutPlane1[ind2]) * xa_p[x];
            uint64_t r2_res = cv::saturate_cast<uint64_t>(lutPlane2[ind1]) * xa1_p[x] + cv::saturate_cast<uint64_t>(lutPlane2[ind2]) * xa_p[x];

            uint64_t res = (r1_res * ya1 + r2_res * ya + W_ROUND) >> (W_BITS * 2);

            // 第三级：还原回 14-bit 原始深度并截断
            dstRow[x] = cv::saturate_cast<T>(res) << shift_;

            if (ind1 != ind2)
            {
                int a = 0;
            }
        }
    }
}

class CLAHE_Impl_Fixed CV_FINAL : public CLAHE_Fixed
{
public:
    CLAHE_Impl_Fixed(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8, int bitShift = 0);

    void apply(cv::InputArray src, cv::OutputArray dst) CV_OVERRIDE;

    void setClipLimit(double clipLimit) CV_OVERRIDE;
    double getClipLimit() const CV_OVERRIDE;

    void setTilesGridSize(cv::Size tileGridSize) CV_OVERRIDE;
    cv::Size getTilesGridSize() const CV_OVERRIDE;

    void setBitShift(int bitShift) CV_OVERRIDE;
    int getBitShift() const CV_OVERRIDE;

    void collectGarbage() CV_OVERRIDE;

private:
    double clipLimit_;
    int tilesX_;
    int tilesY_;
    int bitShift_;

    cv::Mat srcExt_;
    cv::Mat lut_;
};

CLAHE_Impl_Fixed::CLAHE_Impl_Fixed(double clipLimit, int tilesX, int tilesY, int bitShift) :
    clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY), bitShift_(bitShift)
{
}

void CLAHE_Impl_Fixed::apply(cv::InputArray _src, cv::OutputArray _dst)
{
    CV_Assert(_src.type() == CV_8UC1 || _src.type() == CV_16UC1);

    int histSize = _src.type() == CV_8UC1 ? (256 >> bitShift_) : (65536 >> bitShift_);

    cv::Size tileSize;
    cv::_InputArray _srcForLut;

    if (_src.size().width % tilesX_ == 0 && _src.size().height % tilesY_ == 0)
    {
        tileSize = cv::Size(_src.size().width / tilesX_, _src.size().height / tilesY_);
        _srcForLut = _src;
    }
    else
    {
        cv::copyMakeBorder(_src, srcExt_, 0, tilesY_ - (_src.size().height % tilesY_), 0, tilesX_ - (_src.size().width % tilesX_), cv::BORDER_REFLECT_101);
        tileSize = cv::Size(srcExt_.size().width / tilesX_, srcExt_.size().height / tilesY_);
        _srcForLut = srcExt_;
    }
    

    const int tileSizeTotal = tileSize.area();
    const uint64_t lutScale = (cv::saturate_cast<uint64_t>(histSize - 1) << Q) / cv::saturate_cast<uint64_t>(tileSizeTotal);

    int clipLimit = 0;
    if (clipLimit_ > 0.0)
    {
        clipLimit = static_cast<int>(clipLimit_ * tileSizeTotal / histSize);
        clipLimit = std::max(clipLimit, 1);
    }

    cv::Mat src = _src.getMat();
    _dst.create(src.size(), src.type());
    cv::Mat dst = _dst.getMat();
    cv::Mat srcForLut = _srcForLut.getMat();
    lut_.create(tilesX_ * tilesY_, histSize, _src.type());

    cv::Ptr<cv::ParallelLoopBody> calcLutBody;
    if (_src.type() == CV_8UC1)
    {
        calcLutBody = cv::makePtr<CLAHE_CalcLut_Body_Fixed<uchar> >(srcForLut, lut_, tileSize, tilesX_, clipLimit, lutScale, histSize, bitShift_);
    }
    else if (_src.type() == CV_16UC1)
    {
        calcLutBody = cv::makePtr<CLAHE_CalcLut_Body_Fixed<ushort> >(srcForLut, lut_, tileSize, tilesX_, clipLimit, lutScale, histSize, bitShift_);
    }
    else
        CV_Error(cv::Error::StsBadArg, "Unsupported type");

    cv::parallel_for_(cv::Range(0, tilesX_ * tilesY_), *calcLutBody);

    cv::Mat lut_cv;
    {
		lut_cv.create(tilesX_ * tilesY_, histSize, _src.type());
        const float lutScale_cv = static_cast<float>(histSize - 1) / tileSizeTotal;

        cv::Ptr<cv::ParallelLoopBody> calcLutBody_cv;
        if (_src.type() == CV_8UC1)
        {
            calcLutBody_cv = cv::makePtr<CLAHE_CalcLut_Body<uchar> >(srcForLut, lut_cv, tileSize, tilesX_, clipLimit, lutScale_cv, histSize, bitShift_);
        }
        else if (_src.type() == CV_16UC1)
        {
            calcLutBody_cv = cv::makePtr<CLAHE_CalcLut_Body<ushort> >(srcForLut, lut_cv, tileSize, tilesX_, clipLimit, lutScale_cv, histSize, bitShift_);
        }
        else
            CV_Error(cv::Error::StsBadArg, "Unsupported type");

        cv::parallel_for_(cv::Range(0, tilesX_ * tilesY_), *calcLutBody_cv);
    }

    cv::Mat diff_lut;
    cv::absdiff(lut_, lut_cv, diff_lut);
    double minVal, maxVal;
    cv::minMaxLoc(diff_lut, &minVal, &maxVal);
    int nonzero = cv::countNonZero(diff_lut);

    cv::Ptr<cv::ParallelLoopBody> interpolationBody;
    if (_src.type() == CV_8UC1)
    {
        interpolationBody = cv::makePtr<CLAHE_Interpolation_Body_Fixed<uchar> >(src, dst, lut_, tileSize, tilesX_, tilesY_, bitShift_);
    }
    else if (_src.type() == CV_16UC1)
    {
        interpolationBody = cv::makePtr<CLAHE_Interpolation_Body_Fixed<ushort> >(src, dst, lut_, tileSize, tilesX_, tilesY_, bitShift_);
    }

    cv::parallel_for_(cv::Range(0, src.rows), *interpolationBody);

	cv::Mat dst_cv(dst.size(), dst.type());
    {
        cv::Ptr<cv::ParallelLoopBody> interpolationBody_cv;
        if (_src.type() == CV_8UC1)
        {
            interpolationBody_cv = cv::makePtr<CLAHE_Interpolation_Body<uchar> >(src, dst_cv, lut_, tileSize, tilesX_, tilesY_, bitShift_);
        }
        else if (_src.type() == CV_16UC1)
        {
            interpolationBody_cv = cv::makePtr<CLAHE_Interpolation_Body<ushort> >(src, dst_cv, lut_, tileSize, tilesX_, tilesY_, bitShift_);
        }
		cv::parallel_for_(cv::Range(0, src.rows), *interpolationBody_cv);
    }

	cv::Mat diff_dst;
	cv::absdiff(dst, dst_cv, diff_dst);
	double minDstVal, maxDstVal;
	cv::minMaxLoc(diff_dst, &minDstVal, &maxDstVal);
	int nonzeroDst = cv::countNonZero(diff_dst);

	diff_lut.release();
}

void CLAHE_Impl_Fixed::setClipLimit(double clipLimit)
{
    clipLimit_ = clipLimit;
}

double CLAHE_Impl_Fixed::getClipLimit() const
{
    return clipLimit_;
}

void CLAHE_Impl_Fixed::setTilesGridSize(cv::Size tileGridSize)
{
    tilesX_ = tileGridSize.width;
    tilesY_ = tileGridSize.height;
}

cv::Size CLAHE_Impl_Fixed::getTilesGridSize() const
{
    return cv::Size(tilesX_, tilesY_);
}

void CLAHE_Impl_Fixed::setBitShift(int bitShift)
{
    bitShift_ = bitShift;
}

int CLAHE_Impl_Fixed::getBitShift() const
{
    return bitShift_;
}

void CLAHE_Impl_Fixed::collectGarbage()
{
    srcExt_.release();
    lut_.release();
}


cv::Ptr<CLAHE_Fixed> createCLAHE_Fixed(double clipLimit, cv::Size tileGridSize)
{
    return cv::makePtr<CLAHE_Impl_Fixed>(clipLimit, tileGridSize.width, tileGridSize.height);
}


int linear_mapping_fixed(cv::InputArray input, cv::OutputArray output)
{
    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

	auto h = src.rows;
	auto w = src.cols;

    // Step1: 整数 min/max 扫描，无浮点
	uint16_t in_min = src.ptr<uint16_t>()[0];
    uint16_t in_max = src.ptr<uint16_t>()[0];

    for (int i = 1; i < h; ++i)
    {
        const auto* psrc = src.ptr<uint16_t>(i);
		for (int j = 0; j < w; ++j)
        {
            if (psrc[j] < in_min) in_min = psrc[j];
            if (psrc[j] > in_max) in_max = psrc[j];
        }
    }

    uint16_t range = in_max - in_min;
    if (range == 0) range = 1;

    // Step2: 预计算 scale，帧头一次性整数除法
    constexpr int      Q = 16;
    constexpr uint32_t OUT_MAX = 255;
	constexpr uint32_t ROUND = 1 << (Q - 1); // 四舍五入偏移
    uint32_t scale_fixed = ((OUT_MAX << Q) + range / 2) / range;

    // Step3: 逐像素映射
    cv::Mat  dst(src.size(), CV_8UC1);
    for (int i = 1; i < h; ++i)
    {
		const auto* psrc = src.ptr<uint16_t>(i);
		auto* pdst = dst.ptr<uint8_t>(i);
        for (int j = 0; j < w; ++j)
        {
            uint32_t val = cv::saturate_cast<uint32_t>(psrc[j]);
            uint32_t diff = val - cv::saturate_cast<uint32_t>(in_min);
			uint32_t res = (diff * scale_fixed + ROUND) >> Q;
            pdst[j] = cv::saturate_cast<uint8_t>(res);
        }
    }

    output.assign(dst);
    return 0;
}


