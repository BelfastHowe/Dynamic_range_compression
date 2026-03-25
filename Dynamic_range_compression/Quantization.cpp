#include "Quantization.h"
#include <atomic>

namespace fs = std::filesystem;



template <class T>
class CLAHE_CalcLut_Body_Float : public cv::ParallelLoopBody
{
public:
    CLAHE_CalcLut_Body_Float(const cv::Mat& src, const cv::Mat& lut, const cv::Size& tileSize, const int& tilesX, const int& clipLimit, const float& lutScale, const int& histSize, const int& shift) :
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
void CLAHE_CalcLut_Body_Float<T>::operator ()(const cv::Range& range) const
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
class CLAHE_Interpolation_Body_Float : public cv::ParallelLoopBody
{
public:
    CLAHE_Interpolation_Body_Float(const cv::Mat& src, const cv::Mat& dst, const cv::Mat& lut, const cv::Size& tileSize, const int& tilesX, const int& tilesY, const int& shift) :
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
void CLAHE_Interpolation_Body_Float<T>::operator ()(const cv::Range& range) const
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

class CLAHE_Impl_Float CV_FINAL : public CLAHE_Float
{
public:
    CLAHE_Impl_Float(double clipLimit = 40.0, int tilesX = 8, int tilesY = 8, int histSize = 0, int bitShift = 0);

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
    int histSize_;

    cv::Mat srcExt_;
    cv::Mat lut_;
};

CLAHE_Impl_Float::CLAHE_Impl_Float(double clipLimit, int tilesX, int tilesY, int histSize, int bitShift) :
    clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY), histSize_(histSize), bitShift_(bitShift)
{
}

void CLAHE_Impl_Float::apply(cv::InputArray _src, cv::OutputArray _dst)
{
    CV_Assert(_src.type() == CV_8UC1 || _src.type() == CV_16UC1);

    int histSize = histSize_ > 0 ? histSize_ : (_src.type() == CV_8UC1 ? (256 >> bitShift_) : (65536 >> bitShift_));
    //int histSize = _src.type() == CV_8UC1 ? (256 >> bitShift_) : (65536 >> bitShift_);

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
    const float lutScale = static_cast<float>(histSize - 1) / tileSizeTotal;

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
        calcLutBody = cv::makePtr<CLAHE_CalcLut_Body_Float<uchar> >(srcForLut, lut_, tileSize, tilesX_, clipLimit, lutScale, histSize, bitShift_);
    }
    else if (_src.type() == CV_16UC1)
    {
        calcLutBody = cv::makePtr<CLAHE_CalcLut_Body_Float<ushort> >(srcForLut, lut_, tileSize, tilesX_, clipLimit, lutScale, histSize, bitShift_);
    }
    else
        CV_Error(cv::Error::StsBadArg, "Unsupported type");

    cv::parallel_for_(cv::Range(0, tilesX_ * tilesY_), *calcLutBody);

    cv::Ptr<cv::ParallelLoopBody> interpolationBody;
    if (_src.type() == CV_8UC1)
    {
        interpolationBody = cv::makePtr<CLAHE_Interpolation_Body_Float<uchar> >(src, dst, lut_, tileSize, tilesX_, tilesY_, bitShift_);
    }
    else if (_src.type() == CV_16UC1)
    {
        interpolationBody = cv::makePtr<CLAHE_Interpolation_Body_Float<ushort> >(src, dst, lut_, tileSize, tilesX_, tilesY_, bitShift_);
    }

    cv::parallel_for_(cv::Range(0, src.rows), *interpolationBody);
}

void CLAHE_Impl_Float::setClipLimit(double clipLimit)
{
    clipLimit_ = clipLimit;
}

double CLAHE_Impl_Float::getClipLimit() const
{
    return clipLimit_;
}

void CLAHE_Impl_Float::setTilesGridSize(cv::Size tileGridSize)
{
    tilesX_ = tileGridSize.width;
    tilesY_ = tileGridSize.height;
}

cv::Size CLAHE_Impl_Float::getTilesGridSize() const
{
    return cv::Size(tilesX_, tilesY_);
}

void CLAHE_Impl_Float::setBitShift(int bitShift)
{
    bitShift_ = bitShift;
}

int CLAHE_Impl_Float::getBitShift() const
{
    return bitShift_;
}

void CLAHE_Impl_Float::collectGarbage()
{
    srcExt_.release();
    lut_.release();
}

cv::Ptr<CLAHE_Float> createCLAHE_Float(double clipLimit, cv::Size tileGridSize, int histSize)
{
    return cv::makePtr<CLAHE_Impl_Float>(clipLimit, tileGridSize.width, tileGridSize.height, histSize);
}

/*----------------------------------------------------------------------定点计算------------------------------------------------------------------------------------------*/


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
        //uint64_t inv_tw_fixed = (1ULL << W_BITS) / static_cast<uint64_t>(tileSize_.width);
        uint64_t inv_tw_fixed = ((1ULL << (W_BITS + 1)) / static_cast<uint64_t>(tileSize_.width) + 1) >> 1;

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

        result_min_.store(65535);
        result_max_.store(0);
    }

    uint16_t get_min() const { return result_min_.load(); }
    uint16_t get_max() const { return result_max_.load(); }

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

    mutable std::atomic<uint16_t> result_min_;
    mutable std::atomic<uint16_t> result_max_;
};

template <class T>
void CLAHE_Interpolation_Body_Fixed<T>::operator ()(const cv::Range& range) const
{
    uint16_t local_min = 65535;
    uint16_t local_max = 0;

    //uint64_t inv_th_fixed = (1ULL << W_BITS) / static_cast<uint64_t>(tileSize_.height);
    uint64_t inv_th_fixed = ((1ULL << (W_BITS + 1)) / static_cast<uint64_t>(tileSize_.height) + 1) >> 1;

    for (int y = range.start; y < range.end; ++y)
    {
        const T* srcRow = src_.ptr<T>(y);
        T* dstRow = dst_.ptr<T>(y);

        // --- 纵向坐标定点化计算 ---
        // tyf = y * inv_th - 0.5
        int64_t tyf = cv::saturate_cast<int64_t>(cv::saturate_cast<uint64_t>(y) * inv_th_fixed) - cv::saturate_cast<int64_t>(W_ROUND_OFFSET);

        // 算术右移提取 Tile 索引
        int ty1 = static_cast<int>(tyf >> W_BITS);
        int ty2 = ty1 + 1;

        // 缩放到 Q 权重
        uint64_t ya = cv::saturate_cast<uint64_t>(tyf - (cv::saturate_cast<int64_t>(ty1) << W_BITS));
        uint64_t ya1 = (1ULL << W_BITS) - ya;

        // 边界限制
        ty1 = std::max(ty1, 0);
        ty2 = std::min(ty2, tilesY_ - 1);

        const T* lutPlane1 = lut_.ptr<T>(ty1 * tilesX_);
        const T* lutPlane2 = lut_.ptr<T>(ty2 * tilesX_);

        for (int x = 0; x < src_.cols; ++x)
        {
            // 原始像素值
            int srcVal = srcRow[x] >> shift_;

            int ind1 = ind1_p[x] + srcVal;
            int ind2 = ind2_p[x] + srcVal;

            // --- 核心插值公式量化 (乘法累加) ---
            uint64_t r1_res = cv::saturate_cast<uint64_t>(lutPlane1[ind1]) * xa1_p[x] + cv::saturate_cast<uint64_t>(lutPlane1[ind2]) * xa_p[x];
            uint64_t r2_res = cv::saturate_cast<uint64_t>(lutPlane2[ind1]) * xa1_p[x] + cv::saturate_cast<uint64_t>(lutPlane2[ind2]) * xa_p[x];

            uint64_t res = (r1_res * ya1 + r2_res * ya + W_ROUND) >> (W_BITS * 2);

            // 第三级：还原回原始深度并截断
            dstRow[x] = cv::saturate_cast<T>(res) << shift_;

            uint16_t res_16bit = cv::saturate_cast<uint16_t>(res) << shift_;
            if (res_16bit < local_min) local_min = res_16bit;
            if (res_16bit > local_max) local_max = res_16bit;
        }
    }

    uint16_t cur_min = result_min_.load(std::memory_order_relaxed);
    while (local_min < cur_min && !result_min_.compare_exchange_weak(cur_min, local_min, std::memory_order_relaxed))
        ;
    uint16_t cur_max = result_max_.load(std::memory_order_relaxed);
    while (local_max > cur_max && !result_max_.compare_exchange_weak(cur_max, local_max, std::memory_order_relaxed))
        ;
}

class CLAHE_Impl_Fixed CV_FINAL : public CLAHE_Fixed
{
public:
    CLAHE_Impl_Fixed(int clipLimit = 40, int tilesX = 8, int tilesY = 8, int histSize = 0, int bitShift = 0);

    void apply(cv::InputArray src, cv::OutputArray dst) CV_OVERRIDE;

    void setClipLimit(int clipLimit) CV_OVERRIDE;
    int getClipLimit() const CV_OVERRIDE;

    void setTilesGridSize(cv::Size tileGridSize) CV_OVERRIDE;
    cv::Size getTilesGridSize() const CV_OVERRIDE;

    void setBitShift(int bitShift) CV_OVERRIDE;
    int getBitShift() const CV_OVERRIDE;

    void collectGarbage() CV_OVERRIDE;

    uint16_t getGlobalMin() const CV_OVERRIDE;
    uint16_t getGlobalMax() const CV_OVERRIDE;

private:
    int clipLimit_;
    int tilesX_;
    int tilesY_;
    int bitShift_;
	int histSize_;

    cv::Mat srcExt_;
    cv::Mat lut_;

    uint16_t global_min_ = UINT16_MAX;
    uint16_t global_max_ = 0;
};

CLAHE_Impl_Fixed::CLAHE_Impl_Fixed(int clipLimit, int tilesX, int tilesY, int histSize, int bitShift) :
    clipLimit_(clipLimit), tilesX_(tilesX), tilesY_(tilesY), histSize_(histSize), bitShift_(bitShift)
{
}

void CLAHE_Impl_Fixed::apply(cv::InputArray _src, cv::OutputArray _dst)
{
    CV_Assert(_src.type() == CV_8UC1 || _src.type() == CV_16UC1);

	int histSize = histSize_ > 0 ? histSize_ : (_src.type() == CV_8UC1 ? (256 >> bitShift_) : (65536 >> bitShift_));
    //int histSize = _src.type() == CV_8UC1 ? (256 >> bitShift_) : (65536 >> bitShift_);

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
    const uint64_t lutScale = ((cv::saturate_cast<uint64_t>(histSize - 1) << (Q + 1)) / cv::saturate_cast<uint64_t>(tileSizeTotal) + 1) >> 1;

    int clipLimit = 0;
    if (clipLimit_ > 0)
    {
        clipLimit = (((clipLimit_ * tileSizeTotal) << 1) / histSize + 1) >> 1;
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
            calcLutBody_cv = cv::makePtr<CLAHE_CalcLut_Body_Float<uchar> >(srcForLut, lut_cv, tileSize, tilesX_, clipLimit, lutScale_cv, histSize, bitShift_);
        }
        else if (_src.type() == CV_16UC1)
        {
            calcLutBody_cv = cv::makePtr<CLAHE_CalcLut_Body_Float<ushort> >(srcForLut, lut_cv, tileSize, tilesX_, clipLimit, lutScale_cv, histSize, bitShift_);
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
    cv::Mat lut_mask;
    cv::compare(diff_lut, 1, lut_mask, cv::CMP_GT); // CMP_GT 表示大于
    int lut_greater1 = cv::countNonZero(lut_mask);

    //cv::Ptr<cv::ParallelLoopBody> interpolationBody;
    if (_src.type() == CV_8UC1)
    {
        auto body = cv::makePtr<CLAHE_Interpolation_Body_Fixed<uchar> >(src, dst, lut_, tileSize, tilesX_, tilesY_, bitShift_);
        cv::parallel_for_(cv::Range(0, src.rows), *body);
        global_min_ = body->get_min();
        global_max_ = body->get_max();
    }
    else if (_src.type() == CV_16UC1)
    {
        auto body = cv::makePtr<CLAHE_Interpolation_Body_Fixed<ushort> >(src, dst, lut_, tileSize, tilesX_, tilesY_, bitShift_);
        cv::parallel_for_(cv::Range(0, src.rows), *body);
        global_min_ = body->get_min();
        global_max_ = body->get_max();
    }

    //cv::parallel_for_(cv::Range(0, src.rows), *interpolationBody);

    cv::Mat dst_cv(dst.size(), dst.type());
    {
        cv::Ptr<cv::ParallelLoopBody> interpolationBody_cv;
        if (_src.type() == CV_8UC1)
        {
            interpolationBody_cv = cv::makePtr<CLAHE_Interpolation_Body_Float<uchar> >(src, dst_cv, lut_cv, tileSize, tilesX_, tilesY_, bitShift_);
        }
        else if (_src.type() == CV_16UC1)
        {
            interpolationBody_cv = cv::makePtr<CLAHE_Interpolation_Body_Float<ushort> >(src, dst_cv, lut_cv, tileSize, tilesX_, tilesY_, bitShift_);
        }
        cv::parallel_for_(cv::Range(0, src.rows), *interpolationBody_cv);
    }

    cv::Mat diff_dst;
    cv::absdiff(dst, dst_cv, diff_dst);
    double minDstVal, maxDstVal;
    cv::minMaxLoc(diff_dst, &minDstVal, &maxDstVal);
    int nonzeroDst = cv::countNonZero(diff_dst);
    cv::Mat dst_mask;
    cv::compare(diff_dst, 1, dst_mask, cv::CMP_GT); // CMP_GT 表示大于
    int dst_greater1 = cv::countNonZero(dst_mask);

    diff_lut.release();
}

void CLAHE_Impl_Fixed::setClipLimit(int clipLimit)
{
    clipLimit_ = clipLimit;
}

int CLAHE_Impl_Fixed::getClipLimit() const
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

uint16_t CLAHE_Impl_Fixed::getGlobalMin() const
{
    return global_min_;
}

uint16_t CLAHE_Impl_Fixed::getGlobalMax() const
{
    return global_max_;
}


cv::Ptr<CLAHE_Fixed> createCLAHE_Fixed(int clipLimit, cv::Size tileGridSize, int histSize)
{
    return cv::makePtr<CLAHE_Impl_Fixed>(clipLimit, tileGridSize.width, tileGridSize.height, histSize);
}

/*==================================================================================================================================================*/

int linear_mapping_fixed(cv::InputArray input, cv::OutputArray output)
{
    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    auto h = src.rows;
    auto w = src.cols;

    // Step1: 整数 min/max 扫描，无浮点
    uint16_t in_min = src.ptr<uint16_t>()[0];
    uint16_t in_max = src.ptr<uint16_t>()[0];

    for (int i = 0; i < h; ++i)
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
    constexpr int      LQ = 16;
    constexpr uint32_t OUT_MAX = 255;
    constexpr uint32_t ROUND = 1 << (LQ - 1); // 四舍五入偏移
    //uint32_t scale_fixed = ((OUT_MAX << Q) + range / 2) / range;
    uint32_t scale_fixed = ((OUT_MAX << (LQ + 1)) / range + 1) >> 1;

    // Step3: 逐像素映射
    cv::Mat  dst(src.size(), CV_8UC1);
    for (int i = 0; i < h; ++i)
    {
        const auto* psrc = src.ptr<uint16_t>(i);
        auto* pdst = dst.ptr<uint8_t>(i);
        for (int j = 0; j < w; ++j)
        {
            uint32_t val = cv::saturate_cast<uint32_t>(psrc[j]);
            uint32_t diff = val - cv::saturate_cast<uint32_t>(in_min);
            uint32_t res = (diff * scale_fixed + ROUND) >> LQ;
            pdst[j] = cv::saturate_cast<uint8_t>(res);
        }
    }

    output.assign(dst);
    return 0;
}

class Gaussian_Blur_Fixed
{
public:
    enum class BorderType { REFLECT_101, REFLECT };

    Gaussian_Blur_Fixed(int sigma, int gaussQ, BorderType border = BorderType::REFLECT_101)
        : border_(border), gaussQ_(gaussQ)
    {
        build_kernel(sigma);
        gauss_Round_ = 1LL << (2 * gaussQ_ - 1);
        right_move_ = std::max(0, 16 + gaussQ_ - 31);
    }

    // sigma 变化时重新构建核
    void set_sigma(int sigma)
    {
        build_kernel(sigma);
    }

    // 输入输出均为 CV_16UC1
    int apply(cv::InputArray input, cv::OutputArray output) const
    {
		cv::Mat src = input.getMat();
        CV_Assert(src.type() == CV_16UC1);

        int rows = src.rows, cols = src.cols;

        // 水平方向
        cv::Mat tmp(src.size(), CV_32SC1);
        for (int r = 0; r < rows; ++r)
        {
            const auto* ps = src.ptr<uint16_t>(r);
            auto* pt = tmp.ptr<int32_t>(r);
            for (int c = 0; c < cols; ++c)
            {
                int64_t acc = 0;
                for (int k = 0; k < ksize_; ++k)
                {
                    int cc = border_idx(c + k - half_, cols);
                    acc += cv::saturate_cast<int64_t>(ps[cc]) * cv::saturate_cast<int64_t>(kernel_[k]);
                }
                pt[c] = cv::saturate_cast<int32_t>(acc >> right_move_);
            }
        }

        // 垂直方向
        output.create(src.size(), CV_16UC1);
		auto dst = output.getMat();
        for (int r = 0; r < rows; ++r)
        {
            auto* pd = dst.ptr<uint16_t>(r);
            for (int c = 0; c < cols; ++c)
            {
                int64_t acc = 0;
                for (int k = 0; k < ksize_; ++k)
                {
                    int rr = border_idx(r + k - half_, rows);
                    auto tmp_val = cv::saturate_cast<int64_t>(tmp.ptr<int32_t>(rr)[c]) << right_move_;
                    acc += tmp_val * cv::saturate_cast<int64_t>(kernel_[k]);
                }
                pd[c] = cv::saturate_cast<uint16_t>((acc + gauss_Round_) >> (gaussQ_ * 2));
            }
        }

        return 0;
    }

    int ksize() const { return ksize_; }
    int sigma() const { return sigma_; }

    const std::vector<int32_t>& kernel() const { return kernel_; }
	const std::vector<int32_t>& cv_kernel() const { return cv_kernel_; }

private:
    void build_kernel(int sigma)
    {
        sigma_ = sigma;
        ksize_ = std::max(3, cv::saturate_cast<int>(sigma * 8 + 1) | 1);
        half_ = ksize_ / 2;

        bool use_cv = true;
        if(use_cv)
        {
            cv::Mat kx = cv::getGaussianKernel(ksize_, sigma, CV_32F);
            cv_kernel_.resize(ksize_);
            int q_sum = 0;
            for (int i = 0; i < ksize_; ++i)
            {
                cv_kernel_[i] = cv::saturate_cast<int32_t>(kx.at<float>(i) * (1 << gaussQ_));
                q_sum += cv_kernel_[i];
            }
            cv_kernel_[half_] += ((1 << gaussQ_) - q_sum); // 修正舍入误差，保证归一化
		}

        std::vector<float> kf(ksize_);
        float sum = 0;
        for (int i = 0; i < ksize_; ++i)
        {
            float x = cv::saturate_cast<float>(i - half_);
            kf[i] = std::exp(-x * x / (2.0f * sigma * sigma));
            sum += kf[i];
        }

        kernel_.resize(ksize_);
        int q_sum = 0;
        for (int i = 0; i < ksize_; ++i)
        {
            kernel_[i] = cv::saturate_cast<int32_t>(kf[i] / sum * (1 << gaussQ_));
            q_sum += kernel_[i];
        }
        kernel_[half_] += ((1 << gaussQ_) - q_sum); // 修正舍入误差，保证归一化
    }

    inline int border_idx(int i, int n) const
    {
        if (border_ == BorderType::REFLECT_101)
        {
            while (i < 0 || i >= n)
            {
                if (i < 0)  i = -i;
                if (i >= n) i = 2 * (n - 1) - i;
            }
        }
        else
        {
            while (i < 0 || i >= n)
            {
                if (i < 0)  i = -i - 1;
                if (i >= n) i = 2 * n - i - 1;
            }
        }
        return i;
    }

    int                 sigma_;
	int                 gaussQ_;
	int64_t             gauss_Round_;
	int                 right_move_;
    int                 ksize_;
    int                 half_;
    BorderType          border_;
    std::vector<int32_t> kernel_;
	std::vector<int32_t> cv_kernel_;
};

class SSR_Fixed
{
public:
    SSR_Fixed(int logQ = 16, int linearQ = 16, cv::Ptr<Gaussian_Blur_Fixed> gauss_ptr = cv::makePtr<Gaussian_Blur_Fixed>(50, 16)) : LOG_Q_(logQ), LINEAR_Q_(linearQ), gauss_ptr_(gauss_ptr)
    {
        // 预计算 log LUT
        log_lut_.create(1, 16384, CV_32SC1);
        auto* ptr = log_lut_.ptr<int>(0);
        for (int i = 0; i < 16384; ++i)
        {
            double log_norm = 1.0 + cv::saturate_cast<double>(i) / 16384.0;
            double log_val = std::log(log_norm);

            int fixed_log = cv::saturate_cast<int>(log_val * cv::saturate_cast<double>(1 << LOG_Q_));
            ptr[i] = fixed_log;
        }
    }

    int apply(cv::InputArray input, cv::OutputArray output);

private:
    cv::Mat log_lut_;
    int LOG_Q_;
    //int GAUSS_Q_;
    //int sigma_;
    cv::Ptr<Gaussian_Blur_Fixed> gauss_ptr_;
    int LINEAR_Q_;
};

int SSR_Fixed::apply(cv::InputArray input, cv::OutputArray output)
{
    cv::Mat src = input.getMat();
    CV_CheckTypeEQ(src.type(), CV_16UC1, "");

    // 1. 高斯模糊（定点实现）
    cv::Mat gauss;
    //cv::GaussianBlur(src, gauss, cv::Size(0, 0), gauss_ptr_->sigma());
	gauss_ptr_->apply(src, gauss);

    // 2. 对数变换（定点实现）
    cv::Mat log_reflectance(src.size(), CV_32SC1);
	int min_ref = INT_MAX;
	int max_ref = INT_MIN;
    const auto* ptr_lut = log_lut_.ptr<int>(0);
    for (int i = 0; i < src.rows; ++i)
    {
        const auto* psrc = src.ptr<uint16_t>(i);
        const auto* pgauss = gauss.ptr<uint16_t>(i);
        auto* plog = log_reflectance.ptr<int>(i);
        for (int j = 0; j < src.cols; ++j)
        {
			int src_val = cv::saturate_cast<int>(psrc[j]);
			int gauss_val = cv::saturate_cast<int>(pgauss[j]);
            int res = ptr_lut[src_val] - ptr_lut[gauss_val];
			plog[j] = res;

			if (res < min_ref) min_ref = res;
			if (res > max_ref) max_ref = res;
        }
    }

	int range = max_ref - min_ref;
    if (range == 0) range = 1;

    constexpr int OUT_MAX = 255;
    int ROUND = 1 << (LINEAR_Q_ - 1); // 四舍五入偏移
    int scale_fixed = ((OUT_MAX << (LINEAR_Q_ + 1)) / range + 1) >> 1;

    // Step3: 逐像素映射
    cv::Mat  dst(src.size(), CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < src.rows; ++i)
    {
        const auto* plog = log_reflectance.ptr<int>(i);
        auto* pdst = dst.ptr<uint8_t>(i);
        for (int j = 0; j < src.cols; ++j)
        {
            int val = plog[j];
            int diff = val - min_ref;
            int res = (diff * scale_fixed + ROUND) >> LINEAR_Q_;
            pdst[j] = cv::saturate_cast<uint8_t>(res);
        }
    }

    output.assign(dst);
    return 0;
}


/*========================================================================误差计算==============================================================================*/

struct PrecisionReport
{
    double psnr = 0.0;          // 峰值信噪比，>40dB 可接受，>60dB 优秀
    double mae = 0.0;           // 平均绝对误差
    int    max_abs_err = 0;   // 最大绝对误差  ≤ 1LSB  理想情况，仅定点舍入误差
    int    over_1lsb = 0;     // 误差 >1LSB 的像素数
    double over_1lsb_pct = 0.0; // 占比  < 0.1%  可接受
};

PrecisionReport test_precision_14to8(
    cv::InputArray input,     // 原始 14bit 输入
    int clipLimit,
    cv::Size tileSize,
    cv::Ptr<SSR_Fixed> ssr)
{
    cv::Mat img14bit = input.getMat();
    CV_CheckTypeEQ(img14bit.type(), CV_16UC1, "");

    // 浮点参考版本
    cv::Mat ref_out;
    clahe_mapping(img14bit, ref_out, cv::saturate_cast<double>(clipLimit), tileSize);
    //single_scale_retinex(img14bit, ref_out, 50);

    // 定点量化版本
    cv::Mat fixed_out;
    clahe_fixed_mapping(img14bit, fixed_out, clipLimit, tileSize);
    //cv::Ptr<SSR_Fixed> ssr = cv::makePtr<SSR_Fixed>(16, 24, 16, 50);
	//ssr->apply(img14bit, fixed_out);

    //8bit
    CV_Assert(ref_out.size() == fixed_out.size());
    CV_Assert(ref_out.type() == fixed_out.type());

    // 逐像素比对
    const int n = cv::saturate_cast<int>(img14bit.total());

    PrecisionReport rpt{};
    double sse = 0;
    double sum_err = 0;
    const double MAX_VAL = 255.0; // CLAHE 输出归一化到 8bit 等效范围

    cv::Mat residual_map(ref_out.size(), CV_8UC1);

    for (int r = 0; r < ref_out.rows; ++r)
    {
        const auto* pref = ref_out.ptr<uint8_t>(r);
        const auto* pfixed = fixed_out.ptr<uint8_t>(r);
        uint8_t* pres = residual_map.ptr<uint8_t>(r);
        for (int c = 0; c < ref_out.cols; ++c)
        {
            int err = std::abs(cv::saturate_cast<int>(pref[c]) - cv::saturate_cast<int>(pfixed[c]));
            rpt.max_abs_err = std::max(rpt.max_abs_err, err);

            if (err > 1) rpt.over_1lsb++;

            sum_err += err;
            sse += cv::saturate_cast<double>(err) * cv::saturate_cast<double>(err);

            pres[c] = cv::saturate_cast<uint8_t>(err);
        }
    }

    double mse = sse / n;
    rpt.mae = sum_err / n;
    rpt.over_1lsb_pct = 100.0 * rpt.over_1lsb / n;
    rpt.psnr = (mse > 0)
        ? 10.0 * std::log10(MAX_VAL * MAX_VAL / mse)
        : std::numeric_limits<double>::infinity();

    // 打印报告
    printf("=== CLAHE 量化精度报告 ===\n");
    printf("图像尺寸     : %dx%d\n", img14bit.cols, img14bit.rows);
    printf("clipLimit    : %d\n", clipLimit);
    printf("tileSize     : %dx%d\n", tileSize.width, tileSize.height);
    printf("PSNR         : %.2f dB\n", rpt.psnr);
    printf("MAE          : %.4f LSB\n", rpt.mae);
    printf("max_abs_err  : %d LSB\n", rpt.max_abs_err);
    printf(">1LSB 像素   : %d / %d (%.3f%%)\n",
        rpt.over_1lsb, n, rpt.over_1lsb_pct);

    cv::Mat residual_norm;
    cv::normalize(residual_map, residual_norm, 0, 255, cv::NORM_MINMAX);
    imwrite_mdy_private(residual_norm, "residual_map.png");

    return rpt;
}

int test_precision_batch_14to8()
{
    int clipLimit = 3;
    cv::Size tileSize = cv::Size(8, 8);

    fs::path inputDir(IMAGE_DIR);

    if (!fs::exists(inputDir))
    {
        std::cerr << "Input directory not found: " << IMAGE_DIR << std::endl;
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

    PrecisionReport avg{};
    double sum_psnr = 0;
    double sum_mae = 0;
    int    sum_max_abs_err = 0;
    int    sum_over_1lsb = 0;
    double sum_over_1lsb_pct = 0;

    auto gauss_ptr = cv::makePtr<Gaussian_Blur_Fixed>(50, 16);
	auto ssr_ptr = cv::makePtr<SSR_Fixed>(16, 16, gauss_ptr);

    for (int i = 0; i < images.size(); ++i)
    {
        printf("--- 第 %d / %zu 张 ---\n", i + 1, images.size());
        PrecisionReport rpt = test_precision_14to8(images[i], clipLimit, tileSize, ssr_ptr);

        sum_psnr += rpt.psnr;
        sum_mae += rpt.mae;
        sum_max_abs_err = std::max(sum_max_abs_err, rpt.max_abs_err); // 取最差值
        sum_over_1lsb += rpt.over_1lsb;
        sum_over_1lsb_pct += rpt.over_1lsb_pct;
    }

    const int m = static_cast<int>(images.size());
    avg.psnr = sum_psnr / m;
    avg.mae = sum_mae / m;
    avg.max_abs_err = sum_max_abs_err;   // 整批最大误差
    avg.over_1lsb = sum_over_1lsb / m; // 平均每张超标像素数
    avg.over_1lsb_pct = sum_over_1lsb_pct / m;

    printf("=== 批量测试汇总 ( %d 张 ) ===\n", m);
    printf("平均 PSNR        : %.2f dB\n", avg.psnr);
    printf("平均 MAE         : %.4f LSB\n", avg.mae);
    printf("全局 max_abs_err : %d LSB\n", avg.max_abs_err);
    printf("平均 >1LSB 像素  : %d (%.3f%%)\n", avg.over_1lsb, avg.over_1lsb_pct);

    return 0;
}


