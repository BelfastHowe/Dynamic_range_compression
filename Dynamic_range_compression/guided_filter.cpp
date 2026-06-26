#include "guided_filter.h"



#if CV_SSE
namespace
{

    inline bool CPU_SUPPORT_SSE1()
    {
        static const bool is_supported = cv::checkHardwareSupport(CV_CPU_SSE);
        return is_supported;
    }

}  // end
#endif

inline float getFloatSignBit()
{
    union
    {
        int signInt;
        float signFloat;
    };
    signInt = 0x80000000;

    return signFloat;
}

void mul(float* dst, float* src1, float* src2, int w)
{
    int j = 0;
#if CV_SSE
    if (CPU_SUPPORT_SSE1())
    {
        __m128 a, b;
        for (; j < w - 3; j += 4)
        {
            a = _mm_loadu_ps(src1 + j);
            b = _mm_loadu_ps(src2 + j);
            b = _mm_mul_ps(a, b);
            _mm_storeu_ps(dst + j, b);
        }
    }
#endif
    for (; j < w; j++)
        dst[j] = src1[j] * src2[j];
}

// 计算 dst = dst - src1 * src2
void sub_mul(float* dst, float* src1, float* src2, int w)
{
    int j = 0;
#if CV_SSE
    if (CPU_SUPPORT_SSE1())
    {
        __m128 a, b, c;
        for (; j < w - 3; j += 4)
        {
            a = _mm_loadu_ps(src1 + j);
            b = _mm_loadu_ps(src2 + j);
            b = _mm_mul_ps(b, a);
            c = _mm_loadu_ps(dst + j);
            c = _mm_sub_ps(c, b);
            _mm_storeu_ps(dst + j, c);
        }
    }
#endif
    for (; j < w; j++)
        dst[j] -= src1[j] * src2[j];
}

// 计算 dst = dst - src1 * src2 - c0
void sub_mad(float* dst, float* src1, float* src2, float c0, int w)
{
    int j = 0;
#if CV_SSE
    if (CPU_SUPPORT_SSE1())
    {
        __m128 a, b, c;
        __m128 cnst = _mm_set_ps1(c0);
        for (; j < w - 3; j += 4)
        {
            a = _mm_loadu_ps(src1 + j);
            b = _mm_loadu_ps(src2 + j);
            b = _mm_mul_ps(b, a);
            c = _mm_loadu_ps(dst + j);
            c = _mm_sub_ps(c, cnst);
            c = _mm_sub_ps(c, b);
            _mm_storeu_ps(dst + j, c);
        }
    }
#endif
    for (; j < w; j++)
        dst[j] -= src1[j] * src2[j] + c0;
}

// 计算 2x2 矩阵的行列式，矩阵元素分别为 a00, a01, a10, a11，结果存储在 dst 中
void det_2x2(float* dst, float* a00, float* a01, float* a10, float* a11, int w)
{
    int j = 0;
#if CV_SSE
    if (CPU_SUPPORT_SSE1())
    {
        __m128 a, b;
        for (; j < w - 3; j += 4)
        {
            a = _mm_mul_ps(_mm_loadu_ps(a00 + j), _mm_loadu_ps(a11 + j));
            b = _mm_mul_ps(_mm_loadu_ps(a01 + j), _mm_loadu_ps(a10 + j));
            a = _mm_sub_ps(a, b);
            _mm_storeu_ps(dst + j, a);
        }
    }
#endif
    for (; j < w; j++)
        dst[j] = a00[j] * a11[j] - a01[j] * a10[j];
}

// 计算 dst = dst + src1 * src2
void add_mul(float* dst, float* src1, float* src2, int w)
{
    int j = 0;
#if CV_SSE
    if (CPU_SUPPORT_SSE1())
    {
        __m128 a, b, c;
        for (; j < w - 3; j += 4)
        {
            a = _mm_loadu_ps(src1 + j);
            b = _mm_loadu_ps(src2 + j);
            b = _mm_mul_ps(b, a);
            c = _mm_loadu_ps(dst + j);
            c = _mm_add_ps(c, b);
            _mm_storeu_ps(dst + j, c);
        }
    }
#endif
    for (; j < w; j++)
    {
        dst[j] += src1[j] * src2[j];
    }
}

// 计算 a1 = a1 / b1，逐元素除法
void div_1x(float* a1, float* b1, int w)
{
    int j = 0;
#if CV_SSE
    if (CPU_SUPPORT_SSE1())
    {
        __m128 _a1, _b1;
        for (; j < w - 3; j += 4)
        {
            _b1 = _mm_loadu_ps(b1 + j);
            _a1 = _mm_loadu_ps(a1 + j);
            _mm_storeu_ps(a1 + j, _mm_div_ps(_a1, _b1));
        }
    }
#endif
    for (; j < w; j++)
    {
        a1[j] /= b1[j];
    }
}

// 计算 2x2 矩阵的逆矩阵，矩阵元素分别为 a00, a01, a11，结果存储在原地（a00, a01, a11）中
void div_det_2x2(float* a00, float* a01, float* a11, int w)
{
    int j = 0;
#if CV_SSE
    if (CPU_SUPPORT_SSE1())
    {
        const __m128 SIGN_MASK = _mm_set_ps1(getFloatSignBit());

        __m128 a, b, _a00, _a01, _a11;
        for (; j < w - 3; j += 4)
        {
            _a00 = _mm_loadu_ps(a00 + j);
            _a11 = _mm_loadu_ps(a11 + j);
            a = _mm_mul_ps(_a00, _a11);

            _a01 = _mm_loadu_ps(a01 + j);
            _a01 = _mm_xor_ps(_a01, SIGN_MASK);
            b = _mm_mul_ps(_a01, _a01);

            a = _mm_sub_ps(a, b);

            _a01 = _mm_div_ps(_a01, a);
            _a00 = _mm_div_ps(_a00, a);
            _a11 = _mm_div_ps(_a11, a);

            _mm_storeu_ps(a01 + j, _a01);
            _mm_storeu_ps(a00 + j, _a00);
            _mm_storeu_ps(a11 + j, _a11);
        }
    }
#endif
    for (; j < w; j++)
    {
        float det = a00[j] * a11[j] - a01[j] * a01[j];
        a00[j] /= det;
        a11[j] /= det;
        a01[j] /= -det;
    }
}

// 计算 src 中每个元素的倒数，并将结果存储回 src 中
void inv_self(float* src, int w)
{
    int j = 0;
#if CV_SSE
    if (CPU_SUPPORT_SSE1())
    {
        __m128 a;
        for (; j < w - 3; j += 4)
        {
            a = _mm_rcp_ps(_mm_loadu_ps(src + j));
            _mm_storeu_ps(src + j, a);
        }
    }
#endif
    for (; j < w; j++)
    {
        src[j] = 1.0f / src[j];
    }
}

template <typename T>
struct SymArray2D
{
    std::vector<T> vec;
    int sz;

    SymArray2D()
    {
        sz = 0;
    }

    void create(int sz_)
    {
        CV_DbgAssert(sz_ > 0);
        sz = sz_;
        vec.resize(total());
    }

    inline T& operator()(int i, int j)
    {
        CV_DbgAssert(i >= 0 && i < sz && j >= 0 && j < sz);
        if (i < j) std::swap(i, j);
        return vec[i * (i + 1) / 2 + j];
    }

    inline T& operator()(int i)
    {
        return vec[i];
    }

    int total() const
    {
        return sz * (sz + 1) / 2;
    }

    void release()
    {
        vec.clear();
        sz = 0;
    }
};

// 检查输入的一组图像（或单张图像）是否都具有完全相同的尺寸（Size）和数据类型深度（Depth），
// 如果一致，则提取并输出这个统一的尺寸和深度。 
// 如果有任何不一致或者输入类型不支持，程序会直接中断并报错（通过 CV_Assert）
void checkSameSizeAndDepth(cv::InputArrayOfArrays src, cv::Size& sz, int& depth)
{
    CV_Assert(src.isMat() || src.isUMat() || src.isMatVector() || src.isUMatVector());

    if (src.isMat() || src.isUMat())
    {
        CV_Assert(!src.empty());
        sz = src.size();
        depth = src.depth();
    }
    else if (src.isMatVector())
    {
        const std::vector<cv::Mat>& srcv = *static_cast<const std::vector<cv::Mat>*>(src.getObj());
        CV_Assert(srcv.size() > 0);
        for (unsigned i = 0; i < srcv.size(); i++)
        {
            CV_Assert(srcv[i].depth() == srcv[0].depth());
            CV_Assert(srcv[i].size() == srcv[0].size());
        }
        sz = srcv[0].size();
        depth = srcv[0].depth();
    }
    else if (src.isUMatVector())
    {
        const std::vector<cv::UMat>& srcv = *static_cast<const std::vector<cv::UMat>*>(src.getObj());
        CV_Assert(srcv.size() > 0);
        for (unsigned i = 0; i < srcv.size(); i++)
        {
            CV_Assert(srcv[i].depth() == srcv[0].depth());
            CV_Assert(srcv[i].size() == srcv[0].size());
        }
        sz = srcv[0].size();
        depth = srcv[0].depth();
    }
}

// 计算并返回输入数据中所包含的总通道数量（Total Channels）。
// 无论输入的是单张图像（如 RGB 3通道、灰度 1通道），还是多张图像的组合（容器），都能智能地将所有图像的通道数累加起来并返回。
int getTotalNumberOfChannels(cv::InputArrayOfArrays src)
{
    CV_Assert(src.isMat() || src.isUMat() || src.isMatVector() || src.isUMatVector());

    if (src.isMat() || src.isUMat())
    {
        return src.channels();
    }
    else if (src.isMatVector())
    {
        int cnNum = 0;
        const std::vector<cv::Mat>& srcv = *static_cast<const std::vector<cv::Mat>*>(src.getObj());
        for (unsigned i = 0; i < srcv.size(); i++)
            cnNum += srcv[i].channels();
        return cnNum;
    }
    else if (src.isUMatVector())
    {
        int cnNum = 0;
        const std::vector<cv::UMat>& srcv = *static_cast<const std::vector<cv::UMat>*>(src.getObj());
        for (unsigned i = 0; i < srcv.size(); i++)
            cnNum += srcv[i].channels();
        return cnNum;
    }
    else
    {
        return 0;
    }
}

// 从输入的图像（或图像集合）中，提取前 N 个通道（数量由 maxDstCn 决定），并将它们解包分离到一个单通道图像数组 dst 中
template <typename XMat>
static void splitFirstNChannels(cv::InputArrayOfArrays src, std::vector<XMat>& dst, int maxDstCn)
{
    CV_Assert(src.isMat() || src.isUMat() || src.isMatVector() || src.isUMatVector());

    if ((src.isMat() || src.isUMat()) && src.channels() == maxDstCn)
    {
        cv::split(src, dst);
    }
    else
    {
        cv::Size sz;
        int depth, totalCnNum;

        checkSameSizeAndDepth(src, sz, depth);
        totalCnNum = std::min(maxDstCn, getTotalNumberOfChannels(src));

        dst.resize(totalCnNum);
        std::vector<int> fromTo(2 * totalCnNum);
        for (int i = 0; i < totalCnNum; i++)
        {
            fromTo[i * 2 + 0] = i;
            fromTo[i * 2 + 1] = i;

            dst[i].create(sz, CV_MAKE_TYPE(depth, 1));
        }

        cv::mixChannels(src, dst, fromTo);
    }
}


class GuidedFilterImpl : public GuidedFilter
{
public:

    static cv::Ptr<GuidedFilterImpl> create(cv::InputArray guide, int radius, double eps, double scale);

    void filter(cv::InputArray src, cv::OutputArray dst, int dDepth = -1) CV_OVERRIDE;

protected:

    int radius;
    double eps;
    double scale;
    int h, w;
    int hOriginal, wOriginal;

    std::vector<cv::Mat> guideCn;
    std::vector<cv::Mat> guideCnMean;
    std::vector<cv::Mat> guideCnOriginal;

    SymArray2D<cv::Mat> covarsInv;

    int gCnNum;

protected:

    GuidedFilterImpl() {}

    void init(cv::InputArray guide, int radius, double eps, double scale);

    void computeCovGuide(SymArray2D<cv::Mat>& covars);

    void computeCovGuideAndSrc(std::vector<cv::Mat>& srcCn, std::vector<cv::Mat>& srcCnMean, std::vector<std::vector<cv::Mat> >& cov);

    void getWalkPattern(int eid, int& cn1, int& cn2);

    inline void meanFilter(cv::Mat& src, cv::Mat& dst)
    {
        boxFilter(src, dst, CV_32F, cv::Size(2 * radius + 1, 2 * radius + 1), cv::Point(-1, -1), true, cv::BORDER_REFLECT);
    }

    inline void convertToWorkType(cv::Mat& src, cv::Mat& dst)
    {
        src.convertTo(dst, CV_32F);
    }

    inline void subsample(cv::Mat& src, cv::Mat& dst)
    {
        resize(src, dst, cv::Size(w, h), 0, 0, cv::INTER_LINEAR);
    }

    inline void upsample(cv::Mat& src, cv::Mat& dst)
    {
        resize(src, dst, cv::Size(wOriginal, hOriginal), 0, 0, cv::INTER_LINEAR);
    }

private: /*Routines to parallelize boxFilter and convertTo*/

    typedef void (GuidedFilterImpl::* TransformFunc)(cv::Mat& src, cv::Mat& dst);

    struct GFTransform_ParBody : public cv::ParallelLoopBody
    {
		// 通用并行任务分发器 并行处理所有“单输入、单输出”的图像矩阵集合，真正的处理逻辑由 func 指向的成员函数完成
        GuidedFilterImpl& gf;
        mutable std::vector<cv::Mat*> src;
        mutable std::vector<cv::Mat*> dst;
        TransformFunc func;

        GFTransform_ParBody(GuidedFilterImpl& gf_, std::vector<cv::Mat>& srcv, std::vector<cv::Mat>& dstv, TransformFunc func_);
        GFTransform_ParBody(GuidedFilterImpl& gf_, std::vector<std::vector<cv::Mat> >& srcvv, std::vector<std::vector<cv::Mat> >& dstvv, TransformFunc func_);

        void operator () (const cv::Range& range) const CV_OVERRIDE;

        cv::Range getRange() const
        {
            return cv::Range(0, (int)src.size());
        }
    };

	// 多线程处理输入图像集合 src，将每个图像转换为工作类型（CV_32F），并将结果存储到 dst 中
    template<typename V>
    void parConvertToWorkType(V& src, V& dst)
    {
        GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::convertToWorkType);
        parallel_for_(pb.getRange(), pb);
    }

	// 镜面反射边界条件下的均值滤波，处理输入图像集合 src，将结果存储到 dst 中
    template<typename V>
    void parMeanFilter(V& src, V& dst)
    {
        GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::meanFilter);
        parallel_for_(pb.getRange(), pb);
    }

	// 多线程处理输入图像集合 src，将每个图像下采样到w、h（缩小尺寸），并将结果存储到 dst 中
    template<typename V>
    void parSubsample(V& src, V& dst)
    {
        GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::subsample);
        parallel_for_(pb.getRange(), pb);
    }

	// 多线程处理输入图像集合 src，将每个图像上采样到 wOriginal、hOriginal（放大尺寸），并将结果存储到 dst 中
    template<typename V>
    void parUpsample(V& src, V& dst)
    {
        GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::upsample);
        parallel_for_(pb.getRange(), pb);
    }

private: /*Parallel body classes*/

    inline void runParBody(const cv::ParallelLoopBody& pb)
    {
        parallel_for_(cv::Range(0, h), pb);
    }

    struct MulChannelsGuide_ParBody : public cv::ParallelLoopBody
    {
		// 计算引导图各个通道互相点乘的结果，存储在 covars 中
        GuidedFilterImpl& gf;
        SymArray2D<cv::Mat>& covars;

        MulChannelsGuide_ParBody(GuidedFilterImpl& gf_, SymArray2D<cv::Mat>& covars_)
            : gf(gf_), covars(covars_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ComputeCovGuideFromChannelsMul_ParBody : public cv::ParallelLoopBody
    {
		// 计算引导图像（Guide Image）在每个像素邻域上的协方差矩阵。
        // 方差 + eps
        GuidedFilterImpl& gf;
        SymArray2D<cv::Mat>& covars;

        ComputeCovGuideFromChannelsMul_ParBody(GuidedFilterImpl& gf_, SymArray2D<cv::Mat>& covars_)
            : gf(gf_), covars(covars_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ComputeCovGuideInv_ParBody : public cv::ParallelLoopBody
    {
        // 计算引导图像（Guide Image）在每个像素邻域上的协方差矩阵的逆矩阵。
        GuidedFilterImpl& gf;
        SymArray2D<cv::Mat>& covars;

        ComputeCovGuideInv_ParBody(GuidedFilterImpl& gf_, SymArray2D<cv::Mat>& covars_);

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };


    struct MulChannelsGuideAndSrc_ParBody : public cv::ParallelLoopBody
    {
        // 计算dstLine[j] = srcLine[j] * guideLine[j]
        GuidedFilterImpl& gf;
        std::vector<std::vector<cv::Mat> >& cov;
        std::vector<cv::Mat>& srcCn;

        MulChannelsGuideAndSrc_ParBody(GuidedFilterImpl& gf_, std::vector<cv::Mat>& srcCn_, std::vector<std::vector<cv::Mat> >& cov_)
            : gf(gf_), cov(cov_), srcCn(srcCn_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ComputeCovFromSrcChannelsMul_ParBody : public cv::ParallelLoopBody
    {
        // 减去均值的乘积（- E[I] * E[p]）
        GuidedFilterImpl& gf;
        std::vector<std::vector<cv::Mat> >& cov;
        std::vector<cv::Mat>& srcCnMean;

        ComputeCovFromSrcChannelsMul_ParBody(GuidedFilterImpl& gf_, std::vector<cv::Mat>& srcCnMean_, std::vector<std::vector<cv::Mat> >& cov_)
            : gf(gf_), cov(cov_), srcCnMean(srcCnMean_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ComputeAlpha_ParBody : public cv::ParallelLoopBody
    {
		// 计算线性变换系数 alpha，alpha = covSrc * covarsInv
        GuidedFilterImpl& gf;
        std::vector<std::vector<cv::Mat> >& alpha;
        std::vector<std::vector<cv::Mat> >& covSrc;

        ComputeAlpha_ParBody(GuidedFilterImpl& gf_, std::vector<std::vector<cv::Mat> >& alpha_, std::vector<std::vector<cv::Mat> >& covSrc_)
            : gf(gf_), alpha(alpha_), covSrc(covSrc_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ComputeBeta_ParBody : public cv::ParallelLoopBody
    {
        // 计算线性变换系数 beta，beta = srcCnMean - sum(alpha * guideCnMean)
        GuidedFilterImpl& gf;
        std::vector<std::vector<cv::Mat> >& alpha;
        std::vector<cv::Mat>& srcCnMean;
        std::vector<cv::Mat>& beta;

        ComputeBeta_ParBody(GuidedFilterImpl& gf_, std::vector<std::vector<cv::Mat> >& alpha_, std::vector<cv::Mat>& srcCnMean_, std::vector<cv::Mat>& beta_)
            : gf(gf_), alpha(alpha_), srcCnMean(srcCnMean_), beta(beta_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ApplyTransform_ParBody : public cv::ParallelLoopBody
    {
        GuidedFilterImpl& gf;
        std::vector<std::vector<cv::Mat> >& alpha;
        std::vector<cv::Mat>& beta;

        ApplyTransform_ParBody(GuidedFilterImpl& gf_, std::vector<std::vector<cv::Mat> >& alpha_, std::vector<cv::Mat>& beta_)
            : gf(gf_), alpha(alpha_), beta(beta_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };
};

void GuidedFilterImpl::MulChannelsGuide_ParBody::operator()(const cv::Range& range) const
{
    int total = covars.total();

    for (int i = range.start; i < range.end; i++)
    {
        int c1, c2;
        float* cov, * guide1, * guide2;

        for (int k = 0; k < total; k++)
        {
            gf.getWalkPattern(k, c1, c2);

            guide1 = gf.guideCn[c1].ptr<float>(i);
            guide2 = gf.guideCn[c2].ptr<float>(i);
            cov = covars(c1, c2).ptr<float>(i);

            mul(cov, guide1, guide2, gf.w);
        }
    }
}

void GuidedFilterImpl::ComputeCovGuideFromChannelsMul_ParBody::operator()(const cv::Range& range) const
{
    int total = covars.total();
    float diagSummand = (float)(gf.eps);

    for (int i = range.start; i < range.end; i++)
    {
        int c1, c2;
        float* cov, * guide1, * guide2;

        for (int k = 0; k < total; k++)
        {
            gf.getWalkPattern(k, c1, c2);

            guide1 = gf.guideCnMean[c1].ptr<float>(i);
            guide2 = gf.guideCnMean[c2].ptr<float>(i);
            cov = covars(c1, c2).ptr<float>(i);

            if (c1 != c2)
            {
                sub_mul(cov, guide1, guide2, gf.w);
            }
            else
            {
                sub_mad(cov, guide1, guide2, -diagSummand, gf.w);
            }
        }
    }
}

GuidedFilterImpl::ComputeCovGuideInv_ParBody::ComputeCovGuideInv_ParBody(GuidedFilterImpl& gf_, SymArray2D<cv::Mat>& covars_)
    : gf(gf_), covars(covars_)
{
    gf.covarsInv.create(gf.gCnNum);

    if (gf.gCnNum == 3)
    {
        for (int k = 0; k < 2; k++)
            for (int l = 0; l < 3; l++)
                gf.covarsInv(k, l).create(gf.h, gf.w, CV_32FC1);

        ////trick to avoid memory allocation  避免内存分配的技巧
        gf.covarsInv(2, 0).create(gf.h, gf.w, CV_32FC1);
        gf.covarsInv(2, 1) = covars(2, 1);
        gf.covarsInv(2, 2) = covars(2, 2);

        return;
    }

    if (gf.gCnNum == 2)
    {
        gf.covarsInv(0, 0) = covars(1, 1);
        gf.covarsInv(0, 1) = covars(0, 1);
        gf.covarsInv(1, 1) = covars(0, 0);
        return;
    }

    if (gf.gCnNum == 1)
    {
        gf.covarsInv(0, 0) = covars(0, 0);
        return;
    }
}

void GuidedFilterImpl::ComputeCovGuideInv_ParBody::operator()(const cv::Range& range) const
{
    if (gf.gCnNum == 3)
    {
        std::vector<float> covarsDet(gf.w);
        float* det = &covarsDet[0];

        for (int i = range.start; i < range.end; i++)
        {
            for (int k = 0; k < 3; k++)
                for (int l = 0; l <= k; l++)
                {
                    float* dst = gf.covarsInv(k, l).ptr<float>(i);

                    float* a00 = covars((k + 1) % 3, (l + 1) % 3).ptr<float>(i);
                    float* a01 = covars((k + 1) % 3, (l + 2) % 3).ptr<float>(i);
                    float* a10 = covars((k + 2) % 3, (l + 1) % 3).ptr<float>(i);
                    float* a11 = covars((k + 2) % 3, (l + 2) % 3).ptr<float>(i);

                    det_2x2(dst, a00, a01, a10, a11, gf.w);
                }

            for (int k = 0; k < 3; k++)
            {
                float* a = covars(k, 0).ptr<float>(i);
                float* ac = gf.covarsInv(k, 0).ptr<float>(i);

                if (k == 0)
                    mul(det, a, ac, gf.w);
                else
                    add_mul(det, a, ac, gf.w);
            }

            if (gf.eps < 1e-2)
            {
                for (int j = 0; j < gf.w; j++)
                    if (abs(det[j]) < 1e-6f)
                        det[j] = 1.f;
            }

            for (int k = 0; k < gf.covarsInv.total(); k += 1)
            {
                div_1x(gf.covarsInv(k).ptr<float>(i), det, gf.w);
            }
        }
        return;
    }

    if (gf.gCnNum == 2)
    {
        for (int i = range.start; i < range.end; i++)
        {
            float* a00 = gf.covarsInv(0, 0).ptr<float>(i);
            float* a10 = gf.covarsInv(1, 0).ptr<float>(i);
            float* a11 = gf.covarsInv(1, 1).ptr<float>(i);

            div_det_2x2(a00, a10, a11, gf.w);
        }
        return;
    }

    if (gf.gCnNum == 1)
    {
        //divide(1.0, covars(0, 0)(range, Range::all()), gf.covarsInv(0, 0)(range, Range::all()));
        //return;

        for (int i = range.start; i < range.end; i++)
        {
            float* res = covars(0, 0).ptr<float>(i);
            inv_self(res, gf.w);
        }
        return;
    }
}

void GuidedFilterImpl::MulChannelsGuideAndSrc_ParBody::operator()(const cv::Range& range) const
{
    int srcCnNum = (int)srcCn.size();

    for (int i = range.start; i < range.end; i++)
    {
        for (int si = 0; si < srcCnNum; si++)
        {
            int step = (si % 2) * 2 - 1;
            int start = (si % 2) ? 0 : gf.gCnNum - 1;
            int end = (si % 2) ? gf.gCnNum : -1;

            float* srcLine = srcCn[si].ptr<float>(i);

            for (int gi = start; gi != end; gi += step)
            {
                float* guideLine = gf.guideCn[gi].ptr<float>(i);
                float* dstLine = cov[si][gi].ptr<float>(i);

                mul(dstLine, srcLine, guideLine, gf.w);
            }
        }
    }
}

void GuidedFilterImpl::ComputeCovFromSrcChannelsMul_ParBody::operator()(const cv::Range& range) const
{
    int srcCnNum = (int)srcCnMean.size();

    for (int i = range.start; i < range.end; i++)
    {
        for (int si = 0; si < srcCnNum; si++)
        {
            int step = (si % 2) * 2 - 1;
            int start = (si % 2) ? 0 : gf.gCnNum - 1;
            int end = (si % 2) ? gf.gCnNum : -1;

            float* srcMeanLine = srcCnMean[si].ptr<float>(i);

            for (int gi = start; gi != end; gi += step)
            {
                float* guideMeanLine = gf.guideCnMean[gi].ptr<float>(i);
                float* covLine = cov[si][gi].ptr<float>(i);

                sub_mul(covLine, srcMeanLine, guideMeanLine, gf.w);
            }
        }
    }
}

void GuidedFilterImpl::ComputeAlpha_ParBody::operator()(const cv::Range& range) const
{
    int srcCnNum = (int)covSrc.size();

    for (int i = range.start; i < range.end; i++)
    {
        for (int si = 0; si < srcCnNum; si++)
        {
            for (int gi = 0; gi < gf.gCnNum; gi++)
            {
                float* y, * A, * dstAlpha;

                dstAlpha = alpha[si][gi].ptr<float>(i);
                for (int k = 0; k < gf.gCnNum; k++)
                {
                    y = covSrc[si][k].ptr<float>(i);
                    A = gf.covarsInv(gi, k).ptr<float>(i);

                    if (k == 0)
                    {
                        mul(dstAlpha, A, y, gf.w);
                    }
                    else
                    {
                        add_mul(dstAlpha, A, y, gf.w);
                    }
                }
            }
        }
    }
}

void GuidedFilterImpl::ComputeBeta_ParBody::operator()(const cv::Range& range) const
{
    int srcCnNum = (int)srcCnMean.size();
    CV_DbgAssert(&srcCnMean == &beta);

    for (int i = range.start; i < range.end; i++)
    {
        float* _g[4];
        for (int gi = 0; gi < gf.gCnNum; gi++)
            _g[gi] = gf.guideCnMean[gi].ptr<float>(i);

        float* betaDst, * g, * a;
        for (int si = 0; si < srcCnNum; si++)
        {
            betaDst = beta[si].ptr<float>(i);
            for (int gi = 0; gi < gf.gCnNum; gi++)
            {
                a = alpha[si][gi].ptr<float>(i);
                g = _g[gi];

                sub_mul(betaDst, a, g, gf.w);
            }
        }
    }
}

GuidedFilterImpl::GFTransform_ParBody::GFTransform_ParBody(GuidedFilterImpl& gf_, std::vector<cv::Mat>& srcv, std::vector<cv::Mat>& dstv, TransformFunc func_)
    : gf(gf_), func(func_)
{
    CV_DbgAssert(srcv.size() == dstv.size());
    src.resize(srcv.size());
    dst.resize(srcv.size());

    for (int i = 0; i < (int)srcv.size(); i++)
    {
        src[i] = &srcv[i];
        dst[i] = &dstv[i];
    }
}

GuidedFilterImpl::GFTransform_ParBody::GFTransform_ParBody(GuidedFilterImpl& gf_, std::vector<std::vector<cv::Mat> >& srcvv, std::vector<std::vector<cv::Mat> >& dstvv, TransformFunc func_)
    : gf(gf_), func(func_)
{
    CV_DbgAssert(srcvv.size() == dstvv.size());
    int n = (int)srcvv.size();
    int total = 0;

    for (int i = 0; i < n; i++)
    {
        CV_DbgAssert(srcvv[i].size() == dstvv[i].size());
        total += (int)srcvv[i].size();
    }

    src.resize(total);
    dst.resize(total);

    int k = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < (int)srcvv[i].size(); j++)
        {
            src[k] = &srcvv[i][j];
            dst[k] = &dstvv[i][j];
            k++;
        }
    }
}

void GuidedFilterImpl::GFTransform_ParBody::operator()(const cv::Range& range) const
{
    for (int i = range.start; i < range.end; i++)
    {
        (gf.*func)(*src[i], *dst[i]);
    }
}

//给定一个一维索引 eid，根据图像的通道数 gCnNum，返回对应的通道对 (cn1, cn2)
void GuidedFilterImpl::getWalkPattern(int eid, int& cn1, int& cn2)
{
    static int wdata[] = {
        0, -1, -1, -1, -1, -1,
        0, -1, -1, -1, -1, -1,

        0,  0,  1, -1, -1, -1,
        0,  1,  1, -1, -1, -1,

        0,  0,  0,  2,  1,  1,
        0,  1,  2,  2,  2,  1,
    };

    cn1 = wdata[6 * 2 * (gCnNum - 1) + eid];
    cn2 = wdata[6 * 2 * (gCnNum - 1) + 6 + eid];
}

cv::Ptr<GuidedFilterImpl> GuidedFilterImpl::create(cv::InputArray guide, int radius, double eps, double scale)
{
    GuidedFilterImpl* gf = new GuidedFilterImpl();
    gf->init(guide, radius, eps, scale);
    return cv::Ptr<GuidedFilterImpl>(gf);
}

void GuidedFilterImpl::init(cv::InputArray guide, int radius_, double eps_, double scale_)
{
    CV_Assert(!guide.empty() && radius_ >= 0 && eps_ >= 0);
    CV_Assert((guide.depth() == CV_32F || guide.depth() == CV_8U || guide.depth() == CV_16U) && (guide.channels() <= 3));
    CV_Assert(scale_ <= 1.0);

    radius = radius_;
    eps = eps_;
    scale = scale_;

    splitFirstNChannels(guide, guideCnOriginal, 3);
    gCnNum = (int)guideCnOriginal.size();
    hOriginal = guideCnOriginal[0].rows;
    wOriginal = guideCnOriginal[0].cols;
    h = int(hOriginal * scale);
    w = int(wOriginal * scale);

    parConvertToWorkType(guideCnOriginal, guideCnOriginal);
    if (scale < 1.0)
    {
        // 开启快速引导滤波
        guideCn.resize(gCnNum);
        parSubsample(guideCnOriginal, guideCn);
    }
    else
    {
        guideCn = guideCnOriginal;
    }

	// 计算引导图像各通道的均值图像 E[guideCn]
    guideCnMean.resize(gCnNum);
    parMeanFilter(guideCn, guideCnMean);

	// 计算引导图像各通道的协方差矩阵
    SymArray2D<cv::Mat> covars;
    computeCovGuide(covars);
	// 计算协方差矩阵的逆矩阵
    runParBody(ComputeCovGuideInv_ParBody(*this, covars));
    covars.release();
}

void GuidedFilterImpl::computeCovGuide(SymArray2D<cv::Mat>& covars)
{
    covars.create(gCnNum);
    for (int i = 0; i < covars.total(); i++)
        covars(i).create(h, w, CV_32FC1);

    runParBody(MulChannelsGuide_ParBody(*this, covars));

    parMeanFilter(covars.vec, covars.vec);

    runParBody(ComputeCovGuideFromChannelsMul_ParBody(*this, covars));
}

void GuidedFilterImpl::filter(cv::InputArray src, cv::OutputArray dst, int dDepth /*= -1*/)
{
    CV_Assert(!src.empty() && (src.depth() == CV_32F || src.depth() == CV_8U));
    if (src.rows() != hOriginal || src.cols() != wOriginal)
    {
        CV_Error(cv::Error::StsBadSize, "Size of filtering image must be equal to size of guide image");
        return;
    }

    if (dDepth == -1) dDepth = src.depth();
    int srcCnNum = src.channels();

    std::vector<cv::Mat> srcCn(srcCnNum);
    std::vector<cv::Mat>& srcCnMean = srcCn;
    cv::split(src, srcCn);

    if (scale < 1.0)
    {
        parSubsample(srcCn, srcCn);
    }

    if (src.depth() != CV_32F)
    {
        parConvertToWorkType(srcCn, srcCn);
    }

    std::vector<std::vector<cv::Mat> > covSrcGuide(srcCnNum);
    computeCovGuideAndSrc(srcCn, srcCnMean, covSrcGuide);

    std::vector<std::vector<cv::Mat> > alpha(srcCnNum);
    for (int si = 0; si < srcCnNum; si++)
    {
        alpha[si].resize(gCnNum);
        for (int gi = 0; gi < gCnNum; gi++)
            alpha[si][gi].create(h, w, CV_32FC1);
    }
    runParBody(ComputeAlpha_ParBody(*this, alpha, covSrcGuide));
    covSrcGuide.clear();

    std::vector<cv::Mat>& beta = srcCnMean;
    runParBody(ComputeBeta_ParBody(*this, alpha, srcCnMean, beta));

    parMeanFilter(beta, beta);
    parMeanFilter(alpha, alpha);

    if (scale < 1.0)
    {
        parUpsample(beta, beta);
        parUpsample(alpha, alpha);
    }

    parallel_for_(cv::Range(0, hOriginal), ApplyTransform_ParBody(*this, alpha, beta));
    if (dDepth != CV_32F)
    {
        for (int i = 0; i < srcCnNum; i++)
            beta[i].convertTo(beta[i], dDepth);
    }
    cv::merge(beta, dst);
}

// 计算协方差矩阵 covSrcGuide，其中每个元素 covSrcGuide[si][gi] 表示源图像第 si 个通道与引导图像第 gi 个通道的协方差
// 公式为：covSrcGuide[si][gi] = E[srcCn[si] * guideCn[gi]] - E[srcCn[si]] * E[guideCn[gi]]
void GuidedFilterImpl::computeCovGuideAndSrc(std::vector<cv::Mat>& srcCn, std::vector<cv::Mat>& srcCnMean, std::vector<std::vector<cv::Mat> >& cov)
{
    int srcCnNum = (int)srcCn.size();

    cov.resize(srcCnNum);
    for (int i = 0; i < srcCnNum; i++)
    {
        cov[i].resize(gCnNum);
        for (int j = 0; j < gCnNum; j++)
            cov[i][j].create(h, w, CV_32FC1);
    }

    runParBody(MulChannelsGuideAndSrc_ParBody(*this, srcCn, cov));

    parMeanFilter(srcCn, srcCnMean);
    parMeanFilter(cov, cov);

    runParBody(ComputeCovFromSrcChannelsMul_ParBody(*this, srcCnMean, cov));
}



cv::Ptr<GuidedFilter> createGuidedFilter(cv::InputArray guide, int radius, double eps, double scale)
{
    return cv::Ptr<GuidedFilter>(GuidedFilterImpl::create(guide, radius, eps, scale));
}

void guidedFilter(cv::InputArray guide, cv::InputArray src, cv::OutputArray dst, int radius, double eps, int dDepth, double scale)
{
    cv::Ptr<GuidedFilter> gf = createGuidedFilter(guide, radius, eps, scale);
    gf->filter(src, dst, dDepth);
}
