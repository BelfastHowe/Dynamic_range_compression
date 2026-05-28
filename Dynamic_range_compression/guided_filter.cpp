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

    template<typename V>
    void parConvertToWorkType(V& src, V& dst)
    {
        GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::convertToWorkType);
        parallel_for_(pb.getRange(), pb);
    }

    template<typename V>
    void parMeanFilter(V& src, V& dst)
    {
        GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::meanFilter);
        parallel_for_(pb.getRange(), pb);
    }

    template<typename V>
    void parSubsample(V& src, V& dst)
    {
        GFTransform_ParBody pb(*this, src, dst, &GuidedFilterImpl::subsample);
        parallel_for_(pb.getRange(), pb);
    }

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
        GuidedFilterImpl& gf;
        SymArray2D<cv::Mat>& covars;

        MulChannelsGuide_ParBody(GuidedFilterImpl& gf_, SymArray2D<cv::Mat>& covars_)
            : gf(gf_), covars(covars_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ComputeCovGuideFromChannelsMul_ParBody : public cv::ParallelLoopBody
    {
        GuidedFilterImpl& gf;
        SymArray2D<cv::Mat>& covars;

        ComputeCovGuideFromChannelsMul_ParBody(GuidedFilterImpl& gf_, SymArray2D<cv::Mat>& covars_)
            : gf(gf_), covars(covars_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ComputeCovGuideInv_ParBody : public cv::ParallelLoopBody
    {
        GuidedFilterImpl& gf;
        SymArray2D<cv::Mat>& covars;

        ComputeCovGuideInv_ParBody(GuidedFilterImpl& gf_, SymArray2D<cv::Mat>& covars_);

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };


    struct MulChannelsGuideAndSrc_ParBody : public cv::ParallelLoopBody
    {
        GuidedFilterImpl& gf;
        std::vector<std::vector<cv::Mat> >& cov;
        std::vector<cv::Mat>& srcCn;

        MulChannelsGuideAndSrc_ParBody(GuidedFilterImpl& gf_, std::vector<cv::Mat>& srcCn_, std::vector<std::vector<cv::Mat> >& cov_)
            : gf(gf_), cov(cov_), srcCn(srcCn_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ComputeCovFromSrcChannelsMul_ParBody : public cv::ParallelLoopBody
    {
        GuidedFilterImpl& gf;
        std::vector<std::vector<cv::Mat> >& cov;
        std::vector<cv::Mat>& srcCnMean;

        ComputeCovFromSrcChannelsMul_ParBody(GuidedFilterImpl& gf_, std::vector<cv::Mat>& srcCnMean_, std::vector<std::vector<cv::Mat> >& cov_)
            : gf(gf_), cov(cov_), srcCnMean(srcCnMean_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ComputeAlpha_ParBody : public cv::ParallelLoopBody
    {
        GuidedFilterImpl& gf;
        std::vector<std::vector<cv::Mat> >& alpha;
        std::vector<std::vector<cv::Mat> >& covSrc;

        ComputeAlpha_ParBody(GuidedFilterImpl& gf_, std::vector<std::vector<cv::Mat> >& alpha_, std::vector<std::vector<cv::Mat> >& covSrc_)
            : gf(gf_), alpha(alpha_), covSrc(covSrc_) {}

        void operator () (const cv::Range& range) const CV_OVERRIDE;
    };

    struct ComputeBeta_ParBody : public cv::ParallelLoopBody
    {
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



cv::Ptr<GuidedFilter> createGuidedFilter(cv::InputArray guide, int radius, double eps, double scale)
{
    return cv::Ptr<GuidedFilter>(GuidedFilterImpl::create(guide, radius, eps, scale));
}

void guidedFilter(cv::InputArray guide, cv::InputArray src, cv::OutputArray dst, int radius, double eps, int dDepth, double scale)
{
    cv::Ptr<GuidedFilter> gf = createGuidedFilter(guide, radius, eps, scale);
    gf->filter(src, dst, dDepth);
}
