#include "Dynamic_range_compression.h"



class GuidedFilter : public cv::Algorithm
{
public:

    /** @brief Apply (Fast) Guided Filter to the filtering image.

    @param src filtering image with any numbers of channels.

    @param dst output image.

    @param dDepth optional depth of the output image. dDepth can be set to -1, which will be equivalent
    to src.depth().
     */
    virtual void filter(cv::InputArray src, cv::OutputArray dst, int dDepth = -1) = 0;
};


cv::Ptr<GuidedFilter> createGuidedFilter(cv::InputArray guide, int radius, double eps, double scale = 1.0);

void guidedFilter(cv::InputArray guide, cv::InputArray src, cv::OutputArray dst, int radius, double eps, int dDepth = -1, double scale = 1.0);

int selfGuidedFilter(cv::InputArray input, cv::OutputArray output, int radius, double eps);

int sgf_cmodel(cv::InputArray  input,
    cv::OutputArray output,
    int             radius,
    int64_t         eps);


