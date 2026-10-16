#include "Dynamic_range_compression.h"
#include "quantitative_assessment.h"
#include "slider_display.h"


int DDE_improvement();

int calc_Gradient_32F(cv::InputArray src, cv::OutputArray output);

double estimateNoiseWaveletMAD(cv::InputArray input);

