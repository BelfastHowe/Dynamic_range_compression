#pragma once
#include "Dynamic_range_compression.h"


int slider_display();

int showHalfStepHistogram(
    const std::vector<float>& values,
    const std::string& windowName = "HH Coefficient Histogram",
    double xMin = -1.0,
    double xMax = -1.0,
    int plotWidth = 1400,
    int plotHeight = 600);

