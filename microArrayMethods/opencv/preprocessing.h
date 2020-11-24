#pragma once
#include "OpenCVSegmenter.h"

#ifndef _OpenCVSegmenter_preprocessing
#define _OpenCVSegmenter_preprocessing

float calculateContrast(Mat image);
float calculateK(Mat image);
float calculateKv2(Mat image);
Mat contrastEnhance(Mat image, float k);
Mat medianFilter(Mat image);

#endif