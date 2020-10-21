#pragma once
#include "OpenCVSegmenter.h"

#ifndef _OpenCVSegmenter_preprocessing
#define _OpenCVSegmenter_preprocessing

float calculateContrast(Mat image);
float calculateK(Mat image);

#endif