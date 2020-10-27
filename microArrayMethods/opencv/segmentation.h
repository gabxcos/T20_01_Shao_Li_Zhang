#pragma once
#include "OpenCVSegmenter.h"

#ifndef _OpenCVSegmenter_preprocessing
#define _OpenCVSegmenter_preprocessing

Mat resizeSpot(Mat spot);
float calculateSpotK(Mat image);
float getMaxCenter(Mat image);
PCA getPCA(Mat spot);
float eucDist(Mat col1, Mat col2);
bool IKM(Mat spot, PCA pca, Point p1, Point p2);

#endif