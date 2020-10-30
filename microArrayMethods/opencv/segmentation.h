#pragma once
#include "OpenCVSegmenter.h"

#ifndef _OpenCVSegmenter_preprocessing
#define _OpenCVSegmenter_preprocessing

Mat resizeSpot(Mat spot);
float calculateSpotK(Mat image);
float getMaxCenter(Mat image);
PCA getPCA(Mat spot, Mat& PCAset_);
float eucDist(Mat col1, Mat col2);
int getAvgDiameter(vector<int> gridH, vector<int> gridV);
Mat adjustedCluster(Mat cluster, Mat spot, int s);
bool IKM(Mat spot, Mat PCAset, PCA pca, Point p1, Point p2, int diameter);

#endif