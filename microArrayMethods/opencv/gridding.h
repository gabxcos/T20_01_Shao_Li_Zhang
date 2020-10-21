#pragma once
#include "OpenCVSegmenter.h"

#ifndef _OpenCVSegmenter_gridding
#define _OpenCVSegmenter_gridding

struct projections {
	Mat H;
	Mat V;
};

bool getProjections(Mat image, projections* proj_set);
Mat getReconstruction(Mat marker, Mat mask, int kernelSize);
int calculateKernelSize(Mat H, bool horizontal);
bool calculateSignals(projections init_proj, projections* signals, projections* recs);
float getThreshold(Mat M);
float getThresholdV2(Mat M);
bool getBinarySignals(projections* signals, projections* binarySignals);
vector<int> getHlines(Mat H);
vector<int> getVlines(Mat V);

#endif