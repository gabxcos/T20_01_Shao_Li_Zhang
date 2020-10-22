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
bool deleteEmptyLines(Mat image, vector<int>& Hlines, vector<int>& Vlines, bool fourAngles = false, bool first = true);
bool adjustToDevice(Device d, Mat image, vector<int>& Hlines, vector<int>& Vlines);
bool adjustGrid(vector<int>& Hlines, vector<int>& Vlines); // shorthand
bool adjustHgrid(vector<int>& Hlines, int flag=0, int flagend=0, float errorlimit=0.0); // both H and V

#endif