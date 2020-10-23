#pragma once
#include "OpenCVSegmenter.h"

#ifndef _OpenCVSegmenter_gridding
#define _OpenCVSegmenter_gridding

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#include <math.h>

#endif

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
bool deleteEmptyLines(Mat image, vector<int>& Hlines, vector<int>& Vlines, bool first = true);
bool setToAngles(Mat image, vector<int>& Hlines, vector<int>& Vlines, projections binarySignals);
bool reBinaryGrid(vector<int>& Hlines, vector<int>& Vlines, projections binarySignals);
double pdf(double mean, double var, double x);
float calculateAngleProb(int startX, int startY, int endX, int endY, Mat image, vector<int> Hlines, vector<int> Vlines);
bool adjustToDevice(Device d, Mat image, vector<int>& Hlines, vector<int>& Vlines, projections binarySignals);
bool adjustGrid(vector<int>& Hlines, vector<int>& Vlines, bool resizable = true); // shorthand
bool adjustHgrid(vector<int>& Hlines, bool resizable, int flag=0, int flagend=0, float errorlimit=0.0); // both H and V
bool adaptToDeviceSize(int numRows, int numCols, vector<int>& Hlines, vector<int>& Vlines);

#endif