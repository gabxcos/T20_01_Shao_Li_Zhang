#pragma once
#include "OpenCVSegmenter.h"

#ifndef _OpenCVSegmenter_gridding
#define _OpenCVSegmenter_gridding

struct projections {
	Mat H;
	Mat V;
};

bool getProjections(Mat image, projections* proj_set);
Mat getReconstruction(Mat marker, Mat mask);
bool calculateSignals(projections init_proj, projections* signals, projections* recs);

#endif