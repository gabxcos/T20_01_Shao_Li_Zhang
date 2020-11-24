#pragma once
#include "OpenCVSegmenter.h"
#include "../../Device.h"

#include <iostream>
#include <fstream>

#ifndef _OpenCVSegmenter_results
#define _OpenCVSegmenter_results

void printResults(OpenCVSegmenter* seg, vector<vector<Spot>> spotMatrix, Device d, String path);
Mat drawCircles(Mat fullImage, vector<vector<Spot>> spotMatrix);
bool nextToPixel(int x, int y, Mat image);

#endif