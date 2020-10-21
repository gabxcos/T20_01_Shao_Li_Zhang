#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "Device.h"
#include "microArrayMethods/opencv/OpenCVSegmenter.h"

using namespace std;

int main() {
	Device dev(6, 21);
	OpenCVSegmenter service("sample.tif", true);
	service.init();
	service.preprocess();
	service.gridding();

	return 0;
}