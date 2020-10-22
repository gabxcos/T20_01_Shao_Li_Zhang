#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include "Device.h"
#include "microArrayMethods/opencv/OpenCVSegmenter.h"

#define Hc Controls::HYBRIDIZATION
#define Ec Controls::EMPTY
#define Nc Controls::NEGATIVE
#define Pc Controls::PCR
#define Cc Controls::CAPTURE

using namespace std;

int main() {
	vector<vector<int>> controls {
		{ Hc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Nc, Ec, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Hc },
		{ Cc, Cc, Cc, Pc, Nc, Cc, Cc, Cc, Cc, Ec, Ec, Cc, Cc, Cc, Pc, Nc, Cc, Cc, Cc, Cc, Ec },
		{ Ec, Cc, Cc, Cc, Cc, Hc, Cc, Cc, Cc, Ec, Ec, Ec, Cc, Cc, Cc, Cc, Hc, Cc, Cc, Cc, Ec },
		{ Cc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Ec, Nc, Ec, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Ec, Nc },
		{ Nc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Ec, Hc, Ec, Nc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Ec, Hc },
		{ Hc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Nc, Ec, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Hc }
	};
	Device dev(6, 21, controls);
	if (!dev.checkValidity()) return 1;
	else std::printf("Device correctly initialized!\n\n");
	dev.describe();

	OpenCVSegmenter service("sample.tif", dev, true);
	service.init();
	service.preprocess();
	service.gridding();

	return 0;
}