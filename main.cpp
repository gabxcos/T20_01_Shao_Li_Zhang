#include <stdio.h>
#include <io.h>
#include <stdlib.h>
#include <string>
#include <iostream>
#include <regex>

// Windows only
#include <direct.h>

#include "Device.h"
#include "microArrayMethods/opencv/OpenCVSegmenter.h"

#define Hc Controls::HYBRIDIZATION
#define Ec Controls::EMPTY
#define Nc Controls::NEGATIVE
#define Pc Controls::PCR
#define Cc Controls::CAPTURE

using namespace std;

int main() {
	String inFolder = "input";
	String outFolder = "output";

	vector<String> inFiles, outFiles;

	glob(inFolder, inFiles);
	regex e("");
	for (int i = 0; i < inFiles.size(); i++) {
		String s = inFiles[i];
		s = s.substr((inFolder.size() + 1));
		int dotPos = s.find(".");
		s = s.substr(0, dotPos);
		outFiles.push_back(s);
	}

	vector<vector<int>> controls {
		{ Hc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Nc, Ec, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Hc },
		{ Cc, Cc, Cc, Pc, Nc, Cc, Cc, Cc, Cc, Ec, Ec, Cc, Cc, Cc, Pc, Nc, Cc, Cc, Cc, Cc, Ec },
		{ Ec, Cc, Cc, Cc, Cc, Hc, Cc, Cc, Cc, Ec, Ec, Ec, Cc, Cc, Cc, Cc, Hc, Cc, Cc, Cc, Ec },
		{ Cc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Ec, Nc, Ec, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Ec, Nc },
		{ Nc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Ec, Hc, Ec, Nc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Ec, Hc },
		{ Hc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Ec, Nc, Ec, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Cc, Hc }
	};
	Device dev(6, 21, controls, 240, 280, 1120, 660); // ROI calculated manually based on 52 images: 260, 295, 1100, 645
	if (!dev.checkValidity()) return 1;
	else std::printf("Device correctly initialized!\n\n");
	dev.describe();

	for (int i = 0; i < inFiles.size(); i++) {
		String newFolder = outFolder + "\\" + outFiles[i];
		_mkdir(newFolder.c_str());
		OpenCVSegmenter* service = new OpenCVSegmenter(inFiles[i], newFolder, dev, true);
		service->init();
		service->preprocess();
		service->gridding();
		service->segmenting();
		service->produceResults();
		delete service;
	}
	return 0;
}