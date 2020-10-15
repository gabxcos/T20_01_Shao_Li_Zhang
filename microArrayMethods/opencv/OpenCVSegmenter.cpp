#include "OpenCVSegmenter.h"

OpenCVSegmenter::OpenCVSegmenter(std::string path) {
	setPath(path);
}

bool OpenCVSegmenter::init() {
	std::string image_path = samples::findFile(getPath());
	Mat img = imread(image_path, CV_8UC1);

	if (img.empty())
	{
		std::cout << "Could not read the image: " << image_path << std::endl;
		return false;
	}

	// Make sure image is 8-bit
	CV_Assert(img.depth() == CV_8U);
	setImage(img);

	setChannels(img.channels());
	setWidth(img.cols);
	setHeight(img.rows);

	setContinuous(img.isContinuous());
	
	std::cout << "OpenCV segmentation service correctly initialized." << std::endl;
	return true;
}