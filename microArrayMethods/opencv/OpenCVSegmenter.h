#pragma once

#include <iostream>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

#include "tests.h"
# include "gridding.h"

class OpenCVSegmenter
{
private:
	std::string file_path;
	Mat image;
	int channels;
	int width;
	int height;
	bool continuous;
public:
	OpenCVSegmenter(std::string path);
	bool init();

	//Steps
	void gridding();

	// Getters/Setters
	std::string getPath() { return file_path; };
	void setPath(std::string path) { file_path = path; }

	Mat getImage() { return image; }
	void setImage(Mat _image) { image = _image; }

	int getWidth() { return width; }
	void setWidth(int _width) { width = _width; }

	int getHeight() { return height; }
	void setHeight(int _height) { height = _height; }

	int getChannels() { return channels; }
	void setChannels(int _channels) { channels = _channels; }

	bool isContinuous() { return continuous; }
	void setContinuous(bool _continuous) { continuous = _continuous; }

	// Test suite
	static int test(int num_test = 1);
};

