#pragma once

#include <iostream>
#include <array>

using namespace std;

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

#include "tests.h"
# include "gridding.h"

class OpenCVSegmenter
{
private:
	// Construction parameters
	std::string file_path; // the file path of the microArray image file, supposed 8-bit, grayscale, TIFF
	bool visualized; // does the user want to visualize the middle results in a window? for testing purposes

	// Internal status, OpenCV image and its parameters
	Mat image;
	int channels;
	int width;
	int height;
	bool continuous;
public:
	OpenCVSegmenter(std::string path, bool isVisualized=false);
	bool init();

	//Steps
	bool gridding();

	// Getters/Setters
	std::string getPath() { return file_path; };
	void setPath(std::string path) { file_path = path; }

	bool isVisualized() { return visualized; }
	void setVisualized(bool _visualized) { visualized = _visualized; }

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
