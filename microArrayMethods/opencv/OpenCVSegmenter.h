#pragma once

#include <iostream>
#include <array>
using namespace std;

#include "../../Device.h"

#include <opencv2/core/core.hpp>
#include <opencv2/core/core_c.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace cv;

// utilities
double medianMat(cv::Mat Input, cv::Mat mask, int nVals);

struct displacements {
	int top;
	int bottom;
	int left;
	int right;
};

struct circleSizes {
	int width;
	int height;
	int radius;
	int maxRadius;
	int minRadius;
	Point center;
};

struct spotSizes {
	displacements displ;
	circleSizes circle;
};

struct spotQuality {
	// according to: https://www.researchgate.net/publication/228989623_A_Quality_Measure_Model_for_Microarray_Images
	float signalNoise;
	float backgroundNoise;
	float scaleInvariant;
	float sizeRegularity;
	// according to: PMC55840
	float sizeQuality;
	float signalToNoiseRatio;
	float localBackgroundVariability;
	float localBackgroundHighness;
	float saturationQuality;
	float compositeQuality;
};

struct imageQuality {
	// statistics
	float avgSpotDistance;
	float avgSpotRadius;
	float avgSpotArea;
	// quality
	float backgroundNoise;
	float spotAlignment;
};

class Spot {
private:
	Mat spotImage;
	Mat cluster;
	int x, y;
	float signalMedian, backgroundMedian;
	int signalArea, backgroundArea;
	bool filled;
public:
	spotSizes sizes;
	spotQuality quality;

	//Construction
	Spot(Mat spotImg, Mat cluster, int x, int y);
	void init();

	// Getters/Setters
	Mat getSpotImage() { return spotImage; }
	void setSpotImage(Mat _spotImage) { spotImage = _spotImage; }

	Mat getCluster() { return cluster; }
	void setCluster(Mat _cluster) { cluster = _cluster; }

	int getX() { return x; }
	int getY() { return y; }
	void setCoord(int _x, int _y) { x = _x; y = _y; }

	float getSignal() { return signalMedian; }
	void setSignal(float _signal) { signalMedian = _signal; }

	float getBackground() { return backgroundMedian; }
	void setBackground(float _background) { backgroundMedian = _background; }

	int getSignalArea() { return signalArea; }
	void setSignalArea(int _sigArea) { signalArea = _sigArea; }

	int getBackgroundArea() { return backgroundArea; }
	void setBackgroundArea(int _bgArea) { backgroundArea = _bgArea; }

	bool isFilled() { return filled; }
	void setIsFilled(bool _filled) { filled = _filled; }
};

class OpenCVSegmenter
{
private:
	// Construction parameters
	std::string file_path; // the file path of the microArray image file, supposed 8-bit, grayscale, TIFF
	std::string out_folder; // the path to the output folder where to save the results
	Device dev; // the Device object to align the Segmenter to
	bool visualized; // does the user want to visualize the middle results in a window? for testing purposes

	// Internal status, OpenCV image and its parameters
	Mat originalImage, resizedImage, image;
	float bgThresh;
	int resX, resY, resXdist, resYdist;
	vector<int> gridH, gridV;
	int channels;
	int width;
	int height;
	bool continuous;

	// Result status, spot matrix
	vector<vector<Spot>> spotMatrix;
public:
	imageQuality quality;

	OpenCVSegmenter(std::string path, std::string out_folder, Device _dev, bool isVisualized=false);
	OpenCVSegmenter() {};

	bool init();

	// Steps
	bool preprocess();
	bool gridding();
	bool segmenting();
	bool produceResults();

	// Utilities
	Mat resizeImage(bool init);
	float spotScore(Mat spot);

	// Getters/Setters
	std::string getPath() { return file_path; };
	void setPath(std::string path) { file_path = path; }

	std::string getOutPath() { return out_folder; }
	void setOutPath(std::string _out_folder) { out_folder = _out_folder; }

	Device getDevice() { return dev; }
	void setDevice(Device _dev) { dev = _dev; }

	bool isVisualized() { return visualized; }
	void setVisualized(bool _visualized) { visualized = _visualized; }

	Mat getOriginalImage() { return originalImage; }
	void setOriginalImage(Mat _image) { originalImage = _image; }

	Mat getImage() { return image; }
	void setImage(Mat _image) { image = _image; }

	float getBgThresh() { return bgThresh; }
	void setBgThresh(float k) { bgThresh = k; }

	Mat getResImage() { return resizedImage; }
	void setResImage(Mat _image) { resizedImage = _image; }

	void setResRect(int _resX, int _resY, int _resXdist, int _resYdist) { resX = _resX; resY = _resY; resXdist = _resXdist; resYdist = _resYdist; }

	void setGrid(vector<int>& Hlines, vector<int>& Vlines) { gridH = Hlines; gridV = Vlines; }
	vector<int> getGridH() { return gridH; }
	vector<int> getGridV() { return gridV; }

	int getWidth() { return width; }
	void setWidth(int _width) { width = _width; }

	int getHeight() { return height; }
	void setHeight(int _height) { height = _height; }

	int getChannels() { return channels; }
	void setChannels(int _channels) { channels = _channels; }

	bool isContinuous() { return continuous; }
	void setContinuous(bool _continuous) { continuous = _continuous; }

	vector<vector<Spot>> getSpotMatrix() { return spotMatrix; }
	void setSpotMatrix(vector<vector<Spot>> _spMtr) { spotMatrix = _spMtr; }

	// Test suite
	static int test(int num_test = 1);
};

#include "tests.h"
#include "gridding.h"