#include "OpenCVSegmenter.h"

OpenCVSegmenter::OpenCVSegmenter(std::string path, Device _dev, bool isVisualized) {
	setPath(path);
	setDevice(_dev);
	setVisualized(isVisualized);
}

bool OpenCVSegmenter::init() {
	std::string image_path = samples::findFile(getPath());
	Mat img = imread(image_path, CV_8UC1);

	if (img.empty())
	{
		std::cout << "Could not read the image: " << image_path << std::endl;
		return false;
	}

	cv::normalize(img, img, 255, 0, cv::NORM_MINMAX);

	// Make sure image is 8-bit
	CV_Assert(img.depth() == CV_8U);
	// Make 32-bit
	img.convertTo(img, CV_32FC1, 1.0/255.0);

	setOriginalImage(img);
	setImage(img);
	resizeImage();

	std::cout << "OpenCV segmentation service correctly initialized." << std::endl;
	return true;
}

bool OpenCVSegmenter::resizeImage()
{
	Mat image = getImage();
	int width = image.cols, height = image.rows;

	int startX=0, endX=width-1, startY=0, endY=height-1;
	Mat projX, projY;
	reduce(image, projX, 0, CV_REDUCE_SUM);
	reduce(image, projY, 1, CV_REDUCE_SUM);
	// 0.004 = single pixel, lowest intensity
	for (int i = 0; i < width; i++) {
		if (projX.at<float>(0, i) > 0.004) {
			startX = i - 1;
			break;
		}
	}

	for (int i = width - 1; i >= 0; i--) {
		if (projX.at<float>(0, i) > 0.004) {
			endX = i + 1;
			break;
		}
	}

	for (int i = 0; i < height; i++) {
		if (projY.at<float>(i, 0) > 0.004) {
			startY = i - 1;
			break;
		}
	}

	for (int i = height - 1; i >= 0; i--) {
		if (projY.at<float>(i, 0) > 0.004) {
			endY = i + 1;
			break;
		}
	}

	// give a 10 px border
	startX = max(0, startX - 10);
	startY = max(0, startY - 10);
	endX = min(width, endX + 10);
	endY = min(height, endY + 10);

	Rect myROI(startX, startY, (endX-startX), (endY-startY));

	Mat croppedImage = image(myROI);
	setImage(croppedImage);

	setChannels(croppedImage.channels());
	setWidth(croppedImage.cols);
	setHeight(croppedImage.rows);

	setContinuous(croppedImage.isContinuous());

	return true;
}
