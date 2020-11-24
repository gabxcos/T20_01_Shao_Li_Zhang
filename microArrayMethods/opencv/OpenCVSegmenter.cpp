#include "OpenCVSegmenter.h"


Spot::Spot(Mat spotImg, Mat cluster, int x, int y)
{
	setSpotImage(spotImg);
	setCluster(cluster);
	setCoord(x, y);
	init();
}

void Spot::init()
{
	/*Mat cluster = getCluster(), spot = getSpotImage();
	int width = cluster.cols, height = cluster.rows;
	int totArea = width * height, signalArea = max((int)sum(cluster)[0], 1), bgArea = totArea - signalArea;
	float sigMean = sum(spot.mul(cluster))[0] / signalArea;
	Mat notCluster;
	threshold(cluster, notCluster, 0.0, 1.0, THRESH_BINARY_INV);
	float bgMean = sum(spot.mul(notCluster))[0] / bgArea;
	sigMean *= 255.0;
	bgMean *= 255.0;*/
	Mat cluster = getCluster(), spot = getSpotImage();
	spot.convertTo(spot, CV_8UC1, 255.0);
	cluster.convertTo(cluster, CV_8UC1, 255.0);
	int width = cluster.cols, height = cluster.rows;
	int totArea = width * height, signalArea = max((int)(sum(cluster)[0]/255.0), 1), bgArea = totArea - signalArea;
	float sigMedian = (float)medianMat(spot, cluster, 256);
	Mat notCluster;
	threshold(cluster, notCluster, 0.0, 255.0, THRESH_BINARY_INV);
	float bgMedian = (float)medianMat(spot, notCluster, 256);
	/*sigMedian *= 255.0;
	bgMedian *= 255.0;*/

	setSignal(sigMedian);
	setBackground(bgMedian);

	setSignalArea(signalArea);
	setBackgroundArea(bgArea);

	bool filled = (sigMedian - bgMedian) > (0.1 * 255);

	if (filled) { 
		setIsFilled(true); 

		// Y
		int i = 0;
		do {
			if (sum(cluster.row(i))[0]<255) i++;
			else break;
		} while (i < (height - 1));
		int startY = i;
		do {
			if (sum(cluster.row(i))[0] > 254) i++;
			else break;
		} while (i < (height - 1));
		int endY = i - 1;

		// X
		i = 0;
		do {
			if (sum(cluster.col(i))[0] < 255) i++;
			else break;
		} while (i < (width - 1));
		int startX = i;
		do {
			if (sum(cluster.col(i))[0] > 254) i++;
			else break;
		} while (i < (width - 1));
		int endX = i - 1;

		displacements _displ;
		_displ.top = startY;
		_displ.left = startX;
		_displ.bottom = height - endY - 1;
		_displ.right = width - endX - 1;

		circleSizes _circle;
		_circle.width = endX - startX + 1;
		_circle.height = endY - startY + 1;
		_circle.maxRadius = (max(_circle.width, _circle.height)+1)/2;
		_circle.minRadius = min(_circle.width, _circle.height) / 2; // approximate...
		_circle.radius = _circle.minRadius;
		Point center = Point((int)((startX + endX) / 2.0), (int)((startY + endY) / 2.0));

		spotSizes _sizes;
		_sizes.displ = _displ;
		_sizes.circle = _circle;

		this->sizes = _sizes;
	}
	else {
		setIsFilled(false);

		displacements _displ;
		_displ.top = height / 2;
		_displ.bottom = height / 2;
		_displ.left = width / 2;
		_displ.right = width / 2;

		circleSizes _circle;
		_circle.width = 0, _circle.height = 0, _circle.maxRadius = 0, _circle.minRadius = 0;
		_circle.radius = _circle.minRadius;
		_circle.center = Point(_displ.left, _displ.top);

		spotSizes _sizes;
		_sizes.displ = _displ;
		_sizes.circle = _circle;

		this->sizes = _sizes;
	}

	spotQuality q;

	// signalNoise
	if (!filled) q.signalNoise = -1.0;
	else {
		float threshSig = max(sigMedian - 0.1 * 255.0, 0.0);
		Mat noise;
		threshold(spot, noise, threshSig, 255.0, THRESH_BINARY_INV);
		noise = noise - notCluster;
		int signalNoiseArea = sum(noise)[0] / 255.0;
		q.signalNoise = 1 - (float)signalNoiseArea / (float)signalArea;
	}

	// backgroundNoise
	float threshSig = min(bgMedian + 0.1 * 255.0, 255.0);
	Mat noise;
	threshold(spot, noise, threshSig, 255.0, THRESH_BINARY);
	if(!cluster.empty()) noise = noise - cluster;
	int bgNoiseArea = sum(noise)[0] / 255.0;
	q.backgroundNoise = 1 - (float)bgNoiseArea / (float)bgArea;
	
	// scaleInvariant
	if (!filled) q.scaleInvariant = -1.0;
	else q.scaleInvariant = (float)this->sizes.circle.minRadius / (float)this->sizes.circle.maxRadius;

	// sizeRegularity
	if (!filled) q.sizeRegularity = -1.0;
	else {
		int radius = this->sizes.circle.maxRadius;
		Mat testCircle(radius * 2 + 1, radius * 2 + 1, CV_8UC1, 0.0);
		circle(testCircle, Point(radius, radius), radius, 255.0, -1);
		int circleArea = sum(testCircle)[0] / 255.0;
		q.sizeRegularity = (float)min(signalArea, circleArea) / (float)circleArea;
	}

	// sizeQuality has to wait for all spots...

	// signalToNoiseRatio
	if (!filled) q.signalToNoiseRatio = 255.0 / (255.0 + bgMedian);
	else q.signalToNoiseRatio = sigMedian / (sigMedian + bgMedian);

	// localBackgroundVariability
	Scalar mean, stdDev;
	meanStdDev(spot, mean, stdDev, notCluster);
	q.localBackgroundVariability = mean[0] / stdDev[0];
	// needs to be standardized...

	// localBackgroundHighness has to wait for all spots...

	// saturationQuality has to be done outside...

	// compositeQuality waits for other parameters

	this->quality = q;
}

double medianMat(cv::Mat Input, cv::Mat mask, int nVals) {
	// As of: https://answers.opencv.org/question/176494/how-to-calculate-the-median-of-pixel-values-in-opencv-python/?sort=oldest
	if (mask.empty()) return 1.0;
	// COMPUTE HISTOGRAM OF SINGLE CHANNEL MATRIX
	float range[] = { 0, nVals };
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;
	cv::Mat hist;
	calcHist(&Input, 1, 0, mask, hist, 1, &nVals, &histRange, uniform, accumulate);

	// COMPUTE CUMULATIVE DISTRIBUTION FUNCTION (CDF)
	cv::Mat cdf;
	hist.copyTo(cdf);
	for (int i = 1; i <= nVals - 1; i++) {
		cdf.at<float>(i) += cdf.at<float>(i - 1);
	}
	cdf /= (sum(mask)[0]/255.0);

	// COMPUTE MEDIAN
	double medianVal = 0.0;
	for (int i = 0; i <= (nVals - 1); i++) {
		if (cdf.at<float>(i) >= 0.5) { medianVal = i;  break; }
	}
	return medianVal;
}

OpenCVSegmenter::OpenCVSegmenter(std::string path, std::string out_folder, Device _dev, bool isVisualized) {
	setPath(path);
	setOutPath(out_folder);
	std::string printstr = "\nWorking on: " + path + " - " + out_folder + "\n";
	std::printf(printstr.c_str());
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
	resizeImage(true);

	setWidth(img.cols);
	setHeight(img.rows);
	setChannels(img.channels());
	setContinuous(img.isContinuous());

	std::cout << "OpenCV segmentation service correctly initialized." << std::endl;
	return true;
}

Mat OpenCVSegmenter::resizeImage(bool init)
{
	Mat image = getImage().clone();

	if (init) {
		Device d = getDevice();
		image = image(Rect(d.getX(), d.getY(), d.getWidth(), d.getHeight()));
	}

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
	
	if (init) {
		setResImage(croppedImage);
		setResRect(startX, startY, (endX - startX), (endY - startY));
	}

	return croppedImage;
}
