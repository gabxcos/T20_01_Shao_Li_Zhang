#include "segmentation.h"

bool OpenCVSegmenter::segmenting() {
	vector<int> gridH = getGridH(), gridV = getGridV();
	Mat image = getResImage();

	int x = gridV[20], y = gridH[4];
	int xd = gridV[21] - x, yd = gridH[5] - y;

	Mat spot = image(Rect(x, y, xd, yd));
	spot = resizeSpot(spot);

	imshow("Spot", spot);
	waitKey(0);

	float c1 = getMaxCenter(spot), c2 = calculateSpotK(spot);

	Point pos1, pos2;
	minMaxLoc(abs(spot - c1), NULL, NULL, &pos1, NULL);
	minMaxLoc(abs(spot - c2), NULL, NULL, &pos2, NULL);

	PCA pca = getPCA(spot);
	IKM(spot, pca, pos1, pos2);

	return true;
}

Mat resizeSpot(Mat spot) {
	int width = spot.cols, height = spot.rows;

	int startX = 0, endX = width - 1, startY = 0, endY = height - 1;
	Mat projX, projY;
	reduce(spot, projX, 0, CV_REDUCE_SUM);
	reduce(spot, projY, 1, CV_REDUCE_SUM);
	// 0.004 = single pixel, lowest intensity
	for (int i = 0; i < width; i++) {
		if (projX.at<float>(0, i) > 0.004) {
			startX = i;
			break;
		}
	}

	for (int i = width - 1; i >= 0; i--) {
		if (projX.at<float>(0, i) > 0.004) {
			endX = i;
			break;
		}
	}

	for (int i = 0; i < height; i++) {
		if (projY.at<float>(i, 0) > 0.004) {
			startY = i;
			break;
		}
	}

	for (int i = height - 1; i >= 0; i--) {
		if (projY.at<float>(i, 0) > 0.004) {
			endY = i;
			break;
		}
	}

	Rect myROI(startX, startY, (endX - startX + 1), (endY - startY + 1));

	Mat croppedSpot = spot(myROI);
	return croppedSpot;
}

float calculateSpotK(Mat image) {
	int width, height;
	width = image.cols;
	height = image.rows;

	float temp_k = 1.0;

	float numIter = 10.0;
	float lowBound = 0.6, upBound = 0.9;

	int sqSize = 5;
	int randVal = 500;

	for (int i = 0; i < numIter; i++) {
		std::printf("- Iteration %d:\n", i + 1);
		float minMax = upBound;
		for (int j = 0; j < 12; j++) {
			double max;
			Mat stripped;
			// randomize srand for each side, take into consideration 10 px border and 10 px displacement
			if (j < 6) {
				if (j < 3) {
					srand(time(NULL) + randVal);
					randVal = rand() % (width - sqSize);
					stripped = image(Rect(randVal, 0, sqSize, sqSize)); // upper
				}
				else {
					srand(time(NULL) - randVal);
					randVal = rand() % (width - sqSize);
					stripped = image(Rect(randVal, height - sqSize - 1, sqSize, sqSize)); // lower
				}
			}
			else {
				if (j < 9) {
					srand(time(NULL) + randVal / 2);
					randVal = rand() % (height - sqSize);
					stripped = image(Rect(0, randVal, sqSize, sqSize)); // left
				}
				else {
					srand(time(NULL) - randVal / 2);
					randVal = rand() % (height - sqSize);
					stripped = image(Rect(width - sqSize - 1, randVal, sqSize, sqSize)); // right
				}
			}

			minMaxLoc(stripped, NULL, &max);
			std::printf("Max found: %.4f\n", max);
			if (max < minMax/* && max > lowBound && max < upBound*/) minMax = (float)max;
		}
		std::printf("Minmax chosen: %.4f\n\n", minMax);
		//temp_k += minMax;
		temp_k = min(temp_k, minMax);
	}
	float k = temp_k / numIter;

	std::printf("k trovato: %.4f\n\n", k);


	return k;
}

float getMaxCenter(Mat image) {
	int width = image.cols, height = image.rows;

	double max;
	Mat cntr = image(Rect(width / 2 - 2, height / 2 - 2, 4, 4));
	minMaxLoc(cntr, NULL, &max);

	return (float) max;
}

PCA getPCA(Mat spot) {
	int kradius = 5;
	int ksize = kradius * 2 + 1;
	double ksigma = 2.5;

	int width = spot.cols, height = spot.rows;

	Mat PCAset(8, width * height, CV_32FC1, float(0));

	// Features

	// # 1 : pixel intensity
	Mat intensity = spot.clone().reshape(1, 1);
	PCAset.row(0) = intensity;

	// 2 : gaussian filtered image
	Mat gaussian = Mat();
	GaussianBlur(spot, gaussian, Size(ksize, ksize), ksigma, ksigma);

	imshow("Gauss", gaussian);
	waitKey(0);

	PCAset.row(1) = gaussian.clone().reshape(1, 1);

	// #3-4 : mean and STD
	Mat addIntensity = Mat();
	copyMakeBorder(spot, addIntensity, 1, 1, 1, 1, BORDER_CONSTANT, 0);

	Mat intMean = Mat(height, width, CV_32FC1, float(0)), intStd = Mat(height, width, CV_32FC1, float(0));
	Mat square = Mat(3, 3, CV_32FC1, float(0));

	// 5-6 : Euclidan and City-block distance of pixels to clustering center
	Mat Eurodis = Mat(height, width, CV_32FC1, float(0)), Citydis = Mat(height, width, CV_32FC1, float(0));

	Point maxGauss;
	minMaxLoc(gaussian, NULL, NULL, NULL, &maxGauss);
	int mR = maxGauss.y, nR = maxGauss.x;

	// 7-8 : Row and Cols
	Mat Row = Mat(height, width, CV_32FC1, float(0)), Col = Mat(height, width, CV_32FC1, float(0));

	for (int i = 0; i < height; i++) {
		Row.row(i) = i;
		for (int j = 0; j < width; j++) {
			square = addIntensity(Rect(j, i, 3, 3));

			Scalar mean, std;

			meanStdDev(square, mean, std);

			intMean.at<float>(i, j) = mean[0];
			intStd.at<float>(i, j) = std[0];

			Eurodis.at<float>(i, j) = sqrt(pow(i - mR, 2) + pow(j - nR, 2));
			Citydis.at<float>(i, j) = abs(i - mR) + abs(j - nR);
		}
	}

	for (int j = 0; j < width; j++) Col.col(j) = j;

	PCAset.row(2) = intMean.clone().reshape(1, 1);
	PCAset.row(3) = intStd.clone().reshape(1, 1);


	PCAset.row(4) = Eurodis.clone().reshape(1, 1);
	PCAset.row(5) = Citydis.clone().reshape(1, 1);

	PCAset.row(6) = Row.clone().reshape(1, 1);
	PCAset.row(7) = Col.clone().reshape(1, 1);

	PCA pca(PCAset, Mat(), PCA::DATA_AS_ROW, 3);

	return pca;
}

float eucDist(Mat col1, Mat col2) {
	Mat diff = col1 - col2;
	pow(diff, 2, diff);
	Mat result;
	reduce(diff, result, 0, REDUCE_SUM);
	return result.at<float>(0,0);
}

bool IKM(Mat spot, PCA pca, Point p1, Point p2) {
	Mat data = pca.eigenvectors;
	int width = spot.cols, height = spot.rows;
	int posF = width * p1.y + p1.x, posB = width * p2.y + p2.x;

	int k = 2, dataSetLength = width * height, numberOfFeature = 3, numberOfIterations = 10;

	Mat memberShipMatrix = Mat(k, dataSetLength, CV_32FC1, float(0)),
		centroidMatrix = Mat(3, k, CV_32FC1, float(0)),
		objectiveFunctionSum = Mat(numberOfIterations, 1, CV_32FC1, float(0));

	// centroidMatrix initialization
	centroidMatrix.col(0) = data.col(posF); // foreground
	centroidMatrix.col(1) = data.col(posB); // background

	int iterations = 1;

	while (true) {
		memberShipMatrix = Mat(k, dataSetLength, CV_32FC1, float(0));

		for (int i = 0; i < data.cols; i++) {
			Mat currentDistance = Mat(1, k, CV_32FC1, float(0));
			for (int j = 0; j < k; j++) {
				currentDistance.at<float>(0, j) = eucDist(data.col(i), centroidMatrix.col(j));
			}
			double min; Point minLoc;
			minMaxLoc(currentDistance, &min, NULL, &minLoc, NULL);
			memberShipMatrix.at<float>(minLoc.x, i) = 1.0;
			objectiveFunctionSum.at<float>(iterations - 1, 0) += (float)min;
		}
		Mat clusterSum = Mat(numberOfFeature, k, CV_32FC1, float(0));
		Mat numberOfElementsPerCluster = Mat(1, k, CV_32FC1, float(0));

		int clusterNumber = 0;

		for (int y = 0; y < dataSetLength; y++) {
			for (int z = 0; z < k; z++) {
				if (memberShipMatrix.at<float>(z, y) > 0.0) {
					clusterNumber = z;
					numberOfElementsPerCluster.at<float>(0, z) += 1;
				}
			}
			clusterSum.col(clusterNumber) += data.col(y);
		}

		Mat newCentroid = Mat(numberOfFeature, k, CV_32FC1, float(0));
		for (int u = 0; u < k; u++) {
			newCentroid.col(u) = clusterSum.col(u) / numberOfElementsPerCluster.at<float>(0, u);
		}

		centroidMatrix = newCentroid;

		iterations++;
		if (iterations > numberOfIterations) break;
	}

	Mat trueM = Mat(1, dataSetLength, CV_32FC1, float(0));
	for (int i = 0; i < dataSetLength; i++) {
		if (memberShipMatrix.at<float>(0, i) > 0.0) {
			trueM.at<float>(0, i) = 1.0;
		}
		else {
			trueM.at<float>(0, i) = 0.0;
		}
	}
	trueM = trueM.reshape(1, height);
	Mat notTrueM;
	threshold(trueM, notTrueM, 0.5, 1.0, THRESH_BINARY_INV);
	Mat true1 = trueM.mul(spot);
	Mat true0 = notTrueM.mul(spot);
	float true1Mean = (float)(sum(true1)[0] / sum(trueM)[0]);
	float true0Mean = (float)(sum(true0)[0] / sum(notTrueM)[0]);

	if (true1Mean > true0Mean) trueM = notTrueM;

	imshow("kmeans filtered", true1);
	waitKey(0);

	return true;
}