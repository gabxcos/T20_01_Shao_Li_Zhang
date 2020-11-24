#include "preprocessing.h"
#include <time.h>
#include <algorithm>

bool OpenCVSegmenter::preprocess() {
	Mat img = getResImage().clone();
	float k = calculateKv2(img);
	Mat ctrEnhImg = contrastEnhance(img, k);
	ctrEnhImg = medianFilter(ctrEnhImg);

	//normalize(ctrEnhImg, ctrEnhImg, 1.0, 0.0, NORM_MINMAX);

	setImage(ctrEnhImg);
	setBgThresh(k);

	return true;
}

float calculateContrast(Mat image)
{
	Mat _image;
	normalize(image, _image, 255.0, 0.0, NORM_MINMAX);

	Mat delta2_image, M4_image;
	pow((_image - mean(_image)[0]), 4, delta2_image);
	float delta2 = mean(delta2_image)[0]; // moment 2
	float delta = sqrt(delta2); // stddev

	pow(delta2_image, 2, M4_image);
	float M4 = mean(M4_image)[0]; // moment 4

	float alfa4 = M4 / pow(delta2, 2); // kurtosis

	float C = delta / pow(alfa4, (1.0/4.0));
	C = 10000.0 / C;

	std::printf("Contrasto: %.4f\n\n", C);
	return C;
}

float calculateK(Mat image) {
	int width, height;
	width = image.cols;
	height = image.rows;

	float temp_k = 0.0;

	float numIter = 200.0;
	float lowBound = 0.6, upBound = 0.9;

	int sqSize = 15;
	int randVal = 500;

	for (int i = 0; i < numIter; i++) {
		std::printf("- Iteration %d:\n", i + 1);
		float minMax = upBound;
		for(int j = 0; j < 12; j++){
			double max;
			Mat stripped;
			// randomize srand for each side, take into consideration 10 px border and 10 px displacement
			if (j < 6) {
				if (j < 3) {
					srand(time(NULL) + randVal);
					randVal = rand() % (width - sqSize - 40);
					stripped = image(Rect(randVal + 20, 20, sqSize, sqSize)); // upper
				}
				else {
					srand(time(NULL) - randVal);
					randVal = rand() % (width - sqSize - 40);
					stripped = image(Rect(randVal + 20, height - sqSize - 21, sqSize, sqSize)); // lower
				}
			}
			else {
				if (j < 9) {
					srand(time(NULL) + randVal/2);
					randVal = rand() % (height - sqSize - 40);
					stripped = image(Rect(20, randVal + 20, sqSize, sqSize)); // left
				}
				else {
					srand(time(NULL) - randVal / 2);
					randVal = rand() % (height - sqSize - 40);
					stripped = image(Rect(width - sqSize - 21, randVal + 20, sqSize, sqSize)); // right
				}
			}
			/*
			srand(time(NULL) + randValX);
			randValX = rand() % (width - 150);

			srand(time(NULL) + randValY);
			randValY = rand() % (height - 150);

			stripped = image(Rect(randValX, randValY, 150, 150));
			*/
			minMaxLoc(stripped, NULL, &max);
			std::printf("Max found: %.4f\n", max);
			if (max < minMax && max > lowBound && max < upBound) minMax = (float)max;
		}
		std::printf("Minmax chosen: %.4f\n\n", minMax);
		temp_k += minMax;
	}
	float k = temp_k / numIter;

	std::printf("k trovato: %.4f\n\n", k);

	return k;
}

float calculateKv2(Mat image) {
	int width, height;
	width = image.cols;
	height = image.rows;

	float temp_k = 0.0;

	float numIter = 200.0;

	int sqSize = 100;
	int maxX = width / sqSize - 1;
	int maxY = height / sqSize - 1;

	float lowBound = mean(image)[0], upBound = 0.85;


	// T-L
	for (int x = 0; x < maxX; x++) {
		float minMax = 1.0;
		for (int y = 0; y < maxY; y++) {
			double max;
			Mat stripped;
			stripped = image(Rect(x * sqSize, y * sqSize, sqSize, sqSize));
			minMaxLoc(stripped, NULL, &max);
			if (max < lowBound) max = lowBound;
			if (max > upBound) max = upBound;
			if (max < minMax) minMax = (float)max;
		}
		temp_k += minMax;
	}
	// T-R
	for (int x = 0; x < maxX; x++) {
		float minMax = 1.0;
		for (int y = 0; y < maxY; y++) {
			double max;
			Mat stripped;
			stripped = image(Rect(width - 1 - (x + 1) * sqSize, y * sqSize, sqSize, sqSize));
			minMaxLoc(stripped, NULL, &max);
			if (max < lowBound) max = lowBound;
			if (max > upBound) max = upBound;
			if (max < minMax) minMax = (float)max;
		}
		temp_k += minMax;
	}
	// B-R
	for (int x = 0; x < maxX; x++) {
		float minMax = 1.0;
		for (int y = 0; y < maxY; y++) {
			double max;
			Mat stripped;
			stripped = image(Rect(width - 1 - (x + 1) * sqSize, height - 1 - (y + 1) * sqSize, sqSize, sqSize));
			minMaxLoc(stripped, NULL, &max);
			if (max < lowBound) max = lowBound;
			if (max > upBound) max = upBound;
			if (max < minMax) minMax = (float)max;
		}
		temp_k += minMax;
	}
	// B-L
	for (int x = 0; x < maxX; x++) {
		float minMax = 1.0;
		for (int y = 0; y < maxY; y++) {
			double max;
			Mat stripped;
			stripped = image(Rect(x * sqSize, height - 1 - (y + 1) * sqSize, sqSize, sqSize));
			minMaxLoc(stripped, NULL, &max);
			if (max < lowBound) max = lowBound;
			if (max > upBound) max = upBound;
			if (max < minMax) minMax = (float)max;
		}
		temp_k += minMax;
	}



	float k = temp_k / (4.0 * maxX);

	std::printf("k trovato: %.4f\n\n", k);

	return k;
}


Mat contrastEnhance(Mat image, float k) {
	float ctr = calculateContrast(image);
	Mat contrastedImg = image;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (contrastedImg.at<float>(i, j) >= k) {
				float currVal = contrastedImg.at<float>(i, j);
				float newVal = currVal * ctr;
				if (newVal > 1.0) newVal = 1.0;
				contrastedImg.at<float>(i, j) = newVal;
			}
			/*else {
				contrastedImg.at<float>(i, j) /= ctr;
			}*/
		}
	}
	return contrastedImg;
}

Mat medianFilter(Mat image) {
	int width = image.cols, height = image.rows;

	Mat A = image;
	Mat C(height, width, CV_32FC1, float(0));

	for (int i = 1; i < (height - 1); i++) {
		for (int j = 1; j < (width - 1); j++) {
			float B[9];
			for (int k = -1; k < 2; k++) {
				for (int l = -1; l < 2; l++) {
					B[k * 3 + l + 4] = A.at<float>(i + k, j + l);
				}
			}
			std::sort(std::begin(B), std::end(B));
			A.at<float>(i, j) = B[4];
		}
	}

	return A;
}