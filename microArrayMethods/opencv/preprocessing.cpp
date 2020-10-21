#include "preprocessing.h"
#include <time.h>
#include <algorithm>

bool OpenCVSegmenter::preprocess() {
	Mat img = getImage();
	Mat ctrEnhImg = contrastEnhance(img);
	ctrEnhImg = medianFilter(ctrEnhImg);

	setImage(ctrEnhImg);
	resizeImage();

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

	int randValX = 500, randValY = -500;

	for (int i = 0; i < 10; i++) {
		std::printf("- Iteration %d:\n", i + 1);
		float minMax = 0.9;
		for(int j = 0; j < 12; j++){
			double max;
			Mat stripped;
			/*if (j < 6) {
				randVal = rand() % (width - 100);
				if (j < 3) {
					stripped = image(Rect(randVal, 0, 100, 100)); // upper
				}
				else {
					stripped = image(Rect(randVal, height - 101, 100, 100)); // lower
				}
			}
			else {
				randVal = rand() % (height - 100);
				if (j < 9) {
					stripped = image(Rect(0, randVal, 100, 100)); // left
				}
				else {
					stripped = image(Rect(width - 101, randVal, 100, 100)); // right
				}
			}*/
			srand(time(NULL) + randValX);
			randValX = rand() % (width - 150);

			srand(time(NULL) + randValY);
			randValY = rand() % (height - 150);

			stripped = image(Rect(randValX, randValY, 150, 150));

			minMaxLoc(stripped, NULL, &max);
			std::printf("Max found: %.4f\n", max);
			if (max < minMax && max > 0) minMax = (float)max;
		}
		std::printf("Minmax chosen: %.4f\n\n", minMax);
		temp_k += minMax;
	}
	float k = temp_k / 10.0;

	std::printf("k trovato: %.4f\n\n", k);

	return k;
}


Mat contrastEnhance(Mat image) {
	float ctr = calculateContrast(image);
	float k = calculateK(image);
	Mat contrastedImg = image;
	for (int i = 0; i < image.rows; i++) {
		for (int j = 0; j < image.cols; j++) {
			if (contrastedImg.at<float>(i, j) > k) {
				float currVal = contrastedImg.at<float>(i, j);
				contrastedImg.at<float>(i, j) = (float)max(1.0, (double)currVal * ctr);
			}
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