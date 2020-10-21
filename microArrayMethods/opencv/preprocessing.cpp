#include "preprocessing.h"

bool OpenCVSegmenter::preprocess() {
	float ctr = calculateContrast(getImage());
	std::printf("Contrasto: %.4f\n\n", ctr);
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

	return 10000.0 / C;
}

float calculateK(Mat image) {
	int width, height;
	width = image.cols;
	height = image.rows;

	for (int i = 0; i < 10; i++) {
		break;
	}

	return 0.0;
}
