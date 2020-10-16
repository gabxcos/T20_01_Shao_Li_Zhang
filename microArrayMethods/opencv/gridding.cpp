#include "gridding.h"

bool OpenCVSegmenter::gridding() {
	projections proj, reconstructions, signals, binarySignals;
	Mat img = getImage();
	int width = getWidth();
	int height = getHeight();
	bool continuous = isContinuous();

	if (!getProjections(img, &proj)) return false;
	
	if(isVisualized()){
		std::printf("Projections obtained! Testing results:\n\n");

		std::printf("- Calculated rows: %d, actual: %d\n", proj.H.rows, height);
		std::printf("- Calculated cols: %d, actual: %d\n\n", proj.V.cols, width);
		
		// Horizontal
		std::printf("- Horizontal histogram:\n");
		for (int i = 0; i < height; i++) {
			float val = proj.H.at<float>(i, 0);
			for (int m = 0; m < (val*255); m++) std::printf("#");
			std::printf(" %.2f\n", val);
		}
		std::printf("\n\n\n");
		
		// Vertical
		std::printf("- Vertical histogram:\n");
		for (int j = 0; j < width; j++) {
			float val = proj.V.at<float>(0,j);
			for (int n = 0; n < (val*255); n++) std::printf("#");
			std::printf(" %.2f\n", val);
		}
		std::printf("\n\n\n");

	}

	int kernelSize = calculateKernelSize(proj.H, proj.V);
	if (isVisualized()) std::printf("\n\nIdeal kernel size found: %d\n\n", kernelSize);

	if (!calculateSignals(proj, &signals, &reconstructions, kernelSize)) return false;

	if (isVisualized()) {
		std::printf("Showing the values for the signals:\n\n");

		std::printf("Horizontal signal:\n");
		for (int i = 0; i < height; i++) {
			float val = signals.H.at<float>(i, 0);
			std::printf(" %f\n", val);
		}
		std::printf("\n\n");

		std::printf("Vertical signal:\n");
		for (int i = 0; i < width; i++) {
			float val = signals.V.at<float>(0, i);
			std::printf(" %f\n", val);
		}
		std::printf("\n\n");
	}

	getThreshold(signals.H);
	
	if (getBinarySignals(&signals, &binarySignals)) {
		Mat visualizeV(100, binarySignals.V.cols, CV_32FC1);
		repeat(binarySignals.V, 100, 1, visualizeV);
		Mat visualizeH(binarySignals.H.rows, 100, CV_32FC1);
		repeat(binarySignals.H, 1, 100, visualizeH);
		imshow("Image", getImage());
		imshow("V", visualizeV);
		imshow("H", visualizeH);
		waitKey(0);
	}

	return true;
}

bool getProjections(Mat image, projections* proj_set)
{
	/*int width = image.cols;
	int height = image.rows;
	bool continuous = image.isContinuous();*/

	/* OLD DEFINITION
	int y, x;

	int* H = (int*)calloc(height, sizeof(int));
	if (H == NULL) return false;
	int* V = (int*)calloc(width, sizeof(int));
	if (V == NULL) return false;

	uchar* p;

	for (y = 0; y < height; ++y)
	{
		p = continuous ? image.ptr<uchar>(0) : image.ptr<uchar>(y);

		for (x = 0; x < width; ++x)
		{
			int intensity = continuous ? (int)p[y * width + x] : (int)p[x];
			H[y] += intensity;
			V[x] += intensity;
			// intensity - 1 ??
		}
	}*/

	Mat H, V;
	cv::reduce(image, H, 1, CV_REDUCE_AVG, CV_32FC1);
	cv::reduce(image, V, 0, CV_REDUCE_AVG, CV_32FC1);

	proj_set->H = H;
	proj_set->V = V;

	return true;
}

int calculateKernelSize(Mat H, Mat V) {
	float accum = 0;
	float num_nonzero = 0;
	bool wasZero= true, isNonZero;

	// H
	int height = H.rows;
	for (int i = 0; i < height; i++) {
		float val = H.at<float>(i, 0);
		isNonZero = val > 0;
		if (isNonZero) accum++;
		if (!isNonZero && !wasZero) num_nonzero++;
		wasZero = !isNonZero;
	}

	// V
	int width = V.cols;
	for (int i = 0; i < width; i++) {
		float val = V.at<float>(0, i);
		isNonZero = val > 0;
		if (isNonZero) accum++;
		if (!isNonZero && !wasZero) num_nonzero++;
		wasZero = !isNonZero;
	}

	return (int)ceil(accum/num_nonzero);
}


Mat getReconstruction(Mat marker, Mat mask, int kernelSize) {
	// According to: https://answers.opencv.org/question/35224/morphological-reconstruction/
	// With "Opening" method from "morphologyEx"
	Mat m0, m1;


	int morph_elem = 2; // Ellipse
	int morph_size = 2;

	int const max_elem = 2;
	int const max_kernel_size = 21;

	morph_size = max(morph_size, min(kernelSize, max_kernel_size));

	m1 = marker;

	Mat diff;
	int iterations = 0;

	do {
		m0 = m1.clone();
		morphologyEx(m0, m1, MORPH_OPEN, getStructuringElement(morph_elem,
			Size(2 * morph_size + 1, 2 * morph_size + 1),
			Point(morph_size, morph_size)));
		min(m1, mask, m1);

		diff = m0 != m1;

		iterations++;
	} while (countNonZero(diff)!=0);

	std::printf("%d iterations needed.\n\n", iterations);

	return m1;
}

bool calculateSignals(projections init_proj, projections* signals, projections* recs, int kernelSize){
	// N.B. marker = (H - _H), mask = H
	float H_mean = (float)mean(init_proj.H)[0];
	float V_mean = (float)mean(init_proj.V)[0];

	Mat _H, _V;
	Mat H_mark, V_mark;
	Mat H_rec, V_rec;

	subtract(init_proj.H, H_mean, _H);
	subtract(init_proj.V, V_mean, _V);

	// TEMPTATIVE CHANGES ! ! Guarantee no negative values
	max(_H, Mat(_H.rows, 1, CV_32FC1, float(0)), _H);
	max(_V, Mat(1, _V.cols, CV_32FC1, float(0)), _V);
	// ----------------------------------------------------

	H_rec = getReconstruction(_H, init_proj.H, kernelSize);
	V_rec = getReconstruction(_V, init_proj.V, kernelSize);

	recs->H = H_rec;
	recs->V = V_rec;

	subtract(init_proj.H, H_rec, H_mark);
	subtract(init_proj.V, V_rec, V_mark);


	signals->H = H_mark;
	signals->V = V_mark;

	return true;
}

float getThreshold(Mat M) {
	float L;
	double max, min;
	minMaxLoc(M, &min, &max);
	L = (float)max;



	return L;
}

bool getBinarySignals(projections* signals, projections* binarySignals) {

	float threshH = 0.5 * (float) mean(signals->H)[0];
	float threshV = 0.5 * (float) mean(signals->V)[0];

	// H
	int height = (signals->H).rows;
	Mat binH(height, 1, CV_32FC1, float(0));
	for (int i = 0; i < height; i++) {
		if (signals->H.at<float>(i, 0) > threshH) binH.at<float>(i, 0) = 255.0;
	}
	binarySignals->H = binH;

	// H
	int width = (signals->V).cols;
	Mat binV(1, width, CV_32FC1, float(0));
	for (int i = 0; i < width; i++) {
		if (signals->V.at<float>(0, i) > threshV) binV.at<float>(0, i) = 255.0;
	}
	binarySignals->V = binV;

	return true;
}