#include "gridding.h"

bool OpenCVSegmenter::gridding() {
	projections proj;
	projections signals;
	projections reconstructions;
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
			int val = (int)((int)proj.H.at<float>(i, 0) / 1000);
			for (int m = 0; m < val; m++) std::printf("#");
			std::printf(" %d\n", val);
		}
		std::printf("\n\n\n");

		// Vertical
		std::printf("- Vertical histogram:\n");
		for (int j = 0; j < width; j++) {
			int val = (int)((int)proj.V.at<float>(0,j) / 1000);
			for (int n = 0; n < val; n++) std::printf("#");
			std::printf(" %d\n", val);
		}
		std::printf("\n\n\n");
	}

	if (!calculateSignals(proj, &signals, &reconstructions)) return false;

	if (isVisualized()) {
		std::printf("Showing the values for the signals:\n\n");

		std::printf("Horizontal signal:\n");
		for (int i = 0; i < height; i++) {
			double val = (double) signals.H.at<int>(i, 0);
			std::printf(" %f\n", val);
		}
		std::printf("\n\n");

		std::printf("Vertical signal:\n");
		for (int i = 0; i < width; i++) {
			double val = (double) signals.V.at<int>(0, i);
			std::printf(" %f\n", val);
		}
		std::printf("\n\n");
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


Mat getReconstruction(Mat marker, Mat mask) {
	// According to: https://answers.opencv.org/question/35224/morphological-reconstruction/
	Mat m0, m1, geoDilate;

	int dilation_type = MORPH_DILATE;
	int dilation_size = 0;

	m1 = marker;

	Mat diff;
	int iterations = 0;

	do {
		m0 = m1.clone();
		dilate(m0, m1, getStructuringElement(dilation_type,
			Size(2 * dilation_size + 1, 2 * dilation_size + 1),
			Point(dilation_size, dilation_size)));
		min(m1, mask, m1);

		diff = m0 != m1;

		iterations++;
	} while (countNonZero(diff)!=0);

	std::printf("%d iterations needed.\n\n", iterations);

	return m1;
}

bool calculateSignals(projections init_proj, projections* signals, projections* recs){
	// N.B. marker = (H - _H), mask = H
	double H_mean = mean(init_proj.H)[0];
	double V_mean = mean(init_proj.V)[0];

	Mat _H, _V;
	Mat H_mark, V_mark;
	Mat H_rec, V_rec;

	subtract(init_proj.H, H_mean, _H);
	subtract(init_proj.V, V_mean, _V);

	H_rec = getReconstruction(_H, init_proj.H);
	V_rec = getReconstruction(_V, init_proj.V);

	recs->H = H_rec;
	recs->V = V_rec;

	subtract(init_proj.H, H_rec, H_mark);
	subtract(init_proj.V, V_rec, V_mark);


	signals->H = H_mark;
	signals->V = V_mark;

	return true;
}

