#include "gridding.h"

bool OpenCVSegmenter::gridding() {
	projections proj, reconstructions, signals, binarySignals;
	Mat img = getImage().clone();
	int width = img.cols;
	int height = img.rows;
	bool continuous = isContinuous();
	vector<int> Hlines, Vlines;

	if (!getProjections(img, &proj)) return false;
	

	/*if(isVisualized()){
		std::printf("Projections obtained! Testing results:\n\n");

		std::printf("- Calculated rows: %d, actual: %d\n", proj.H.rows, height);
		std::printf("- Calculated cols: %d, actual: %d\n\n", proj.V.cols, width);
		
		// Horizontal
		std::printf("- Horizontal histogram:\n");
		for (int i = 0; i < height; i++) {
			float val = proj.H.at<float>(i, 0);
			for (int m = 0; m < (val/255); m++) std::printf("#");
			std::printf(" %.2f\n", val);
		}
		std::printf("\n\n\n");
		
		// Vertical
		std::printf("- Vertical histogram:\n");
		for (int j = 0; j < width; j++) {
			float val = proj.V.at<float>(0,j);
			for (int n = 0; n < (val/255); n++) std::printf("#");
			std::printf(" %.2f\n", val);
		}
		std::printf("\n\n\n");

	}*/

	
	if (!calculateSignals(proj, &signals, &reconstructions)) return false;
	

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
	
	if (getBinarySignals(&signals, &binarySignals)) {
		Mat visualizeV(100, binarySignals.V.cols, CV_32FC1);
		repeat(binarySignals.V, 100, 1, visualizeV);
		Mat visualizeH(binarySignals.H.rows, 100, CV_32FC1);
		repeat(binarySignals.H, 1, 300, visualizeH);
		//imshow("Image", getImage());
		//imshow("V", visualizeV);
		//imshow("H", visualizeH);
		//waitKey(0);
	}

	Hlines = getHlines(binarySignals.H);
	Vlines = getVlines(binarySignals.V);

	adjustToDevice(this, getDevice(), getImage(), Hlines, Vlines, binarySignals);

	Mat rgbImage;
	Mat ogImage = getImage().clone();
	ogImage.convertTo(ogImage, CV_8U, 255);
	cvtColor(ogImage, rgbImage, COLOR_GRAY2BGR);

	vector<int>::iterator it;
	for (it = Hlines.begin(); it != Hlines.end(); it++) line(rgbImage, Point(0, *it), Point(width - 1, *it), Scalar(0, 0, 255));
	for (it = Vlines.begin(); it != Vlines.end(); it++) line(rgbImage, Point(*it, 0), Point(*it, height - 1), Scalar(0, 0, 255));

	//imshow("Image", rgbImage);
	//waitKey(0);
	String temp_file = getOutPath() + "\\" + "grid.png";
	rgbImage.convertTo(rgbImage, CV_8U);
	imwrite(temp_file, rgbImage);

	setGrid(Hlines, Vlines);

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

	//cv::normalize(H, H, 1.0, 0.0, cv::NORM_MINMAX);
	//cv::normalize(V, V, 1.0, 0.0, cv::NORM_MINMAX);

	proj_set->H = H;
	proj_set->V = V;

	return true;
}

int calculateKernelSize(Mat M, bool horizontal) {
	float accum = 0;
	float num_nonzero = 0;
	bool wasZero= true, isNonZero;

	if (horizontal) {
		// H
		int height = M.rows;
		for (int i = 0; i < height; i++) {
			float val = M.at<float>(i, 0);
			isNonZero = val > 0;
			if (isNonZero) accum++;
			if (!isNonZero && !wasZero) num_nonzero++;
			wasZero = !isNonZero;
		}
	}
	else {
		// V
		int width = M.cols;
		for (int i = 0; i < width; i++) {
			float val = M.at<float>(0, i);
			isNonZero = val > 0;
			if (isNonZero) accum++;
			if (!isNonZero && !wasZero) num_nonzero++;
			wasZero = !isNonZero;
		}
	}
	return (int)ceil(accum/num_nonzero);
}


Mat getReconstruction(Mat marker, Mat mask, int kernelSize) {
	// According to: https://answers.opencv.org/question/35224/morphological-reconstruction/	
	
	// With "Dilation" method from: https://it.mathworks.com/help/images/understanding-morphological-reconstruction.html
	Mat m0, m1;


	int morph_elem = MORPH_ELLIPSE;
	int morph_size = 2;

	int const max_kernel_size = 21;

	morph_size = max(morph_size, min(kernelSize, max_kernel_size));

	m1 = marker;

	Mat diff;
	int iterations = 0;
	int maxIterations = 15;

	do {
		m0 = m1.clone();
		dilate(m0, m1, getStructuringElement(morph_elem,
			Size(2 * morph_size + 1, 2 * morph_size + 1)));/* ,
			Point(morph_size, morph_size)));*/
		min(m1, mask, m1);

		diff = m0 != m1;

		iterations++;
	} while (countNonZero(diff) != 0 && iterations < maxIterations);

	std::printf("%d iterations needed.\n\n", iterations);

	return m1;
	
	/*// With "Opening" method from "morphologyEx"
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
	
	return m1;*/
}

bool calculateSignals(projections init_proj, projections* signals, projections* recs){
	// N.B. marker = (H - _H), mask = H
	float H_mean = (float)mean(init_proj.H)[0];
	float V_mean = (float)mean(init_proj.V)[0];

	Mat _H, _V;
	Mat H_mark, V_mark;
	Mat H_rec, V_rec;

	subtract(init_proj.H, H_mean, _H);
	subtract(init_proj.V, V_mean, _V);

	// TEMPTATIVE CHANGES ! ! Guarantee no negative values
	//max(_H, Mat(_H.rows, 1, CV_32FC1, float(0)), _H);
	//max(_V, Mat(1, _V.cols, CV_32FC1, float(0)), _V);
	// ----------------------------------------------------

	H_rec = getReconstruction(_H, init_proj.H, calculateKernelSize(_H, true));
	V_rec = getReconstruction(_V, init_proj.V, calculateKernelSize(_V, false));

	recs->H = H_rec;
	recs->V = V_rec;

	subtract(init_proj.H, H_rec, H_mark);
	subtract(init_proj.V, V_rec, V_mark);

	//cv::normalize(H_mark, H_mark, 1.0, 0.0, cv::NORM_MINMAX);
	//cv::normalize(V_mark, V_mark, 1.0, 0.0, cv::NORM_MINMAX);

	signals->H = H_mark;
	signals->V = V_mark;

	return true;
}

float getThreshold(Mat M) {
	Mat M_;
	M.convertTo(M_, CV_8UC1, 255, 0); // checked, converts correctly

	int L;
	double max, min;
	minMaxLoc(M_, &min, &max);
	L = (int)max;

	Mat hold_count(1, L + 1, CV_32FC1, float(0));
	int n = 0;
	for (int i = 0; i < M_.rows; i++) {
		for (int j = 0; j < M_.cols; j++) {
			int val = (int)M_.at<uchar>(i, j);
			hold_count.at<float>(0, val) += 1.0;
			n++;
		}
	}
	hold_count /= (float)n;

	Mat w(1, L + 1, CV_32FC1, float(0));
	float temp_w = 0;

	// first-order
	float u = 0; // mean
	Mat ut(1, L + 1, CV_32FC1, float(0));

	for (int i = 1; i < (L + 1); i++) {
		float pi = hold_count.at<float>(0, i);
		u += pi * i;
		ut.at<float>(0, i) = u;

		temp_w += pi;
		w.at<float>(0, i) = temp_w;
	}

	// second-order
	float u2 = 0; // mean
	Mat ut2(1, L + 1, CV_32FC1, float(0));

	for (int i = 1; i < (L + 1); i++) {
		float pi = hold_count.at<float>(0, i);

		u2 += pi * pow((i - u), 2);
		// u2 += pow(pi * (i - u), 2); // in Matlab
		float ut2_temp = 0;
		for (int j = 1; j < i + 1; j++) {
			ut2_temp += pi * pow((j - ut.at<float>(0, i)), 2);
			// ut2_temp += pow(pi * (i - ut.at<float>(0, i)), 2); // in Matlab
		}
		ut2_temp = sqrt(ut2_temp / i);
		ut2.at<float>(0, i) = ut2_temp;
	}
	u2 = sqrt(u2 / L);

	Mat d_cand(1, L + 1, CV_32FC1, float(0));
	for (int i = 0; i < (L + 0); i++) { // varying i=t
		float w_, ut2_;
		w_ = w.at<float>(0, i);
		ut2_ = ut2.at<float>(0, i);

		d_cand.at<float>(0, i) = u2 * pow((w_ - ut2_), 2) / (w_ * (1.0 - w_));
	}

	Point min_loc, max_loc;
	minMaxLoc(d_cand, &min, &max, &min_loc, &max_loc);

	float my_max = (float)max_loc.x / 255.0;

	std::printf("\n\nThreshold: %.4f\n\n", my_max);
	/*std::printf("Candidates: ");
	for (int i = 1; i < (L + 1); i++) std::printf("%.4f ", d_cand.at<float>(0, i));*/

	return my_max;
}

float getThresholdV2(Mat M) {
	// according to: https://www.sciencedirect.com/topics/engineering/class-variance
	Mat M_;
	M.convertTo(M_, CV_8UC1, 255, 0); // checked, converts correctly
	// number of levels = 256 ( 0 - 255 )
	Mat BCV(1, 256, CV_32FC1, float(0)); // Between-Class Variance

	Mat frequencies(1, 256, CV_32FC1, float(0)); // frequencies
	int n = M_.rows * M_.cols;
	for (int i = 0; i < M_.rows; i++) {
		for (int j = 0; j < M_.cols; j++) {
			int val = (int)M_.at<uchar>(i, j);
			frequencies.at<float>(0, val) += 1.0;
		}
	}
	frequencies /= n;

	Mat addFreq(1, 256, CV_32FC1, float(0)); // PI0, PI1 obtained as 1-PI0
	float cumulativeFreq = 0.0;
	for (int i = 0; i < 256; i++) {
		cumulativeFreq += frequencies.at<float>(0, i);
		addFreq.at<float>(0, i) = cumulativeFreq;
	}

	Mat mi0(1, 256, CV_32FC1, float(0));
	Mat mi1(1, 256, CV_32FC1, float(0));

	float temp_mi0 = 0.0;
	float temp_mi1 = 0.0;
	for (int i = 1; i < 256; i++) {
		temp_mi0 += i * frequencies.at<float>(0, i) / addFreq.at<float>(0, i);
		mi0.at<float>(0, i) = temp_mi0;

		int j = 256 - i;
		temp_mi1 += j * frequencies.at<float>(0, j) / (1 - addFreq.at<float>(0, j));
		mi1.at<float>(0, j) = temp_mi1;
	}

	// get BCV
	for (int i = 1; i < 256; i++) {
		float my_pi0, my_mi0, my_mi1;
		my_pi0 = addFreq.at<float>(0, i);
		my_mi0 = mi0.at<float>(0, i);
		my_mi1 = mi1.at<float>(0, i);

		BCV.at<float>(0, i) = my_pi0 * (1 - my_pi0)* pow((my_mi1 - my_mi0), 2);
	}
	double min, max;
	Point minLoc, maxLoc;
	minMaxLoc(BCV, &min, &max, &minLoc, &maxLoc);

	float threshold = (float)maxLoc.x / 255.0;

	std::printf("\n\nThreshold: %.4f\n\n", threshold);
	/*std::printf("Candidates: ");
	for (int i = 1; i < 256; i++) std::printf("%.4f ", BCV.at<float>(0, i));
	*/
	return threshold;
}

bool getBinarySignals(projections* signals, projections* binarySignals) {

	/*float threshH = getThresholdV2(signals->H); // naive: 0.5 * (float) mean(signals->H)[0];
	float threshV = getThresholdV2(signals->V); // naive: 0.5 * (float) mean(signals->V)[0];

	// H
	int height = (signals->H).rows;
	Mat binH(height, 1, CV_32FC1, float(0));
	for (int i = 0; i < height; i++) {
		if (signals->H.at<float>(i, 0) > threshH) {
			if ((i == 0 || i == (height - 1)) || (i > 0 && signals->H.at<float>(i - 1, 0) > threshH) || (i < (height - 1) && signals->H.at<float>(i + 1, 0) > threshH)) binH.at<float>(i, 0) = float(255);
		}
	}
	binarySignals->H = binH;

	// H
	int width = (signals->V).cols;
	Mat binV(1, width, CV_32FC1, float(0));
	for (int i = 0; i < width; i++) {
		if (signals->V.at<float>(0, i) > threshV) {
			if((i == 0 || i == (width - 1)) || (i>0 && signals->V.at<float>(0, i-1) > threshV) || (i < (width - 1) && signals->V.at<float>(0, i + 1) > threshV)) binV.at<float>(0, i) = float(255);
		}
	}
	binarySignals->V = binV;
	*/

	Mat ogH = signals->H.clone(), ogV = signals->V.clone();
	ogH.convertTo(ogH, CV_8UC1, 255);
	ogV.convertTo(ogV, CV_8UC1, 255);

	threshold(ogH, ogH, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);
	threshold(ogV, ogV, 0.0, 255.0, cv::THRESH_BINARY | cv::THRESH_OTSU);

	ogH.convertTo(ogH, CV_32FC1);
	ogV.convertTo(ogV, CV_32FC1);

	if (ogH.at<float>(0, 0) > 254) {
		int i = 0;
		while (ogH.at<float>(i, 0) > 254) {
			ogH.at<float>(i, 0) = 0.0;
			i++;
		}
	}
	if (ogH.at<float>((ogH.rows - 1), 0) > 254) {
		int i = ogH.rows - 1;
		while (ogH.at<float>(i, 0) > 254) {
			ogH.at<float>(i, 0) = 0.0;
			i--;
		}
	}

	if (ogV.at<float>(0, 0) > 254) {
		int i = 0;
		while (ogV.at<float>(0, i) > 254) {
			ogH.at<float>(0, i) = 0.0;
			i++;
		}
	}
	if (ogV.at<float>(0, (ogV.cols - 1)) > 254) {
		int i = ogV.cols - 1;
		while (ogV.at<float>(0, i) > 254) {
			ogV.at<float>(0, i) = 0.0;
			i--;
		}
	}

	binarySignals->H = ogH;
	binarySignals->V = ogV;

	return true;
}

vector<int> getHlines(Mat H) {
	int mm = 20;
	int height = H.rows;

	vector<int> hspot, hlspot;

	int t = 0;
	int flag = 0;

	for (int i = 0; i < height; i++) {
		if (flag == 0) {
			if (H.at<float>(i, 0) > 254){ // && H.at<float>(i + 1, 0) > 254 && H.at<float>(i + 2, 0) > 254) {
				hspot.push_back(i);
				flag++;
				t = 0;
			}
		}
		if (flag == 1) {
			t++;
			if (H.at<float>(i+1, 0) < 1 && t > 4) {
				hspot.push_back(i);
				flag = 0;
			}
		}
	}

	hlspot.push_back(max(hspot.at(0) - mm, 0));
	for (int i = 1; i < (hspot.size() - 1); i += 2) {
		hlspot.push_back((int)round((hspot.at(i)+hspot.at(i+1)) / 2));
	}
	hlspot.push_back(min(hspot.at(hspot.size() - 1) + mm, height-1));

	return hlspot;
}

vector<int> getVlines(Mat V) {
	int mm = 20;
	int width = V.cols;

	vector<int> vspot, vlspot;

	int t = 0;
	int flag = 0;

	for (int i = 0; i < width; i++) {
		if (flag == 0) {
			if (V.at<float>(0, i) > 254){ // && V.at<float>(0, i + 1) > 254 && V.at<float>(0, i + 2) > 254) {
				vspot.push_back(i);
				flag++;
				t = 0;
			}
		}
		if (flag == 1) {
			t++;
			if (V.at<float>(0, i + 1) < 1 && t > 4) {
				vspot.push_back(i);
				flag = 0;
			}
		}
	}

	vlspot.push_back(max(vspot.at(0) - mm, 0));
	for (int i = 1; i < (vspot.size() - 1); i += 2) {
		vlspot.push_back((int)round((vspot.at(i) + vspot.at(i + 1)) / 2));
	}
	vlspot.push_back(min(vspot.at(vspot.size() - 1) + mm, width - 1));

	return vlspot;
}

bool deleteEmptyLines(Mat image, vector<int>& Hlines, vector<int>& Vlines, bool first) {
	bool hasEliminated = false;
	float thresh = 0.005;

	int numHlines = Hlines.size(), numVlines = Vlines.size();
	printf("H = %d, V = %d\n", numHlines, numVlines);

	float minMean = 1.0;
	int minIndex = -1;
	// Horizontal strips
	std::printf("\nHorizontal elimination:\n");
	int startV = Vlines[0], endV = Vlines[numVlines - 1];
	for (int i = 0; i < (numHlines - 1); i++) {
		Mat testImg = image(Rect(startV, Hlines[i], endV - startV, Hlines[i+1] - Hlines[i]));
		float curr_mean = mean(testImg)[0];
		std::printf("- Mean: %.4f\n", curr_mean);
		if (curr_mean < minMean) {
			minMean = curr_mean;
			minIndex = i;
		}
	}
	std::printf("- - Final: mean: %.4f, index: %d\n", minMean, minIndex);
	if (minMean < thresh && (minIndex == 0 || minIndex == (numHlines - 2))) {
		std::printf("\nDeleting horizontal line...\n");

		if (minIndex == 0) Hlines.erase(Hlines.begin() + minIndex);
		else if (minIndex == (numHlines - 2)) Hlines.erase(Hlines.begin() + minIndex + 1);

		numHlines--;
		hasEliminated = true;
	}

	minMean = 1.0;
	minIndex = -1;
	// Vertical strips
	std::printf("\nVertical elimination:\n");
	int startH = Hlines[0], endH = Hlines[numHlines - 1];
	for (int i = 0; i < (numVlines - 1); i++) {
		Mat testImg = image(Rect(Vlines[i], startH, Vlines[i + 1] - Vlines[i], endH - startH));
		float curr_mean = mean(testImg)[0];
		std::printf("- Mean: %.4f\n", curr_mean);
		if (curr_mean < minMean) {
			minMean = curr_mean;
			minIndex = i;
		}
	}
	std::printf("- - Final: mean: %.4f, index: %d\n", minMean, minIndex);
	if (minMean < thresh && (minIndex == 0 || minIndex == (numVlines - 2))) {
		std::printf("\nDeleting vertical line...\n");

		if (minIndex == 0) Vlines.erase(Vlines.begin() + minIndex);
		else if (minIndex == (numVlines - 2)) Vlines.erase(Vlines.begin() + minIndex + 1);

		numVlines--;
		hasEliminated = true;
	}
	

	if (!first || !hasEliminated) return true;
	else return deleteEmptyLines(image, Hlines, Vlines, false);
}

bool setToAngles(OpenCVSegmenter* seg, Mat image, vector<int>& Hlines, vector<int>& Vlines, projections binarySignals) {
	int startX = 0, startY = 0, endX = Vlines.size() - 1, endY = Hlines.size() - 1;

	/* OLD
	
	bool resize;

	float Pbias = 1.01;
	float Nbias = 0.99;


	// INTUITION: totally empty rows or cols can influence the spot probability algorithm
	float tempRes = 100.0;
	float minThresh = 50.0;
	do { // remove top
		int _x = Vlines[startX], _y = Hlines[startY], _w = Vlines[startX + 1] - _x, _h = Hlines[startY + 1] - _y;
		Mat leftImg = image(Rect(_x, _y, _w, _h));
		float left = seg->spotScore(leftImg);

		_x = Vlines[endX - 1], _w = Vlines[endX] - _x;
		Mat rightImg = image(Rect(_x, _y, _w, _h));
		float right = seg->spotScore(leftImg);
		tempRes = max(left, right);

		if (tempRes < minThresh) {
			startY++;
		}
	} while (tempRes < minThresh);

	do { // remove bottom
		int _x = Vlines[startX], _y = Hlines[endY - 1], _w = Vlines[startX + 1] - _x, _h = Hlines[endY] - _y;
		Mat leftImg = image(Rect(_x, _y, _w, _h));
		float left = seg->spotScore(leftImg);

		_x = Vlines[endX - 1], _w = Vlines[endX] - _x;
		Mat rightImg = image(Rect(_x, _y, _w, _h));
		float right = seg->spotScore(leftImg);
		tempRes = max(left, right);

		if (tempRes < minThresh) {
			endY--;
		}
	} while (tempRes < minThresh);

	do { // remove left
		int _x = Vlines[startX], _y = Hlines[startY], _w = Vlines[startX + 1] - _x, _h = Hlines[startY + 1] - _y;
		Mat topImg = image(Rect(_x, _y, _w, _h));
		float top = seg->spotScore(topImg);

		_y = Hlines[endY - 1], _h = Hlines[endY] - _y;
		Mat botImg = image(Rect(_x, _y, _w, _h));
		float bot = seg->spotScore(botImg);
		tempRes = max(top, bot);

		if (tempRes < minThresh) {
			startX++;
		}
	} while (tempRes < minThresh);

	do { // remove right
		int _x = Vlines[endX - 1], _y = Hlines[startY], _w = Vlines[endX] - _x, _h = Hlines[startY + 1] - _y;
		Mat topImg = image(Rect(_x, _y, _w, _h));
		float top = seg->spotScore(topImg);

		_y = Hlines[endY - 1], _h = Hlines[endY] - _y;
		Mat botImg = image(Rect(_x, _y, _w, _h));
		float bot = seg->spotScore(botImg);
		tempRes = max(top, bot);

		if (tempRes < minThresh) {
			endX--;
		}
	} while (tempRes < minThresh);

	/////////////////////////////////////////////////////////////////////////
	// angle probability algorithm

	float bestMean = calculateAngleProb(seg, startX, startY, endX, endY, image, Hlines, Vlines);
	int newStartX = startX, newStartY = startY, newEndX = endX, newEndY = endY;
	do{
		startX = newStartX, startY = newStartY, endX = newEndX, endY = newEndY;
		resize = false;
		float currMean;
		float bias;
		// L-R
		currMean = calculateAngleProb(seg, startX + 1, startY, endX, endY, image, Hlines, Vlines);
		currMean > 0 ? bias = Pbias : bias = Nbias;
		if (bias * currMean > bestMean) {
			bestMean = currMean;
			newStartX = startX + 1;

			newStartY = startY, newEndX = endX, newEndY = endY;
			resize = true;
			continue;
		}
		currMean = calculateAngleProb(seg, startX, startY, endX - 1, endY, image, Hlines, Vlines);
		currMean > 0 ? bias = Pbias : bias = Nbias;
		if (bias * currMean > bestMean) {
			bestMean = currMean;
			newEndX = endX-1;

			newStartX = startX, newStartY = startY, newEndY = endY;
			resize = true;
			continue;
		}
		// T-B
		currMean = calculateAngleProb(seg, startX, startY + 1, endX, endY, image, Hlines, Vlines);
		currMean > 0 ? bias = Pbias : bias = Nbias;
		if (bias * currMean > bestMean) {
			bestMean = currMean;
			newStartY = startY+1;

			newStartX = startX, newEndX = endX, newEndY = endY;
			resize = true;
			continue;
		}
		currMean = calculateAngleProb(seg, startX, startY, endX, endY - 1, image, Hlines, Vlines);
		currMean > 0 ? bias = Pbias : bias = Nbias;
		if (bias * currMean > bestMean) {
			bestMean = currMean;
			newEndY = endY-1;

			newStartX = startX, newStartY = startY, newEndX = endX;
			resize = true;
			continue;
		}
		// TL-BR
		currMean = calculateAngleProb(seg, startX + 1, startY + 1, endX, endY, image, Hlines, Vlines);
		currMean > 0 ? bias = Pbias : bias = Nbias;
		if (bias * currMean > bestMean) {
			bestMean = currMean;
			newStartX = startX+1;
			newStartY = startY+1;

			newEndX = endX, newEndY = endY;
			resize = true;
			continue;
		}
		currMean = calculateAngleProb(seg, startX, startY, endX - 1, endY - 1, image, Hlines, Vlines);
		currMean > 0 ? bias = Pbias : bias = Nbias;
		if (bias * currMean > bestMean) {
			bestMean = currMean;
			newEndX = endX-1;
			newEndY = endY-1;

			newStartX = startX, newStartY = startY;
			resize = true;
			continue;
		}
		// TR-BL
		currMean = calculateAngleProb(seg, startX, startY + 1, endX - 1, endY, image, Hlines, Vlines);
		currMean > 0 ? bias = Pbias : bias = Nbias;
		if (bias * currMean > bestMean) {
			bestMean = currMean;
			newStartY = startY+1;
			newEndX = endX-1;

			newStartX = startX, newEndY = endY;
			resize = true;
			continue;
		}
		currMean = calculateAngleProb(seg, startX + 1, startY, endX, endY - 1, image, Hlines, Vlines);
		currMean > 0 ? bias = Pbias : bias = Nbias;
		if (bias * currMean > bestMean) {
			bestMean = currMean;
			newStartX = startX+1;
			newEndY = endY-1;

			newStartY = startY, newEndX = endX;
			resize = true;
			continue;
		}
		// All angles
		currMean = calculateAngleProb(seg, startX + 1, startY + 1, endX - 1, endY - 1, image, Hlines, Vlines);
		currMean > 0 ? bias = Pbias : bias = Nbias;
		if (bias * currMean > bestMean) {
			bestMean = currMean;
			newStartX = startX+1;
			newStartY = startY+1;
			newEndX = endX-1;
			newEndY = endY-1;
			resize = true;
			continue;
		}
		
	} while (resize && newStartX < (newEndX-1) && newStartY < (newEndY-1));
	startX = newStartX, startY = newStartY, endX = newEndX, endY = newEndY;
	*/

	// NEW
	Mat map(endY, endX, CV_32FC1, float(0));
	for (int i = 0; i < endX; i++) {
		for (int j = 0; j < endY; j++) {
			int x = Vlines[i], w = Vlines[i + 1] - x, y = Hlines[j], h = Hlines[j + 1] - y;
			Mat spot = image(Rect(x, y, w, h));
			map.at<float>(j, i) = seg->spotScore(spot);
		}
	}
	Mat visualMap = map / 100.0;

	threshold(visualMap, visualMap, 0.2, 1.0, THRESH_BINARY);
	int i = 0;
	float tot;
	do { // LEFT
		tot = sum(visualMap.col(i))[0];
		if (tot < 0.004) {
			i++;
			startX++;
		}
	} while (tot < 0.004 && startX < (endX - 1));
	i = endX - 1;
	do { // RIGHT
		tot = sum(visualMap.col(i))[0];
		if (tot < 0.004) {
			i--;
			endX--;
		}
	} while (tot < 0.004 && startX < (endX - 1));
	i = 0;
	do { // TOP
		tot = sum(visualMap.row(i))[0];
		if (tot < 0.004) {
			i++;
			startY++;
		}
	} while (tot < 0.004 && startY < (endY - 1));
	i = endY - 1;
	do { // BOT
		tot = sum(visualMap.row(i))[0];
		if (tot < 0.004) {
			i--;
			endY--;
		}
	} while (tot < 0.004 && startY < (endY - 1));

	// FIND THE LARGEST RECTANGLE WITH LARGEST PROBABILITY
	// Suppose max 2 steps lines of error for any direction
	int bestRes = numAngles(visualMap, startX, startY, endX, endY);
	int tempRes, bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
	if((endX - startX)>4 && (endY - startY) > 4)
		for (int i = 0; i < 2; i++) {
			if (bestRes == 4) break;
			// L
			tempRes = numAngles(visualMap, startX + 1, startY, endX, endY);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bStartX++;
				if (bestRes == 4) break;
			}
			// T
			tempRes = numAngles(visualMap, startX, startY + 1, endX, endY);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bStartY++;
				if (bestRes == 4) break;
			}
			// R
			tempRes = numAngles(visualMap, startX, startY, endX - 1, endY);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bEndX--;
				if (bestRes == 4) break;
			}
			// B
			tempRes = numAngles(visualMap, startX, startY, endX, endY - 1);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bEndY--;
				if (bestRes == 4) break;
			}
			// TL
			tempRes = numAngles(visualMap, startX + 1, startY + 1, endX, endY);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bStartX++;
				bStartY++;
				if (bestRes == 4) break;
			}
			// BR
			tempRes = numAngles(visualMap, startX, startY, endX - 1, endY - 1);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bEndX--;
				bEndY--;
				if (bestRes == 4) break;
			}
			// TR
			tempRes = numAngles(visualMap, startX, startY + 1, endX - 1, endY);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bEndX--;
				bStartY++;
				if (bestRes == 4) break;
			}
			// BL
			tempRes = numAngles(visualMap, startX + 1, startY, endX, endY - 1);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bStartX++;
				bEndY--;
				if (bestRes == 4) break;
			}
			// ALL
			tempRes = numAngles(visualMap, startX + 1, startY + 1, endX - 1, endY - 1);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX+1, bStartY = startY+1, bEndX = endX-1, bEndY = endY-1;
				if (bestRes == 4) break;
			}
			/******************************************************************************/
			// 2x

			// L
			tempRes = numAngles(visualMap, startX + 2, startY, endX, endY);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX + 2, bStartY = startY, bEndX = endX, bEndY = endY;
				if (bestRes == 4) break;
			}
			// T
			tempRes = numAngles(visualMap, startX, startY + 2, endX, endY);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY + 2, bEndX = endX, bEndY = endY;
				if (bestRes == 4) break;
			}
			// R
			tempRes = numAngles(visualMap, startX, startY, endX - 2, endY);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX - 2, bEndY = endY;
				if (bestRes == 4) break;
			}
			// B
			tempRes = numAngles(visualMap, startX, startY, endX, endY - 2);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY - 2;
				if (bestRes == 4) break;
			}
			// TL
			tempRes = numAngles(visualMap, startX + 2, startY + 2, endX, endY);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bStartX+=2;
				bStartY+=2;
				if (bestRes == 4) break;
			}
			// BR
			tempRes = numAngles(visualMap, startX, startY, endX - 2, endY - 2);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bEndX-=2;
				bEndY-=2;
				if (bestRes == 4) break;
			}
			// TR
			tempRes = numAngles(visualMap, startX, startY + 2, endX - 2, endY);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bEndX-=2;
				bStartY+=2;
				if (bestRes == 4) break;
			}
			// BL
			tempRes = numAngles(visualMap, startX + 2, startY, endX, endY - 2);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX, bStartY = startY, bEndX = endX, bEndY = endY;
				bStartX+=2;
				bEndY-=2;
				if (bestRes == 4) break;
			}
			// ALL
			tempRes = numAngles(visualMap, startX + 2, startY + 2, endX - 2, endY - 2);
			if (tempRes > bestRes) {
				bestRes = tempRes;
				bStartX = startX + 2, bStartY = startY + 2, bEndX = endX - 2, bEndY = endY - 2;
				if (bestRes == 4) break;
			}

			startX = bStartX, startY = bStartY, endX = bEndX, endY = bEndY;
		}
	startX = bStartX, startY = bStartY, endX = bEndX, endY = bEndY;

	// H adjust
	//for (int i = 0; i < Vlines[startX]; i++) binarySignals.V.at<float>(0, i) = 0.0;
	//for (int i = Vlines[endX] + 1; i < binarySignals.V.cols; i++) binarySignals.V.at<float>(0, i) = 0.0;

	if (endX < (Vlines.size() - 1)) for (int i = (Vlines.size() - 1); i > endX; i--) Vlines.erase(Vlines.begin() + i);
	
	if (startX > 0) for (int i = (startX - 1); i >= 0; i--) Vlines.erase(Vlines.begin() + i);

	// V adjust
	//for (int i = 0; i < Hlines[startY]; i++) binarySignals.H.at<float>(i, 0) = 0.0;
	//for (int i = Hlines[endY] + 1; i < binarySignals.H.rows; i++) binarySignals.H.at<float>(i, 0) = 0.0;

	if (endY < (Hlines.size() - 1)) for (int i = (Hlines.size() - 1); i > endY; i--) Hlines.erase(Hlines.begin() + i);
	
	if (startY > 0) for (int i = (startY - 1); i >= 0; i--) Hlines.erase(Hlines.begin() + i);
	

	reBinaryGrid(Hlines, Vlines, binarySignals);

	/*deleteEmptyLines(image, Hlines, Vlines, true);

	reBinaryGrid(Hlines, Vlines, binarySignals);*/

	return true;
}

int numAngles(Mat image, int startX, int startY, int endX, int endY) {
	int res = 0;
	// We need at least two opposite angles
	if (image.at<float>(startY, startX) > 0 && image.at<float>(endY - 1, endX - 1) > 0) {
		res = 2;
		if (image.at<float>(startY, endX - 1) > 0) res++;
		if (image.at<float>(endY - 1, startX) > 0) res++;
		return res;
	}
	if (image.at<float>(startY, endX - 1) > 0 && image.at<float>(endY - 1, startX) > 0) {
		res = 2;
		if (image.at<float>(endY - 1, endX - 1) > 0) res++;
		if (image.at<float>(startY, startX) > 0) res++;
		return res;
	}
	return res;
}

bool reBinaryGrid(vector<int> &Hlines, vector<int> &Vlines, projections binarySignals) {
	int displ = 15;
	int minDispl = 15;
	// move left to right
	bool isOff = true;
	int cursor = Vlines[0];
	if (binarySignals.V.at<float>(0, cursor) < 100) {
		do {
			cursor++;
			isOff = binarySignals.V.at<float>(0, cursor) < 100;
		} while (isOff);
		Vlines[0] = max(0, cursor - displ);
	}
	else {
		do {
			cursor--;
			isOff = binarySignals.V.at<float>(0, cursor) < 100;
		} while (!isOff);
		Vlines[0] = max(0, cursor - minDispl);
	}
	// move right to left
	isOff = true;
	cursor = Vlines[Vlines.size() - 1];
	if (binarySignals.V.at<float>(0, cursor) < 100) {
		do {
			cursor--;
			isOff = binarySignals.V.at<float>(0, cursor) < 100;
		} while (isOff);
		Vlines[Vlines.size() - 1] = min(binarySignals.V.cols - 1, cursor + displ);
	}
	else {
		do {
			cursor++;
			isOff = binarySignals.V.at<float>(0, cursor) < 100;
		} while (!isOff);
		Vlines[Vlines.size() - 1] = min(binarySignals.V.cols - 1, cursor + minDispl);
	}

	// move up to down
	isOff = true;
	cursor = Hlines[0];
	if (binarySignals.H.at<float>(cursor, 0) < 100) {
		do {
			cursor++;
			isOff = binarySignals.H.at<float>(cursor, 0) < 100;
		} while (isOff);
		Hlines[0] = max(0, cursor - displ);
	}
	else {
		do {
			cursor--;
			isOff = binarySignals.H.at<float>(cursor, 0) < 100;
		} while (!isOff);
		Hlines[0] = max(0, cursor - minDispl);
	}
	// move down to up
	isOff = true;
	cursor = Hlines[Hlines.size() - 1];
	if (binarySignals.H.at<float>(cursor, 0) < 100) {
		do {
			cursor--;
			isOff = binarySignals.H.at<float>(cursor, 0) < 100;
		} while (isOff);
		Hlines[Hlines.size() - 1] = min(binarySignals.H.rows - 1, cursor + displ);
	}
	else {
		do {
			cursor++;
			isOff = binarySignals.H.at<float>(cursor, 0) < 100;
		} while (!isOff);
		Hlines[Hlines.size() - 1] = min(binarySignals.H.rows - 1, cursor + minDispl);
	}

	return true;
}

double pdf(double mean, double var, double x) { // Gaussian PDF
	double result = 1.0;
	result /= var;
	result /= sqrt(2 * M_PI);
	// result *= exp(-0.5 * pow((x - mean) / var, 2));
	double Z = x - mean;
	Z /= var;
	Z = pow(Z, 2);
	result *= exp(-0.5 * Z);
	return result;
}

float calculateAngleProb(OpenCVSegmenter* seg, int startX, int startY, int endX, int endY, Mat image, vector<int> Hlines, vector<int> Vlines) {
	/*
	OLD METHOD
	double totMean = 0.0;
	double currMean;
	double e = 0.4, v = 0.1; // expected, variance

	// temptative: add a negative bias towards "empty" spots
	currMean = corrMean(image(Rect(Vlines[startX], Hlines[startY], Vlines[startX + 1] - Vlines[startX], Hlines[startY + 1] - Hlines[startY])));
	if (currMean < 0.005) totMean -= (1 - pdf(e, v, currMean));
	else totMean += pdf(e, v, currMean);

	currMean = corrMean(image(Rect(Vlines[endX - 1], Hlines[startY], Vlines[endX] - Vlines[endX - 1], Hlines[startY + 1] - Hlines[startY])));
	if (currMean < 0.005) totMean -= (1 - pdf(e, v, currMean));
	else totMean += pdf(e, v, currMean);

	currMean = corrMean(image(Rect(Vlines[startX], Hlines[endY - 1], Vlines[startX + 1] - Vlines[startX], Hlines[endY] - Hlines[endY - 1])));
	if (currMean < 0.005) totMean -= (1 - pdf(e, v, currMean));
	else totMean += pdf(e, v, currMean);

	currMean = corrMean(image(Rect(Vlines[endX - 1], Hlines[endY - 1], Vlines[endX] - Vlines[endX - 1], Hlines[endY] - Hlines[endY - 1])));
	if (currMean < 0.005) totMean -= (1 - pdf(e, v, currMean));
	else totMean += pdf(e, v, currMean);

	totMean /= 4.0;
	
	return (float)totMean;
	*/
	double totMean = 0.0;
	double currMean;

	currMean = seg->spotScore(image(Rect(Vlines[startX], Hlines[startY], Vlines[startX + 1] - Vlines[startX], Hlines[startY + 1] - Hlines[startY]))); // TL
	totMean += currMean;

	currMean = seg->spotScore(image(Rect(Vlines[endX - 1], Hlines[startY], Vlines[endX] - Vlines[endX - 1], Hlines[startY + 1] - Hlines[startY]))); // TR
	totMean += currMean;

	currMean = seg->spotScore(image(Rect(Vlines[startX], Hlines[endY - 1], Vlines[startX + 1] - Vlines[startX], Hlines[endY] - Hlines[endY - 1]))); // BL
	totMean += currMean;

	currMean = seg->spotScore(image(Rect(Vlines[endX - 1], Hlines[endY - 1], Vlines[endX] - Vlines[endX - 1], Hlines[endY] - Hlines[endY - 1]))); // BR
	totMean += currMean;

	currMean = seg->spotScore(image(Rect(Vlines[endX - 1], Hlines[endY - 2], Vlines[endX] - Vlines[endX - 1], Hlines[endY - 1] - Hlines[endY - 2]))); // BR + 1
	totMean += currMean;

	totMean /= 5.0;

	return (float)totMean;
}

float corrMean(Mat image) {
	int startX = 0, startY = 0, endX = image.cols - 1, endY = image.rows - 1;
	bool empty;
	if (mean(image)[0] < 0.005) return 0.0;
	// L to R
	do {
		empty = false;
		if (mean(image.col(startX))[0] < 0.005) {
			empty = true;
			startX++;
		}
		if (startX == image.cols) return 0.0;
	} while (empty);
	// R to L
	do {
		empty = false;
		if (mean(image.col(endX))[0] < 0.005) {
			empty = true;
			endX--;
		}
		if (endX == 0) return 0.0;
	} while (empty);
	// T to B
	do {
		empty = false;
		if (mean(image.row(startY))[0] < 0.005) {
			empty = true;
			startY++;
		}
		if (startY == image.rows) return 0.0;
	} while (empty);
	// B to T
	do {
		empty = false;
		if (mean(image.row(endY))[0] < 0.005) {
			empty = true;
			endY--;
		}
		if (endY == 0) return 0.0;
	} while (empty);
	if (startX >= endX || startY >= endY) return 0.0;
	image = image(Rect(startX, startY, endX - startX, endY - startY));
	return mean(image)[0];
}

bool adjustToDevice(OpenCVSegmenter* seg, Device d, Mat image, vector<int>& Hlines, vector<int>& Vlines, projections binarySignals) {

	int numRows = d.numRows();
	int numCols = d.numCols();

	int vecNumRows = Vlines.size() - 1;
	int vecNumCols = Hlines.size() - 1;

	adjustGrid(Hlines, Vlines, false);

	bool hasAngles = d.hasAngles();
	vector<vector<bool>> angles;
	if (hasAngles) angles = d.getAngles();
	bool hasFourAngles = angles[0][0] && angles[0][1] && angles[1][0] && angles[1][1];
	if (hasFourAngles) setToAngles(seg, image, Hlines, Vlines, binarySignals);
	//deleteEmptyLines(image, Hlines, Vlines);
	
	//adjustGrid(Hlines, Vlines, true);
	// old: adaptToDeviceSize(numRows, numCols, Hlines, Vlines);
	alignToDevice(numRows, numCols, Hlines, Vlines, true);

	return true;
}

bool alignToDevice(int numRows, int numCols, vector<int>& Hlines, vector<int>& Vlines, bool fineAdjust) {
	if ((Hlines.size() - 1) != numRows || (Vlines.size() - 1) != numCols) {
		int startH = Hlines[0], endH = Hlines[Hlines.size() - 1];
		int distH = endH - startH;
		int mean = distH / numRows;
		int displ = distH - mean * numRows;
		vector<int> newH;
		int iterH = startH;
		newH.push_back(iterH);
		for (int i = 1; i < numRows; i++) {
			iterH += mean;
			if (i == 1) iterH += displ/2;
			newH.push_back(iterH);
		}
		newH.push_back(endH);
		Hlines = newH;

		int startV = Vlines[0], endV = Vlines[Vlines.size() - 1];
		int distV = endV - startV;
		mean = distV / numCols;
		displ = distV - mean * numCols;
		vector<int> newV;
		int iterV = startV;
		newV.push_back(iterV);
		for (int i = 1; i < numCols; i++) {
			iterV += mean;
			if (i == 1) iterV += displ / 2;
			newV.push_back(iterV);
		}
		newV.push_back(endV);
		Vlines = newV;
	}

	return true;
}

bool adjustGrid(vector<int> &Hlines, vector<int> &Vlines, bool resizable) { 
	
	adjustHgrid(Hlines, resizable);
	// -------------------------------------------------------------------
	adjustHgrid(Vlines, resizable);

	return true;
}

bool adjustHgrid(vector<int>& Hlines, bool resizable, int flag, int flagend, float errorlimit) {
	// Valid for both H and V
	vector<int> Hdist, Herr, HerrLines;
	float HmeanDist = 0.0, HerrMax = 0.0, HmeanErr = 0.0, HMSE = 0.0;
	int HmaxErrIndex = -1, numHerr = 0;

	// Horizontal
	//
	int numHlines = Hlines.size();
	for (int i = 0; i < (numHlines - 1); i++) {
		float dist = Hlines[i + 1] - Hlines[i];
		Hdist.push_back(dist); // Get distances
		HmeanDist += dist;
	}
	HmeanDist /= (numHlines - 1); // Get mean distance
	for (int i = 0; i < (numHlines - 1); i++) {
		float error = abs(Hdist[i] - HmeanDist);
		if (error > HerrMax) {
			HerrMax = error;
			HmaxErrIndex = i;
		}
		Herr.push_back(error); // Get errors
		HmeanErr += error;
		HMSE += pow(error, 2);
	}
	HmeanErr /= (numHlines - 1); // Get mean error
	HMSE /= (numHlines - 1);
	HMSE = sqrt(HMSE); // Get MSE
	// Count number of wrong lines
	for (int i = 1; i < (numHlines - 2); i++) {
		if (Herr[i] > (HMSE + HmeanDist / 10)) {
			numHerr++;
			HerrLines.push_back(i);
		}
	}
	if (flag == 0) {
		flagend = numHerr;
		errorlimit = 2 * HmeanErr;
	}
	// Find the maximum error
	if (flagend > 0 && errorlimit > HmeanErr) {
		if (numHerr > 0) {
			if (HmaxErrIndex == 0) { // first line
				int j = HmaxErrIndex;
				if (Hdist[j] < HmeanDist) {
					if (resizable && Herr[j + 1] < (HMSE + HmeanDist / 10)) {
						// redundant grid line 1
						Hlines[j] = (Hlines[j] + Hlines[j + 1]) / 2;
						Hlines.erase(Hlines.begin() + j + 1);
					}else if (Herr[j + 1] > (HMSE + HmeanDist / 10)) {
						// redundant grid line 2
						Hlines[j + 1] = (Hlines[j + 1] + Hlines[j + 2]) / 2;
						Hlines.erase(Hlines.begin() + j + 2);
					}
					else if (Hdist[j + 1] > HmeanDist) {
						// locate at wrong place
						Hlines[j + 1] = (Hlines[j] + Hlines[j + 2]) / 2;
					}
				}
				else if (Hdist[j] > HmeanDist) {
					if (Herr[j + 1] < (HMSE + HmeanDist / 10)) {
						// missing grid line 1
						Hlines.insert(Hlines.begin() + j + 1, (Hlines[j] + Hlines[j + 1]) / 2);
					}
					else if (Herr[j + 1] > (HMSE + HmeanDist / 10)) {
						// missing grid line 2
						Hlines[j + 1] = Hlines[j] + (Hlines[j + 2] - Hlines[j]) / 3;
						Hlines.insert(Hlines.begin() + j + 2, Hlines[j] + 2 * (Hlines[j + 2] - Hlines[j]) / 3);
					}
					else if (Hdist[j + 1] < HmeanDist) {
						// locate at wrong place
						Hlines[j + 1] = (Hlines[j] + Hlines[j + 2]) / 2;
					}
				}
			}
			else if (HmaxErrIndex == (numHlines - 2)) { // last line
				int j = HmaxErrIndex;
				if (Hdist[j] < HmeanDist) {
					if (resizable && Herr[j - 1] < (HMSE + HmeanDist / 10)) {
						// redundant grid line 1
						Hlines[j] = (Hlines[j] + Hlines[j + 1]) / 2;
						Hlines.erase(Hlines.begin() + j + 1);
					}
					else if (Herr[j - 1] > (HMSE + HmeanDist / 10)) {
						// redundant grid line 2
						Hlines[j - 1] = (Hlines[j - 1] + Hlines[j]) / 2;
						Hlines.erase(Hlines.begin() + j);
					}
					else if (Hdist[j - 1] > HmeanDist) {
						// locate at wrong place
						Hlines[j] = (Hlines[j - 1] + Hlines[j + 1]) / 2;
					}
				}
				else if (Hdist[j] > HmeanDist) {
					if (Herr[j - 1] < (HMSE + HmeanDist / 10)) {
						// missing grid line 1
						Hlines.insert(Hlines.begin() + j + 1, (Hlines[j] + Hlines[j + 1]) / 2);
					}
					else if (Herr[j - 1] > (HMSE + HmeanDist / 10)) {
						// missing grid line 2
						Hlines[j] = Hlines[j - 1] + (Hlines[j + 1] - Hlines[j - 1]) / 3;
						Hlines.insert(Hlines.begin() + j + 1, Hlines[j - 1] + 2 * (Hlines[j + 1] - Hlines[j - 1]) / 3);
					}
					else if (Hdist[j - 1] < HmeanDist) {
						// locate at wrong place
						Hlines[j] = (Hlines[j - 1] + Hlines[j + 1]) / 2;
					}
				}
			}
			else { // internal line, left side
				int j = HmaxErrIndex;
				float thresh = (HMSE + HmeanDist / 10);
				if (Hdist[j] < HmeanDist) {
					if (((Herr[j - 1] < thresh && Herr[j - 1] < thresh) || Hdist[j + 1] < HmeanDist)) {
						// redundant grid line 1
						Hlines[j] = (Hlines[j] + Hlines[j + 1]) / 2;
						Hlines.erase(Hlines.begin() + j + 1);
					}
					else if ((Herr[j - 1] > thresh || Herr[j + 1] > thresh)) {
						// redundant grid line 2
						Hlines[j + 1] = (Hlines[j + 1] + Hlines[j + 2]) / 2;
						Hlines.erase(Hlines.begin() + j + 2);
					}
				}
				else if (Hdist[j] > HmeanDist) {
					if ((Herr[j - 1] < thresh && Herr[j + 1] < thresh)) {
						// missing grid line 1
						if (true) Hlines.insert(Hlines.begin() + j + 1, (Hlines[j] + Hlines[j + 1]) / 2);
						else Hlines[j + 1] = (Hlines[j] + 2 * Hlines[j + 1]) / 3; // not addable
					}
					else if ((Herr[j - 1] > thresh || Herr[j + 1] > thresh)) {
						// missing grid line 2
						if (Hdist[j - 1] > HmeanDist) {
							Hlines[j] = Hlines[j - 1] + (Hlines[j + 1] - Hlines[j - 1]) / 3;
							if (true) Hlines.insert(Hlines.begin() + j + 1, Hlines[j - 1] + 2 * (Hlines[j + 1] - Hlines[j - 1]) / 3);
						}
						else {
							Hlines[j + 1] = Hlines[j] + (Hlines[j + 2] - Hlines[j]) / 3;
							if (true) Hlines.insert(Hlines.begin() + j + 2, Hlines[j] + 2 * (Hlines[j + 2] - Hlines[j]) / 3);
						}
					}
				}
			}

			flag++;
			flagend--;
			return adjustHgrid(Hlines, resizable, flag, flagend, errorlimit);
		}else return true;
	}else return true;
}

bool adaptToDeviceSize(int numRows, int numCols, vector<int>& Hlines, vector<int>& Vlines) {
	int numHlines = Hlines.size(), numVlines = Vlines.size();

	if (numRows == (numHlines - 1) && numCols == (numVlines - 1)) return true;
	else {
		int minDist, maxDist;
		int index;
		int currDist;
		if (numRows < (numHlines - 1)) { // elimina righe
			minDist = -1;
			index = -1;
			for (int i = 0; i < (numHlines - 1); i++) {
				currDist = Hlines[i + 1] - Hlines[i];
				if (currDist < minDist || minDist == -1) {
					minDist = currDist;
					index = i + 1;
				}
			}
			Hlines.erase(Hlines.begin() + index);
		}
		else if (numRows > (numHlines - 1)) { // aggiungi righe
			maxDist = -1;
			index = -1;
			for (int i = 0; i < (numHlines - 1); i++) {
				currDist = Hlines[i + 1] - Hlines[i];
				if (currDist > maxDist || maxDist == -1) {
					maxDist = currDist;
					index = i + 1;
				}
			}
			Hlines.insert(Hlines.begin() + index, Hlines[index] - maxDist / 2);
		}

		if (numCols < (numVlines - 1)) { // elimina colonne
			minDist = -1;
			index = -1;
			for (int i = 0; i < (numVlines - 1); i++) {
				currDist = Vlines[i + 1] - Vlines[i];
				if (currDist < minDist || minDist == -1) {
					minDist = currDist;
					index = i + 1;
				}
			}
			Vlines.erase(Vlines.begin() + index);
		}
		else if (numCols > (numVlines - 1)) { // aggiungi colonne
			maxDist = -1;
			index = -1;
			for (int i = 0; i < (numVlines - 1); i++) {
				currDist = Vlines[i + 1] - Vlines[i];
				if (currDist > maxDist || maxDist == -1) {
					maxDist = currDist;
					index = i + 1;
				}
			}
			Vlines.insert(Vlines.begin() + index, Vlines[index] - maxDist / 2);
		}
	}
	return adaptToDeviceSize(numRows, numCols, Hlines, Vlines);
}