#include "tests.h"

int OpenCVSegmenter::test(int num_test) {
	switch (num_test)
	{
	case 1:
		return testSegmenter();
	default:
		return testSegmenter();
	}
}

int testSegmenter()
{
	// Esempio 1 OpenCV

	std::string image_path = samples::findFile("sample.tif");
	Mat img = imread(image_path, CV_8UC1);

	if (img.empty())
	{
		std::cout << "Could not read the image: " << image_path << std::endl;
		return 1;
	}

	ScanImageAndReduceC(img);

	imshow("Display window", img);
	int k = waitKey(0); // Wait for a keystroke in the window
	if (k == 's')
	{
		imwrite("sample.tif", img);
	}
	return 0;
}

Mat& ScanImageAndReduceC(Mat& I) //, const uchar* const table)
{
	// accept only char type matrices
	CV_Assert(I.depth() == CV_8U);
	int channels = I.channels();
	printf("%d canali.\n\n", channels);
	int nRows = I.rows;
	int nCols = I.cols * channels;

	printf("L'immagine e' %dpx per %dpx.\n\n", nCols, nRows);

	bool continuous;
	if (I.isContinuous())
	{
		printf("L'immagine e' continua.\n\n");
		continuous = true;
		/*nCols *= nRows;
		nRows = 1;*/
	}
	int i, j;
	uchar* p;

	int minval = 255;
	int maxval = 0;

	for (i = 0; i < nRows; ++i)
	{
		p = continuous ? I.ptr<uchar>(0) : I.ptr<uchar>(i);

		int row_intensity = 0;

		for (j = 0; j < nCols; ++j)
		{
			//p[j] = table[p[j]];
			int intensity = continuous ? (int)p[i * nCols + j] : (int)p[j];
			if (intensity < minval) minval = intensity;
			if (intensity > maxval) maxval = intensity;
			row_intensity += (intensity - 1);
		}
		printf("%d\n", row_intensity);
	}
	printf("Min: %d ; Max: %d", minval, maxval);
	return I;
}
