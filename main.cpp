#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "tiffio.h"

using namespace cv;
using namespace std;

Mat& ScanImageAndReduceC(Mat& I) //, const uchar* const table)
{
	// accept only char type matrices
	CV_Assert(I.depth() == CV_8U);
	int channels = I.channels();
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
	for (i = 0; i < nRows; ++i)
	{
		p = continuous ? I.ptr<uchar>(0) : I.ptr<uchar>(i);

		int row_intensity = 0;

		for (j = 0; j < nCols; ++j)
		{
			//p[j] = table[p[j]];
			int intensity = continuous ? (int)p[i*nCols+j] : (int)p[j];
			row_intensity += (intensity-1);
		}
		printf("%d\n", row_intensity);
	}
	return I;
}

int main() {
	/* Esempio libtiff
	
	TIFF* tif = TIFFOpen("sample.tif", "r");
	if (tif) {
		uint32 imagelength;
		tsize_t scanline;
		tdata_t* buf;
		uint32 row, col;

		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
		scanline = TIFFScanlineSize(tif);
		*buf = _TIFFmalloc(scanline);
		for (row = 0; row < imagelength; row++) {
			TIFFReadScanline(tif, buf, row, 0);
			for (col = 0; col < scanline; col++) printf("%d", (int) buf[col]);
			printf("\n");
		}
		_TIFFfree(buf);
		TIFFClose(tif);
	}
	*/

	// Esempio OpenCV

	std::string image_path = samples::findFile("sample.tif");
	Mat img = imread(image_path, IMREAD_GRAYSCALE);

	if (img.empty())
	{
		std::cout << "Could not read the image: " << image_path << std::endl;
		return 1;
	}

	ScanImageAndReduceC(img);

	imshow("Display window", img);
	int k = waitKey(0); // Wait for a keystroke in the window
	/*if (k == 's')
	{
		imwrite("sample.png", img);
	}*/

	return 0;
}