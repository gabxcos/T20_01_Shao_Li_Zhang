#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "tiffio.h"

using namespace cv;
using namespace std;

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

	Mat image = Mat::zeros(300, 600, CV_8UC3);
	circle(image, Point(250, 150), 100, Scalar(0, 255, 128), -100);
	circle(image, Point(350, 150), 100, Scalar(255, 255, 255), -100);
	imshow("Display Window", image);
	waitKey(0);

	return 0;
}