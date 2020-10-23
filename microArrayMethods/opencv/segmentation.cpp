#include "segmentation.h"

bool OpenCVSegmenter::segmenting() {
	vector<int> gridH = getGridH(), gridV = getGridV();
	Mat image = getResImage();

	int x = gridV[0], y = gridH[0];
	int xd = gridV[1] - x, yd = gridH[1] - y;

	Mat spot = image(Rect(x, y, xd, yd));

	spot = resizeSpot(spot);
	imshow("Spot", spot);
	waitKey(0);

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
			startX = i - 1;
			break;
		}
	}

	for (int i = width - 1; i >= 0; i--) {
		if (projX.at<float>(0, i) > 0.004) {
			endX = i + 1;
			break;
		}
	}

	for (int i = 0; i < height; i++) {
		if (projY.at<float>(i, 0) > 0.004) {
			startY = i - 1;
			break;
		}
	}

	for (int i = height - 1; i >= 0; i--) {
		if (projY.at<float>(i, 0) > 0.004) {
			endY = i + 1;
			break;
		}
	}

	Rect myROI(startX, startY, (endX - startX + 1), (endY - startY + 1));

	Mat croppedSpot = spot(myROI);
	return croppedSpot;
}

bool IKM(Mat spot) {
	return true;
}