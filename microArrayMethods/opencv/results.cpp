#include "results.h"
#include <iomanip>
#include <numeric>

bool OpenCVSegmenter::produceResults() {
	vector<vector<Spot>> spotMatrix = getSpotMatrix();

	printResults(this, spotMatrix, getDevice() ,getOutPath());
	Mat circledImage = drawCircles(getResImage().clone(), spotMatrix);
	//imshow("circled", circledImage);
	//waitKey(0);
	String temp_file = getOutPath() + "\\" + "circles.png";
	imwrite(temp_file, circledImage);

	return true;
}

void printResults(OpenCVSegmenter* seg, vector<vector<Spot>> spotMatrix, Device d, String path) {
	int width = spotMatrix[0].size(), height = spotMatrix.size();
	path = path + "\\";

	vector<vector<int>> controls = d.getControls();

	String signalPath = path + "signal.txt";
	String bgPath = path + "background.txt";
	String resultPath = path + "results.txt";
	String statsPath = path + "statistics.txt";

	ofstream sFile, bFile, rFile, stFile;

	sFile.open(signalPath);
	sFile << std::fixed << std::setprecision(2);

	std::printf("\n\nRisultati:\n----------------------------------------------\n\n");

	// Signal mean
	std::printf("\nSignal mean:\n");
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float signal = spotMatrix[i][j].getSignal();
			printf("%.2f\t", signal);
			sFile << to_string(signal) << "\t";
		}
		printf("\n");
		sFile << "\n";
	}
	sFile.close();


	bFile.open(bgPath);
	bFile << std::fixed << std::setprecision(2);
	// BG mean
	std::printf("\nBackground mean:\n");
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			float bg = spotMatrix[i][j].getBackground();
			printf("%.2f\t", bg);
			bFile << to_string(bg) << "\t";
		}
		printf("\n");
		bFile << "\n";
	}
	bFile.close();

	rFile.open(resultPath);
	rFile << std::fixed << std::setprecision(2);
	stFile.open(statsPath);
	stFile << std::fixed << std::setprecision(2);
	stFile << "ROW\tCOL\tON\tDEVICE\tSIG_Q\tBG_Q\tSCALE_I\tSIZE_R\tSIZE_Q\tSIGN_R\tLOCBG_V\tLOCBG_H\tSAT_Q\tCOMP_Q\n";
	bool hybridCntrl = true; int hybridNum = 0, hybridTot = 0;
	bool negativeCntrl = true; int negNum = 0, negTot = 0;
	bool PCRcntrl = true; int PCRnum = 0, PCRtot = 0;
	// Result

	vector<float> spotStats[10];

	std::printf("\nResult:\n");
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			bool filled = spotMatrix[i][j].isFilled();

			spotQuality q = spotMatrix[i][j].quality;
			stFile << i + 1 << "\t" << j + 1 << "\t";

			if (filled) {
				rFile << "1\t";
				stFile << "1\t";
				std::printf("1\t");
			}
			else {
				rFile << "0\t";
				stFile << "0\t";
				std::printf("0\t");
			}

			switch (controls[i][j]) {
			case Controls::HYBRIDIZATION:
				stFile << "HYB\t";
				hybridTot++;
				if (!filled) hybridCntrl = false;
				else hybridNum++;
				break;
			case Controls::NEGATIVE:
				stFile << "NEG\t";
				negTot++;
				if (filled) negativeCntrl = false;
				else negNum++;
				break;
			case Controls::PCR:
				stFile << "PCR\t";
				PCRtot++;
				if (!filled) PCRcntrl = false;
				else PCRnum++;
				break;
			case Controls::EMPTY:
				stFile << "EMP\t";
				break;
			case Controls::CAPTURE:
				stFile << "CAP\t";
				break;
			default:
				std::printf("Find a non-valid Control...\n");
				stFile << "NULL\t";
			}

			if(q.signalNoise < 0) stFile << "NULL\t";
			else stFile << q.signalNoise << "\t";				spotStats[0].push_back(q.signalNoise);
			stFile << q.backgroundNoise << "\t";				spotStats[1].push_back(q.backgroundNoise);
			if (q.scaleInvariant < 0) stFile << "NULL\t";
			else stFile << q.scaleInvariant << "\t";			spotStats[2].push_back(q.scaleInvariant);
			if (q.sizeRegularity < 0) stFile << "NULL\t";
			else stFile << q.sizeRegularity << "\t";			spotStats[3].push_back(q.sizeRegularity);
			if (q.sizeQuality < 0) stFile << "NULL\t";
			else stFile << q.sizeQuality << "\t";				spotStats[4].push_back(q.sizeQuality);
			stFile << q.signalToNoiseRatio << "\t";				spotStats[5].push_back(q.signalToNoiseRatio);
			stFile << q.localBackgroundVariability << "\t";		spotStats[6].push_back(q.localBackgroundVariability);
			stFile << q.localBackgroundHighness << "\t";		spotStats[7].push_back(q.localBackgroundHighness);
			stFile << q.saturationQuality << "\t";				spotStats[8].push_back(q.saturationQuality);
			stFile << q.compositeQuality << "\n";				spotStats[9].push_back(q.compositeQuality);

		}
		rFile << "\n";
		printf("\n");
	}

	stFile.close();

	rFile << "\nHybridization control: " << (hybridCntrl ? "VALID" : "NOT VALID"); rFile << " : " << hybridNum << " spots out of " << hybridTot;
	rFile << "\nNegative control: " << (negativeCntrl ? "VALID" : "NOT VALID"); rFile << " : " << negNum << " spots out of " << negTot;
	rFile << "\nPCR control: " << (PCRcntrl ? "VALID" : "NOT VALID"); rFile << " : " << PCRnum << " spots out of " << PCRtot;

	rFile << "\n\n";

	imageQuality iq = seg->quality;
	rFile << "Average distance between spots: " << iq.avgSpotDistance << "\n";
	rFile << "Average spot radius: " << iq.avgSpotRadius << "\n";
	rFile << "Average spot area: " << iq.avgSpotArea << "\n\n";

	for (int i = 0; i < 10; i++) spotStats[i].erase(remove_if(spotStats[i].begin(), spotStats[i].end(), [](float x) {return x < 0; }), spotStats[i].end());

	float sum[10], mean[10], stdDev[10];
	for (int i = 0; i < 10; i++) {
		sum[i] = std::accumulate(spotStats[i].begin(), spotStats[i].end(), 0.0);
		mean[i] = sum[i] / spotStats[i].size();
		vector<float> diff(spotStats[i].size());
		float _mean = mean[i];
		std::transform(spotStats[i].begin(), spotStats[i].end(), diff.begin(), [_mean](float x) { return x - _mean; });
		float sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
		stdDev[i] = std::sqrt(sq_sum / spotStats[i].size());
	}


	rFile << "Spot statistics:\n";
	rFile << "STAT\tSIG_Q\tBG_Q\tSCALE_I\tSIZE_R\tSIZE_Q\tSIGN_R\tLOCBG_V\tLOCBG_H\tSAT_Q\tCOMP_Q\n";
	rFile << "AVG";
	for (int i = 0; i < 10; i++) rFile << "\t" << mean[i];
	rFile << "\n";
	rFile << "STDEV";
	for (int i = 0; i < 10; i++) rFile << "\t" << stdDev[i];
	rFile << "\n";

	rFile.close();
}

Mat drawCircles(Mat fullImage, vector<vector<Spot>> spotMatrix) {
	Mat newImage;
	fullImage.convertTo(fullImage, CV_8U, 255);
	cvtColor(fullImage, newImage, COLOR_GRAY2BGR);

	for (int i = 0; i < spotMatrix.size(); i++) {
		for (int j = 0; j < spotMatrix[0].size(); j++) {
			Spot s = spotMatrix[i][j];
			Mat cluster = s.getCluster();
			int x = s.getX(), y = s.getY();
			int width = cluster.cols, height = cluster.rows;
			for (int m = 0; m < width; m++) {
				for (int n = 0; n < height; n++) {
					if (nextToPixel(m, n, cluster)) newImage.at<Vec3b>(y+n, x+m) = Vec3b(0, 0, 255);
				}
			}
		}
	}
	return newImage;
}

bool nextToPixel(int x, int y, Mat image) {
	int width = image.cols, height = image.rows;
	if (image.at<float>(y, x) > 0.0) return false;
	if (x > 0 && image.at<float>(y, x - 1) > 0.0) return true;
	if (y > 0 && image.at<float>(y - 1, x) > 0.0) return true;
	if (y < (height - 1) && image.at<float>(y + 1, x) > 0.0) return true;
	if (x < (width - 1) && image.at<float>(y, x + 1) > 0.0) return true;
	return false;
}