#include "Device.h"

using namespace std;

Device::Device(int rows, int cols, vector<vector<int>> _controls)
{
	setSize(rows, cols);
	setControls(_controls);
	// feature checks
	checkHasAngles();
	checkIsDuplicated();
}

Device::Device() {
	setSize(0, 0);
	setControls(vector<vector<int>>());
}

bool Device::checkValidity()
{
	vector<vector<int>> myControls = getControls();
	if (myControls.empty()) return false;

	int myRows = myControls.size();
	if (numRows() <= 0 || numRows() != myRows) return false;

	int myCols = myControls[0].size();
	if (numCols() <= 0 || numCols() != myCols ) return false;
	for (int i = 1; i < myRows; i++) {
		int temp_cols = myControls[i].size();
		if (myCols != temp_cols) return false;
	}

	return true;
}

bool Device::checkHasAngles() {
	int rows = numRows(), cols = numCols();
	vector<vector<int>> controls = getControls();
	bool temp_hasAngles = false;
	vector<vector<bool>> angles = {{false, false}, {false, false}};
	for (int i = 0; i < 2; i++) {
		for (int j = 0; j < 2; j++) {
			if (controls[i*(rows-1)][j*(cols-1)] == Controls::HYBRIDIZATION) {
				temp_hasAngles = true;
				angles[i][j] = true;
			}
		}
	}
	setHasAngles(temp_hasAngles);
	setAngles(angles);

	return temp_hasAngles;
	
}

bool Device::checkIsDuplicated() {
	int rows = numRows(), cols = numCols();
	int oddDispl = cols % 2;
	int halfSize = (cols - (oddDispl)) / 2;

	vector<vector<int>> controls = getControls();

	bool temp_isDuplicated = true;
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < halfSize; j++) {
			int temp_cntrlA = controls[i][j];
			int temp_cntrlB = controls[i][j + halfSize + oddDispl];
			if (temp_cntrlA == Controls::HYBRIDIZATION || temp_cntrlA == Controls::EMPTY) {
				if (temp_cntrlB != Controls::HYBRIDIZATION && temp_cntrlB != Controls::EMPTY) temp_isDuplicated = false;
			}
			else {
				if (temp_cntrlA != temp_cntrlB) temp_isDuplicated = false;
			}
		}
		if (!temp_isDuplicated) break;
	}
	setIsDuplicated(temp_isDuplicated);
	return temp_isDuplicated;
}

void Device::describe() {
	std::printf("Il device ha %d righe e %d colonne.\n", numRows(), numCols());
	std::printf("Il device ");
	if (hasAngles()) {
		std::printf("ha il check sui seguenti angoli:\n");
		vector<vector<bool>> angles = getAngles();
		for (int i = 0; i < 4; i++) {
			int x = i % 2;
			int y = (i - x) / 2;
			bool check = angles[x][y];
			if (check) {
				switch (i) {
				case 0:
					std::printf("- Top Left\n");
					break;
				case 1:
					std::printf("- Top Right\n");
					break;
				case 2:
					std::printf("- Bottom Left\n");
					break;
				case 3:
					std::printf("- Bottom Right\n");
					break;
				default:
					break;
				}
			}
		}
	}
	else std::printf("non ha check sugli angoli.\n");
	std::printf("Il device ");
	if (!isDuplicated()) std::printf("non ");
	std::printf("e' sdoppiato orizzontalmente.\n\n");
}