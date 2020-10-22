#pragma once
#include <vector>

using namespace std;

enum Controls : int {
	HYBRIDIZATION,
	EMPTY,
	NEGATIVE,
	PCR,
	CAPTURE
};

class Device
{
private:
	int rows;
	int cols;
	vector<vector<int>> controls;
	// features
	bool _hasAngles;
	vector<vector<bool>> angles; // {{Top L, Top R}, {Bot L, Bot R}}
	bool _isDuplicated;
public:
	Device(int rows, int cols, vector<vector<int>> _controls);
	Device();

	// Steps
	bool checkValidity();

	// Features
	bool checkHasAngles();
	bool checkIsDuplicated();

	// Getters / Setters
	int numRows() { return rows; };
	int numCols() { return cols; };
	void setSize(int n_rows, int n_cols) { rows = n_rows; cols = n_cols; }

	vector<vector<int>> getControls() { return controls; };
	void setControls(vector<vector<int>> _controls) { controls = _controls; }

	bool hasAngles() { return _hasAngles; }
	void setHasAngles(bool __hasAngles) { _hasAngles = __hasAngles; }

	vector<vector<bool>> getAngles() { return angles; }
	void setAngles(vector<vector<bool>> _angles) { angles = _angles; }

	bool isDuplicated() { return _isDuplicated; }
	void setIsDuplicated(bool __isDuplicated) { _isDuplicated = __isDuplicated; }

	// utilities
	void describe();
};

