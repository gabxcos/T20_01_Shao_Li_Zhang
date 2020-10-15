#pragma once
class Device
{
private:
	int rows;
	int cols;
public:
	Device(int rows, int cols);
	void SetSize(int n_rows, int n_cols);

	int numRows();
	int numCols();
};

