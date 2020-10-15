#include "Device.h"

Device::Device(int rows, int cols)
{
	SetSize(rows, cols);
}

void Device::SetSize(int n_rows, int n_cols)
{
	rows = n_rows;
	cols = n_cols;
}

int Device::numRows()
{
	return rows;
}

int Device::numCols()
{
	return cols;
}
