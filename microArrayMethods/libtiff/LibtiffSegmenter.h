#pragma once

#include <iostream>

using namespace std;

#include "tiffio.h"

class LibtiffSegmenter
{
private:
	std::string file_path;
public:
	LibtiffSegmenter(std::string path);

	// Getters/Setters
	std::string getPath() { return file_path; };
	void setPath(std::string path) { file_path = path; }

	// Test suite
	static int test();
};

