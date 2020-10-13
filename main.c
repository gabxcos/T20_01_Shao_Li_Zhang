#include <stdio.h>
#include <stdlib.h>

#include "tiffio.h"

int main() {
	TIFF* tif = TIFFOpen("sample.tif", "r");
	if (tif) {
		uint32 imagelength;
		tsize_t scanline;
		tdata_t* buf;
		uint32 row, col;

		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &imagelength);
		scanline = TIFFScanlineSize(tif);
		buf = _TIFFmalloc(scanline);
		for (row = 0; row < imagelength; row++) {
			TIFFReadScanline(tif, buf, row, 0);
			for (col = 0; col < scanline; col++) printf("%d", (int) buf[col]);
			printf("\n");
		}
		_TIFFfree(buf);
		TIFFClose(tif);
	}

	exit(0);
}