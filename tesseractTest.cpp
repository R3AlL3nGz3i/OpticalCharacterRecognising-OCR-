#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <opencv2/core/base.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include "tesseract.h"
 
using namespace cv;
using namespace std;


void tesseractmain(Mat image, int number)
{
	const char* outText;
	Mat imgOpenTess, imggauTess;
	int n = number;

	tesseract::TessBaseAPI* api = new tesseract::TessBaseAPI();
	// Initialize tesseract-ocr with English, without specifying tessdata path
	if (api->Init("C:\\Users\\Alex\\Desktop\\year 2 sem 2\\Imaging and Special Effects\\ISE Assignment\\ISE Assignment\\Tesseract-OCR\\tessdata", "eng")) {
		fprintf(stderr, "Could not initialize tesseract.\n");	
		exit(1);
	}
	

	api->SetVariable("tessedit_char_whitelist", "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ");

	Mat plate = image.clone();

	// Open input image with leptonica libr	ary
	api->SetImage((uchar*)plate.data, plate.size().width, plate.size().height, plate.channels(), plate.step1());
	api->Recognize(0);
	api->SetPageSegMode(tesseract::PSM_AUTO);

	// Get OCR result
	outText = (api->GetUTF8Text());


	if (strlen(outText)>5)
	cout << "Number Plate " << n << ":" << outText << endl;

	 
	 
	// Destroy used object and release memory
	api->End();
	return;
}
