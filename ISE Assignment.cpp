//includes the neccessary function for opencv
#include "opencv2/imgproc/imgproc.hpp"
#include <tesseract/baseapi.h>
#include <iostream>
#include <algorithm>
#include <math.h>
#include <string.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include "tesseract.h"

using namespace cv;
using namespace std;

Mat invertion(Mat Grey)
{
	Mat Invertimg = Mat::zeros(Grey.size(), CV_8UC1);



	for (int i = 0; i < Grey.rows; i++)
	{
		for (int j = 0; j < Grey.cols; j++)
		{
			Invertimg.at<uchar>(i, j) = 255 - Grey.at<uchar>(i, j);
		}
	}



	return Invertimg;
}

//mat function for converts RGB image to Grey
Mat ISERGBtoGrey(Mat Rgbimg) {

	Mat Greyimg = Mat::zeros(Rgbimg.size(), CV_8UC1);

	for (int i = 0; i < Rgbimg.rows; i++) {

		for (int j = 0; j < Rgbimg.cols * 3; j = j + 3) {

			Greyimg.at<uchar>(i, j / 3) = (Rgbimg.at<uchar>(i, j) + Rgbimg.at<uchar>(i, j + 1) + Rgbimg.at<uchar>(i, j + 2)) / 3;

		}
	}
	return Greyimg;
}

//mat function for converts grey image to Binary
Mat ISEGreytoBinary(Mat Greyimg, int threshold) {

	//create grey image which is full of zeros (a black image)
	Mat Binaryimg = Mat::zeros(Greyimg.size(), CV_8UC1);
	//int a, b;
	for (int i = 0; i < Greyimg.rows; i++) {


		for (int j = 0; j < Greyimg.cols; j = j + 1) {

			if (Greyimg.at<uchar>(i, j) > threshold) {
				Binaryimg.at<uchar>(i, j) = 255;
			}

		}
	}
	return Binaryimg;
}

//mat function for stepping the greyimage
Mat ISEStep(Mat Greyimg, int th1, int th2) {

	// create a image which got same size with Grey Image
	Mat Stepimage = Mat(Greyimg.size(), CV_8UC1);

	for (int i = 0; i < Greyimg.rows; i++) {

		for (int j = 0; j < Greyimg.cols; j = j + 1) {

			if (Greyimg.at<uchar>(i, j) > th1 && Greyimg.at<uchar>(i, j) < th2) {
				Stepimage.at<uchar>(i, j) = 255;
			}
			else {
				Stepimage.at<uchar>(i, j) = 0;
			}
		}
	}

	return Stepimage;
}

// Mat function for convert greyimage to blur image
Mat ISEBlur(Mat Greyimg, int n) {

	Mat BlurImage = Mat(Greyimg.size(), CV_8UC1);

	for (int i = n; i < Greyimg.rows - n; i++) {
		for (int j = n; j < Greyimg.cols - n; j = j + 1) {
			int sum = 0;
			for (int ii = -n; ii <= n; ii++) {
				for (int jj = -n; jj <= n; jj++) {
					sum += Greyimg.at<uchar>(i + ii, j + jj);
				}
			}
			BlurImage.at<uchar>(i, j) = sum / ((2 * n + 1) * (2 * n + 1));
		}
	}

	return BlurImage;
}

//Mat function for vertical edge detection
Mat ISEVerticalEdge(Mat Greyimg) {
	Mat Image = Mat::zeros(Greyimg.size(), CV_8UC1);
	for (int i = 1; i < Greyimg.rows - 1; i++) {
		for (int j = 1; j < Greyimg.cols - 1; j++) {
			int leftside = -1 * Greyimg.at<uchar>(i - 1, j - 1) + -2 * Greyimg.at<uchar>(i, j - 1) + -1 * Greyimg.at<uchar>(i + 1, j - 1);
			int rightside = Greyimg.at<uchar>(i - 1, j + 1) + 2 * Greyimg.at<uchar>(i, j + 1) + Greyimg.at<uchar>(i + 1, j + 1);
			if (abs(leftside + rightside) > 50) {
				Image.at<uchar>(i, j) = 255;
			}
		}
	}
	return Image;
}

//Mat function for increase white pixels
Mat ISEDilation(Mat Greyimg, int n) {
	//create black image
	Mat Image = Mat::zeros(Greyimg.size(), CV_8UC1);

	for (int i = n; i < Greyimg.rows - n; i++) {
		for (int j = n; j < Greyimg.cols - n; j++) {
			//check neighbours
			for (int ii = -n; ii <= n; ii++) {
				for (int jj = -n; jj <= n; jj++) {
					//if one of the neighbour is 255 (white)
					if (Greyimg.at<uchar>(i + ii, j + jj) == 255) {
						Image.at<uchar>(i, j) = 255;
						break;
					}
				}
			}
		}
	}

	return Image;
}
//Mat function for decrease white pixels
Mat ISEErosion(Mat Greyimg, int n) {
	//create white image
	//Mat Image = Mat(Grey.size(), CV_8UC1, Scalar(255, 255, 255));

	Mat Image = Mat::zeros(Greyimg.size(), CV_8UC1);
	Image = Greyimg.clone();

	for (int i = n; i < Greyimg.rows - n; i++) {
		for (int j = n; j < Greyimg.cols - n; j++) {
			//check neighbours
			for (int ii = -n; ii <= n; ii++) {
				for (int jj = -n; jj <= n; jj++) {
					//if one of the neighbour is 0 (white)
					if (Greyimg.at<uchar>(i + ii, j + jj) == 0) {
						Image.at<uchar>(i, j) = 0;
						break;
					}
				}
			}

		}
	}

	return Image;
}

//Mat function for filer all the colors to black except white pixels
Mat ISEcolorFilter(Mat LPRImage) {
	Mat colorFilteredimg;
	cvtColor(LPRImage, colorFilteredimg, COLOR_BGR2HSV);
	Mat m1, m2, m3, m4, m5, m6, m7;
	inRange(colorFilteredimg, Scalar(0, 0, 80), Scalar(180, 255, 255), m1);
	inRange(colorFilteredimg, Scalar(100, 80, 80), Scalar(180, 210, 230), m2); //red color
	inRange(colorFilteredimg, Scalar(17, 88, 86), Scalar(50, 255, 255), m3); //red color
	inRange(colorFilteredimg, Scalar(0, 5, 60), Scalar(140, 70, 210), m4); //black color
	inRange(colorFilteredimg, Scalar(75, 5, 60), Scalar(180, 85, 195), m5); //black color
	inRange(colorFilteredimg, Scalar(80, 35, 100), Scalar(100, 65, 220), m6); //black color
	inRange(colorFilteredimg, Scalar(20, 66, 100), Scalar(25, 90, 120), m7); //black color

	Mat RGB;
	//17 127 225
	colorFilteredimg.setTo(Scalar(0, 0, 0), ~m1);
	colorFilteredimg.setTo(Scalar(0, 0, 0), m2);
	colorFilteredimg.setTo(Scalar(0, 0, 0), m3);
	colorFilteredimg.setTo(Scalar(0, 0, 0), m4);
	colorFilteredimg.setTo(Scalar(0, 0, 0), m5);
	colorFilteredimg.setTo(Scalar(0, 0, 0), m6);
	colorFilteredimg.setTo(Scalar(0, 0, 0), m7);
	cvtColor(colorFilteredimg, RGB, COLOR_HSV2BGR);
	return RGB;
}

void plateDetection(string carplate, int number) {

	//Original image of LPRImage
	Mat LPRImage;
	LPRImage = imread(carplate);
	//imshow("Number plate", LPRImage);
	//LPRimage to Grey Image
	Mat Greyimage = ISERGBtoGrey(LPRImage);
	//Color Filtered the LPRImage 
	Mat colorfilteredImage = ISEcolorFilter(LPRImage);
	//imshow("Number Plate Filtered", colorfilteredImage);

	Mat GreyImage2 = ISERGBtoGrey(colorfilteredImage);
	//imshow("GreyImage2", GreyImage2);

	Mat BluredGreyImage = ISEBlur(GreyImage2, 1);
	//imshow("Blur", BluredGreyImage);

	//Edge Detection
	Mat VImage = ISEVerticalEdge(BluredGreyImage);
	//imshow("Vertical Edge Image", VImage);
	Mat DImage = ISEDilation(VImage, 2);
	//imshow("Dilation Image ", DImage);
	Mat EImage = ISEErosion(DImage, 5);
	//imshow("Erosion Image", EImage);
	Mat DImage2 = ISEDilation(EImage, 7);
	//imshow("Diltation Image 2", DImage2);
	Mat EImage2 = ISEErosion(DImage2, 4);
	//imshow("Erosion Image 2", EImage2); 

	Mat blob;
	blob = EImage2.clone();
	vector<vector<Point>> segments;
	vector<Vec4i> hierarchy1;
	findContours(EImage2, segments, hierarchy1, RETR_EXTERNAL,
		CHAIN_APPROX_NONE, Point(0, 0));

	Mat dst = Mat::zeros(Greyimage.size(), CV_8UC3); //8UC3->RGB
	if (!segments.empty()) { //if segments!= 0 
		for (int i = 0; i < segments.size(); i++) {
			//Assign random color to each segment
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			//fill color to each segment
			drawContours(dst, segments, i, colour, -1, 8, hierarchy1);
		}
		//imshow("Segmentation", dst);
	}


	//Convert small segments to black color (clear noise)
	Rect rect;
	Scalar black = CV_RGB(0, 0, 0);
	Mat plate;
	//go through each segment
	for (int i = 0; i < segments.size(); i++)
	{
		// filter contours
		rect = boundingRect(segments[i]);
		Mat mask = Mat::zeros(GreyImage2.size(), CV_8UC1);
		Mat maskROI(mask, rect);
		maskROI = Scalar(0, 0, 0);
		// fill the contour
		drawContours(mask, segments, i, Scalar(255, 255, 255), FILLED);
		// ratio of non-zero pixels in the filled region
		double r = (double)countNonZero(maskROI) / (rect.width * rect.height);

		//find bounding box
		//convert small/high segments to black
		if (rect.width > GreyImage2.size().width / 6 ||
			rect.width < 42 ||
			rect.height < 15 ||
			(abs(rect.width - rect.height) > 20 && abs(rect.width - rect.height) < 25) ||
			rect.width * rect.height < 1080 ||
			rect.width > 123 ||
			rect.height > 47 ||
			rect.height > rect.width ||
			(rect.width / rect.height < 2 && (rect.width > 50 || rect.height < 27)) ||
			(rect.width / rect.height >= 2 && (rect.width < 68 || rect.height > 41)) ||
			rect.x < GreyImage2.cols * 0.1 || rect.x > GreyImage2.cols * 0.9 || //remove the unwanted most left & most right
			rect.y < GreyImage2.rows * 0.1 || rect.y > GreyImage2.rows * 0.92 || //remove the unwanted most top & bottom
			r <= .4) //remove the unwanted if the non zero pixel is less than 40%

		{
			drawContours(blob, segments, i, black, -1, 8, hierarchy1); // noise
		}
		else {

			plate = Greyimage(rect);
			imshow("Final Plate", plate);

			int n = number;
			tesseractmain(plate, n);

		}
	}
	LPRImage.release();
	return;
}

void plateDetection2(string carplate, int number) {

	//Original image of LPRImage
	Mat LPRImage;
	LPRImage = imread(carplate);
	//imshow("Number plate", LPRImage);
	//LPRimage to Grey Image
	Mat Greyimage = ISERGBtoGrey(LPRImage);
	//imshow("RGB to Grey", Greyimage);
	//Color Filtered the LPRImage 
	Mat colorfilteredImage = ISEcolorFilter(LPRImage);
	//imshow("Number Plate Filtered", colorfilteredImage);
	Mat GreyImage2 = ISERGBtoGrey(colorfilteredImage);
	//imshow("GreyImage2", GreyImage2);
	Mat BluredGreyImage = ISEBlur(GreyImage2, 1);
	//imshow("Blur", BluredGreyImage);

	//Edge Detection
	Mat VImage = ISEVerticalEdge(BluredGreyImage);
	//imshow("Vertical Edge Image", VImage);
	Mat DImage = ISEDilation(VImage, 2);
	//imshow("Dilation Image ", DImage);
	Mat EImage = ISEErosion(DImage, 5);
	//imshow("Erosion Image", EImage);
	Mat DImage2 = ISEDilation(EImage, 7);
	//imshow("Dilation Image 2", DImage2);
	Mat EImage2 = ISEErosion(DImage2, 4);
	//imshow("Erosion Image 2", EImage2); 

	Mat blob;
	blob = EImage2.clone();
	vector<vector<Point>> segments;
	vector<Vec4i> hierarchy1;
	findContours(EImage2, segments, hierarchy1, RETR_EXTERNAL,
		CHAIN_APPROX_NONE, Point(0, 0));

	Mat dst = Mat::zeros(Greyimage.size(), CV_8UC3); //8UC3->RGB
	if (!segments.empty()) { //if segments!= 0 
		for (int i = 0; i < segments.size(); i++) {
			//Assign random color to each segment
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			//fill color to each segment
			drawContours(dst, segments, i, colour, -1, 8, hierarchy1);
		}
		//imshow("Segmentation", dst);
	}

	//Convert small segments to black color (clear noise)
	Rect rect;
	Scalar black = CV_RGB(0, 0, 0);
	Mat plate;
	//go through each segment
	for (int i = 0; i < segments.size(); i++)
	{
		// filter contours
		rect = boundingRect(segments[i]);
		Mat mask = Mat::zeros(GreyImage2.size(), CV_8UC1);
		Mat maskROI(mask, rect);
		maskROI = Scalar(0, 0, 0);
		// fill the contour
		drawContours(mask, segments, i, Scalar(255, 255, 255), FILLED);
		// ratio of non-zero pixels in the filled region
		double r = (double)countNonZero(maskROI) / (rect.width * rect.height);

		//find bounding box
		//convert small/high segments to black
		if (rect.width > GreyImage2.size().width / 6 ||
			rect.width < 42 ||
			rect.height < 15 ||
			(abs(rect.width - rect.height) > 20 && abs(rect.width - rect.height) < 25) ||
			rect.width * rect.height < 1080 ||
			rect.width > 123 ||
			rect.height > 47 ||
			rect.height > rect.width ||
			(rect.width / rect.height < 2 && (rect.width > 50 || rect.height < 27)) ||
			(rect.width / rect.height >= 2 && (rect.width < 68 || rect.height > 41)) ||
			rect.x < GreyImage2.cols * 0.1 || rect.x > GreyImage2.cols * 0.9 || //remove the unwanted most left & most right
			rect.y < GreyImage2.rows * 0.1 || rect.y > GreyImage2.rows * 0.92 || //remove the unwanted most top & bottom
			r <= .4) //remove the unwanted if the non zero pixel is less than 40%

		{
			drawContours(blob, segments, i, black, -1, 8, hierarchy1); // noise
		}
		else {
			plate = Greyimage(rect);
			imshow("Final Plate", plate);

			Mat resizeimg;
			resize(plate, resizeimg, cv::Size(), 0.75, 0.75);

			Mat biliteralimg;
			bilateralFilter(resizeimg, biliteralimg, 5, 75, 75);

			int n = number;
			tesseractmain(biliteralimg, n);
			waitKey();
		}
	}
	LPRImage.release();
	return;
}

void plateDetection3(string carplate, int number) {

	//Original image of LPRImage
	Mat LPRImage;
	LPRImage = imread(carplate);
	//imshow("Number plate", LPRImage);
	//LPRimage to Grey Image
	Mat Greyimage = ISERGBtoGrey(LPRImage);
	//imshow("RGB to Grey", Greyimage);
	//Color Filtered the LPRImage 
	Mat colorfilteredImage = ISEcolorFilter(LPRImage);
	//imshow("Number Plate Filtered", colorfilteredImage);
	Mat GreyImage2 = ISERGBtoGrey(colorfilteredImage);
	//imshow("GreyImage2", GreyImage2);
	Mat BluredGreyImage = ISEBlur(GreyImage2, 1);
	//imshow("Blur", BluredGreyImage);

	//Edge Detection
	Mat VImage = ISEVerticalEdge(BluredGreyImage);
	//imshow("Vertical Edge Image", VImage);
	Mat DImage = ISEDilation(VImage, 2);
	//imshow("Dilation Image ", DImage);
	Mat EImage = ISEErosion(DImage, 5);
	//imshow("Erosion Image", EImage);
	Mat DImage2 = ISEDilation(EImage, 7);
	//imshow("Dilation Image 2", DImage2);
	Mat EImage2 = ISEErosion(DImage2, 4);
	//imshow("Erosion Image 2", EImage2); 

	Mat blob;
	blob = EImage2.clone();
	vector<vector<Point>> segments;
	vector<Vec4i> hierarchy1;
	findContours(EImage2, segments, hierarchy1, RETR_EXTERNAL,
		CHAIN_APPROX_NONE, Point(0, 0));

	Mat dst = Mat::zeros(Greyimage.size(), CV_8UC3); //8UC3->RGB
	if (!segments.empty()) { //if segments!= 0 
		for (int i = 0; i < segments.size(); i++) {
			//Assign random color to each segment
			Scalar colour((rand() & 255), (rand() & 255), (rand() & 255));
			//fill color to each segment
			drawContours(dst, segments, i, colour, -1, 8, hierarchy1);
		}
		//imshow("Segmentation", dst);
	}

	//Convert small segments to black color (clear noise)
	Rect rect;
	Scalar black = CV_RGB(0, 0, 0);
	Mat plate;
	//go through each segment
	for (int i = 0; i < segments.size(); i++)
	{
		// filter contours
		rect = boundingRect(segments[i]);
		Mat mask = Mat::zeros(GreyImage2.size(), CV_8UC1);
		Mat maskROI(mask, rect);
		maskROI = Scalar(0, 0, 0);
		// fill the contour
		drawContours(mask, segments, i, Scalar(255, 255, 255), FILLED);
		// ratio of non-zero pixels in the filled region
		double r = (double)countNonZero(maskROI) / (rect.width * rect.height);

		//find bounding box
		//convert small/high segments to black
		if (rect.width > GreyImage2.size().width / 6 ||
			rect.width < 42 ||
			rect.height < 15 ||
			(abs(rect.width - rect.height) > 20 && abs(rect.width - rect.height) < 25) ||
			rect.width * rect.height < 1080 ||
			rect.width > 123 ||
			rect.height > 47 ||
			rect.height > rect.width ||
			(rect.width / rect.height < 2 && (rect.width > 50 || rect.height < 27)) ||
			(rect.width / rect.height >= 2 && (rect.width < 68 || rect.height > 41)) ||
			rect.x < GreyImage2.cols * 0.1 || rect.x > GreyImage2.cols * 0.9 || //remove the unwanted most left & most right
			rect.y < GreyImage2.rows * 0.1 || rect.y > GreyImage2.rows * 0.92 || //remove the unwanted most top & bottom
			r <= .4) //remove the unwanted if the non zero pixel is less than 40%

		{
			drawContours(blob, segments, i, black, -1, 8, hierarchy1); // noise
		}
		else {
			plate = Greyimage(rect);
			imshow("Final Plate", plate);

			Mat resizeimg, img_gau;
			resize(plate, resizeimg, cv::Size(), 1.5, 1.5);
			imshow("Resized Plate", resizeimg);

			Mat sharp;
			Mat sharpening_kernel = (Mat_<double>(3, 3) << -1, -1, -1, -1, 9, -1, -1, -1, -1);
			filter2D(resizeimg, sharp, -1, sharpening_kernel);

			int n = number;
			tesseractmain(sharp, n);
			waitKey();
		}
	}

	LPRImage.release();
	return;
}

int main() {

	//First Algorithm of Plate Detection and Tesseract Recognition OCR

	string carimages[7] = {
	"pic1.jpg",
	"pic4.jpg",
	"pic7.jpg",
	"pic10.jpg",
	"pic11.jpg",
	"pic12.jpg",
	"pic17.jpg" };

	int number[7] = {
		1,4,7,10,11,12,17
	};


	for (int i = 0; i < 7; i++) {
		plateDetection(carimages[i], number[i]);
		waitKey();
	}
	waitKey();

	//Second Algorithm of Plate Detection and Tesseract Recognition OCR

	string carimages2[2] = {
	"pic6.jpg",
	"pic14.jpg" };

	int number2[2] = {
		6,14
	};


	for (int i = 0; i < 2; i++) {
		plateDetection2(carimages2[i], number2[i]);
		waitKey();
	}
	waitKey();

	//Third Algorithm of Plate Detection and Tesseract Recognition OCR

	string carimages3[10] = {
	"pic2.jpg",
	"pic3.jpg",
	"pic5.jpg",
	"pic8.jpg",
	"pic9.jpg",
	"pic13.jpg",
	"pic15.jpg",
	"pic16.jpg",
	"pic18.jpg",
	"pic19.jpg" };

	int number3[10] = {
		2,3,5,8,9,13,15,16,18,19
	};


	for (int i = 0; i < 10; i++) {
		plateDetection3(carimages3[i], number3[i]);
		waitKey();
	}
	waitKey();
}
