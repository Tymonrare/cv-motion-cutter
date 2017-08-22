//objectTrackingTutorial.cpp

//Written by  Kyle Hounslow 2013

//Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software")
//, to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, 
//and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

//The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
//IN THE SOFTWARE.

#include <sstream>
#include <string>
#include <iostream>
//opencv
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv/cv.h>
#include "opencv2/core.hpp"
#include "opencv2/core/ocl.hpp"
#include "opencv2/core/utility.hpp"

//C
#include <stdio.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

using namespace cv;
using namespace std;
//initial min and max HSV filter values.
//these will be changed using trackbars
int H_MIN = 0;
int H_MAX = 256;
int S_MIN = 0;
int S_MAX = 256;
int V_MIN = 0;
int V_MAX = 256;
//default capture width and height
const int FRAME_WIDTH = 640;
const int FRAME_HEIGHT = 480;
//max number of objects to be detected in frame
const int MAX_NUM_OBJECTS = 50;
//minimum and maximum object area
const int MIN_OBJECT_AREA = 20 * 20;
const int MAX_OBJECT_AREA = FRAME_HEIGHT*FRAME_WIDTH / 1.5;
//names that will appear at the top of each window
const string windowName = "Original Image";
const string windowName1 = "HSV Image";
const string windowName2 = "Thresholded Image";
const string windowName3 = "After Morphological Operations";
const string trackbarWindowName = "Trackbars";

void backFilterImageBased();

// Global variables
Mat frame; //current frame
Mat fgMaskMOG2; //fg mask fg mask generated by MOG2 method
Ptr<BackgroundSubtractor> pMOG2; //MOG2 Background subtractor
int keyboard; //input from keyboard

#define M_MOG2 2
#define M_KNN  3

int medianBlurStrng = 5, BlurStrng = 3, SecondBlurStrng = 1;
int nearestEvenInt(int to)
{
	return (to % 2 != 0) ? to : (to + 1);
}
int main(int argc, char* argv[])
{
	cout << __FILENAME__;
	backFilterImageBased();
	return EXIT_SUCCESS;
}

void backFilterImageBased() {
	bool useCamera = true;
	string file = "source.avi";
	int m = true ? M_MOG2 : M_KNN;

	VideoCapture cap;
	if (useCamera)
		cap.open(0);
	else
		cap.open(file);

	if (!cap.isOpened())
	{
		cout << "can not open camera or video file" << endl;
		return;
	}

	UMat frame, fgmask, fgimg;
	cout << "init back image\n";
	UMat bgImg = imread("bgimg.jpg", IMREAD_COLOR).getUMat(ACCESS_READ);
	cout << "copy capture to frame\n";
	cap >> frame;
	cout << "create fg image\n";
	fgimg.create(frame.size(), frame.type());
	Ptr<BackgroundSubtractorKNN> knn = createBackgroundSubtractorKNN();
	Ptr<BackgroundSubtractorMOG2> mog2 = createBackgroundSubtractorMOG2();
	cout << "resize back image\n";
	resize(bgImg, bgImg, frame.size());

	cout << "init windows\n";
	namedWindow("Params", WINDOW_NORMAL);
	createTrackbar("medianBlur", "Params", &medianBlurStrng, 35);
	createTrackbar("PrevBlur", "Params", &BlurStrng, 10);
	createTrackbar("PostBlur", "Params", &SecondBlurStrng, 10);

	switch (m)
	{
	case M_KNN:
		knn->apply(frame, fgmask);
		break;

	case M_MOG2:
		mog2->apply(frame, fgmask);
		break;
	}
	bool running = true;
	while(running)
	{
		cap >> frame;
		if (frame.empty())
			break;

		int64 start = getTickCount();


		//update the model
		switch (m)
		{
		case M_KNN:
			knn->apply(frame, fgmask);
			break;

		case M_MOG2:
			mog2->apply(frame, fgmask);
			break;
		}
		//Filter model
		medianBlur(fgmask, fgmask, nearestEvenInt(medianBlurStrng));
		blur(fgmask, fgmask, Size(BlurStrng+1, BlurStrng + 1));
		cv::threshold(fgmask, fgmask, 250, 255, cv::THRESH_BINARY);
		blur(fgmask, fgmask, Size(SecondBlurStrng+1, SecondBlurStrng+1));

		double fps = getTickFrequency() / (getTickCount() - start);
		std::cout << "FPS : " << fps << std::endl;
		std::cout << fgimg.size() << std::endl;
		fgimg.setTo(Scalar::all(0));
		bgImg.copyTo(fgimg);
		frame.copyTo(fgimg, fgmask);

		imshow("image", frame);
		imshow("foreground mask", fgmask);
		imshow("foreground image", fgimg);

		char key = (char)waitKey(30);

		switch (key)
		{
		case 27:
			running = false;
			break;
		case 'm':
		case 'M':
			ocl::setUseOpenCL(!ocl::useOpenCL());
			cout << "Switched to " << (ocl::useOpenCL() ? "OpenCL enabled" : "CPU") << " mode\n";
			break;
		}
	}
}

