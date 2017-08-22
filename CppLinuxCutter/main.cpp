/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   main.cpp
 * Author: x
 *
 * Created on October 27, 2016, 10:41 PM
 */

#include <sstream>
#include <string>
#include <iostream>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>
#include <opencv/cv.h>
#include <opencv2/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace std;

#define M_MOG2 2
#define M_KNN  3

int maxThresholdVal = 255, minThresholdVal = 250;

int nearestNonEvenInt(int to)
{
	return (to % 2 != 0) ? to : (to + 1);
}

void onMaxThresholdChanged(int, void*){
    setTrackbarMax("min threshold", "Params", maxThresholdVal);
}

void onMinThresholdChanged(int, void*){
    setTrackbarMin("max threshold", "Params", minThresholdVal);
}

/*
 * 
 */
int main(int argc, char** argv) {
        bool useCamera = false;
        int prevMedianBlurStrng = 5, postMedianBlurStrng = 3, BlurStrng = 3, SecondBlurStrng = 1;
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
		return 0;
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
	//resize(bgImg, bgImg, frame.size());

	cout << "init windows\n";
	namedWindow("Params", WINDOW_NORMAL);
	createTrackbar("prev medianBlur", "Params", &prevMedianBlurStrng, 35);
        createTrackbar("post medianBlur", "Params", &postMedianBlurStrng, 35);
	createTrackbar("PrevBlur", "Params", &BlurStrng, 10);
	createTrackbar("PostBlur", "Params", &SecondBlurStrng, 10);
        createTrackbar("min threshold", "Params", &minThresholdVal, 250, onMinThresholdChanged);
        createTrackbar("max threshold", "Params", &maxThresholdVal, 255, onMaxThresholdChanged);
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
		medianBlur(fgmask, fgmask, nearestNonEvenInt(prevMedianBlurStrng));
		blur(fgmask, fgmask, Size(BlurStrng+1, BlurStrng + 1));
		cv::threshold(fgmask, fgmask, minThresholdVal, maxThresholdVal, cv::THRESH_BINARY);
                medianBlur(fgmask, fgmask, nearestNonEvenInt(postMedianBlurStrng));
		blur(fgmask, fgmask, Size(SecondBlurStrng+1, SecondBlurStrng+1));

		double fps = getTickFrequency() / (getTickCount() - start);
		std::cout << "FPS : " << fps << std::endl;
		std::cout << fgimg.size() << std::endl;
		fgimg.setTo(Scalar::all(0));
		//bgImg.copyTo(fgimg);
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
    return 0;
}

