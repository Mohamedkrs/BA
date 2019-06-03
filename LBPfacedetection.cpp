#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Global variables */
String faceCascadePath;
String VideoPath;
CascadeClassifier faceCascade;
std::vector<Rect> faces;

void detectionfunction(CascadeClassifier faceCascade, Mat &frame)
{
	//Start detecting faces using LBP face detection function
	faceCascade.detectMultiScale(frame, faces);

	//For each face an ellipse will be drawn arround it
	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 1.9, faces[i].height / 1.2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
	}
}


int main(int argc)
{   //load the LBP cascade model file
	faceCascadePath = "./lbpcascade_frontalface_improved.xml";

	if (!faceCascade.load(faceCascadePath)) { printf("Error loading face cascade\n"); return -1; };
	faceCascade.load(faceCascadePath);
		
	//Capture video from camera or saved video, either 1 to read cameras stream or give the video path
	VideoCapture source;
	if (argc == 1)
		source.open(0);
	else
		source.open(VideoPath);
	Mat frame;

	double Endtime = 0;
	double fps = 0;
	while (1)
	{
		source >> frame;
		if (frame.empty())
			break;
		double t = cv::getTickCount();

		//Enter the frame to the detection function of LBP
		detectionfunction(faceCascade, frame);

		//Calculate computation time
		Endtime = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

		//Calculate FPS
		fps = 1 / Endtime;

		putText(frame, format("LBP ; FPS = %.2f", fps), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);

		//Show Video 
		imshow("LBP Face Detection", frame);
		
	}
}