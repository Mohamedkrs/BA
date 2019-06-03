#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <dlib/image_transforms.h>
#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace cv;
using namespace std;
using namespace dlib;

Mat frame;

void detectionfunction(frontal_face_detector hogFaceDetector, Mat &frameDlibHog )
{
	// Convert OpenCV image format to Dlib's image format
	cv_image<bgr_pixel> dlibIm(frameDlibHog);
	
	// Detect faces in the image
	std::vector<dlib::rectangle> faceRects = hogFaceDetector(dlibIm);

	//Draw an ellipse arround the face
	for (size_t i = 0; i < faceRects.size(); i++)
	{
		int x1 = (int)(faceRects[i].left() );
		int y1 = (int)(faceRects[i].top());
		int x2 = (int)(faceRects[i].right());
		int y2 = (int)(faceRects[i].bottom());
		ellipse(frame, cv::Point((((x2 - x1) / 2) + x1 ), (((y2 - y1) / 2) +y1)), cv::Point(((x2 - x1) / 2), ((y2 - y1) / 2)), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		
	}
}

int main(int argc, const char** argv)
{
	String Videopath;
	
	//Read Config file
	frontal_face_detector hogFaceDetector = get_frontal_face_detector();

	//Capture video from camera or saved video, either 1 to read cameras stream or give the video path
	VideoCapture source;
	if (argc == 1)
		source.open(0);
	else
		source.open(Videopath);


	double Endtime = 0;
	double fps = 0;
	while (1)
	{
		source >> frame;
		if (frame.empty())
			break;
		double t = cv::getTickCount();

		//Enter the frame to the detection function of HOG
		detectionfunction(hogFaceDetector, frame);

		//Calculate computation time
		Endtime = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

		//Calculate FPS
		fps = 1 / Endtime;

		putText(frame, format("HOG ; FPS = %.2f", fps), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);

		//Show Video 
		imshow("HOG Face Detection", frame);
	}
		
	
	
