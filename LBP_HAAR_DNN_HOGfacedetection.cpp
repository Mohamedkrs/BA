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
#include <stdio.h>
#include "opencv2/objdetect.hpp"
#include "opencv2/videoio.hpp"
#include <opencv2/dnn.hpp>

using namespace cv;
using namespace std;
using namespace dlib;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 116.7, 123.0);

const std::string caffeConfigFile = "./models/deploy.prototxt";
const std::string caffeWeightFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

String faceCascadePath;
CascadeClassifier faceCascade;
Mat frame;
std::vector<Rect> faces;
int Meth;//Choose wich method to use; 1 for HAAR, 2 for LBP, 3 for DNN and 4 for HOG

void HAARorLBPdetectionfunction(CascadeClassifier faceCascade, Mat &frame)
{
	//Start detecting faces using LBP face detection function
	faceCascade.detectMultiScale(frame, faces);

	//For each face an ellipse will be drawn arround it
	for (size_t i = 0; i < faces.size(); i++)
	{
		if (Meth = 2) {
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			ellipse(frame, center, Size(faces[i].width / 1.9, faces[i].height / 1.2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		}
		else {
			Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
			ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		}
	}
}

void DNNdetectionfunction(Net net, Mat &frameOpenCVDNN)
{
	int frameHeight = frameOpenCVDNN.rows;
	int frameWidth = frameOpenCVDNN.cols;

	//Convert frame to a Blob
	cv::Mat inputBlob = cv::dnn::blobFromImage(frameOpenCVDNN, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, false, false);

	//Feed the Blob to the neural network
	net.setInput(inputBlob, "data");
	cv::Mat detection = net.forward("detection_out");
	cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());


	for (int i = 0; i < detectionMat.rows; i++)
	{
		//Get detection confidence from each frame
		float confidence = detectionMat.at<float>(i, 2);

		//Filter and draw an ellipse arround the face
		if (confidence > confidenceThreshold)
		{
			int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * frameWidth);
			int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * frameHeight);
			int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * frameWidth);
			int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * frameHeight);

			ellipse(frameOpenCVDNN, cv::Point(((x2 - x1) / 2) + x1, ((y2 - y1) / 2) + y1), cv::Point((x2 - x1) / 2, (y2 - y1) / 2), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);
		}
	}

}

void HOGdetectionfunction(frontal_face_detector hogFaceDetector, Mat &frameDlibHog )
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
{	String Videopath;


	//HAAR or LBP : Read Config file
	faceCascadePath = "./lbpcascade_frontalface_improved.xml";
	faceCascadePath = "./haarcascade_frontalface_default.xml";
	faceCascade.load(faceCascadePath);

	//HOG Read Config file
	frontal_face_detector hogFaceDetector = get_frontal_face_detector();

	//DNN Read Config and Weight files
	Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);

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
		if (Meth = 1) { HAARorLBPdetectionfunction(faceCascade, frame);
		} else if (Meth = 2) { HAARorLBPdetectionfunction(faceCascade, frame); 
		} else if (Meth = 3) { DNNdetectionfunction(net, frame); 
		} else if (Meth = 4) { HOGdetectionfunction(hogFaceDetector, frame);}

		//Calculate computation time
		Endtime = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

		//Calculate FPS
		fps = 1 / Endtime;

		//ADD text and show video
		if (Meth = 1) {
			putText(frame, format("HAAR ; FPS = %.2f", fps), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
			imshow("HAAR Face Detection", frame);
		} else if (Meth = 2) {
			putText(frame, format("LBP ; FPS = %.2f", fps), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
			imshow("LBP Face Detection", frame);
		} else if (Meth = 3) {
			putText(frame, format("DNN ; FPS = %.2f", fps), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
			imshow("DNN Face Detection", frame);
		} else if (Meth = 4) {
			putText(frame, format("HOG ; FPS = %.2f", fps), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);
			imshow("HOG Face Detection", frame);
		}
		
	}
		
	
	
