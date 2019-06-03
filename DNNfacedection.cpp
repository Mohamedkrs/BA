#include <iostream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>


using namespace cv;
using namespace std;
using namespace cv::dnn;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(104.0, 116.7, 123.0);



const std::string caffeConfigFile = "./models/deploy.prototxt";
const std::string caffeWeightFile = "./models/res10_300x300_ssd_iter_140000_fp16.caffemodel";

void detectionfunction(Net net, Mat &frameOpenCVDNN)
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


int main(int argc)
{
	String Videopath;

	//Read Config and Weight files
	Net net = cv::dnn::readNetFromCaffe(caffeConfigFile, caffeWeightFile);

	//Capture video from camera or saved video, either 1 to read cameras stream or give the video path
	VideoCapture source;
	if (argc == 1)
		source.open(0);
	else
		source.open(Videopath);
	Mat frame;

	double Endtime = 0;
	double fps = 0;
	while (1)
	{
		source >> frame;
		if (frame.empty())
			break;
		double t = cv::getTickCount();

		//Enter the frame to the detection function of DNN
		detectionfunction(net, frame);

		//Calculate computation time
		Endtime = ((double)cv::getTickCount() - t) / cv::getTickFrequency();

		//Calculate FPS
		fps = 1 / Endtime;

		putText(frame, format("DNN ; FPS = %.2f", fps), Point(10, 50), FONT_HERSHEY_SIMPLEX, 1.4, Scalar(0, 0, 255), 4);

		//Show Video 
		imshow("DNN Face Detection", frame);

	}
}