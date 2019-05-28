#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "header.hpp"

int main(int argc, char** argv) {
	/***** Generate motion parameters ******/
	std::vector<cv::Mat> motionvec;
	motionMat(motionvec, image_count, rfactor, true);
	for (size_t i = 0; i < image_count; i++) {
		Src[i] = cv::imread("../Images/PCB/LR" + boost::lexical_cast<std::string>(i + 1) + ".tif", CV_LOAD_IMAGE_ANYDEPTH);
		//degimage[i] = imread("../ibm_16_300/LR"+ boost::lexical_cast<std::string> (i+1) + ".tif", CV_LOAD_IMAGE_ANYDEPTH);
		Src[i].convertTo(Src[i], CV_32F);
		dest = cv::Mat(Src[0].rows * rfactor, Src[0].cols * rfactor, CV_16UC1);
		std::cout << "Src[0].rows, Src[0].cols = " << Src[0].rows << " , " << Src[0].cols << ", rfactor =" << rfactor
				<< ", dest size = " << dest.size() << std::endl;
		cv::resize(Src[0], dest, dest.size(), 0, 0, INTER_CUBIC);
		/***** Generate Matrices A = DHF, inverse A = DHFT and B = DHF2, invere B = DHFT2 ******/
		GenerateAT(Src[i], dest, i, motionvec, ker, rfactor, DMatrix, HMatrix, MMatrix, A, AT, A2, AT2, DHF, DHFT, DHF2, DHFT2);
		std::cout << "Matrices of image " << (i + 1) << " done." << std::endl;
	}

	/***** Implement optimizator (ADMM) to solve given cost function, including calculate the gradient ******/
	std::cout << "------------------------------------------------" << std::endl;
	initialize();
//	updateX();
	admm();
//	test();
	output();
	return 0;
}
