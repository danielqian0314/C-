#ifndef SRC_HEADER_HPP_
#define SRC_HEADER_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <eigen3/Eigen/Sparse>
#include <Core/Assert.hpp>
#include <Core/Time.hpp>
#include <Core/TimeSpan.hpp>
#include <boost/lexical_cast.hpp>
#include <math.h>
#include <stdio.h>
#include <viennacl/linalg/inner_prod.hpp>
#include <viennacl/linalg/norm_1.hpp>
#include <viennacl/linalg/norm_2.hpp>
#include <viennacl/backend/opencl.hpp>
#include <viennacl/compressed_matrix.hpp>
#include "variables.hpp"
#include "preparation.hpp"
#include "function.hpp"
#include "climpl.hpp"
#include "init.hpp"
#include "admm.hpp"

using namespace cv;

hMat Dmatrix(cv::Mat& Src, cv::Mat & Dest, int rfactor);
hMat Hmatrix(cv::Mat & Dest, const cv::Mat& kernel);
hMat Mmatrix(cv::Mat &Dest, float deltaX, float deltaY);
void motionMat(std::vector<Mat>& motionVec, size_t image_count, size_t rfactor, bool clockwise);
hMat sparseMatSq(hMat& src);
hMat ComposeSystemMatrix(cv::Mat& Src, cv::Mat& Dest, const cv::Point2f delta, int rfactor, const cv::Mat& kernel, hMat& DMatrix,
		hMat &HMatrix, hMat &MMatrix);
void Normalization(hMat& src, hMat& dst);
void Gaussiankernel(cv::Mat& dst);
void GenerateAT(cv::Mat& Src, cv::Mat& Dest, int imgindex, std::vector<Mat>& motionVec, cv::Mat &kernel, size_t rfactor,
		hMat& DMatrix, hMat &HMatrix, hMat &MMatrix, hMat& A, hMat& AT, hMat& A2, hMat& AT2, std::vector<mat>& DHF,
		std::vector<mat>& DHFT, std::vector<mat> &DHF2, std::vector<mat> &DHFT2);

#define sign_float(a,b) (a>b)?1.0f:(a<b)?-1.0f:0.0f
#define max_float(a,b) (a>b)?a:b
#define ROUND_UINT(d) ( (unsigned int) ((d) + ((d) > 0 ? 0.5 : -0.5)) )

/////////////////////////////////////////////////////////////////////////////////////

#endif /* SRC_HEADER_HPP_ */
