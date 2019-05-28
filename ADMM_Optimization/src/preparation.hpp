#ifndef SRC_PREPARATION_HPP_
#define SRC_PREPARATION_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <eigen3/Eigen/Dense>
#include <opencv2/core/eigen.hpp>
#include "header.hpp"

using namespace cv;

hMat Dmatrix(cv::Mat& Src, cv::Mat & Dest, int rfactor) {
	int dim_srcvec = Src.rows * Src.cols, dim_dstvec = Dest.rows * Dest.cols;
	hMat _Dmatrix(dim_srcvec, dim_dstvec);
	for (int i = 0; i < Src.rows; i++)
		for (int j = 0; j < Src.cols; j++) {
			int LRindex = i * Src.cols + j;
			for (int m = rfactor * i; m < (i + 1) * rfactor; m++)
				for (int n = rfactor * j; n < (j + 1) * rfactor; n++) {
					int HRindex = m * Dest.cols + n;
					_Dmatrix.coeffRef(LRindex, HRindex) = 1.0 / rfactor / rfactor;
					//std::cout<<"_Dmatrix.coeffRef(LRindex,HRindex) = "<<1.0/rfactor/rfactor<<", rfactor = "<<rfactor<<std::endl;
				}
		}
	return _Dmatrix;
}

hMat Hmatrix(cv::Mat & Dest, const cv::Mat& kernel) {
	int dim_dstvec = Dest.rows * Dest.cols;
	hMat _Hmatrix(dim_dstvec, dim_dstvec);
	for (int i = 0; i < Dest.rows; i++)
		for (int j = 0; j < Dest.cols; j++) {
			int index = i * Dest.cols + j, UL = (i - 1) * Dest.cols + (j - 1), UM = (i - 1) * Dest.cols + j, UR = (i - 1)
					* Dest.cols + (j + 1), ML = i * Dest.cols + (j - 1), MR = i * Dest.cols + (j + 1), BL = (i + 1) * Dest.cols
					+ (j - 1), BM = (i + 1) * Dest.cols + j, BR = (i + 1) * Dest.cols + (j + 1);
			if (i - 1 >= 0 && j - 1 >= 0 && UL < dim_dstvec) _Hmatrix.coeffRef(index, UL) = kernel.at<float>(0, 0);
			if (i - 1 >= 0 && UM < dim_dstvec) _Hmatrix.coeffRef(index, UM) = kernel.at<float>(0, 1);
			if (i - 1 >= 0 && j + 1 < Dest.cols && UR < dim_dstvec) _Hmatrix.coeffRef(index, UR) = kernel.at<float>(0, 2);
			if (j - 1 >= 0 && ML < dim_dstvec) _Hmatrix.coeffRef(index, ML) = kernel.at<float>(1, 0);
			if (j + 1 < Dest.cols && MR < dim_dstvec) _Hmatrix.coeffRef(index, MR) = kernel.at<float>(1, 2);
			if (j - 1 >= 0 && i + 1 < Dest.rows && BL < dim_dstvec) _Hmatrix.coeffRef(index, BL) = kernel.at<float>(2, 0);
			if (i + 1 < Dest.rows && BM < dim_dstvec) _Hmatrix.coeffRef(index, BM) = kernel.at<float>(2, 1);
			if (i + 1 < Dest.rows && j + 1 < Dest.cols && BR < dim_dstvec) _Hmatrix.coeffRef(index, BR) = kernel.at<float>(2, 2);
			_Hmatrix.coeffRef(index, index) = kernel.at<float>(1, 1);
		}
	return _Hmatrix;
}

hMat Mmatrix(cv::Mat &Dest, float deltaX, float deltaY) {
	int dim_dstvec = Dest.rows * Dest.cols;
	hMat _Mmatrix(dim_dstvec, dim_dstvec);
	for (int i = 0; i < Dest.rows; i++)
		for (int j = 0; j < Dest.cols; j++)
			if (i < (Dest.rows - std::floor(deltaY)) && j < (Dest.cols - std::floor(deltaX)) && (i + std::floor(deltaY) >= 0)
					&& (j + std::floor(deltaX) >= 0)) {
				int index = i * Dest.cols + j, neighborUL = (i + std::floor(deltaY)) * Dest.cols + (j + std::floor(deltaX)),
						neighborUR = (i + std::floor(deltaY)) * Dest.cols + (j + std::floor(deltaX) + 1), neighborBR = (i
								+ std::floor(deltaY) + 1) * Dest.cols + (j + std::floor(deltaX) + 1), neighborBL = (i
								+ std::floor(deltaY) + 1) * Dest.cols + (j + std::floor(deltaX));
				if (neighborUL >= 0 && neighborUL < dim_dstvec) _Mmatrix.coeffRef(index, neighborUL) = (i + std::floor(deltaY) + 1
						- (i + deltaY)) * (j + std::floor(deltaX) + 1 - (j + deltaX));
				if (neighborUR >= 0 && neighborUR < dim_dstvec) _Mmatrix.coeffRef(index, neighborUR) = (i + std::floor(deltaY) + 1
						- (i + deltaY)) * (j + deltaX - (j + std::floor(deltaX)));
				if (neighborBR >= 0 && neighborBR < dim_dstvec) _Mmatrix.coeffRef(index, neighborBR) = (i + deltaY
						- (i + std::floor(deltaY))) * (j + deltaX - (j + std::floor(deltaX)));
				if (neighborBL >= 0 && neighborBL < dim_dstvec) _Mmatrix.coeffRef(index, neighborBL) = (i + deltaY
						- (i + std::floor(deltaY))) * (j + std::floor(deltaX) + 1 - (j + deltaX));
			}
	return _Mmatrix;
}

void motionMat(std::vector<Mat>& motionVec, size_t image_count, size_t rfactor, bool clockwise) {
	size_t quotient, remainder;
	if (clockwise) for (size_t i = 0; i < image_count; i++) {
		Mat motionvec = Mat::zeros(3, 3, CV_32F);
		motionvec.at<float>(0, 0) = 1, motionvec.at<float>(0, 1) = 0, motionvec.at<float>(1, 0) = 0, motionvec.at<float>(1, 1) = 1;
		motionvec.at<float>(2, 0) = 0, motionvec.at<float>(2, 1) = 0, motionvec.at<float>(2, 2) = 1;
		quotient = floor(i / 1.0 / rfactor), remainder = i % rfactor;
		if (quotient % 2 == 0) motionvec.at<float>(0, 2) = remainder / 1.0 / rfactor;
		else motionvec.at<float>(0, 2) = (rfactor - remainder - 1) / 1.0 / rfactor;
		motionvec.at<float>(1, 2) = quotient / 1.0 / rfactor, motionVec.push_back(motionvec);
		std::cout << "image i = " << i << ", x motion = " << motionvec.at<float>(0, 2) << ", y motion = "
				<< motionvec.at<float>(1, 2) << std::endl;
	}
	else for (size_t i = 0; i < image_count; i++) {
		Mat motionvec = Mat::zeros(3, 3, CV_32F);
		motionvec.at<float>(0, 0) = 1, motionvec.at<float>(0, 1) = 0, motionvec.at<float>(1, 0) = 0, motionvec.at<float>(1, 1) = 1;
		motionvec.at<float>(2, 0) = 0, motionvec.at<float>(2, 1) = 0, motionvec.at<float>(2, 2) = 1;
		quotient = floor(i / 1.0 / rfactor), remainder = i % rfactor;
		if (quotient % 2 == 0) motionvec.at<float>(1, 2) = remainder / 1.0 / rfactor;
		else motionvec.at<float>(1, 2) = (rfactor - remainder - 1) / 1.0 / rfactor;
		motionvec.at<float>(0, 2) = quotient / 1.0 / rfactor, motionVec.push_back(motionvec);
		std::cout << "image i = " << i << ", x motion = " << motionvec.at<float>(0, 2) << ", y motion = "
				<< motionvec.at<float>(1, 2) << std::endl;
	}
}

hMat sparseMatSq(hMat& src) {
	hMat A2(src.rows(), src.cols());
	for (int k = 0; k < src.outerSize(); ++k)
		for (typename hMat::InnerIterator innerit(src, k); innerit; ++innerit)
			//A2.insert(innerit.row(), innerit.col()) = innerit.value() * innerit.value();
			A2.insert(k, innerit.index()) = innerit.value() * innerit.value();
	//A2.insert(innerit.row(), innerit.col()) = 0;
	A2.makeCompressed();
	return A2;
}

hMat ComposeSystemMatrix(cv::Mat& Src, cv::Mat& Dest, const cv::Point2f delta, int rfactor, const cv::Mat& kernel, hMat& DMatrix,
		hMat &HMatrix, hMat &MMatrix) {
	int dim_srcvec = Src.rows * Src.cols, dim_dstvec = Dest.rows * Dest.cols;
	//float maxPsfRadius = 3 * rfactor * psfWidth;
	hMat _DHF(dim_srcvec, dim_dstvec);
	DMatrix = Dmatrix(Src, Dest, rfactor), HMatrix = Hmatrix(Dest, kernel), MMatrix = Mmatrix(Dest, delta.x, delta.y);
	_DHF = DMatrix * (HMatrix * MMatrix), _DHF.makeCompressed();
	return _DHF;
}

void Normalization(hMat& src, hMat& dst) {
	for (Eigen::Index c = 0; c < src.rows(); ++c) {
		float colsum = 0.0;
		for (typename hMat::InnerIterator itL(src, c); itL; ++itL)
			colsum += itL.value();
		for (typename hMat::InnerIterator itl(src, c); itl; ++itl)
			dst.coeffRef(itl.row(), itl.col()) = src.coeffRef(itl.row(), itl.col()) / colsum;
	}
}

void Gaussiankernel(cv::Mat& dst) {
	int klim = int((dst.rows - 1) / 2);
	for (int i = -klim; i <= klim; i++)
		for (int j = -klim; j <= klim; j++) {
			float dist = i * i + j * j;
			dst.at<float>(i + klim, j + klim) = 1 / (2 * M_PI) * exp(-dist / 2);
		}
	float normF = cv::sum(dst)[0];
	dst = dst / normF;
}

void GenerateAT(cv::Mat& Src, cv::Mat& Dest, int imgindex, std::vector<Mat>& motionVec, cv::Mat &kernel, size_t rfactor,
		hMat& DMatrix, hMat &HMatrix, hMat &MMatrix, hMat& A, hMat& AT, hMat& A2, hMat& AT2, std::vector<mat>& DHF,
		std::vector<mat>& DHFT, std::vector<mat> &DHF2, std::vector<mat> &DHFT2) {
	Gaussiankernel(kernel);
	cv::Point2f Shifts;
	Shifts.x = motionVec[imgindex].at<float>(0, 2) * rfactor, Shifts.y = motionVec[imgindex].at<float>(1, 2) * rfactor;
	A = ComposeSystemMatrix(Src, Dest, Shifts, rfactor, kernel, DMatrix, HMatrix, MMatrix);
	Normalization(A, A), A2 = sparseMatSq(A), AT = A.transpose(), AT2 = A2.transpose();
	A *= I_0, A2 *= I_0, AT *= I_0, AT2 *= I_0;
	mat tmp_vcl(A.rows(), A.cols(), A.nonZeros()), tmp_vclT(AT.rows(), AT.cols(), AT.nonZeros());
	hA.push_back(A), hB.push_back(A2);
	viennacl::copy(A, tmp_vcl), viennacl::copy(AT, tmp_vclT), DHF.push_back(tmp_vcl), DHFT.push_back(tmp_vclT);
	viennacl::copy(A2, tmp_vcl), viennacl::copy(AT2, tmp_vclT), DHF2.push_back(tmp_vcl), DHFT2.push_back(tmp_vclT);
}

#endif /* SRC_PREPARATION_HPP_ */
