#ifndef SRC_VARIABLES_HPP_
#define SRC_VARIABLES_HPP_

#include "header.hpp"

typedef viennacl::compressed_matrix<float> mat;
typedef viennacl::vector<float> vec;
typedef viennacl::scalar<float> fl;
typedef Eigen::SparseMatrix<float, Eigen::RowMajor, int> hMat;
typedef Eigen::VectorXf hVec;

size_t image_count = 4; // M
int rfactor = 2; //magnification factor
float psfWidth = 3;
int I_0 = 6e4;
int w = 1; // window size, can be tuned but odd number
float alpha = 0.4; // alpha in matrix gamma, can be tuned
float sigma = 100; // sigma can be 100, 150, 200,...500
float beta = 0.1; // needs to be tuned
std::vector<cv::Mat> Src(image_count);
cv::Mat dest, ker = cv::Mat::zeros(cv::Size(psfWidth, psfWidth), CV_32F);
std::vector<mat> DHF, DHFT, DHF2, DHFT2;
hMat DMatrix, HMatrix, MMatrix, AT, A, AT2, A2;

int K = 10, M = image_count, I = M + (2 * w + 1) * (2 * w + 1), n, N;
const int p[13] = { 0, 0, 0, 0, -1, -1, -1, 0, 0, 1, 1, 1, 0 }, q[13] = { 0, 0, 0, 0, -1, 0, 1, -1, 1, -1, 0, 1, 0 };
const float r = 1.0, c1 = 1e-4, c2 = 0.9, delta = 1, MAX = std::numeric_limits<float>::max();
const float aix = 2, aiz = 2, dx = 1, dz = 1;
vec x, sigma2, one;
hVec hsigma2;
std::vector<vec> yi, zi, pi;
std::vector<mat> Ti, Tti;
std::vector<fl> ri;
std::vector<float> hri;
std::vector<hMat> hA, hB;
std::vector<hVec> hy, hz;

#endif /* SRC_VARIABLES_HPP_ */
