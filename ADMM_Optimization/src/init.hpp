#ifndef SRC_INIT_HPP_
#define SRC_INIT_HPP_

#include "header.hpp"

void initialize() {
	n = A.rows(), N = A.cols(), hsigma2 = hVec::Constant(n, sigma * sigma);
	sigma2.resize(n), viennacl::copy(hsigma2, sigma2);
	hVec temp = hVec::Constant(n, 1);
	one.resize(n), viennacl::copy(temp, one);
	Eigen::MatrixXf y0, y1;
	clinit();

	// initialize x
	x.resize(N);
	cv::Mat dest2 = cv::Mat(Src[1].rows * rfactor, Src[1].cols * rfactor, CV_16UC1);
	cv::resize(Src[1], dest2, dest2.size(), 0, 0, INTER_CUBIC), cv2eigen(dest2, y1);
	hVec y1e = hVec(Eigen::Map<hVec>(y1.data(), y1.size()));
	viennacl::copy(y1e, x);

	// initialize yi
	for (int i = 0; i < M; i++) {
		Eigen::MatrixXf ys;
		cv2eigen(Src[i], ys);
		vec y(n);
		hVec yb = hVec(Eigen::Map<hVec>(ys.data(), ys.size()));
		viennacl::copy(yb, y), yi.push_back(y), hy.push_back(yb);
	}

	std::cout << "Loading" << std::flush;
	for (int i = 0; i < I; i++) {
		std::cout << "." << std::flush;
		vec z(N);
		zi.push_back(z);

		// initialize pi
		vec p(N);
		p.clear(), pi.push_back(p);

		// initialize Ti & Tti
		mat T(N, N), Tt(N, N);
		hMat t = Eigen::MatrixXf::Identity(N, N).sparseView();
		if (i >= M && i <= M + (2 * w + 1) * (2 * w + 1) - 2) t -= Mmatrix(dest, p[i], q[i]);
		viennacl::copy(t, T), viennacl::copy(hMat(t.transpose()), Tt);
		Ti.push_back(T), Tti.push_back(Tt);

		// initialize ri
		ri.push_back(fl(r)), hri.push_back(r);
	}

	//initialize zi
	cv2eigen(dest, y0);
	hVec y0e = hVec(Eigen::Map<hVec>(y0.data(), y0.size()));
	viennacl::copy(y0e, zi[0]);
	zi[0] = -1 * element_log(zi[0] / I_0);
	hVec zb(N);
	viennacl::copy(zi[0], zb);
	for (int i = 0; i < I; i++) {
		hVec zbb = zb * 1;
		hz.push_back(zbb);
	}
	clCopyZ();

	std::cout << " Initialization complete." << std::endl;
}

void output() {
	hVec xe(N);
	viennacl::copy(x, xe);
	Eigen::MatrixXf out = Eigen::MatrixXf(Eigen::Map<Eigen::MatrixXf>(xe.data(), dest.rows, dest.cols));
	cv::Mat output;
	eigen2cv(out, output), output.convertTo(output, CV_16UC1), cv::imwrite("../Images/PCB/output.tif", output);
}

#endif /* SRC_INIT_HPP_ */
