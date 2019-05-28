#ifndef SRC_ADMM_HPP_
#define SRC_ADMM_HPP_

#include "header.hpp"

using namespace viennacl::linalg;

void updateX() {
	std::cout << "X, " << std::flush;
	float a = aix;
	while (abs(a) > dx) {
		vec d = dV(x);
		d /= -(float) norm_2(d);
		float m = 0, v = MAX;
		while (true) {
			if (phix(a, d) > phix(0, d) + a * c1 * phixd(0, d)) v = a;
			else if (phixd(a, d) < c2 * phixd(0, d)) m = a;
			else break;
			if (v < MAX) a = (m + v) / 2;
			else a *= 2;
		}
		x += a * d;
	}
}

void updateZA() {
	std::cout << "ZA, " << std::flush;
	for (int i = 0; i < M; i++) {
		std::cout << i << std::flush;
		float a = aiz;
		vec d = dF(i, zi[i]);
		d /= -(float) norm_2(d);
		float m = 0, v = MAX;
		for (int k = 0; k < 10; k++) {
			if (phiz(i, a, d) > phiz(i, 0, d) + a * c1 * phizd(i, 0, d)) v = a;
			else if (phizd(i, a, d) < c2 * phizd(i, 0, d)) m = a;
			else break;
			if (v < MAX) a = (m + v) / 2;
			else a *= 2;
		}
		zi[i] += a * d;
		viennacl::copy(zi[i], hz[i]);
	}
}

void updateZB() {
	std::cout << "ZB, " << std::flush;
	for (int i = M; i < I - 1; i++) {
		clUpdateZB(i, beta * ga(i) / hri[i]);
		viennacl::copy(zi[i], hz[i]);
	}
}

void updateZC() {
	std::cout << "ZC, " << std::flush;
	clUpdateZC();
	viennacl::copy(zi[12], hz[12]);
}

void updateZ() {
	updateZA(), updateZB(), updateZC();
}

void updateP() {
	std::cout << "P, " << std::flush;
	for (int i = 0; i < I; i++)
		pi[i] += ri[i] * (prod(Ti[i], x) - zi[i]);
}

void admm() {
	std::cout << "Start optimizing using ADMM..." << std::endl;
	for (int k = 0; k < K; k++)
		updateX(), updateZ(), updateP(), std::cout << " Round " << k << " complete." << std::endl;
}

void test() {
	std::cout << "Testfield:" << std::endl;
	vec t1 = dG(0, zi[0]);
	std::cout << "dG0(z0) = " << norm_2(t1) << std::endl;
	vec t2 = dH(0, zi[0]);
	std::cout << "dH0(z0) = " << norm_2(t2) << std::endl;
	vec t3 = dU(0, zi[0]);
	std::cout << "dU0(z0) = " << norm_2(t3) << std::endl;
}

#endif /* SRC_ADMM_HPP_ */
