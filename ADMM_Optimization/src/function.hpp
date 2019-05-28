#ifndef SRC_FUNCTION_HPP_
#define SRC_FUNCTION_HPP_

#include "header.hpp"

using namespace viennacl::linalg;
using namespace Eigen;

float V(vec xx) {
	float V = 0;
	for (int i = 0; i < I; i++)
		V += (hri[i] / 2) * pow(norm_2(prod(Ti[i], xx) - zi[i] + pi[i] / ri[i]), 2);
	return V;
}

vec dV(vec xx) {
	vec dV(N);
	dV.clear();
	for (int i = 0; i < I; i++) {
		vec temp = prod(Ti[i], xx) - zi[i] + (pi[i] / ri[i]);
		dV += hri[i] * prod(Tti[i], temp);
	}
	return dV;
}

float phix(float a, vec d) {
	return V(x + a * d);
}

float phixd(float a, vec d) {
	return inner_prod(dV(x + a * d), d);
}

float ga(int i) {
	return pow(alpha, abs(p[i]) + abs(q[i]));
}

hVec ne(hVec z) {
	return (-z).array().exp();
}

float G(int i, vec zz) {
	float sum = 0;
	for (int k = 0; k < n; k++)
		sum += std::pow(hy[i][k] - hA[i].row(k) * ne(hz[i]), 2) / (hB[i].row(k) * ne(hz[i]) + sigma * sigma);
	return sum / 2;
}

float H(int i, vec zz) {
	vec nez = element_exp(-zz);
	return inner_prod(element_log(prod(DHF2[i], nez) + sigma2), one) / 2;
}

float U(int i, vec zz) {
	vec temp = prod(Ti[i], x);
	return (hri[i] / 2) * pow(norm_2(zz - temp - pi[i] / ri[i]), 2);
}

float F(int i, vec zz) {
	return G(i, zz) + H(i, zz) + U(i, zz);
}

vec MM(int i, vec zz) {
	vec nez = element_exp(-zz);
	vec temp = prod(DHF[i], nez);
	vec deno = prod(DHF2[i], nez) + sigma2;
	return element_div(yi[i] - temp, element_prod(deno, deno));
}

vec dG(int i, vec zz) {
	vec mm = MM(i, zz);
	vec nez = element_exp(-zz);
	viennacl::matrix<float> diag = viennacl::diag(nez, 0);
	vec temp1 = prod(DHFT[i], mm);
	vec temp2 = prod(diag, temp1);
	vec temp3 = element_prod(mm, mm);
	vec temp4 = prod(DHFT2[i], temp3);
	vec temp5 = prod(diag, temp4);
	return temp2 + temp5 / 2;
}

vec dH(int i, vec zz) {
	vec mm = MM(i, zz);
	vec nez = element_exp(-zz);
	viennacl::matrix<float> diag = viennacl::diag(nez, 0);
	vec temp1 = prod(DHFT2[i], mm);
	return prod(diag, temp1) / -2;
}

vec dU(int i, vec zz) {
	vec temp = prod(Ti[i], x);
	return ri[i] * (zz - temp - pi[i] / ri[i]);
}

vec dF(int i, vec zz) {
	return dG(i, zz) + dH(i, zz) + dU(i, zz);
}

float phiz(int i, float a, vec d) {
	return F(i, zi[i] + a * d);
}

float phizd(int i, float a, vec d) {
	return inner_prod(dF(i, x + a * d), d);
}

#endif /* SRC_FUNCTION_HPP_ */
