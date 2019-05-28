#ifndef SRC_CLIMPL_HPP_
#define SRC_CLIMPL_HPP_

#include <string>
#include <fstream>
#include <streambuf>
#include "header.hpp"

using namespace viennacl::ocl;

cl_uint cl_N;

void clinit() {
	cl_N = static_cast<cl_uint>(N);
	std::ifstream s("Kernel.cl");
	std::string src((std::istreambuf_iterator<char>(s)), std::istreambuf_iterator<char>());
	current_context().add_program(src, "prog");
	std::cout << "Init CL complete." << std::endl;
}

kernel& k(std::string const & name) {
	return current_context().get_program("prog").get_kernel(name);
}

void clCopyZ() {
	enqueue(k("_copyZ")(zi[0], zi[1], zi[2], zi[3], zi[4], zi[5], zi[6], zi[7], zi[8], zi[9], zi[10], zi[11], zi[12], cl_N));
}

void clUpdateZC() {
	enqueue(k("_updateZC")(x, pi[12], zi[12], hri[12], cl_N));
}

void clUpdateZB(int i, float bgr) {
	enqueue(k("_updateZB")(x, pi[i], zi[i], hri[i], bgr, cl_N));
}

#endif /* SRC_CLIMPL_HPP_ */
