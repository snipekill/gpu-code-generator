#include <iostream>
#include <cassert>
#include <cstdlib>
#include "gpu_run.cuh"

void print_matrix(float *matrix, int D1) {
	for (int i = 0; i < D1; i++) {
		std::cout << matrix[i] << " ";
	}
}

void initialize_matrix(float *matrix, int D1) {
	for (int i = 0; i < D1; i++) {
		matrix[i] = rand() % 10;
	}
}

void cpu_verify(float* input1, float* input2, float* output, int D1) {
	float temp;
	for (int i = 0; i < D1; i++) {
		temp += input1[i]*input2[i];
	}
	assert (output[0] == temp);
	std::cout << "MAC executed correctly!\n\n";
}

int main() {
	
	float *input1 = new float[INPUT_SIZE];
	float *input2 = new float[INPUT_SIZE];
	float *output = new float[1];

	initialize_matrix(input1, INPUT_SIZE);
	initialize_matrix(input2, INPUT_SIZE);
	
	gpu_run(input1, input2, output);

	cpu_verify(input1, input2, output, INPUT_SIZE);

	delete[] input1;
	delete[] input2;
	delete[] output;

	return 0;
}