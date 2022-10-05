#include <iostream>
#include <cassert>
#include <cstdlib>
#include "gpu_run.cuh"

void print_matrix(float *matrix, int R, int C) {
	for (int i = 0; i < R; i++) {
		for (int j = 0; j < C; j++) {
			std::cout << matrix[C*i + j] << " ";
		}
		std::cout << "\n";		
	}
	std::cout << "\n";
}

void initialize_matrix(float *matrix, int R, int C) {
	for (int i = 0; i < R; i++) {
		for (int j = 0; j < C; j++) {
			matrix[C*i + j] = rand() % 10;
		}
	}
}

void cpu_verify(float* input1, float* input2, float* output, int R1, int C1R2, int C2) {
	for (int i = 0; i < R1; i++) {
		for (int j = 0; j < C2; j++) {
			int temp = 0;
			for (int k = 0; k < C1R2; k++) {
				temp += input1[C1R2*i + k] * input2[C2*k + j];
			}
			assert(temp == output[C2*i + j]);
		}
	}
	std::cout << "GEMM executed correctly!\n\n";
}

int main() {
	
	float *input1 = new float[M*K];
	float *input2 = new float[K*N];
	float *output = new float[M*N];

	initialize_matrix(input1, M, K);
	initialize_matrix(input2, K, N);
	
	gpu_run(input1, input2, output);
	
	cpu_verify(input1, input2, output, M, K, N);

	delete[] input1;
	delete[] input2;
	delete[] output;

	return 0;
}