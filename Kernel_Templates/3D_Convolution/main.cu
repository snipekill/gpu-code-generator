#include <iostream>
#include <cassert>
#include <cstdlib>
#include "gpu_run.cuh"

void print_matrix(float *matrix, int D1, int D2, int D3, int D4) {
	for (int j = 0; j < D3; j++) {
		for (int i = 0; i < D4; i++) {
			for (int k = 0; k < D2; k++) {
				for (int l = 0; l < D1; l++) {
					std::cout << matrix[i*D3*D2*D1 + j*D2*D1 + k*D1 + l] << " ";
				}
				std::cout << "\n";
			}
			std::cout << "\n";
		}
		std::cout << "\n";
	}
}

void initialize_matrix(float *matrix, int D1, int D2, int D3, int D4) {
	for (int j = 0; j < D3; j++) {
		for (int i = 0; i < D4; i++) {
			for (int k = 0; k < D2; k++) {
				for (int l = 0; l < D1; l++) {
					matrix[i*D3*D2*D1 + j*D2*D1 + k*D1 + l] =  static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
				}
			}
		}
	}
}

void cpu_verify(float* input1, float* input2, float* output) {
	int b_offset;
	int c_offset;
	float temp;
	for (int d = 0; d < OC; d++) {
		for (int a = 0; a < BS; a++) {
			for (int b = 0; b < OS_Y; b++) {
				for (int c = 0; c < OS_X; c++) {
					temp = 0;
					for(int e = 0; e < IC; e++) {
						for (int f = 0; f < FS; f++) {
							for (int g = 0; g < FS; g++) {
								b_offset = b + f;
								c_offset = c + g;
								temp += input1[a*IC*IS_Y*IS_X + e*IS_Y*IS_X + b_offset*IS_X + c_offset] * input2[d*IC*FS*FS + e*FS*FS + f*FS + g];
							}
						}
					}
					float diff = output[d*BS*OS_Y*OS_X + a*OS_Y*OS_X + b*OS_X + c] - temp;
      				if(diff>0.001f || diff <-0.001f){
						  assert(false);
					}
					// assert (output[d*BS*OS_Y*OS_X + a*OS_Y*OS_X + b*OS_X + c] == temp);
				}
			}
		}
	}
	std::cout << "\nConvolution Executed Correctly!\n\n";
}

int main() {
	
	float *input1 = new float[IS_X*IS_Y*IC*BS];
	float *input2 = new float[FS*FS*IC*OC];
	float *output = new float[OS_X*OS_Y*OC*BS];
	
	initialize_matrix(input1, IS_X, IS_Y, IC, BS);
//	print_matrix(input1, IS_X, IS_Y, IC, BS);
	
	initialize_matrix(input2, FS, FS, IC, OC);
//	print_matrix(input2, FS, FS, IC, OC);
	
	gpu_run(input1, input2, output);
//	print_matrix(output, OS_X, OS_Y, OC, BS);
	
	cpu_verify(input1, input2, output);

	delete[] input1;
	delete[] input2;
	delete[] output;

	return 0;
}