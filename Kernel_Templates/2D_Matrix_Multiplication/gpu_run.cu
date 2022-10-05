// ------------------------------- STATIC (except datatype) -------------------------------
#include "gpu_run.cuh"
void gpu_run(float* input1, float* input2, float* output) {
	float* input1_d;
	size_t s_input1 = M*K*sizeof(float);
	float* input2_d;
	size_t s_input2 = K*N*sizeof(float);
	float* output_d;
	size_t s_output = M*N*sizeof(float);
	cudaMalloc(&input1_d, s_input1);
	cudaMalloc(&input2_d, s_input2);
	cudaMalloc(&output_d, s_output);
	cudaMemcpy(input1_d, input1, s_input1, cudaMemcpyHostToDevice);
	cudaMemcpy(input2_d, input2, s_input2, cudaMemcpyHostToDevice);
	dim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);
	dim3 grid_dim(BLOCKS_X, BLOCKS_Y);
	kernel<<<grid_dim, block_dim>>>(input1_d, input2_d, output_d);
	cudaMemcpy(output, output_d, s_output, cudaMemcpyDeviceToHost);	
	cudaFree(input1_d);
	cudaFree(input2_d);
	cudaFree(output_d);
}
// ------------------------------- STATIC (except datatype) -------------------------------