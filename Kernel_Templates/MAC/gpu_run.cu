// ------------------------------- STATIC (except datatype) -------------------------------
#include "gpu_run.cuh"
void gpu_run(float* input1, float* input2, float* output) {
	float* input1_d;
	size_t s_input1 = INPUT_SIZE*sizeof(float);
	float* input2_d;
	size_t s_input2 = INPUT_SIZE*sizeof(float);
	float* output_d;
	size_t s_output = INPUT_SIZE*sizeof(float);
	cudaMalloc(&input1_d, s_input1);
	cudaMalloc(&input2_d, s_input2);
	cudaMalloc(&output_d, s_output);
	cudaMemcpy(input1_d, input1, s_input1, cudaMemcpyHostToDevice);
	cudaMemcpy(input2_d, input2, s_input2, cudaMemcpyHostToDevice);
	dim3 block_dim(THREADS_PER_BLOCK);
	dim3 grid_dim(BLOCKS);
	kernel<<<grid_dim, block_dim>>>(input1_d, input2_d, output_d);
// ------------------------------- STATIC (except datatype) -------------------------------

// ------------------------------- PROGRAM -------------------------------
// This step is not required when INPUT_SIZE <= THREADS_PER_BLOCK (i.e. INPUT_SIZE <= MAX_THREADS_PER_BLOCK)
	kernel<<<1, block_dim>>>(output_d, output_d);
// ------------------------------- PROGRAM -------------------------------

// ------------------------------- STATIC (except datatype) -------------------------------	
	float *output_all = new float[INPUT_SIZE];
	cudaMemcpy(output_all, output_d, s_output, cudaMemcpyDeviceToHost);	
	cudaFree(input1_d);
	cudaFree(input2_d);
	cudaFree(output_d);
	output[0] = output_all[0];
}
// ------------------------------- STATIC (except datatype) -------------------------------