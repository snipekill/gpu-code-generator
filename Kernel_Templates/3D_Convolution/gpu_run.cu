// ------------------------------- STATIC (except datatype) -------------------------------
#include "gpu_run.cuh"
#include <iostream>
#include <chrono>
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;
void gpu_run(float* input1, float* input2, float* output) {
	float* input1_d;
	size_t s_input1 = IS_X*IS_Y*IC*BS*sizeof(float);
	float* input1_gemm_d;
	size_t s_input1_gemm = K*N*sizeof(float);
	float* input2_d;
	size_t s_input2 = FS*FS*IC*OC*sizeof(float);
	float* input2_gemm_d;
	size_t s_input2_gemm = M*K*sizeof(float);
	float* output_d;
	size_t s_output = OS_X*OS_Y*OC*BS*sizeof(float);
	cudaMalloc(&input1_d, s_input1);
	cudaMalloc(&input1_gemm_d, s_input1_gemm);
	cudaMalloc(&input2_d, s_input2);
	cudaMalloc(&input2_gemm_d, s_input2_gemm);
	cudaMalloc(&output_d, s_output);
	cudaMemcpy(input1_d, input1, s_input1, cudaMemcpyHostToDevice);
	dim3 block_dim_ri(THREADS_PER_BLOCK_X_RI, THREADS_PER_BLOCK_Y_RI, THREADS_PER_BLOCK_Z_RI);
	dim3 grid_dim_ri(BLOCKS_X_RI, BLOCKS_Y_RI, BLOCKS_Z_RI);
	reformat_input<<<grid_dim_ri, block_dim_ri>>>(input1_d, input1_gemm_d);
	cudaMemcpy(input2_d, input2, s_input2, cudaMemcpyHostToDevice);
	dim3 block_dim_rf(THREADS_PER_BLOCK_X_RF, THREADS_PER_BLOCK_Y_RF, THREADS_PER_BLOCK_Z_RF);
	dim3 grid_dim_rf(BLOCKS_X_RF, BLOCKS_Y_RF, BLOCKS_Z_RF);
	reformat_filter<<<grid_dim_rf, block_dim_rf>>>(input2_d, input2_gemm_d);
	const auto begin = steady_clock::now();
	dim3 block_dim_gemm(THREADS_PER_BLOCK_GEMM, THREADS_PER_BLOCK_GEMM);
	dim3 grid_dim_gemm(BLOCKS_X_GEMM, BLOCKS_Y_GEMM);
	kernel<<<grid_dim_gemm, block_dim_gemm>>>(input2_gemm_d, input1_gemm_d, output_d);
	cudaDeviceSynchronize();
	const auto end = steady_clock::now();  uint64_t run_time_us = duration_cast<microseconds>(end - begin).count();
  	float gflops = float(IC) * OC * OS_X * OS_Y * FS * FS * BS* 2
    	/ (run_time_us * 1e3);
  	std::clog << "Time: " << run_time_us * 1e-6 << " s\n";
  	std::clog << "Perf: " << gflops << " GFlops\n";
	cudaMemcpy(output, output_d, s_output, cudaMemcpyDeviceToHost);
	cudaFree(input1_d);
	cudaFree(input1_gemm_d);
	cudaFree(input2_d);
	cudaFree(input2_gemm_d);
	cudaFree(output_d);
}
// ------------------------------- STATIC (except datatype) -------------------------------