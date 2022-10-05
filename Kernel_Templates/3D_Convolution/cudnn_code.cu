#include <iomanip>
#include <iostream>
#include <cstdlib>
#include <vector>

#include <cuda.h>
#include <cudnn.h>
#include <iostream>
#include <chrono>
using std::chrono::duration_cast;
using std::chrono::microseconds;
using std::chrono::steady_clock;

const int in_n = 3;
const int in_c = 64;
const int in_h = 224;
const int in_w = 224;
const int filt_k = 64;
const int filt_c = 64;
const int filt_h = 3;
const int filt_w = 3;
const int pad_h = 1;
const int pad_w = 1;
const int str_h = 1;
const int str_w = 1;
const int dil_h = 0;
const int dil_w = 0;
#define CUDA_CALL(f) { \
  cudaError_t err = (f); \
  if (err != cudaSuccess) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

#define CUDNN_CALL(f) { \
  cudnnStatus_t err = (f); \
  if (err != CUDNN_STATUS_SUCCESS) { \
    std::cout \
        << "    Error occurred: " << err << std::endl; \
    std::exit(1); \
  } \
}

__global__ void dev_const(float *px, float k) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = k;
}

__global__ void dev_iota(float *px) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  px[tid] = tid;
}

void fill_convolution_shared_simple(float *synapse, float *neuron_i) {
  for(int yy = 0; yy < filt_k; ++yy) {
    for(int xx = 0; xx < filt_c; ++xx) {
      for(int nn = 0; nn < filt_h; ++nn) {
        for(int ni = 0; ni < filt_w; ++ni) {
          synapse[yy*filt_c*filt_h*filt_w + xx*filt_h*filt_w + nn*filt_w + ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
        } } } }
  for(int bb = 0; bb < in_n; bb++){
  for(int yy = 0; yy < in_c; ++yy) {
    for(int xx = 0; xx < in_h; ++xx) {      
      for(int ni = 0; ni < in_w; ++ni) {
        neuron_i[bb*in_c*in_h*in_w + yy*in_h*in_w + xx*in_w + ni] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX) - 0.5f;
  }  }  } }
}

void print(const float *data, int n, int c, int h, int w) {
  std::vector<float> buffer(1 << 20);
  CUDA_CALL(cudaMemcpy(
        buffer.data(), data,
        n * c * h * w * sizeof(float),
        cudaMemcpyDeviceToHost));
  int a = 0;
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < c; ++j) {
      std::cout << "n=" << i << ", c=" << j << ":" << std::endl;
      for (int k = 0; k < h; ++k) {
        for (int l = 0; l < w; ++l) {
          std::cout << std::setw(4) << std::right << buffer[a];
          ++a;
        }
        std::cout << std::endl;
      }
    }
  }
  std::cout << std::endl;
}

int main() {
  cudnnHandle_t cudnn;
  CUDNN_CALL(cudnnCreate(&cudnn));

  // input
  std::cout << "in_n: " << in_n << std::endl;
  std::cout << "in_c: " << in_c << std::endl;
  std::cout << "in_h: " << in_h << std::endl;
  std::cout << "in_w: " << in_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t in_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&in_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        in_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
        in_n, in_c, in_h, in_w));

  float *in_data;
  float *cpu_in_data;
  CUDA_CALL(cudaMalloc(
        &in_data, in_n * in_c * in_h * in_w * sizeof(float)));
  cpu_in_data = (float *)malloc(sizeof(float) * in_n * in_c * in_h * in_w);

  // filter
  std::cout << "filt_k: " << filt_k << std::endl;
  std::cout << "filt_c: " << filt_c << std::endl;
  std::cout << "filt_h: " << filt_h << std::endl;
  std::cout << "filt_w: " << filt_w << std::endl;
  std::cout << std::endl;

  cudnnFilterDescriptor_t filt_desc;
  CUDNN_CALL(cudnnCreateFilterDescriptor(&filt_desc));
  CUDNN_CALL(cudnnSetFilter4dDescriptor(
        filt_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NHWC,
        filt_k, filt_c, filt_h, filt_w));

  float *filt_data;
  float *cpu_filt_data;
  cpu_filt_data = (float *)malloc(sizeof(float) * filt_k * filt_c * filt_h * filt_w );
  CUDA_CALL(cudaMalloc(
      &filt_data, filt_k * filt_c * filt_h * filt_w * sizeof(float)));

  // convolution
  std::cout << "pad_h: " << pad_h << std::endl;
  std::cout << "pad_w: " << pad_w << std::endl;
  std::cout << "str_h: " << str_h << std::endl;
  std::cout << "str_w: " << str_w << std::endl;
  std::cout << "dil_h: " << dil_h << std::endl;
  std::cout << "dil_w: " << dil_w << std::endl;
  std::cout << std::endl;

  cudnnConvolutionDescriptor_t conv_desc;
  CUDNN_CALL(cudnnCreateConvolutionDescriptor(&conv_desc));
  CUDNN_CALL(cudnnSetConvolution2dDescriptor(
        conv_desc,
        pad_h, pad_w, str_h, str_w, dil_h, dil_w,
        CUDNN_CONVOLUTION, CUDNN_DATA_FLOAT));

  // output
  int out_n;
  int out_c;
  int out_h;
  int out_w;
  
  CUDNN_CALL(cudnnGetConvolution2dForwardOutputDim(
        conv_desc, in_desc, filt_desc,
        &out_n, &out_c, &out_h, &out_w));

  std::cout << "out_n: " << out_n << std::endl;
  std::cout << "out_c: " << out_c << std::endl;
  std::cout << "out_h: " << out_h << std::endl;
  std::cout << "out_w: " << out_w << std::endl;
  std::cout << std::endl;

  cudnnTensorDescriptor_t out_desc;
  CUDNN_CALL(cudnnCreateTensorDescriptor(&out_desc));
  CUDNN_CALL(cudnnSetTensor4dDescriptor(
        out_desc, CUDNN_TENSOR_NHWC, CUDNN_DATA_FLOAT,
        out_n, out_c, out_h, out_w));

  float *out_data;
  CUDA_CALL(cudaMalloc(
        &out_data, out_n * out_c * out_h * out_w * sizeof(float)));
  // algorithm
  cudnnConvolutionFwdAlgo_t algo;
  CUDNN_CALL(cudnnGetConvolutionForwardAlgorithm(
        cudnn,
        in_desc, filt_desc, conv_desc, out_desc,
        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &algo));


  std::cout << "Convolution algorithm: " << CUDNN_CONVOLUTION_FWD_ALGO_GEMM << std::endl;
  std::cout<< "hey dj "<<CUDNN_STATUS_BAD_PARAM<<std::endl;
  std::cout << std::endl;

  // workspace
  size_t ws_size;
  CUDNN_CALL(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn, in_desc, filt_desc, conv_desc, out_desc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, &ws_size));

  float *ws_data;
  CUDA_CALL(cudaMalloc(&ws_data, ws_size));

  std::cout << "Workspace size: " << ws_size << std::endl;
  std::cout << std::endl;

  // perform
  float alpha = 1.f;
  float beta = 0.f;
  // dev_iota<<<in_w * in_h, in_n * in_c>>>(in_data);
  // dev_const<<<filt_w * filt_h, filt_k * filt_c>>>(filt_data, 1.f);
  fill_convolution_shared_simple(cpu_filt_data, cpu_in_data);

  cudaMemcpy(filt_data, cpu_filt_data, sizeof(float) * filt_k * filt_c * filt_h * filt_w, cudaMemcpyHostToDevice);
  cudaMemcpy(in_data, cpu_in_data, sizeof(float) * in_n * in_c * in_h * in_w, cudaMemcpyHostToDevice);
  const auto begin = steady_clock::now();
  CUDNN_CALL(cudnnConvolutionForward(
      cudnn,
      &alpha, in_desc, in_data, filt_desc, filt_data,
      conv_desc, CUDNN_CONVOLUTION_FWD_ALGO_GEMM, ws_data, ws_size,
      &beta, out_desc, out_data));
  cudaDeviceSynchronize();
  const auto end = steady_clock::now();  
	uint64_t run_time_us = duration_cast<microseconds>(end - begin).count();
  float gflops = float(in_n) * in_c * out_c * out_h * out_w * filt_h * filt_w * 2
    / (run_time_us * 1e3);
  std::clog << "Time: " << run_time_us * 1e-6 << " s\n";
  std::clog << "Perf: " << gflops << " GFlops\n";
  // results
  std::cout << "in_data:" << std::endl;
  // print(in_data, in_n, in_c, in_h, in_w);
  
  std::cout << "filt_data:" << std::endl;
  // print(filt_data, filt_k, filt_c, filt_h, filt_w);
  
  std::cout << "out_data:" << std::endl;
  // print(out_data, out_n, out_c, out_h, out_w);

  // finalizing
  CUDA_CALL(cudaFree(ws_data));
  CUDA_CALL(cudaFree(out_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(out_desc));
  CUDNN_CALL(cudnnDestroyConvolutionDescriptor(conv_desc));
  CUDA_CALL(cudaFree(filt_data));
  CUDNN_CALL(cudnnDestroyFilterDescriptor(filt_desc));
  CUDA_CALL(cudaFree(in_data));
  CUDNN_CALL(cudnnDestroyTensorDescriptor(in_desc));
  CUDNN_CALL(cudnnDestroy(cudnn));
  return 0;
}
