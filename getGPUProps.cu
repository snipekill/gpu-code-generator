#include <iostream>
using namespace std;
int main(int argc, char *argv[])
{
  int nDevices;

  cudaGetDeviceCount(&nDevices);
  for (int i = 0; i < nDevices; i++) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, i);
    cout<< "SHMEM_PER_BLOCK = " << prop.sharedMemPerBlock<<"\n";
    cout<< "WARP_SIZE = " << prop.warpSize<<"\n";
    cout<< "MAX_THREADS_PER_BLOCK = " << prop.maxThreadsPerBlock<<"\n"; 

  }
  
  return 0;
}