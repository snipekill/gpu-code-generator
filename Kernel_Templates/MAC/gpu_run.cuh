// ------------------------------- STATIC (except datatype) -------------------------------
#ifndef GPU_RUN
#define GPU_RUN
// ------------------------------- STATIC (except datatype) -------------------------------

	// ------------------------------- PROGRAM -------------------------------
	// Get INPUT_SIZE from the configuration file
	#define INPUT_SIZE 4096
	// If INPUT_SIZE < MAX_THREADS_PER_BLOCK, THREADS_PER_BLOCK = INPUT_SIZE. Else, THREADS_PER_BLOCK = MAX_THREADS_PER_BLOCK
	#define THREADS_PER_BLOCK 1024
	// ------------------------------- PROGRAM -------------------------------
	
// ------------------------------- STATIC (except datatype) -------------------------------
	#define BLOCKS ((INPUT_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)
	void gpu_run(float*, float*, float*);
	__global__ void kernel(float*, float*, float*);
	__global__ void kernel(float*, float*);
#endif
// ------------------------------- STATIC (except datatype) -------------------------------