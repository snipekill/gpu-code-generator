// ------------------------------- STATIC (except datatype) -------------------------------
#ifndef GPU_RUN
#define GPU_RUN
// ------------------------------- STATIC (except datatype) -------------------------------

// ------------------------------- PROGRAM -------------------------------
	// M, K and N are matrix sizes that should be loaded from the configuration file
	#define M 96
	#define K 64
	#define N 32
	// TILE_X*TILE_Y should not exceed Shared Memory Size. TILE_Y cannot exceed 32. TILE_X should be the largest possible value such that shared memory can hold 2*TILE_X*TILE_Y values
	#define TILE_X 64
	#define TILE_Y 16
	// THREADS_PER_BLOCK is same as TILE_Y
	#define THREADS_PER_BLOCK 16
// ------------------------------- PROGRAM -------------------------------

// ------------------------------- STATIC (except datatype) -------------------------------
	#define BLOCKS_X ((N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)
	#define BLOCKS_Y ((M + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)
	void gpu_run(float*, float*, float*);
	__global__ void kernel(float*, float*, float*);
#endif
// ------------------------------- STATIC (except datatype) -------------------------------
