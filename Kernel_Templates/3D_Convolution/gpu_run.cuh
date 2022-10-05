// ------------------------------- STATIC (except datatype) -------------------------------
#ifndef GPU_RUN
#define GPU_RUN
// ------------------------------- STATIC (except datatype) -------------------------------

// ------------------------------- PROGRAM -------------------------------
	// IS_X, IS_Y, FS, IC, OC and BS are convolution sizes that should be loaded from the configuration file
	#define IS_X 224
	#define IS_Y 224
	#define FS 3
	#define IC 64
	#define OC 64
	#define BS 3
	// THREADS_PER_BLOCK is same as TILE_Y
	#define THREADS_PER_BLOCK_GEMM 32
	// TILE_X*TILE_Y should not exceed Shared Memory Size. TILE_Y cannot exceed 32. TILE_X should be the largest possible value such that shared memory can hold 2*TILE_X*TILE_Y values
	#define TILE_X 32
	#define TILE_Y 32	
// ------------------------------- PROGRAM -------------------------------

// ------------------------------- STATIC (except datatype) -------------------------------
	#define OS_X (IS_X-FS+1)
	#define OS_Y (IS_Y-FS+1)
	#define M OC
	#define K (FS*FS*IC)
	#define N (OS_X*OS_Y*BS)
	#define BLOCKS_X_GEMM ((N + THREADS_PER_BLOCK_GEMM - 1)/THREADS_PER_BLOCK_GEMM)
	#define BLOCKS_Y_GEMM ((M + THREADS_PER_BLOCK_GEMM - 1)/THREADS_PER_BLOCK_GEMM)
	#define THREADS_PER_BLOCK_X_RI FS
	#define THREADS_PER_BLOCK_Y_RI FS
	#define THREADS_PER_BLOCK_Z_RI IC
	#define BLOCKS_X_RI OS_X
	#define BLOCKS_Y_RI OS_Y
	#define BLOCKS_Z_RI BS
	#define THREADS_PER_BLOCK_X_RF FS
	#define THREADS_PER_BLOCK_Y_RF FS
	#define THREADS_PER_BLOCK_Z_RF 8
	#define BLOCKS_X_RF 1
	#define BLOCKS_Y_RF 1
	#define BLOCKS_Z_RF ((OC*IC + THREADS_PER_BLOCK_Z_RF - 1)/THREADS_PER_BLOCK_Z_RF)
	void gpu_run(float*, float*, float*);
	__global__ void kernel(float*, float*, float*);
	__global__ void reformat_input(float*, float*);
	__global__ void reformat_filter(float*, float*);	
#endif
// ------------------------------- STATIC (except datatype) -------------------------------