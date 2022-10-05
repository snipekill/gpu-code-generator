import math as m

# GPU parameters (Should get this based on gpu being used)
SHMEM_PER_BLOCK = 49152
MAX_THREADS_PER_BLOCK = 1024
WARP_SIZE = 32


#####################################################################################################################################
# MAC Algorithm Parameters (From configuration file)
INPUT_SIZE = 4096
DATATYPE_SIZE = 4

# MAC CUDA Parameters (The final values being printed are what would be used in gpu_run.cuh part of the generated code)
tpb_a = MAX_THREADS_PER_BLOCK
tpb_b = INPUT_SIZE
tpb_c = int(SHMEM_PER_BLOCK/DATATYPE_SIZE)
THREADS_PER_BLOCK = min(tpb_a, tpb_b, tpb_c)
print('THREADS_PER_BLOCK:', THREADS_PER_BLOCK)


#####################################################################################################################################
# 2D_GEMM Algorithm Parameters (From configuration file)
M = 16
K = 96
N = 32
DATATYPE_SIZE = 4

# 2D_GEMM CUDA Parameters (The final values being printed are what would be used in gpu_run.cuh part of the generated code)
tpb_a = int(m.sqrt(MAX_THREADS_PER_BLOCK))
tpb_b = M
THREADS_PER_BLOCK = min(tpb_a, tpb_b)
TILE_Y = THREADS_PER_BLOCK
max_tile_x = int(SHMEM_PER_BLOCK/(2*TILE_Y*DATATYPE_SIZE))
if (max_tile_x > K):
    TILE_X = K
else:
    TILE_X = int(max_tile_x/THREADS_PER_BLOCK)*THREADS_PER_BLOCK
print('THREADS_PER_BLOCK:', THREADS_PER_BLOCK)
print('TILE_X:', TILE_X)
print('TILE_Y:', TILE_Y)


#####################################################################################################################################
# 3D_Convolution Algorithm Parameters (From configuration file)
IS_X = 224
IS_Y = 224
FS = 3
IC = 64
OC = 64
BS = 3
DATATYPE_SIZE = 4

# 3D_Convolution CUDA Parameters (The final values being printed are what would be used in gpu_run.cuh part of the generated code)
M = OC
K = FS*FS*IC
tpb_a = int(m.sqrt(MAX_THREADS_PER_BLOCK))
tpb_b = M
THREADS_PER_BLOCK = min(tpb_a, tpb_b)
TILE_Y = THREADS_PER_BLOCK
max_tile_x = int(SHMEM_PER_BLOCK/(2*TILE_Y*DATATYPE_SIZE))
if (max_tile_x > K):
    TILE_X = K
else:
    TILE_X = int(max_tile_x/THREADS_PER_BLOCK)*THREADS_PER_BLOCK
print('THREADS_PER_BLOCK:', THREADS_PER_BLOCK)
print('TILE_X:', TILE_X)
print('TILE_Y:', TILE_Y)