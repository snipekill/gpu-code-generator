import os
import re
import math as m

class extracting_parameters:
    def __init__(self, file_name):
        print('inside extracting results')
        with open(file_name) as file:
            self.operation = "2D_GEMM"
            self.M = 16
            self.K = 96
            self.data_type = "float"
            self.N = 32
            self.input_size = 4096
            self.IS_X = 224
            self.IS_Y = 224
            self.FS = 3
            self.IC = 64
            self.OC = 64
            self.BS = 3
            for line in file:
                if re.search("Type_operation", line):
                    #print('yes')
                    value = line.split('=')
                    #print(value[0])
                    #print(value[1].split(' '))
                    #print(line)
                    #print(type(line))
                    self.operation = (value[1].split(' ')[1])
                    continue

                if re.search("M", line):
                    value = line.split('=')
                    print('type of line is: ', type(line))
                    print("value is:", value)
                    print("type of value is: ", type(value))
                    self.M = int(value[1].split(' ')[1])
                    continue

                if re.search("K", line):
                    value = line.split('=')
                    self.K = int(value[1].split(' ')[1])
                    continue

                if re.search("data_type", line):
                    value = line.split('=')
                    self.data_type = (value[1].split(' ')[1].split("\n")[0])
                    continue

                if re.search("N", line):
                    value = line.split('=')
                    self.N = int(value[1].split(' ')[1])
                    continue

                if re.search("Input_size", line):
                    value = line.split('=')
                    self.input_size = int(value[1].split(' ')[1])
                    continue

                if re.search("IS_X", line):
                    value = line.split('=')
                    self.IS_X = int(value[1].split(' ')[1])
                    continue

                if re.search("IS_Y", line):
                    value = line.split('=')
                    self.IS_Y = int(value[1].split(' ')[1])
                    continue

                if re.search("FS", line):
                    value = line.split('=')
                    self.FS = int(value[1].split(' ')[1])
                    continue

                if re.search("IC", line):
                    value = line.split('=')
                    self.IC = int(value[1].split(' ')[1])
                    continue

                if re.search("OC", line):
                    value = line.split('=')
                    self.OC = int(value[1].split(' ')[1])
                    continue

                if re.search("BS", line):
                    value = line.split('=')
                    self.BS = int(value[1].split(' ')[1])
                    continue

class extracting_gpu_parameters:
    def __init__(self, file_name):
        print('inside GPU extracting results')
        with open(file_name) as file:
            for line in file:
                if re.search("SHMEM_PER_BLOCK", line):
                    value = line.split('=')
                    self.SHMEM_PER_BLOCK = int(value[1].split(' ')[1])
                    continue

                if re.search("WARP_SIZE", line):
                    value = line.split('=')
                    self.WARP_SIZE = int(value[1].split(' ')[1])
                    continue

                if re.search("MAX_THREADS_PER_BLOCK", line):
                    value = line.split('=')
                    self.MAX_THREADS_PER_BLOCK = int(value[1].split(' ')[1])
                    continue

#main
extracted_parameters = extracting_parameters('parameter_file.txt')
type_operation = (extracted_parameters.operation).split('\n')[0] 
M = extracted_parameters.M
N = extracted_parameters.N
K = extracted_parameters.K
data_type = extracted_parameters.data_type
#THREADS_PER_BLOCK = 16 #needs to be changed
#TILE_X = 256 #needs to be changed
#TILE_Y = 32 #needs to be changed later
input_size = extracted_parameters.input_size
IS_X = extracted_parameters.IS_X
IS_Y = extracted_parameters.IS_Y
FS = extracted_parameters.FS
IC = extracted_parameters.IC
OC = extracted_parameters.OC
BS = extracted_parameters.BS

os.system(f"nvcc getGPUProps.cu -o gpu_parameters")
os.system(f"./gpu_parameters > gp")
extracted_gpu_parameters = extracting_gpu_parameters('gp')
#SHMEM_PER_BLOCK = 49152
#MAX_THREADS_PER_BLOCK = 1024
#WARP_SIZE = 32

SHMEM_PER_BLOCK = extracted_gpu_parameters.SHMEM_PER_BLOCK
MAX_THREADS_PER_BLOCK = extracted_gpu_parameters.MAX_THREADS_PER_BLOCK
WARP_SIZE = extracted_gpu_parameters.WARP_SIZE
print(f"SHMEM is: {SHMEM_PER_BLOCK}")
print(f"MAX Threads per block: {MAX_THREADS_PER_BLOCK}")
print(f"Warp size is: {WARP_SIZE}")

#print(f"M is {M}")
#print(f"Data type is {data_type}")

#generating gpu_run.cuh
if (type_operation == '2D_GEMM'):
    DATATYPE_SIZE = 4
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
    folder_name = f"{type_operation}_{M}_{N}_{K}" #creates a unique folder name
    os.system(f"mkdir {folder_name}")
    #os.system(f"cd ./{folder_name}") 
    with open (f'./{folder_name}/gpu_run.cuh', 'w') as gpu_header:
        gpu_header.write(f"#ifndef GPU_RUN\n")
        gpu_header.write(f"#define GPU_RUN\n")
        gpu_header.write(f"\t#define M {M}\n")
        gpu_header.write(f"\t#define K {K}\n")
        gpu_header.write(f"\t#define N {N}\n")
        gpu_header.write(f"\t#define TILE_X {TILE_X}\n") #need to make it modular
        gpu_header.write(f"\t#define TILE_Y {TILE_Y}\n") #need to make it modular
        gpu_header.write(f"\t#define THREADS_PER_BLOCK {TILE_Y}\n") #it is the same as TILE_Y ()
        gpu_header.write(f"\t#define BLOCKS_X ((N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)\n")
        gpu_header.write(f"\t#define BLOCKS_Y ((M + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)\n")
        gpu_header.write(f"\tvoid gpu_run({data_type}*, {data_type}*, {data_type}*);\n")
        gpu_header.write(f"\t__global__ void kernel({data_type}*, {data_type}*, {data_type}*);\n")
        gpu_header.write('#endif\n')

    #generating gpu_run.cu
    with open (f'./{folder_name}/gpu_run.cu', 'w') as gpu_run:
        #gpu_run.write(f"#define M {M}\n")
        #gpu_run.write(f"#define K {K}\n")
        #gpu_run.write(f"#define N {N}\n")
        #gpu_run.write(f"#define THREADS_PER_BLOCK {THREADS_PER_BLOCK}\n")
        gpu_run.write(f"//---------main body---------\n")
        #gpu_run.write(f"#define BLOCKS_X ((N + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK)\n")
        #gpu_run.write(f"#define BLOCKS_Y ((M + THREADS_PER_BLOCK -1)/THREADS_PER_BLOCK)\n")
        gpu_run.write(f'#include "gpu_run.cuh"\n')
        gpu_run.write(f"void gpu_run({data_type}* input1, {data_type}* input2, {data_type}* output)"+" {\n")
        gpu_run.write(f"\t{data_type}* input1_d;\n")
        gpu_run.write(f"\tsize_t s_input1 = M*K*sizeof({data_type});\n")
        gpu_run.write(f"\t{data_type}* input2_d;\n")
        gpu_run.write(f"\tsize_t s_input2 = K*N*sizeof({data_type});\n")
        gpu_run.write(f"\t{data_type}* output_d;\n")
        gpu_run.write(f"\tsize_t s_output = M*N*sizeof({data_type});\n")
        gpu_run.write(f"\tcudaMalloc(&input1_d, s_input1);\n")
        gpu_run.write(f"\tcudaMalloc(&input2_d, s_input2);\n")
        gpu_run.write(f"\tcudaMalloc(&output_d, s_output);\n")
        gpu_run.write(f"\tcudaMemcpy(input1_d, input1, s_input1, cudaMemcpyHostToDevice);\n")
        gpu_run.write(f"\tcudaMemcpy(input2_d, input2, s_input2, cudaMemcpyHostToDevice);\n")
        gpu_run.write(f"\tdim3 block_dim(THREADS_PER_BLOCK, THREADS_PER_BLOCK);\n")
        gpu_run.write(f"\tdim3 grid_dim(BLOCKS_X, BLOCKS_Y);\n")
        gpu_run.write(f"\tkernel<<<grid_dim, block_dim>>>(input1_d, input2_d, output_d);\n")
        gpu_run.write(f"\tcudaMemcpy(output, output_d, s_output, cudaMemcpyDeviceToHost);\n")
        gpu_run.write(f"\tcudaFree(input1_d);\n")
        gpu_run.write(f"\tcudaFree(input2_d);\n")
        gpu_run.writelines(f"\tcudaFree(output_d);\n")
        gpu_run.write("}\n")

    #generating kernel.cu
    with open (f'./{folder_name}/kernel.cu', 'w') as kernel:
        #kernel.write(f"#define M {M}\n")
        #kernel.write(f"#define K {K}\n")
        #kernel.write(f"#define N {N}\n")
        #kernel.write(f"#define TILE_X {TILE_X}\n")
        #kernel.write(f"#define TILE_Y {TILE_Y}\n")
        #kernel.write(f"#define THREADS_PER_BLOCK {THREADS_PER_BLOCK}\n")
        kernel.write("//---------main body---------\n")
        kernel.write(f'#include "gpu_run.cuh"\n')
        kernel.write(f"__global__ void kernel({data_type}* input1, {data_type}* input2, {data_type}* output)" + " {\n")
        kernel.write(f"\t__shared__ {data_type} input1_s[TILE_X*TILE_Y];\n")
        kernel.write(f"\t__shared__ {data_type} input2_s[TILE_X*TILE_Y];\n")
        kernel.write(f"\t{data_type} temp = 0;\n")
        kernel.write(f"\tint col_idx = blockIdx.x*blockDim.x + threadIdx.x;\n")
        kernel.write(f"\tint row_idx = blockIdx.y*blockDim.y + threadIdx.y;\n")
        kernel.write(f"\tfor (int i=0; i<K; i+=TILE_X)"+" {\n")
        kernel.write(f"\t\tfor (int t=0; t<TILE_X; t+=THREADS_PER_BLOCK)" + " {\n")
        kernel.write(f"\t\t\tinput1_s[threadIdx.y*TILE_X + threadIdx.x + t] = input1[row_idx*K + i + threadIdx.x + t];\n")
        kernel.write(f"\t\t\tinput2_s[(threadIdx.y+t)*TILE_Y + threadIdx.x] = input2[(i+threadIdx.y+t)*N + col_idx];\n")
        kernel.write("\t\t}\n")
        kernel.write("\t\t__syncthreads();\n")
        kernel.write(f"\t\tfor (int j=0; j<TILE_X; j++)" + " {\n")
        kernel.write(f"\t\t\ttemp += input1_s[threadIdx.y*TILE_X + j] * input2_s[j*TILE_Y + threadIdx.x];" + "\n\t\t}\n")
        kernel.write(f"\t\t__syncthreads();\n" + "\t}\n")
        kernel.write(f"\tif (row_idx < M && col_idx < N)" +" {\n")
        kernel.write(f"\t\toutput[row_idx*N + col_idx] = temp;\n" + "\t}\n}")

if (type_operation == 'MAC'):
    DATATYPE_SIZE = 4
    tpb_a = MAX_THREADS_PER_BLOCK
    tpb_b = input_size
    tpb_c = int(SHMEM_PER_BLOCK/DATATYPE_SIZE)
    THREADS_PER_BLOCK = min(tpb_a, tpb_b, tpb_c)
    print('THREADS_PER_BLOCK:', THREADS_PER_BLOCK)
    folder_name = f"{type_operation}_{input_size}" #creates a unique folder name
    os.system(f"mkdir {folder_name}")
    print('hi')
    #MAX_THREADS_PER_BLOCK = 1024
    '''if (input_size < MAX_THREADS_PER_BLOCK):
        THREADS_PER_BLOCK = input_size #where is max_threads_per_block
    else:
        THREADS_PER_BLOCK = MAX_THREADS_PER_BLOCK'''
    #THREADS_PER_BLOCK = 1024 
    #generating gpu_run.cuh for MAC operation
    with open (f'./{folder_name}/gpu_run.cuh', 'w') as gpu_header2:
        gpu_header2.write(f"#ifndef GPU_RUN\n")
        gpu_header2.write(f"#define GPU_RUN\n")
        gpu_header2.write(f'\t#define INPUT_SIZE {input_size}\n')
        gpu_header2.write(f'\t#define THREADS_PER_BLOCK {THREADS_PER_BLOCK}\n')
        gpu_header2.write(f'#define BLOCKS ((INPUT_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)\n')
        gpu_header2.write(f"\tvoid gpu_run({data_type}*, {data_type}*, {data_type}*);\n")
        gpu_header2.write(f"\t__global__ void kernel({data_type}*, {data_type}*, {data_type}*);\n")
        gpu_header2.write(f"\t__global__ void kernel({data_type}*, {data_type}*);\n")
        gpu_header2.write('#endif\n')
    
    #generating gpu_run.cu for MAC operation
    with open (f'./{folder_name}/gpu_run.cu', 'w') as gpu_run2:
        #gpu_run2.write(f'#define INPUT_SIZE {input_size}\n')
        #gpu_run2.write(f'#define THREADS_PER_BLOCK {THREADS_PER_BLOCK}\n')
        #gpu_run2.write(f'#define BLOCKS ((INPUT_SIZE + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)\n')
        gpu_run2.write(f'#include "gpu_run.cuh"\n')
        gpu_run2.write(f'void gpu_run({data_type}* input1, {data_type}* input2, {data_type}* output) ' + "{\n")
        gpu_run2.write(f'\t{data_type}* input1_d;\n')
        gpu_run2.write(f'\tsize_t s_input1 = INPUT_SIZE*sizeof({data_type});\n')
        gpu_run2.write(f'\t{data_type}* input2_d;\n')
        gpu_run2.write(f'\tsize_t s_input2 = INPUT_SIZE*sizeof({data_type});\n')
        gpu_run2.write(f'\t{data_type}* output_d;\n')
        gpu_run2.write(f'\tsize_t s_output = INPUT_SIZE*sizeof({data_type});\n')
        gpu_run2.write(f'\tcudaMalloc(&input1_d, s_input1);\n')
        gpu_run2.write(f'\tcudaMalloc(&input2_d, s_input2);\n')
        gpu_run2.write(f'\tcudaMalloc(&output_d, s_output);\n')
        gpu_run2.write(f'\tcudaMemcpy(input1_d, input1, s_input1, cudaMemcpyHostToDevice);\n')
        gpu_run2.write(f'\tcudaMemcpy(input2_d, input2, s_input2, cudaMemcpyHostToDevice);\n')
        gpu_run2.write(f'\tdim3 block_dim(THREADS_PER_BLOCK);\n')
        gpu_run2.write(f'\tdim3 grid_dim(BLOCKS);\n')
        gpu_run2.write(f'\tkernel<<<grid_dim, block_dim>>>(input1_d, input2_d, output_d);\n')
        if (input_size > THREADS_PER_BLOCK):
            gpu_run2.write(f'\tkernel<<<1, block_dim>>>(output_d, output_d);\n')
        gpu_run2.write(f'\t{data_type} *output_all = new {data_type}[INPUT_SIZE];\n')
        gpu_run2.write(f'\tcudaMemcpy(output_all, output_d, s_output, cudaMemcpyDeviceToHost);\n')
        gpu_run2.write(f'\tcudaFree(input1_d);\n')
        gpu_run2.write(f'\tcudaFree(input2_d);\n')
        gpu_run2.write(f'\tcudaFree(output_d);\n')
        gpu_run2.write(f'\toutput[0] = output_all[0];\n' + '}')

    #generating kernel.cu for MAC operation
    #warp_size = 32
    loop_i = WARP_SIZE
    with open (f'./{folder_name}/kernel.cu', 'w') as kernel2:
        #kernel2.write(f'#define THREADS_PER_BLOCK {THREADS_PER_BLOCK}\n')
        kernel2.write(f'#include "gpu_run.cuh"\n')
        kernel2.write(f'__device__ void singleWarpReduce(volatile {data_type}* input, int idx) ' + "{\n") 
        while(loop_i >= 1):
            kernel2.write(f"\tinput[idx] += input[idx + {loop_i}];\n")
            loop_i = loop_i >> 1

        kernel2.write('}\n')
        kernel2.write(f'__global__ void kernel({data_type}* input1, {data_type}* input2, {data_type}* output) ' + "{\n")
        kernel2.write(f'\t__shared__ {data_type} partial_sum[THREADS_PER_BLOCK];\n')
        kernel2.write(f'\tint idx = blockIdx.x * blockDim.x + threadIdx.x;\n')
        kernel2.write(f'\tpartial_sum[threadIdx.x] = input1[idx]*input2[idx];\n')
        kernel2.write(f'\t__syncthreads();\n')
        kernel2.write(f'\tfor (int s = THREADS_PER_BLOCK/2; s > {WARP_SIZE}; s = s>>1) ' + '{\n')
        kernel2.write(f'\t\tif (threadIdx.x < s) ' + '{\n')
        kernel2.write(f'\t\t\tpartial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];\n')
        kernel2.write('\t\t}\n')
        kernel2.write(f'\t\t__syncthreads();\n')
        kernel2.write('\t}\n')
        kernel2.write(f'\tif (threadIdx.x < {WARP_SIZE}) ' + '{\n')
        kernel2.write(f'\t\tsingleWarpReduce(partial_sum, threadIdx.x);\n' + '\t}\n')
        kernel2.write('\tif (threadIdx.x == 0) {\n')
        kernel2.write('\t\toutput[blockIdx.x] = partial_sum[0];\n\t}\n}\n')
        kernel2.write(f'__global__ void kernel({data_type}* input, {data_type}* output) ' + '{\n')
        kernel2.write(f'\t__shared__ {data_type} partial_sum[THREADS_PER_BLOCK];\n')
        kernel2.write(f'\tint idx = blockIdx.x * blockDim.x + threadIdx.x;\n')
        kernel2.write(f'\tpartial_sum[threadIdx.x] = input[idx];\n')
        kernel2.write(f'\t__syncthreads();\n')
        kernel2.write(f'\tfor (int s = THREADS_PER_BLOCK/2; s > {WARP_SIZE}; s = s>>1) ' + '{\n')
        kernel2.write(f'\t\tif (threadIdx.x < s) ' + '{\n')
        kernel2.write(f'\t\t\tpartial_sum[threadIdx.x] += partial_sum[threadIdx.x + s];\n')
        kernel2.write('\t\t}\n')
        kernel2.write(f'\t\t__syncthreads();\n')
        kernel2.write('\t}\n')
        kernel2.write(f'\tif (threadIdx.x < {WARP_SIZE}) ' + '{\n')
        kernel2.write(f'\t\tsingleWarpReduce(partial_sum, threadIdx.x);\n')
        kernel2.write('\t}\n')
        kernel2.write('\tif (threadIdx.x == 0) {\n')
        kernel2.write('\t\toutput[blockIdx.x] = partial_sum[0];\n\t}\n}\n')

if (type_operation == '3D_Convolution'):
    #IS_X = 224 #All these parameters need to be changed
    #IS_Y = 224
    #FS = 3
    #IC = 64
    #OC = 64
    #BS = 3
    #TILE_X = 16
    #TILE_Y = 8
    DATATYPE_SIZE = 4
    M = OC
    K = FS*FS*IC
    tpb_a = int(m.sqrt(MAX_THREADS_PER_BLOCK))
    tpb_b = M
    THREADS_PER_BLOCK_GEMM = min(tpb_a, tpb_b)
    TILE_Y = THREADS_PER_BLOCK_GEMM
    max_tile_x = int(SHMEM_PER_BLOCK/(2*TILE_Y*DATATYPE_SIZE))
    if (max_tile_x > K):
        TILE_X = K
    else:
        TILE_X = int(max_tile_x/THREADS_PER_BLOCK_GEMM)*THREADS_PER_BLOCK_GEMM
    print('THREADS_PER_BLOCK:', THREADS_PER_BLOCK_GEMM)
    print('TILE_X:', TILE_X)
    print('TILE_Y:', TILE_Y)
    #THREADS_PER_BLOCK_GEMM = 8 #should be the same as TILE_Y
    folder_name = f"{type_operation}_{IS_X}_{IS_Y}" #creates a unique folder name
    os.system(f"mkdir {folder_name}")
    #generating gpu_run.cuh for 3D Convolution operation
    with open (f'./{folder_name}/gpu_run.cuh', 'w') as gpu_header:
        gpu_header.write(f"#ifndef GPU_RUN\n")
        gpu_header.write(f"#define GPU_RUN\n")
        gpu_header.write(f"\t#define IS_X {IS_X}\n")
        gpu_header.write(f"\t#define IS_Y {IS_Y}\n")
        gpu_header.write(f"\t#define FS {FS}\n")
        gpu_header.write(f"\t#define IC {IC}\n")
        gpu_header.write(f"\t#define OC {OC}\n")
        gpu_header.write(f"\t#define BS {BS}\n")
        gpu_header.write(f"\t#define THREADS_PER_BLOCK_GEMM {THREADS_PER_BLOCK_GEMM}\n")
        gpu_header.write(f"\t#define TILE_X {TILE_X}\n")
        gpu_header.write(f"\t#define TILE_Y {TILE_Y}\n")
        gpu_header.write(f"\t#define OS_X (IS_X-FS+1)\n")
        gpu_header.write(f"\t#define OS_Y (IS_Y-FS+1)\n")
        gpu_header.write(f"\t#define M OC\n")
        gpu_header.write(f"\t#define K (FS*FS*IC)\n")
        gpu_header.write(f"\t#define N (OS_X*OS_Y*BS)\n")
        gpu_header.write(f"\t#define BLOCKS_X_GEMM ((N + THREADS_PER_BLOCK_GEMM - 1)/THREADS_PER_BLOCK_GEMM)\n")
        gpu_header.write(f"\t#define BLOCKS_Y_GEMM ((M + THREADS_PER_BLOCK_GEMM - 1)/THREADS_PER_BLOCK_GEMM)\n")
        gpu_header.write(f"\t#define THREADS_PER_BLOCK_X_RI FS\n")
        gpu_header.write(f"\t#define THREADS_PER_BLOCK_Y_RI FS\n")
        gpu_header.write(f"\t#define THREADS_PER_BLOCK_Z_RI IC\n")
        gpu_header.write(f"\t#define BLOCKS_X_RI OS_X\n")
        gpu_header.write(f"\t#define BLOCKS_Y_RI OS_Y\n")
        gpu_header.write(f"\t#define BLOCKS_Z_RI BS\n")
        gpu_header.write(f"\t#define THREADS_PER_BLOCK_X_RF FS\n")
        gpu_header.write(f"\t#define THREADS_PER_BLOCK_Y_RF FS\n")
        gpu_header.write(f"\t#define THREADS_PER_BLOCK_Z_RF 8\n") #check if this is eight
        gpu_header.write(f"\t#define BLOCKS_X_RF 1\n")
        gpu_header.write(f"\t#define BLOCKS_Y_RF 1\n")
        gpu_header.write(f"\t#define BLOCKS_Z_RF ((OC*IC + THREADS_PER_BLOCK_Z_RF - 1)/THREADS_PER_BLOCK_Z_RF)\n")
        gpu_header.write(f"\tvoid gpu_run({data_type}*, {data_type}*, {data_type}*);\n")
        gpu_header.write(f"\t__global__ void kernel({data_type}*, {data_type}*, {data_type}*);\n")
        gpu_header.write(f"\t__global__ void reformat_input({data_type}*, {data_type}*);\n")
        gpu_header.write(f"\t__global__ void reformat_filter({data_type}*, {data_type}*);\n")
        gpu_header.write(f"#endif\n")

    #generating gpu_run.cu for 3D Convolution operation
    with open (f'./{folder_name}/gpu_run.cu', 'w') as gpu_run:
        gpu_run.write(f'#include "gpu_run.cuh"\n')
        gpu_run.write(f'#include <iostream>\n')
        gpu_run.write(f'void gpu_run({data_type}* input1, {data_type}* input2, {data_type}* output) ' + "{\n")
        gpu_run.write(f'\t{data_type}* input1_d;\n')
        gpu_run.write(f'\tsize_t s_input1 = IS_X*IS_Y*IC*BS*sizeof({data_type});\n')
        gpu_run.write(f'\t{data_type}* input1_gemm_d;\n')
        gpu_run.write(f'\tsize_t s_input1_gemm = K*N*sizeof({data_type});\n')
        gpu_run.write(f'\t{data_type}* input2_d;\n')
        gpu_run.write(f'\tsize_t s_input2 = FS*FS*IC*OC*sizeof({data_type});\n')
        gpu_run.write(f'\t{data_type}* input2_gemm_d;\n')
        gpu_run.write(f'\tsize_t s_input2_gemm = M*K*sizeof({data_type});\n')
        gpu_run.write(f'\t{data_type}* output_d;\n')
        gpu_run.write(f'\tsize_t s_output = OS_X*OS_Y*OC*BS*sizeof({data_type});\n')
        gpu_run.write(f'\tcudaMalloc(&input1_d, s_input1);\n')
        gpu_run.write(f'\tcudaMalloc(&input1_gemm_d, s_input1_gemm);\n')
        gpu_run.write(f'\tcudaMalloc(&input2_d, s_input2);\n')
        gpu_run.write(f'\tcudaMalloc(&input2_gemm_d, s_input2_gemm);\n')
        gpu_run.write(f'\tcudaMalloc(&output_d, s_output);\n')
        gpu_run.write(f'\tcudaMemcpy(input1_d, input1, s_input1, cudaMemcpyHostToDevice);\n')
        gpu_run.write(f'\tdim3 block_dim_ri(THREADS_PER_BLOCK_X_RI, THREADS_PER_BLOCK_Y_RI, THREADS_PER_BLOCK_Z_RI);\n')
        gpu_run.write(f'\tdim3 grid_dim_ri(BLOCKS_X_RI, BLOCKS_Y_RI, BLOCKS_Z_RI);\n')
        gpu_run.write(f'\treformat_input<<<grid_dim_ri, block_dim_ri>>>(input1_d, input1_gemm_d);\n')
        gpu_run.write(f'\tcudaMemcpy(input2_d, input2, s_input2, cudaMemcpyHostToDevice);\n')
        gpu_run.write(f'\tdim3 block_dim_rf(THREADS_PER_BLOCK_X_RF, THREADS_PER_BLOCK_Y_RF, THREADS_PER_BLOCK_Z_RF);\n')
        gpu_run.write(f'\tdim3 grid_dim_rf(BLOCKS_X_RF, BLOCKS_Y_RF, BLOCKS_Z_RF);\n')
        gpu_run.write(f'\treformat_filter<<<grid_dim_rf, block_dim_rf>>>(input2_d, input2_gemm_d);\n')
        gpu_run.write(f'\tcudaDeviceSynchronize();\n')
        gpu_run.write(f'\tdim3 block_dim_gemm(THREADS_PER_BLOCK_GEMM, THREADS_PER_BLOCK_GEMM);\n')
        gpu_run.write(f'\tdim3 grid_dim_gemm(BLOCKS_X_GEMM, BLOCKS_Y_GEMM);\n')
        gpu_run.write(f'\tkernel<<<grid_dim_gemm, block_dim_gemm>>>(input2_gemm_d, input1_gemm_d, output_d);\n')
        gpu_run.write(f"\tcudaDeviceSynchronize();\n")
        gpu_run.write(f'\tcudaMemcpy(output, output_d, s_output, cudaMemcpyDeviceToHost);\n')
        gpu_run.write(f'\tcudaFree(input1_d);\n')
        gpu_run.write(f'\tcudaFree(input1_gemm_d);\n')
        gpu_run.write(f'\tcudaFree(input2_d);\n')
        gpu_run.write(f'\tcudaFree(input2_gemm_d);\n')
        gpu_run.write(f'\tcudaFree(output_d); ' + "\n}\n") 

    #generating kernel.cu for 3D Convolution operation
    with open (f'./{folder_name}/kernel.cu', 'w') as kernel:
        kernel.write(f'#include "gpu_run.cuh"\n')
        kernel.write(f'__global__ void reformat_input({data_type}* input, {data_type}* output) ' + "{\n")
        kernel.write(f'\tint row_idx = threadIdx.z*FS*FS + threadIdx.y*FS + threadIdx.x;\n')
        kernel.write(f'\tint col_idx = blockIdx.z*OS_X*OS_Y + blockIdx.y*OS_X + blockIdx.x;\n')
        kernel.write(f'\tif ( (threadIdx.y+blockIdx.y)<IS_Y && (threadIdx.x+blockIdx.x)<IS_X )\n')
        kernel.write(f'\t\toutput[row_idx*N + col_idx] = input[blockIdx.z*IC*IS_X*IS_X + threadIdx.z*IS_X*IS_Y + (threadIdx.y+blockIdx.y)*IS_X + (threadIdx.x+blockIdx.x)];\n')
        kernel.write('}\n')
        kernel.write(f"__global__ void reformat_filter({data_type}* input, {data_type}* output) " + "{\n")
        kernel.write(f"\tint idx_z = blockIdx.z*blockDim.z + threadIdx.z;\n")
        kernel.write(f"\tint idx_y = blockIdx.y*blockDim.y + threadIdx.y;\n")
        kernel.write(f"\tint idx_x = blockIdx.x*blockDim.x + threadIdx.x;\n")
        kernel.write(f"\tint idx_all = idx_z*FS*FS + idx_y*FS + idx_x;\n")
        kernel.write(f"\tint row_idx = idx_z/IC;\n")
        kernel.write(f"\tint col_idx = idx_all - row_idx*FS*FS*IC;\n")
        kernel.write(f"\toutput[row_idx*K + col_idx] = input[idx_all];\n" + "}\n")
        kernel.write(f"__global__ void kernel({data_type}* input1, {data_type}* input2, {data_type}* output) " + "{\n")
        kernel.write(f"\t__shared__ {data_type} input1_s[TILE_X*TILE_Y];\n")
        kernel.write(f"\t__shared__ {data_type} input2_s[TILE_X*TILE_Y];\n")
        kernel.write(f"\t{data_type} temp = 0;\n")
        kernel.write(f"\tint col_idx = blockIdx.x*blockDim.x + threadIdx.x;\n")
        kernel.write(f"\tint row_idx = blockIdx.y*blockDim.y + threadIdx.y;\n")
        kernel.write(f"\tfor (int i=0; i<K; i+=TILE_X) " + "{\n")
        kernel.write(f"\t\tfor (int t=0; t<TILE_X; t+=THREADS_PER_BLOCK_GEMM) " + "{\n")
        kernel.write(f"\t\t\tinput1_s[threadIdx.y*TILE_X + threadIdx.x + t] = input1[row_idx*K + i + threadIdx.x + t];\n")
        kernel.write(f"\t\t\tinput2_s[(threadIdx.y+t)*TILE_Y + threadIdx.x] = input2[(i+threadIdx.y+t)*N + col_idx];\n" + "\t\t}\n")
        kernel.write(f"\t\t__syncthreads();\n")
        kernel.write(f"\t\tfor (int j=0; j<TILE_X; j++) " + "{\n")
        kernel.write(f"\t\t\ttemp += input1_s[threadIdx.y*TILE_X + j] * input2_s[j*TILE_Y + threadIdx.x];" + "\n\t\t}\n")
        kernel.write(f"\t\t__syncthreads();\n" + "\t}\n")
        kernel.write(f"\tif (row_idx < M && col_idx < N) " + "{\n")
        kernel.write(f"\t\toutput[row_idx*N + col_idx] = temp;\n" + "\t}\n")
        kernel.write("}\n")  





