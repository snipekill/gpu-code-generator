import os
import re
import matplotlib.pyplot as plt

start_path = '.' # current directory
file_list = []
#Here, we are generating a list of .cu files
for path,dirs,files in os.walk(start_path):
    for filename in files:
        if re.search(".cu", filename):
            if(filename.find(".cuh") == -1):
                print (filename)
                file_list.append(filename)
print(file_list)

class scan_input:
    def __init__(self):
        self.type_op = input('Pls mention the type of operation: \n')

scan_op = scan_input()
type_op = scan_op.type_op
#type_op = "2D_GEMM"
print(type_op)

#This is compiling and executing the CUDA files
os.system(f"nvcc {file_list[0]} {file_list[1]} {file_list[2]} -o {type_op}")
type_op1 = type_op+".txt"
os.system(f'nvprof --log-file {type_op1} ./{type_op}')