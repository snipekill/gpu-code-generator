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

#This function generates a list of files with a given extension
def list_extract(list_name, ext):
    for path,dirs,files in os.walk(start_path):
        for filename in files:
            if re.search(f"{ext}", filename):
                f = open(filename, 'r')
                data = f.read()
                if(data.find("Error") == -1):
                    #print (filename)
                    list_name.append(filename)

#This function is generating a dictionary of our desired output parameter
def result_extraction(file_name, grep_dict):
    os.system(f"grep -i  -A1 'gpu activities' {file_name} > result.txt")
    with open("result.txt") as file1:
        f = file1.readlines()
        key = file_name.split(".txt")[0]
        value = float(f[1].split("us")[0].split("%")[1].split("  ")[1])
        print("value is: ", value)
        grep_dict[key] = value

#This function is used for plotting
def graph_plot(x_axis, y_axis, x_label, y_label, title):
    plt.bar(x_axis, y_axis, color = ['green', 'yellow', 'maroon'], width=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

#This will compile the CUDA files and store the result in a text file with the same name
for file in file_list:
    output_file = file.split(".cu")[0]
    output_file_1 = output_file+".txt"
    #print(output_file)
    os.system(f"nvcc {file} -o {output_file}")
    os.system(f"nvprof --log-file {output_file_1} ./{output_file}")

file_list = []
list_extract(file_list, ".txt") #This gives a list of the files which I got from nvprof result
print("file list is: ", file_list)
aa={} 
for x in file_list:
    result_extraction(x, aa) #getting the execution time from all the result ".txt" files and putting them in a dictionary
print(aa)

plt.figure(1)
graph_plot(list(aa.keys()), list(aa.values()), "Run", "Execution Time (in us)", "Showing the Execution Time")
plt.show()
plt.savefig("letssee.jpg")  