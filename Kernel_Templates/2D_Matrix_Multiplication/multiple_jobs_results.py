import os
from pickle import GLOBAL
import re
import matplotlib.pyplot as plt
import pandas as pd

start_path = '.' # current directory

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
    os.system(f"grep 'kernel\|reformat' {file_name} > result.txt")
    with open("result.txt") as file1:
        f = file1.readlines()
        key = file_name.split(".txt")[0]
        time = 0
        for data in f:
            #temp_value = f[0].split('s')[0].split("%  ")[1]
            #print(data)
            #temp_value = data.split('s')[0].split("%  ")[1]
            temp_value = data.split('%  ')[1].split("s")[0]
            if re.search("u", temp_value):
                time = time + 0.001*float(temp_value.split("u")[0])
            else:
                time = time + float(temp_value.split("m")[0])
        value = time
        print("value is: ", value)
        grep_dict[key] = value

#This function is used for plotting
def graph_plot(x_axis, y_axis, d, x_label, y_label, title):
    plt.bar(x_axis, y_axis, data = d, color = ['green', 'pink', 'maroon', 'yellow', 'magenta'], width=0.5)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)

file_list = []
list_extract(file_list, ".txt") #This gives a list of the files which I got from nvprof result
print("file list is: ", file_list)
aa={}

for x in file_list:
    result_extraction(x, aa) #getting the execution time from all the result ".txt" files and putting them in a dictionary
print(aa)

plt.figure(1)
x_axis = list(aa.keys())
y_axis = list(aa.values())

df = pd.DataFrame({"Configuration":x_axis, "Run_Time":y_axis})
df_sorted = df.sort_values('Run_Time')

#graph_plot(list(aa.keys()), list(aa.values()), "Run", "Execution Time (in ms)", "Showing the Execution Time")
graph_plot("Configuration", "Run_Time", df_sorted, "Threads Per Block -->", "Execution Time (in ms)", "Execution Time vs Threads Per Block")
plt.show()
plt.savefig("letssee.jpg")
os.system("rm -rf result.txt") 

