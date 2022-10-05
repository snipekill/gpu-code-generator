import numpy as np
import matplotlib.pyplot as plt

# set width of bar
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8))

####### Plots for MAC

# set height of bar
tetra = [3.8400, 3.4560, 4.0960, 7.296, 7.36, 7.263]
aws = [4.5120, 4.9280, 6.1120, 10.7, 10.9, 11]

# Set position of bar on X axis
bar1 = np.arange(len(tetra))
bar2 = [x + barWidth for x in bar1]

# Make the plot
plt.bar(bar1, tetra, color ='b', width = barWidth,
		edgecolor ='grey', label ='Titan V')
plt.bar(bar2, aws, color ='g', width = barWidth,
		edgecolor ='grey', label ='Tesla M60')

# Adding Xticks
plt.xlabel('Input Size', fontweight ='bold', fontsize = 15)
plt.ylabel('Execution Time (us)', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(tetra))],
		['256', '512', '1024', '2048', '4096', '8192'])

plt.legend()
plt.show()


####### Plots for GEMM

# set height of bar
tetra = [115.78, 880.44, 4.7670, 7.0720, 10.112, 24.608]
aws = [728.68, 5540.5, 6.4320, 9.4400, 15.872, 133.25]

# Set position of bar on X axis
bar1 = np.arange(len(tetra))
bar2 = [x + barWidth for x in bar1]

# Make the plot
plt.bar(bar1, tetra, color ='b', width = barWidth,
		edgecolor ='grey', label ='Titan V')
plt.bar(bar2, aws, color ='g', width = barWidth,
		edgecolor ='grey', label ='Tesla M60')

# Adding Xticks
plt.xlabel('Input Size', fontweight ='bold', fontsize = 15)
plt.ylabel('Execution Time (us)', fontweight ='bold', fontsize = 15)
plt.xticks([r + barWidth for r in range(len(tetra))],
		['32', '64', '128', '256', '512', '1024'])

plt.legend()
plt.show()



####### Plots for 3D

# set height of bar
tetra = [0.485, 0.945, 1.8836, 3.7683, 2.5294, 5.0363, 9.9818, 19.668]
cudnn = [0.529, 0.628, 1.036, 1.865, 2.02, 4.136, 4.142, 1.336]
aws = [2.5321, 6.1919, 11.888, 24.580, 14.044, 29.696, 60.905, 98.880]

# Set position of bar on X axis
bar1 = np.arange(len(tetra))
bar2 = [x + barWidth for x in bar1]
bar3 = [x + barWidth for x in bar2]

# Make the plot
plt.bar(bar1, tetra, color ='b', width = barWidth,
		edgecolor ='grey', label ='Titan V')
plt.bar(bar2, cudnn, color ='r', width = barWidth,
		edgecolor ='grey', label ='cuDNN')
plt.bar(bar3, aws, color ='g', width = barWidth,
		edgecolor ='grey', label ='Tesla M60')        

# Adding Xticks
plt.xlabel('Input Size', fontweight ='bold', fontsize = 15)
plt.ylabel('Execution Time (ms)', fontweight ='bold', fontsize = 15)
####### Update the x-ticks accordingly
plt.xticks([r + barWidth for r in range(len(tetra))],
		['32', '64', '128', '256', '512', '1024', '2048', '4096'])

plt.legend()
plt.show()
