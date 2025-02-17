# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 13:08:01 2024

@author: 17154
"""

import tifffile as tiff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the TIFF file
tiff_path = r"D:\BaiduSyncdisk\Research\Succolarity\Test\3D-de-melo.tif"
tiff_data = tiff.imread(tiff_path)

# Define the 3D volume data after transformation
transformed_data = np.transpose(tiff_data, (2, 1, 0))

# Set default font settings
plt.rcParams['font.size'] = 12
plt.rcParams['font.family'] = 'Arial'

# Create the figure with a custom size
fig = plt.figure(figsize=(32, 24))  

# Create 3D axes
ax = fig.add_subplot(111, projection='3d')

# Set up the colors and transparencies
colors = np.empty(transformed_data.shape, dtype=object)
colors[transformed_data == 255] = '#303D90'  
colors[transformed_data == 0] = '#F4B183'    
blue_transparencies = [1, 0.6, 0.1, 0.08, 0.06, 0.06]  

# Iterate over each voxel in the transformed data and plot it
for x in range(transformed_data.shape[0]):
    for y in range(transformed_data.shape[1]):
        for z in range(transformed_data.shape[2]):
            alpha = blue_transparencies[y] if transformed_data[x, y, z] == 255 else 1.0
            color = colors[x, y, z]
            ax.bar3d(x, y, z, 1, 1, 1, color=color, alpha=alpha)

# Setting the axes limits to match the data bounds
ax.set_xlim([0, transformed_data.shape[0]])
ax.set_ylim([0, transformed_data.shape[1]])
ax.set_zlim([0, transformed_data.shape[2]])

# Customize axis labels with font settings
ax.set_xlabel('Y', fontsize=18, fontweight='bold',labelpad=20)
ax.set_ylabel('X', fontsize=18, fontweight='bold',labelpad=20)
ax.set_zlabel('Z', fontsize=18, fontweight='bold',labelpad=10)

# Optionally, hide grid lines and panes for a cleaner look
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.grid(False)

# Customize tick parameters
ax.tick_params(axis='both', which='major', labelsize=18)

# Adjust the view angle
ax.view_init(elev=195, azim=315)

# Set the plot title with custom font settings
# ax.text2D(0.5, 0.9, '3D Synthetic Volume', transform=ax.transAxes, ha='center', fontsize=30, fontweight='bold')
# Adjust the gap between axes and the main figure
ax.set_position([0.1, 0.5, 0.4, 2])  # left, bottom, width, height

plt.subplots_adjust(left=0.5, right=0.9, top=0.9, bottom=0.1)
# Display the plot
plt.show()

# Describing the data
'''data_description = {
    'shape': tiff_data.shape,
    'data_type': tiff_data.dtype,
    'min_value': tiff_data.min(),
    'max_value': tiff_data.max(),
    'mean_value': tiff_data.mean(),
    'non_zero_count': np.count_nonzero(tiff_data),
    'volume_list' : tiff_data.tolist()
}

print(data_description)'''