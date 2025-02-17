# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 15:09:10 2023

@author: 17154
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Create empty volume 
vol = np.full((6,6,6), 255, dtype=np.uint8)

# Set specific voxel values
vol[2,0,5] = 0 
vol[3,0,5] = 0
vol[2,0,4] = 0
vol[2,1,4] = 0
vol[2,2,4] = 0
vol[3,0,4] = 0
vol[3,1,4] = 0
vol[3,2,4] = 0
vol[2,2,3] = 0
vol[2,3,3] = 0
vol[3,2,3] = 0
vol[3,3,3] = 0
vol[1,1,2] = 0
vol[1,2,2] = 0
vol[2,1,2] = 0
vol[2,2,2] = 0
vol[2,3,2] = 0
vol[2,4,2] = 0
vol[3,1,2] = 0
vol[3,2,2] = 0
vol[3,3,2] = 0
vol[3,4,2] = 0
vol[2,2,1] = 0
vol[2,3,1] = 0
vol[3,2,1] = 0
vol[3,3,1] = 0
vol[4,2,1] = 0
vol[4,3,1] = 0
vol[2,1,0] = 0
vol[2,2,0] = 0
vol[2,3,0] = 0
vol[3,1,0] = 0
vol[3,2,0] = 0
vol[3,3,0] = 0
vol[4,1,0] = 0
vol[4,2,0] = 0
vol[4,3,0] = 0

print (vol[:,:,0])
print (vol[:,:,1])
print (vol[:,:,2])
print (vol[:,:,3])
print (vol[:,:,4])
print (vol[:,:,5])

folder = r'D:\17154\OneDrive - University of Leeds\Documents\Research\Succolarity\Test'

# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)

# Iterate through slices
for z in range(vol.shape[2]):
    # Create a PIL Image from the 2D slice
    slice_img = Image.fromarray(vol[:, :, z])

    # Show the slice using matplotlib
    plt.imshow(slice_img, cmap='gray')
    plt.title(f'Slice {z}')
    plt.show()

    # Save the slice as a TIFF image
    tiff_path = os.path.join(folder, f'slice_{z}.tiff')
    slice_img.save(tiff_path)

    # Print a message to confirm saving
    print(f"Slice {z} saved as '{tiff_path}'")