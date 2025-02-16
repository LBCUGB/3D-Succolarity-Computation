# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:43:41 2023

@author: 17154
"""


import numpy as np
from scipy.ndimage import distance_transform_edt as edt
from matplotlib import pyplot as plt
import porespy as ps # Porespy needs to be installed to use the Gstick 2017 algorithm for watershed
import pandas as pd
import math
from skimage.measure import label
import time


# Generic function to aid boxesOfSIze

def divisors(n):
    '''A function to compute the divisors of a number
    @parameters: number
    @returns: a list containing its divisors
    '''
    divisors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    divisors.sort()
    return divisors

####################
#### 3D FUNCTIONS
####################

def selectConnectedComponents3D( imageBinary3D,  direction='tb', con=2 ):
    '''A function to return a binay 3D array containing the pore regions connected to the inlet
    @parameters:    the original 3D binary image as a np array
                    the direction to be applied tb (top-to-bottom), bt (bottom-to-top), lr (left-to-right), rl (right-to-left), fb (front-to-back), bf (back-to-front)
                    (!): Warning: the coordinates in python/np may be different than in other software

    @returns: the 3D binary imge containing only the pore regions connected to a specific inlet, as an np array (1/0)
    '''
    labeled_image = label(imageBinary3D, connectivity=con)
    
    result = np.zeros(labeled_image.shape);
    if direction=='tb':
        acceptedRegions = np.unique(labeled_image[0,:,:])   
        sliceO = labeled_image[0,:,:]
    elif direction=='bt':
        acceptedRegions = np.unique(labeled_image[-1,:,:])
        sliceO = labeled_image[-1,:,:]
    elif direction=='lr':
        acceptedRegions = np.unique(labeled_image[:,0,:])
        sliceO = labeled_image[:,0,:]
    elif direction=='rl':
        acceptedRegions = np.unique(labeled_image[:,-1,:])
        sliceO = labeled_image[:,-1,:]
    elif direction=='fb':
        acceptedRegions = np.unique(labeled_image[:,:,0])
        sliceO = labeled_image[:,:,0]
    elif direction=='bf':
        acceptedRegions = np.unique(labeled_image[:,:,-1])
        sliceO = labeled_image[:,:,-1]
    else:
        print('Unknown direction')
    #print(sliceO.shape)
    #plt.imshow(sliceO); plt.show();
    #print(len(np.unique(sliceO)))
    
    
    if len(np.unique(sliceO))>=2:
        #background_label = labeled_image[sliceO == 0][0] if 0 in sliceO else None
        background_label = 0 if 0 in sliceO else None
        if background_label is not None:
            acceptedRegions = acceptedRegions[acceptedRegions != background_label]
        #for i in acceptedRegions:
        #    print(i)
        #    result = result + 1*(labeled_image==i)
        #Above was changed to be vectorized - faster
        mask = np.isin(labeled_image, acceptedRegions)
        result[mask] = 1

    else:
        if np.unique(sliceO)[0]==1:
            for i in acceptedRegions:
                result = result + 1*(labeled_image==i)
            
            
    result = 1*(1*result!=0)  
    return result

# Using Sliding Window (Convolution) Approach
from scipy.signal import convolve as sig_conv
def slidingWindowOccupancy3D(image, window_size):
    '''
    Computes occupancy (without padding) using optimised convolutions from scipy signal library.
    
    Parameters
    ----------
    image : np.array (0/1)
        The connected geometry applicable.
    window_size : TYPE
        The convokution Window size.

    Returns
    -------
    np.array
        The occupancy value at each pixel space in the raser. To avoid padding we use 'valid' convolution stype which will cut the array
        The approach uses regular convolutions

    '''
    if window_size >= np.min(image.shape):
        print('ISSUE - WINDOW SIZE LARGER THAN THE LENGTH OF THE SMALLEST IMAGE SIDE')
        return 0
    height, width, depth = image.shape
    kernel = np.ones((window_size, window_size, window_size))
    result = sig_conv(image, kernel, mode='valid') #mode reflect considers what happens at the image edges
    return result/window_size/window_size/window_size #The last division scales by occupancy


from scipy.signal import fftconvolve as fft_conv
def slidingWindowOccupancy3DFFT(image, window_size):
    '''
    Computes occupancy (without padding) using optimised FFT convolutions.
    

    Parameters
    ----------
    image : np.array (0/1)
        The connected geometry applicable.
    window_size : TYPE
        The convolution Window size.

    Returns
    -------
    np.array
        The occupancy value at each pixel space in the raser. To avoid padding we use 'valid' convolution stype which will cut the array
        The approach uses FFT which may be faster than regular convolutions in some cases

    '''
    if window_size >= np.min(image.shape):
        print('ISSUE - WINDOW SIZE LARGER THAN THE LENGTH OF THE SMALLEST IMAGE SIDE')
        return 0
    height, width, depth = image.shape
    kernel = np.ones((window_size, window_size, window_size))
    result = fft_conv(image, kernel, mode='valid') #mode reflect considers what happens at the image edges
    return result/window_size/window_size/window_size #The last division scales by occupancy


def slidingWindowVirtualP3D(volume, direction='tb', window_size=1):
    '''
    Computes the virtual pressure field in 3D for a given direction.

    Parameters
    ----------
    volume : np array
        An array with 0 and 1 values that has the connected geometry to be used - only used to get the size.
    direction : TYPE, optional
        The direction to be applied: tb, bt, lr, rl, fb, bf. The default is 'tb'.
    window_size : TYPE, optional
        The size of the convolution window or box. The default is 1 but that is not very informative.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    TYPE: np.array
        A 'virtual pressure array' as described by Melo et al, that has the correct size to be multiplied with the occupancy array (a 'valid' style (without padding) convolution.

    '''
    depth, height, width = volume.shape
    Pressure = np.zeros_like(volume, dtype=float)
    
    if direction == 'tb':  # Top-to-Bottom
        Pressure[:, :, :] = np.arange(0, height)[:, np.newaxis, np.newaxis] + 1

    elif direction == 'bt':  # Bottom-to-Top
        Pressure[:, :, :] = (height - np.arange(0, height)[:, np.newaxis, np.newaxis]) - 1

    elif direction == 'fb':  # Left-to-Right
        Pressure[:, :, :] = np.arange(0, width)[np.newaxis, np.newaxis, :] + 1

    elif direction == 'bf':  # Right-to-Left
        Pressure[:, :, :] = (width - np.arange(0, width)[np.newaxis, np.newaxis, :]) - 1

    elif direction == 'lr':  # Front-to-Back (Depth)
        Pressure[:, :, :] = np.arange(0, depth)[np.newaxis, :, np.newaxis] + 1

    elif direction == 'rl':  # Back-to-Front (Depth)
        Pressure[:, :, :] = (depth - np.arange(0, depth)[np.newaxis, :, np.newaxis]) - 1

    else:
        raise ValueError('Invalid direction')
    
    # Calculate the amount to slice from each side
    slice_amount = (window_size - 1) // 2
    additional_slice = (window_size - 1) % 2

    # Adjust for window size to mimic 'valid' convolution
    Pressure = Pressure[slice_amount:depth - slice_amount - additional_slice, 
                        slice_amount:height - slice_amount - additional_slice, 
                        slice_amount:width - slice_amount - additional_slice] - 0.5


    return Pressure #So the result is between integers

def computePc3D(volume, sigma=0.072, theta=180, g=9.8, deltaRho=1000, voxelSize=1e-8, useWashburn=False, orientation='tb'):
    '''
    A function that computes the capillary pressure of each pore in a 3D volume based on the image and some fluid parameters.
    @parameters:
        volume: the 3D (0/1) image as a np array that only contains the connected component from a specified inlet
        sigma, theta, g, deltaRho: typical; g set to 0 to ignore gravity effects
        voxelSize: in m
        useWashburn: a boolean parameter that controls whether to use the default Washburn equation with sigma and theta provided or use the Young-Laplace approach calculated below
        orientation: direction of flow through the medium ('tb', 'bt', 'lr', 'rl', 'fb', 'bf')
    @returns:
        An array with the capillary pressure attributed to each voxel
    '''

    # Rotate the volume based on the specified orientation
    if orientation == 'bt':
        volume = np.rot90(volume, k=2, axes=(0, 1))  # Top to bottom becomes bottom to top
    elif orientation == 'rl':
        volume = np.rot90(volume, k=2, axes=(1, 2))  # Left to right becomes right to left
    elif orientation == 'lr':
        volume = np.rot90(volume, k=2, axes=(2, 1))  # Right to left becomes left to right
    elif orientation == 'fb':
        volume = np.rot90(volume, k=2, axes=(0, 2))  # Front to back
    elif orientation == 'bf':
        volume = np.rot90(volume, k=2, axes=(2, 0))  # Back to front

    # Compute Euclidean distance transform (EDT) to get pore sizes in meters
    dt = edt(volume) * voxelSize

    # Compute capillary pressure using the Young-Laplace equation
    pc = -2 * sigma * np.cos(np.deg2rad(theta)) / dt

    if useWashburn:
        # Use a library like PoreSpy to compute drainage with Washburn equation
        pc_volume = ps.simulations.drainage(im=volume, voxel_size=voxelSize)
    else:
        # Alternatively, use the Young-Laplace approach for the entire volume
        pc_volume = ps.simulations.drainage(pc=pc, im=volume, voxel_size=voxelSize)

    result = pc_volume.im_pc

    # Rotate the result back to the original orientation, if necessary
    if orientation == 'bt':
        result = np.rot90(result, k=2, axes=(0, 1))
    elif orientation == 'rl':
        result = np.rot90(result, k=2, axes=(1, 2))
    elif orientation == 'lr':
        result = np.rot90(result, k=2, axes=(2, 1))
    elif orientation == 'fb':
        result = np.rot90(result, k=2, axes=(0, 2))
    elif orientation == 'bf':
        result = np.rot90(result, k=2, axes=(2, 0))

    return result

def CutOriginal3D(volume, window_size=1):
    '''
    Produces an array of the correct size to be multiplied with un-paded convolution results fro PR or Occupancy

    Parameters
    ----------
    volume : np.array
        Connectied ement array with values 0/1.
    window_size : int, optional
        DESCRIPTION. The default is 1 - must be the same as the other operations.

    Returns
    -------
    np.array
        The array cut in the 'vaild' convolution styl that can be multiplied with the occupancy or pressure arrays.

    '''
    depth, height, width = volume.shape
    # Calculate the amount to slice from each side
    slice_amount = (window_size - 1) // 2
    additional_slice = (window_size - 1) % 2
    # Adjust for window size to mimic 'valid' convolution
    New = volume[slice_amount:depth - slice_amount - additional_slice, 
                        slice_amount:height - slice_amount - additional_slice, 
                        slice_amount:width - slice_amount - additional_slice] - 0.5
    return New #So the result is between integers

#BOX COUNTING APPROACH
def countingBoxVirtualP3D(volume, box_size, direction='tb'):
    '''
    Virtual pressue optimised approach for the conting box approach

    '''
    fullPArray = slidingWindowVirtualP3D(volume, direction='tb')
    center_values = fullPArray[box_size//2::box_size, box_size//2::box_size, box_size//2::box_size] #only get the value for the box centers
    if box_size%2 ==0:
        center_values = center_values - 1
    else:
        center_values-=0.5 #Adding 0.5 to account for the extra 0.5 distance travelled
         

def computeSuccolarity3DSlidingWindowForSize( connected, boxSize,  direction='tb', option = 'fftconv'):
    '''
    A function that computs the 3D succolarity value across using a sliding window/convolution aproach

    Parameters
    ----------
    connected : np.array (0/1)
        COnnected array.
    boxSize : int
        the window size to be used in the convolutions. As they are unpadded the result will be cut appropriately
    direction : TYPE, optional
        the direction to be applied tb (top-to-bottom), bt (bottom-to-top), lr (left-to-right), rl (right-to-left), fb (front-to-back), bf (back-to-front)
. The default is 'tb'.
    option : str, optional
        'conv' for tratitional convolutions, otherwise FFT approach is used. The default is 'conv'.

    Returns
    -------
    rezult : float
        Returns the Succolarity value computed as in Melo et al.

    '''
    if option == 'conv':
        OccupancyField = slidingWindowOccupancy3D(connected, boxSize)
    else:
        OccupancyField = slidingWindowOccupancy3DFFT(connected, boxSize)
    
    PressureField = slidingWindowVirtualP3D(connected, direction=direction, window_size = boxSize)  #Not multiplied by porosity value at that point (?)
    
    maxOPPR = np.sum(PressureField) #Assuming occupancy of 1 at all possible places
    
    #Computing Occupancy times virtual pressure:
    OPPR = OccupancyField*PressureField;
    sumOPPR = np.sum(OPPR)
    
    rezult = sumOPPR / maxOPPR
    return rezult

def computeSuccolaritySlidingWindow3D(v,sizes, plot=True):
    '''A function that computs the 3D succolarity values across using a sliding window/convolution aproach
    @parameters:    the 3D (0/1) imge as a np array that only contains the connected component from a specified inlet
                    plot: a parameter that will eithrt display or not the log-log plot

    @returns: 6 lists representing the succolarity values at different box sizes, from all 6 possible inlets
    '''

    #commonDiv = divisors(sliceOne.shape[0])
    commonDiv = sizes# since now we can use any sizes, do not need to be exactly divizible
    tb = []
    #computing top to bottom

    for div in commonDiv:
        print('bt - ', div)
        connected = selectConnectedComponents3D(v,  direction='tb', con=2    )
        tb.append(computeSuccolarity3DSlidingWindowForSize( connected, div,  direction='tb', option='FFT' ))
        #if div == commonDiv[0]:
        #    plt.imshow(connected[:,20,:]);plt.show();

    bt = []
    #computing top to bottom
    for div in commonDiv:
        print('tb - ', div)
        connected = selectConnectedComponents3D( v,  direction='bt', con=2   )
        bt.append(computeSuccolarity3DSlidingWindowForSize( connected, div,  direction='bt', option='FFT' ))
        #if div == commonDiv[0]:
        #    plt.imshow(connected[:,20,:]);plt.show();

    lr = []
    #computing top to bottom
    for div in commonDiv:
        print('lr - ', div)
        connected = selectConnectedComponents3D( v,  direction='lr', con=2    )
        lr.append(computeSuccolarity3DSlidingWindowForSize(  connected, div,  direction='lr', option='FFT' ))
        #if div == commonDiv[0]:
        #    plt.imshow(connected[:,20,:]);plt.show();

    rl = []
    #computing top to bottom
    for div in commonDiv:
        print('rl - ', div)
        connected = selectConnectedComponents3D( v,  direction='rl', con=2    )
        rl.append(computeSuccolarity3DSlidingWindowForSize(  connected, div,  direction='rl', option='FFT' ))
        #if div == commonDiv[0]:
        #    plt.imshow(connected[:,20,:]);plt.show();
            
    fb = []
    #computing top to bottom
    for div in commonDiv:
        print('fb - ', div)
        connected = selectConnectedComponents3D( v,  direction='fb', con=2    )
        fb.append(computeSuccolarity3DSlidingWindowForSize(  connected, div,  direction='fb', option='FFT' ))
        #if div == commonDiv[0]:
        #    plt.imshow(connected[:,:,20]);plt.show();

    bf = []
    #computing top to bottom
    for div in commonDiv:
        print('bf - ', div)
        connected = selectConnectedComponents3D( v,  direction='bf', con=2    )
        bf.append(computeSuccolarity3DSlidingWindowForSize(  connected, div,  direction='bf', option='FFT' ))
        #if div == commonDiv[0]:
        #    plt.imshow(connected[:,:,20]);plt.show();

    if plot ==True:

        plt.plot(np.log(v.shape[0]/np.array(commonDiv)), np.log(100*np.array(rl)), marker='o', label='RL')
        plt.plot(np.log(v.shape[0]/np.array(commonDiv)), np.log(100*np.array(lr)), marker='s', label='LR')
        plt.plot(np.log(v.shape[0]/np.array(commonDiv)), np.log(100*np.array(tb)), marker='^', label='TB')
        plt.plot(np.log(v.shape[0]/np.array(commonDiv)), np.log(100*np.array(bt)), marker='D', label='BT')
        plt.plot(np.log(v.shape[0]/np.array(commonDiv)), np.log(100*np.array(fb)), marker='x', label='FB')
        plt.plot(np.log(v.shape[0]/np.array(commonDiv)), np.log(100*np.array(bf)), marker='*', label='BF')
        # Add title and axis labels
        plt.title('Log-Log Plot of Succolarity')
        plt.ylabel('Log (100 * Succolarity)')
        plt.xlabel('Log (d)')

        # Add legend to the right of the plot
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        # Adjust the layout to accommodate the legend
        plt.subplots_adjust(right=0.7)  # You might need to adjust this value

        # Show the plot
        plt.show()
    return commonDiv, tb, bt, lr, rl, fb, bf

def plot_2d_slice(array_3d, slice_index, title):
    """
    Plots a 2D slice from a 3D array.

    Parameters:
    array_3d (numpy.ndarray): The 3D array from which to take the slice.
    slice_index (int): The index of the slice to plot.
    title (str): Title for the plot.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(array_3d[:, :, slice_index], cmap='viridis')
    plt.colorbar()
    plt.title(title)
    plt.show()

###############
# 3D Succolarity
#################
start_time = time.time()
file_path = r"D:\BaiduSyncdisk\Research\Succolarity\Bentheimer sandstone\Water-wet\Oil\Resample\fw_0.05_Oil_512.raw"
volumeOne = np.fromfile(file_path, dtype=np.uint8)
volumeOne = volumeOne.reshape(512,512,512)
# volumeOne = 1*(volumeOne==0)
# volumeOne = volumeOne[:100,:100,:100]

#Now for multiple window sizes
sizesToBeUsed = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# SMult3D = computeSuccolaritySlidingWindow3D(volumeOne,sizesToBeUsed, plot=True) #initial volume is to be used not the connected components

# Extracting the results
commonDiv, tb_results, bt_results, lr_results, rl_results, fb_results, bf_results = computeSuccolaritySlidingWindow3D(volumeOne, sizesToBeUsed, plot=True)

# Preparing the data
data = {
    'Sizes': commonDiv,
    'Top to Bottom': tb_results,
    'Bottom to Top': bt_results,
    'Left to Right': lr_results,
    'Right to Left': rl_results,
    'Front to Back': fb_results,
    'Back to Front': bf_results
}

# Creating a DataFrame
df = pd.DataFrame(data)

# Saving to Excel
excel_file_path = r"D:\BaiduSyncdisk\Research\Succolarity\Bentheimer sandstone\Water-wet\Oil\Resample\fw_0.05_OilSu1.xlsx"
df.to_excel(excel_file_path, index=False)

end_time = time.time()

print(f"Execution time: {end_time - start_time} seconds")