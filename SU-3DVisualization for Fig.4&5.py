# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 19:34:21 2024

@author: 17154
"""
import numpy as np
import tifffile as tiff
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

def CutOriginal3D(volume, window_size=1):
    '''
    Produces an array of the correct size to be multiplied with un-paded convolution results for PR or Occupancy

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

start_time = time.time()
file_path = r"D:\17154\OneDrive - University of Leeds\Documents\Research\Succolarity\Test\3D-samples\CT-TC\TC-1.raw"
volumeOne = np.fromfile(file_path, dtype=np.uint8)
volumeOne = volumeOne.reshape(500,500,500)
volumeOne = 1*(volumeOne==0)
# volumeOne = volumeOne[:100,:100,:100]
#Step by step
window_size = 16
direction='rl'
#Connected components
conV = selectConnectedComponents3D( volumeOne,  direction=direction, con=2 )
ConVizualization = 1*(volumeOne!=0)+1*(conV!=0)
ToBeTransparent = 2*(volumeOne!=0)-2*(conV!=0)
NotTransparent = 1*(conV!=0)

#Occupancy
ow = slidingWindowOccupancy3D(conV, window_size)
#Virtual Pressure filed
PR = slidingWindowVirtualP3D(conV, direction=direction, window_size=window_size)

OCut = CutOriginal3D(conV)
#Multiplication
OWPR = ow*PR

Suc = computeSuccolarity3DSlidingWindowForSize( conV, window_size,  direction=direction, option = 'fftconv') 
print('Succolarity value for an array of size ', conV.shape, ' and window size ', window_size, ' direction ', direction, ' is: ', Suc)

#Now for multiple window sizes
sizesToBeUsed = [1, 2, 4, 8, 16, 32, 64, 128, 256]
# SMult3D = computeSuccolaritySlidingWindow3D(volumeOne,sizesToBeUsed, plot=True) #initial volume is to be used not the connected components

owpr_normalized = (OWPR - np.min(OWPR)) / (np.max(OWPR) - np.min(OWPR))
owpr_scaled = (owpr_normalized * 255).astype(np.uint8)

base_path = r"D:\17154\OneDrive - University of Leeds\Documents\Research\Succolarity\Test\3D-samples\CT-TC\Modified"

file_path_NTransP = fr"{base_path}\TC-{direction}-NTransP.tif"
NotTransparent = NotTransparent/np.max(NotTransparent)*255
tiff.imwrite(file_path_NTransP, NotTransparent.astype(np.uint8))

file_path_OCut = fr"{base_path}\TC-{direction}-Ocut.tif"
OCut = OCut/np.max(OCut)*255
tiff.imwrite(file_path_OCut, OCut.astype(np.uint8))

file_path_TransP = fr"{base_path}\TC-{direction}-TransP.tif"
ToBeTransparent = ToBeTransparent/np.max(ToBeTransparent)*255
tiff.imwrite(file_path_TransP, ToBeTransparent.astype(np.uint8))

file_path_con_Visual = fr"{base_path}\TC-{direction}-con_Visual.tif"
ConVizualization = ConVizualization/np.max(ConVizualization)*255
tiff.imwrite(file_path_con_Visual, ConVizualization.astype(np.uint8))

file_path_conV = fr"{base_path}\TC-{direction}-conV.tif"
file_path_PR = fr"{base_path}\TC-{direction}-PR.tif"
file_path_ow = fr"{base_path}\TC-{direction}16-OW.tif"
file_path_OWPR = fr"{base_path}\TC-{direction}16-OWPR.tif"


conV = conV/np.max(conV)*255
tiff.imwrite(file_path_conV, conV.astype(np.uint8))
PR = PR/np.max(PR)*255
tiff.imwrite(file_path_PR, PR.astype(np.uint8))
ow = ow/np.max(ow)*255
tiff.imwrite(file_path_ow, ow.astype(np.uint8))
OWPR = OWPR/np.max(OWPR)*255
tiff.imwrite(file_path_OWPR, OWPR.astype(np.uint8))
# tiff.imwrite( r"D:\17154\OneDrive - University of Leeds\Documents\Research\Succolarity\Test\3D-samples\CT-TC\MCT-TC-bf16-DST.tif", PR)
# tiff.imwrite( r"D:\17154\OneDrive - University of Leeds\Documents\Research\Succolarity\Test\3D-samples\CT-TC\MCT-TC-bf16-NOWPR.tif", owpr_normalized)