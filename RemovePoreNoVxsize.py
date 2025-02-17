# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:29:09 2024

@author: 17154
"""

import numpy as np
import matplotlib.pyplot as plt
import porespy as ps
import pandas as pd
import os
import time
from scipy.stats import gaussian_kde

def read_raw_file(file_path, shape, dtype=np.uint8):
    data = np.fromfile(file_path, dtype=dtype)
    return data.reshape(shape)

def remove_pores_in_diameter_range(DiameterRange, A0All, dimensions=3):
    min_diameter, max_diameter = DiameterRange
    idx = (A0All.network['pore.equivalent_diameter'] >= min_diameter) & \
          (A0All.network['pore.equivalent_diameter'] <= max_diameter)
    regionLabels = A0All.network['pore.region_label'][idx]
    
    if dimensions == 3:
        copyToRemove = A0All.regions[3:-3, 3:-3, 3:-3].copy()
    else:
        copyToRemove = A0All.regions[3:-3, 3:-3].copy()
    
    mask = np.isin(copyToRemove, regionLabels)
    copyToRemove[mask] = 0
    copyToRemove = 1 * (copyToRemove != 0)
    
    return copyToRemove

def plot_and_save_distribution(data, original_data, sample_name, output_directory):
    if len(data) < 2:
        print(f"Not enough data points to perform KDE for {sample_name}.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    kde = gaussian_kde(data)
    x_pdf = np.linspace(data.min(), data.max(), 1000)
    pdf = kde(x_pdf)
    axes[0].plot(x_pdf, pdf)
    axes[0].set_title(f'Probability Density Function - {sample_name}')
    axes[0].set_xlabel('Pore Diameter (voxels)')
    axes[0].set_ylabel('Density')

    sorted_data = np.sort(data)
    cdf = np.arange(len(sorted_data)) / float(len(sorted_data))
    axes[1].plot(sorted_data, cdf)
    axes[1].set_title(f'Cumulative Distribution Function - {sample_name}')
    axes[1].set_xlabel('Pore Diameter (voxels)')
    axes[1].set_ylabel('Cumulative Probability')

    hist, bin_edges = np.histogram(data, bins=20, density=True)
    axes[2].hist(data, bins=20, edgecolor='black', density=True)
    axes[2].set_title(f'Histogram - {sample_name}')
    axes[2].set_xlabel('Pore Diameter (voxels)')
    axes[2].set_ylabel('Frequency Density')

    fig_path = os.path.join(output_directory, f'{sample_name}_pore_distribution_plots.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.show()

    # Save the data to an Excel file with separate sheets
    excel_path = os.path.join(output_directory, f'{sample_name}_distribution_data.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        # Save PDF data
        pdf_data = pd.DataFrame({'Pore Diameter (voxels)': x_pdf, 'Density': pdf})
        pdf_data.to_excel(writer, sheet_name='PDF', index=False)
        
        # Save CDF data
        cdf_data = pd.DataFrame({'Pore Diameter (voxels)': sorted_data, 'Cumulative Probability': cdf})
        cdf_data.to_excel(writer, sheet_name='CDF', index=False)
        
        # Save Histogram data
        hist_data = pd.DataFrame({'Bin Center (voxels)': (bin_edges[:-1] + bin_edges[1:]) / 2, 'Frequency Density': hist})
        hist_data.to_excel(writer, sheet_name='Histogram', index=False)
        
        # Save original pore size distribution data
        original_data_df = pd.DataFrame({'Original Pore Diameter (voxels)': original_data})
        original_data_df.to_excel(writer, sheet_name='Original Data', index=False)

def plot_slices(original_slice, removed_slice, sample_name, DiameterRange, output_directory):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].imshow(original_slice, cmap='binary')
    axes[0].set_title(f'Original Slice - {sample_name}')

    axes[1].imshow(removed_slice, cmap='binary')
    axes[1].set_title(f'Removed {DiameterRange[0]}-{DiameterRange[1]} voxels - {sample_name}')

    fig_path = os.path.join(output_directory, f'{sample_name}_comparison_slices_{DiameterRange[0]}-{DiameterRange[1]}.png')
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300)
    plt.show()

def main():
    start_time = time.time()
    
    raw_file_path = r"D:\BaiduSyncdisk\Research\Succolarity\Bentheimer sandstone\Mixed-wet\Oil\Resample\fw_0.02_Oil_512.raw"
    output_directory = r"D:\BaiduSyncdisk\Research\Succolarity\Bentheimer sandstone\Mixed-wet\Oil\Resample\fw0.02RPO"
    
    image_shape = (512, 512, 512)
    resolution = 1
    
    try:
        imgMask = read_raw_file(raw_file_path, image_shape, dtype=np.uint8)
    except FileNotFoundError:
        print(f"File not found: {raw_file_path}")
        return

    A0 = imgMask.astype(np.uint8)
    A0All = ps.networks.snow2(A0, voxel_size=resolution)

    DiameterRange = (0, 34)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    removed = remove_pores_in_diameter_range(DiameterRange, A0All, dimensions=3)

    sample_name = os.path.basename(raw_file_path).replace('.raw', '')
    modified_volume_path = os.path.join(output_directory, f"{sample_name}_{DiameterRange[0]}-{DiameterRange[1]}.raw")
    removed.astype(np.uint8).tofile(modified_volume_path)

    # Filter out the removed pore diameters from the original data
    pore_diameters = A0All.network['pore.equivalent_diameter']
    remaining_pore_diameters = pore_diameters[pore_diameters > DiameterRange[1]]

    # Verify removal of specified pores
    if np.all(remaining_pore_diameters >= DiameterRange[1]):
        print(f"Successfully removed all pores smaller than {DiameterRange[1]} voxels.")
    else:
        print(f"There are still pores smaller than {DiameterRange[1]} voxels.")

    # Plot and save the distribution after removal
    plot_and_save_distribution(remaining_pore_diameters, pore_diameters, f"{sample_name}_{DiameterRange[0]}-{DiameterRange[1]}Vx", output_directory)

    plot_slices(A0[256, :, :], A0[256, :, :]*2-removed[256, :, :], sample_name, DiameterRange, output_directory)

    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()

