'''
THIS script produces a label with a 3rd category corresponding to shoreline, i.e. water pixels and
land neighboring the other class within a 8-pixel neighborhood.  Also, it produces a 3rd mask/label
with a distance to shoreline.
'''

import numpy as np
import cupy as cp
import os
from PIL import Image
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--chip-dir',required=True)
parser.add_argument('--out-dir',required=True)

############################################################
# FUNCTIONS
############################################################
# C KERNEL FOR SHORELINE
shoreline_kernel = cp.RawKernel(r'''
extern "C" __global__ void check_mismatch(const unsigned char* img, unsigned char* out, int width, int height, unsigned char threshold) {
	//Device indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Out of bounds
    if (x >= width || y >= height) return;
    
    // Index of pixel in flattened img array
    int idx = y * width + x;
    unsigned char pixel = img[idx];
    bool flag = false;
    
    // Iterate over 8 neighboring pixels
    for (int dy = -1; dy <= 1; ++dy){
        for (int dx = -1; dx <= 1; ++dx){
            if (dx == 0 && dy == 0) continue;
            // Index of neighbor in flattened img array
            int nx = x + dx;
            int ny = y + dy;
            if (nx >= 0 && nx < width && ny >= 0 && ny < height){
                int n_idx = ny * width + nx;
                if (img[n_idx]!=pixel){
                    flag = true;
                    goto wow_my_first_goto;
                }
            }
        }
    }
    
    wow_my_first_goto:
    out[idx] = flag ? 1 : 0;
}
''', 'check_mismatch')

#C KERNEL FOR 
distance_kernel = cp.RawKernel(r'''
extern "C" __global__ void find_nearest(const unsigned char* img, unsigned char* out, int width, int height){

    //Device indices
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Out of bounds
    if (x >= width || y >= height) return;

    // Index of this pixel in flattened img array
    int idx = y * width + x;
    unsigned char pixel = img[idx];

    //If shoreline set distance to zero 
    if (pixel==1){
        out[idx] = 0;
    //Else look for closest shore px
    }else{
        //Iterate through each of the pixel in shoreline set
        int N = width * height;
        int closest_idx = 0;
        for(int i=0;i<=N;++i){ 
            if (img[i] == 1){

            //check it is not the same pixel

            }else{
                continue;
            }
        }

    }


}
''','find_nearest')

def get_shoreline(img):
    """
    Applies a CuPy kernel to check if a pixel's neighbors belong to a different class
    
    Args:
        img (cupy.ndarray): Input 2D image as a CuPy array of type uint8.

    Returns:
        out (cupy.ndarray): Binary mask where 1 indicates a neighboring pixel meets condition
    """
    height, width = img.shape
    out = cp.zeros_like(img, dtype=cp.uint8)
    
    # Define thread/block configuration
    block_size = (16, 16)
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])
    
    # Launch kernel
    shoreline_kernel(grid_size, block_size, (img, out, width, height, threshold))
    return out


def get_distances(img):
    """
    Applies a CuPy kernel to find a pixel's distance to the nearest pixel set to 1 (shoreline).
    
    Args:
        img (cupy.ndarray): Input 2D image as a CuPy array of type uint8.
    
    Returns:
        out (cupy.ndarray): Binary mask where 1 indicates a neighboring pixel meets condition
    """    
    height, width = img.shape
    # out = cp.zeros_like(img, dtype=cp.uint8)
    out = cp.zeros_like(img, dtype=cp.float32)

    # Define thread/block configuration
    block_size = (16, 16)
    grid_size = ((width + block_size[0] - 1) // block_size[0],
                 (height + block_size[1] - 1) // block_size[1])

    g
    
    pass


############################################################
# MAIN
############################################################
if __name__ == '__main__':

	dir_path = args.chip_dir 
	chips = glob(f"{dir_path}/*_LBL.tif")

	# LOAD A CHIP
	idx = 0
	lbl = Image.open(f'{chips[idx]}')

	# TO NUMPY
	lbl = np.array(lbl)

	# NUMPY TO CUPY
	lbl_gpu = cp.asarray(lbl)

	# ESTIMATE SHORELINE
	shore_mask_gpu = get_shoreline(img)
	shore_mask_cpu = cp.asnumpy(shore_mask)

	# CHECK OUTPUT
	pass
	
	# SAVE .TIF

