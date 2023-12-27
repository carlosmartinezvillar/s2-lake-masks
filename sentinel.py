import rasterio as rio
# import geopandas as gpd
import matplotlib.pyplot as plt
# import pandas as pd
import numpy as np
# import cv2 as cv
import os
import xml.etree.ElementTree as ET
import time
import sys
from rasterio.windows import Window

################################################################################
# TYPING
################################################################################
from typing import Tuple, List
ndarray = np.ndarray #quick fix...

################################################################################
# GLOBAL VARIABLES
################################################################################
ns = {
	'n1':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd",
	'other':"http://www.w3.org/2001/XMLSchema-instance",
	'another':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"
	}

plt.style.use('fast')
DATA_DIR  = './dat/'  #<---- change this to argparse? maybe env
TILE_SIZE = 256

################################################################################
# FUNCTIONS
################################################################################
def parse_xml(path: str) -> Tuple[str, int, List[int]]:
	"""
	Get the datastrip id, band offset, and band quantification value from the
	xml metadata file found in path.

	Parameters
	----------
	path : str
		The string path to the xml file.

	Returns
	-------
	datastrip_id: str
		The extracted datastrip id
	offsets: List[int]
		Bottom-of-atmosphere offsets to shift values in the corresponding in a 
		raster by band.
	quant_val: int
		Product quantification value, meaning the correct divisor for all bands 
		to normalize them.

	Details
	-------
	xml file is organized like this:
	
	<n1:General_Info>
		<Product_Info>
			<Product_Organisation>
				<Granule_List>
					<Granule datastripIdentifier="" granuleIdentifier="">

	Hence, we parse with

	root.find('n1:General_Info',ns)
		.find('Product_Info')
			.find('Product_Organisation')
				.find('Granule_List')
					.find('Granule').attrib['datastripIdentifier']

	"""

	#fail if wrong path
	assert os.path.isfile(path), "No file found in path %s" % path

	root      = ET.parse(path).getroot()
	prod_info = root.find('n1:General_Info',namespaces=ns).find('Product_Info')
	granule   = prod_info.find('Product_Organisation').find('Granule_List').find('Granule')
	# granule   = [e for e in prod_info.iter('Granule')][0]x
	datastrip = granule.attrib['datastripIdentifier'].split('_')[-2][1:]
	

	### ADD BAND OFFSET CHECK AND RETURN IF EXISTS !!!
	'''
	<Product_Image_Characteristics>
		<QUANTIFICATION_VALUES LIST>
			<BOA_QUANTIFICATION_VALUE>
		<Reflectance_Conversion>
		....

	<Product_Image_Characteristics>
		<QUANTIFICATION_VALUES LIST>
			<BOA_QUANTIFICATION_VALUE>
		<BOA_ADD_OFFSET_VALUES_LIST>
			<BOA_ADD_OFFSET band_id="0">-1000</BOA_ADD_OFFSET>
			<BOA_ADD_OFFSET band_id="1">-1000</BOA_ADD_OFFSET>
			<BOA_ADD_OFFSET band_id="2">-1000</BOA_ADD_OFFSET>
			...
			<BOA_ADD_OFFSET band_id="12">-1000</BOA_ADD_OFFSET>			
		<Reflectance_Conversion>
		....

	something like:
	'''
	prod_char = root.find('n1:General_Info',namespaces=ns).find('Product_Image_Characteristics')
	boa_quant = prod_char.find('QUANTIFICATION_VALUES_LIST').find('BOA_QUANTIFICATION_VALUE') #get item
	quant_val = int(boa_quant.text)

	boa_add = prod_char.find('BOA_ADD_OFFSET_VALUES_LIST') #None | Element 
	if boa_add is not None:
		offsets = [int(e.text) for e in boa_add[1:4]] + [int(boa_add[7].text)]
	else:
		offsets = [0]*4

	return datastrip,quant_val,offsets


def get_gee_id(s2_img_id: str, datastrip: str) -> str:
	'''
	Read a Sentinel-2 image id string and returns the respective DynamicWorld product id.
	'''
	date,tile = s2_img_id.split('_')[2:6:3]
	gee_id = '_'.join([date,datastrip,tile])
	return gee_id


def get_band_file_path(s2_img_id: str, band: str) -> str:
	'''
	Given a Sentinel-2 product name and band, return the band image file name.
	'''
	date = s2_img_id[11:26]
	tile = s2_img_id[38:44]
	return '_'.join([tile, date, band, '10m.jp2'])


def get_band_file_paths(s2_img_id: str, bands: [str]) -> [str]:
	'''
	Given a Sentinel-2 product name and bands, return all band image file names.
	'''
	return list(map(get_band_file,[s2_img_id]*len(bands),bands))


def remove_borders_as_array(dw_array):
	'''
	Take the ndarray of a dynamic world image as input and return a copy without its zero-valued 
	borders, and a dictionary with the indices of the first and last row and first and last column 
	in the array without the borders.
	'''
	top,bottom,left,right = 0,dw_array.shape[0]-1,0,dw_array.shape[1]-1

	while(dw_array[top,:].sum() == 0):
		top += 1

	while(dw_array[bottom,:].sum() == 0):
		bottom -= 1

	while(dw_array[:,left].sum() == 0):
		left += 1

	while(dw_array[:,right].sum() == 0):
		right -= 1

	return dw_array[top:bottom+1,left:right+1],{'top':top,'bottom':bottom,'left':left,'right':right}


def remove_borders(src):
	'''
	Take a rasterio reader object for a dynamicworld image and get the indices 
	of the first non-zero values counting from the top, bottom, left, and 
	right.
	'''
	top,bottom,left,right = 0,src.height-1,0,src.width-1

	while(True):
		row = src.read(1,window=rio.windows.Window(0,top,src.width,1))
		if row.sum() == 0:
			top += 1
		else:
			break

	while(True):
		row = src.read(1,window=rio.windows.Window(0,bottom,src.width,1))
		if row.sum() == 0:
			bottom -= 1
		else:
			break

	while(True):
		col = src.read(1,window=rio.windows.Window(left,0,1,src.height))
		if col.sum() == 0:
			left += 1
		else:
			break

	while(True):
		col = src.read(1,window=rio.windows.Window(right,0,1,src.height))
		if col.sum() == 0:
			right -= 1
		else:
			break

	return {'top':top, 'bottom':bottom, 'left':left, 'right':right}


def dw_idx_to_s2(dw,s2,idx_dict):
	# s2 indexes:
	# (18, 3391) (10959, 3391) (10959, 10958) (18, 10958) before cleaning
	# (19, 3391) (10958, 3391) (10958, 10958) (19, 10958) after
	dw_xy_ul = dw.xy(idx_dict['top'],idx_dict['left'],offset='center')
	dw_xy_ll = dw.xy(idx_dict['bottom'],idx_dict['left'],offset='center')
	dw_xy_lr = dw.xy(idx_dict['bottom'],idx_dict['right'],offset='center')
	dw_xy_ur = dw.xy(idx_dict['top'],idx_dict['right'],offset='center')
	s2_idx_ul = s2.index(dw_xy_ul[0],dw_xy_ul[1],op=math.floor)
	s2_idx_ll = s2.index(dw_xy_ll[0],dw_xy_ll[1],op=math.floor)
	s2_idx_lr = s2.index(dw_xy_lr[0],dw_xy_lr[1],op=math.floor)
	s2_idx_ur = s2.index(dw_xy_ur[0],dw_xy_ur[1],op=math.floor)

	#return corners
	return s2_idx_ul,s2_idx_ll,s2_idx_lr,s2_idx_ur


def get_windows(s2_rdr, dw_rdr, borders):
	'''
	Given a set of starting and stopping boundaries, returns a list of block 
	indices i,j and window objects to pass to a rasterio DatasetReader.

	order in window object: Window(col_off,row_off,width,height)
	'''

	n_rows = borders['bottom'] + 1 - borders['top']
	n_cols = borders['right'] + 1 - borders['left']

	block_rows = n_rows // TILE_SIZE
	block_cols = n_cols // TILE_SIZE

	N = block_rows*block_cols

	W = []

	for k in N:
		block_i = k // block_cols
		block_j = k % block_cols

		row_start = block_i * TILE_SIZE + borders['top']
		row_stop  = row_start + TILE_SIZE
		col_start = block_j * TILE_SIZE + borders['left']
		col_stop  = col_start + TILE_SIZE

		W += [((block_i,block_j),Window(col_start,row_start,TILE_SIZE,TILE_SIZE))]
		# W += [Window.from_slices((row_start,row_stop),(col_start,col_stop))]

	result = []
	return result


def chip_image_cpu(img: ndarray, windows: List[Tuple]) -> ndarray:
	"""
	Parameters
	----------
	img: numpy.ndarray
		Input raster image.
	chp_size: int
		The size of the chips. The resulting images will be of size chp_size x 
		chp_size.

	Returns
	-------
	result: numpy.ndarray
		An array with dimensions (N,chp_size,chp_size) containing the resulting 
		images, where N is the number of resulting chips from the image.
	"""
	return


def chip_image_gpu(img: ndarray, chp_size: int=256) -> ndarray:
	pass


def upsample_mask(mask: ndarray) -> ndarray:
	"""
	Duplicate the size of the array containing the Sentinel-2 SCL 20m masks.
	"""
	# t_mask = torch.tensor(mask) ----> type error here
	# return torch.nn.functional.upsample(t_mask,scale_factor=2,mode='nearest')
	return np.repeat(np.repeat(mask,2,axis=0),2,axis=1)


################################################################################
# BAND ARITHMETIC
################################################################################
def clip_tail(img: ndarray, bottom: int=1, top: int=99) -> ndarray:
	"""
	Remove the values below the 'bottom' and  above 'top' percent in an image.

	Parameters
	----------
	img: numpy.ndarray	
		The raster image.
	bottom: int
		Bottom amount to be removed
	top: int

	Returns
	-------
	result: numpy.ndarray
	"""

	#input check
	try:
		assert bottom < 100 and bottom >= 0, \
			"'bottom' must be between 0 and 99 inclusive."
		assert top <= 100 and top > 0, \
			"Int 'top' must be between 1 and 100 inclusive."
		assert top > bottom, \
			"Upper boundary 'top' must be greater than 'bottom'."
	except AssertionError as e:
		print("In clip_tail():")
		raise AssertionError from e


def unit_normalize(img: ndarray, by_band: bool=True) -> ndarray:
	"""
	Unit-normalize a set of bands individually, or across all bands.
	
	Parameters
	----------
	img: numpy.ndarray
		A 1-band or 3-band raster image with the first axis being the bands.
	by_band: bool
		Boolean flag for the type of normalization. If false a single pair of 
		min and max values for all bands is used to normalize all bands. If 
		true, the min and max values in each band are used to normalize the 
		corresponding bands.

	Returns
	-------
	result: numpy.ndarray
		The normalized image

	"""
	if by_band is False:
		return (img - img.min()) / (img.max() - img.min())

	else:
		B_n  = img.shape[0] #assuming bands is outermost axis
		mins = np.array([band.min() for band in img]).reshape((B_n,1,1))
		maxs = np.array([band.max() for band in img]).reshape((B_n,1,1))
		return (img - mins) / (maxs - mins)


def calculate_ndwi(b3: ndarray, b8: ndarray) -> ndarray:
	"""
	Parameters
	----------
	b3 : ndarray
		Green band.

	b8: ndarray
		NIR band.

	Returns
	-------
	result: ndarray
		A single band ndarray with the calculated NDWI with values in the range 
		(-1,1).
	"""
	result = (b3-b8)/(b3+b8)
	return result


def plot_ndwi(b3,b4,b8):
	#NIR->R, R->G, G->B -- colour ir for plotting.
	pass

################################################################################
# PLOTTING, HISTOGRAMS, ET CETERA
################################################################################
def plot_img(path: str, img: ndarray, lib: str='opencv') -> None:
	if lib == 'opencv':
		#order is BGR, [M,N,Chans]
		cv.imwrite(path,img)

	elif lib == 'pil':
		if len(img.shape) == 2:
			Image.fromarray(np.uint8(unit_normalize(img)*255)).save(path)

	elif lib == 'pyplot':
		if len(img.shape) == 2:
			plt.imsave(path,img,cmap=Greys)
		if len(img.shape) == 3:
			plt.imsave(path,img[:,:,::-1]) #flip BGR to RGB

	else:
		print("Please specify a library for plot_img().")


def single_hist(path: str, band: ndarray, title: str, n_bins: int) -> None:
	fig,ax = plt.subplots()
	ax.set_title(title)
	ax.hist(band.flatten(),bins=n_bins)
	pass
	plt.savefig(path)


def multip_hist(path: str, img: ndarray, title: str, subtitle: List[str], n_bins: int) -> None:
	fig, axs = plt.subplots(nrows=1,ncols=bands.shape[0],sharey=True,tight_layout=True)
	fig.suptitle(title)
	colors = ['r','g','b','darkred']
	for i in range(img.shape[0]):
		axs[i].hist(img[i].flatten(),bins=n_bins)
		axs[i].set_title(subtitle[i])
	plt.savefig(path)


################################################################################
# MAIN
################################################################################

if __name__ == '__main__':

	#.SAFE folders in data directory
	folders  = [d for d in os.listdir(DATA_DIR) if d[-5:]=='.SAFE']
	
	# A SINGLE FOLDER
	safe_dir = folders[0]
	xml_file = [f for f in os.listdir(DATA_DIR + safe_dir) if f[-4:]=='.xml'][0] #bc MTD, MTD_L2A
	xml_path = DATA_DIR + '/'.join([safe_dir,xml_file])

	datastrip,quant_val,offset_vals = parse_xml(xml_path)

	gee_id = get_gee_id(safe_dir,datastrip)
	gee_path = DATA_DIR + '/'.join(["dynamicworld",gee_id]) + '.tif'

	b02_path = get_band_file_path(safe_dir,'B02')	

	# b02 = rio.open(b02_path,'r',tiled=True,blockxsize=256,blockysize=256)
	lbl = rio.open(gee_path,'r',tiled=True,blockxsize=256,blockysize=256)


	# # indices of first and last non-zero pixels in each direction in dw
	idx_dict = remove_borders(lbl)
	print(idx_dict)


	# # indices in s2 img of index dict from dw
	# ul,ll,lr,ur = dw_idx_to_s2(dw,s2,idx_dict)
	# print(ul,ll,lr,ur)

	# #Some lost notes...
	# b02 = b02.squeeze(axis=0)
	# bgr = np.moveaxis(np.stack((b02,b03,b04)),0,-1)
	# cv.imwrite('bgr_eq.png',bgr)
	# # RIGHT SHIFT -- np.right_shift() bitwise
	#RGB COMPOSITE--np.concatenate([b04,b03,b02],axis=0)
