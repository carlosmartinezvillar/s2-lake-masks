import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import os
# import glob

################################################################################
# TYPING
################################################################################
from typing import Tuple, List
ndarray = np.ndarray #quick fix...

################################################################################
# GLOBAL VARIABLES
################################################################################
plt.style.use('fast')
DATA_DIR  = './dat/'
TILE_SIZE = 512

################################################################################
# FUNCTIONS
################################################################################
def parse_xml(path: str) -> Tuple[str, List[int], int]:
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
	offset_value: List[int]
		Bottom-of-atmosphere offsets to shift values in the corresponding in a 
		raster by band.
	quant_value: int
		Product quantification value, meaning the correct divisor for all bands 
		to normalize them.
	"""
	path = DATA_DIR + row[1] + '/MTD.xml'
	print("--> Parsing %s..." % path)

	# if this throws AssertError something's very wrong...
	assert os.path.isfile(path), "No file found in path %s" % path

	root         = ET.parse(path).getroot()
	gral_info    = root.find('n1:General_Info',MTD_NS)
	product_info = gral_info.find('Product_Info')
	product_char = gral_info.find('Product_Image_Characteristics')
	#INSIDE TAG <Product_Info> -- ALWAYS in file
	granule_attrib = list(product_info.iter('Granule'))[0].attrib
	datastrip      = granule_attrib['datastripIdentifier']
	granule        = granule_attrib['granuleIdentifier']

	pass #----------------------------------------------------------> ADD PARSER

	return datastrip


def remove_borders(image, label):
	'''
	
	'''
	pass

def chip_image_cpu(img: ndarray, chp_size: int=512) -> ndarray:
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
	


def chip_image_gpu(img: ndarray, chp_size: int=512) -> ndarray:
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
	#NIR->R, R->G, G->B -- colour ir for plotting.
	result = (b3-b8)/(b3+b8)
	return result


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


def plot_single_hist(path: str, band: ndarray, title: str, n_bins: int) -> None:
	fig,ax = plt.subplots()
	ax.set_title(title)
	ax.hist(band.flatten(),bins=n_bins)
	pass
	plt.savefig(path)


def plot_multip_hist(path: str, img: ndarray, title: str, subtitle: List[str], n_bins: int) -> None:
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
	files = os.listdir(DATA_DIR + folders[0])
	bands = [b for b in files if b.endswith('10m.jp2')]
	scl   = '_'.join(bands[0].split('_')[0:2]) + "_SCL_20m.jp2"

	print(bands)
	pass

	# #OPEN IMAGE POINTER
	# new_b02 = rio.open(NEW_B02_PATH)
	# new_b03 = rio.open(NEW_B03_PATH)
	# new_b04 = rio.open(NEW_B04_PATH)
	# new_b8a = rio.open(NEW_B8A_PATH)
	# new_tci = rio.open(NEW_TCI_PATH)
	# #LOAD IMAGE
	# b02 = new_b02.read()
	# b03 = new_b03.read()
	# b04 = new_b04.read()
	# b8a = new_b8a.read()
	# tci = new_tci.read()
	# b02 = b02.squeeze(axis=0)
	# b03 = b03.squeeze(axis=0)
	# b04 = b04.squeeze(axis=0)
	# b8a = b8a.squeeze(axis=0)
	# bgr = np.moveaxis(np.stack((b02,b03,b04)),0,-1)
	# cv.imwrite('bgr_eq.png',bgr)
	# # RIGHT SHIFT -- np.right_shift() bitwise

	#RGB COMPOSITE--np.concatenate([b04,b03,b02],axis=0)
