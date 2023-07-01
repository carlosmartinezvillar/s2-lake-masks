import rasterio as rio
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv

################################################################################
# UTILITY FUNCs
################################################################################
def parse_xml(path: str) -> Tuple[str, list, int]:
	"""
	Get the datastrip id, band offset, and band quantification value from the xml metadata file 
	found in path.

	Parameters
	----------
	path : str
		The path to the xml file.

	Returns
	-------
	datastrip_id: str
		Extracted datastrip id

	offset_value: list
		Bottom-of-atmosphere offsets to shift all values in the corresponding bands in a raster.

	quant_value: int
		Product quantification value, meaning the correct divisor for all bands to normalize them.
	"""
	path = DATA_DIR + row[1] + '/MTD.xml'
	print("--> Parsing %s..." % path)

	# if this throws AssertError something's very wrong...
	assert os.path.isfile(path), "No file found in path %s" % path

	root         = ET.parse(path).getroot()
	gral_info    = root.find('n1:General_Info',MTD_NS)
	product_info = gral_info.find('Product_Info')
	product_char = gral_info.find('Product_Image_Characteristics')
	#INSIDE TAG <Product_Info> -- ALWAYS in XML
	granule_attrib = list(product_info.iter('Granule'))[0].attrib
	datastrip      = granule_attrib['datastripIdentifier']
	granule        = granule_attrib['granuleIdentifier']

	return datastrip


################################################################################
# BAND ARITHMETIC
################################################################################
def clip_tail(img: numpy.ndarray, bottom: int=1,top: int=99) -> numpy.ndarray:
	'''
	Removes the data in img whose values are below the 'bottom' percent and above 'top' percent.

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

	'''
	#input check
	try:
		assert bottom < 100 and bottom >= 0, "Int 'bottom' must be between 0 and 99 inclusive."
		assert top <= 100 and top > 0, "Int 'top' must be between 1 and 100 inclusive."
		assert top > bottom, "Upper boundary 'top' must be greater than 'bottom'."
	except AssertionError as e:
		print("In clip_tail():")
		raise AssertionError from e


def unit_normalize(b: numpy.ndarray) -> numpy.ndarray:
	"""
	Unit-normalize a set of bands individually or across all bands.
	
	Parameters
	----------
	

	Returns
	-------

	"""
	return (b - b.min())/(b.max()-b.min())


################################################################################
# PLOTTING, HISTOGRAMS, ET CETERA
################################################################################
def hist_eq(img,bpp):
	L = 2**bpp
	M = img.shape[0]
	N = img.shape[1]
	u,i,c = np.unique(img,return_inverse=True,return_counts=True)
	print(u.shape,i.shape,c.shape)
	cdf   = np.cumsum(c)

	#Method 1
	new_img = cdf[i].reshape((M,N))
	new_img = np.uint16((new_img-cdf.min())/(M*N-1) * (L-1))
	
	#Method 2
	# p = cdf/(M*N)
	# new_img = p[i] * (L-1)
	
	#Return array of size MxN with equalized image
	return new_img


def plot_img(filename,img,lib='opencv'):
	if lib == 'opencv':
		#order is BGR, [M,N,Chans]
		cv.imwrite(filename,img)
	elif lib == 'pil':
		if len(img.shape) == 2:
			img_8bit = np.uint8(unit_norm(img)*255) #floor
			Image.fromarray(img_8bit).save(filename)
	elif lib == 'pyplot':
		if len(img.shape) == 2:
			plt.imsave(filename,img,cmap=Greys)
		if len(img.shape) == 3:
			plt.imsave(filename,img[:,:,::-1]) #flip BGR to RGB
	else:
		print("Please specify a library for plot_img().")


def plot_single_hist(band):
	fig,axs = p
	pass


def plot_multi_hist():
	pass

####################################################################################################
# MAIN
####################################################################################################

if __name__ == '__main__':
	#OPEN IMAGE POINTER
	new_b02 = rio.open(NEW_B02_PATH)
	new_b03 = rio.open(NEW_B03_PATH)
	new_b04 = rio.open(NEW_B04_PATH)
	new_b8a = rio.open(NEW_B8A_PATH)
	new_tci = rio.open(NEW_TCI_PATH)

	#LOAD IMAGE
	b02 = new_b02.read()
	b03 = new_b03.read()
	b04 = new_b04.read()
	b8a = new_b8a.read()
	tci = new_tci.read()

	b02 = b02.squeeze(axis=0)
	b03 = b03.squeeze(axis=0)
	b04 = b04.squeeze(axis=0)
	b8a = b8a.squeeze(axis=0)
	#UNIT NORM AND STRETCH TO BPP
	bpp = 16
	L   = 2**bpp
	b02,b03,b04,b8a = [np.uint16(unit_norm(i)*(L-1)) for i in [b02,b03,b04,b8a]]

	#EQUALIZE
	b02,b03,b04 = [hist_eq(_) for _ in (b02,b03,b04)]

	bgr = np.moveaxis(np.stack((b02,b03,b04)),0,-1)
	cv.imwrite('bgr_eq.png',bgr)

	#STRETCH

	b02_stretched = np.uint16( unit_norm(b03) * (L-1) )
	b03_stretched = np.uint16( unit_norm(b03) * (L-1) )
	b04_stretched = np.uint16( unit_norm(b04) * (L-1) )
	b8a_stretched = np.uint16( unit_norm(b8a) * (L-1) )

	# bins = 2**bpp
	# fig, axs = plt.subplots(1,4,sharey=True,tight_layout=True)
	# axs[0].hist(b04_stretched.flatten(),color='r',bins=bins)
	# axs[1].hist(b03_stretched.flatten(),color='g',bins=bins)
	# axs[2].hist(b02_stretched.flatten(),color='b',bins=bins)
	# axs[3].hist(b8a_stretched.flatten(),color='r',bins=bins)
	# plt.title('Histogram of Bands - Raw Images, Stretch, bins=%i' % bins)
	# plt.savefig('./figures/band_hist_0.png')

	#RIGHT SHIFT
	b02_shifted = np.right_shift(b02_stretched,4) #to 12-bits
	b03_shifted = np.right_shift(b03_stretched,4)
	b04_shifted = np.right_shift(b04_stretched,4)
	b8a_shifted = np.right_shift(b8a_stretched,4)

	bins = 2**12
	fig, axs = plt.subplots(1,4,sharey=True,tight_layout=True,figsize=(20,5))
	axs[0].hist(b04_shifted.flatten(),color='r',bins=bins)
	axs[1].hist(b03_shifted.flatten(),color='g',bins=bins)
	axs[2].hist(b02_shifted.flatten(),color='b',bins=bins)
	axs[3].hist(b8a_shifted.flatten(),color='r',bins=bins)
	plt.title('Histogram of Bands - Raw Image, R Shift ,bins=%i' % bins)
	plt.savefig('./fig/band_hist_1.png')


	#NIR->R, R->G, G->B -- colour ir
	#(B3-B8)/(B3+B8) -- NDWI
	# ndwi = (b03_shifted - b8a_shifted)/(b03_shifted + b8a_shifted)
	# plt.figure()
	# plt.imshow(ndwi[0],cmap='RdBu')
	# plt.title("NDWI")
	# plt.savefig("./figures/ndwi.png")



	#RGB COMPOSITE
	# rgb  = np.concatenate([b04_shifted,b03_shifted,b02_shifted],axis=0)
	# plt.figure()
	# plt.imshow(rgb)
	# plt.title("RGB from Bands")
	# plt.savefig("./figures/rgb.png")
