# import cv2 as cv
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import xml.etree.ElementTree as ET
import time
import sys
import rasterio as rio
from rasterio.windows import Window

####################################################################################################
# TYPING
####################################################################################################
from typing import Tuple, List
ndarray = np.ndarray #quick fix...

####################################################################################################
# GLOBAL VARIABLES
####################################################################################################
ns = {
	'n1':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd",
	'other':"http://www.w3.org/2001/XMLSchema-instance",
	'another':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"
	}

plt.style.use('fast')


CHIP_SIZE = 256
WATER_MIN = 128*64
WATER_MAX = CHIP_SIZE*CHIP_SIZE-WATER_MIN

DATA_DIR = os.getenv('DATA_DIR')
if DATA_DIR is None:
	DATA_DIR = './dat'

LABEL_DIR = DATA_DIR + '/dynamicworld'
CHIP_DIR  = DATA_DIR + '/chp'

####################################################################################################
# BAND ARITHMETIC
####################################################################################################
def minmax_normalize(img: ndarray, by_band: bool=True) -> ndarray:
	"""
	Unit-normalize a set of bands individually, or across all bands, using min and max values.
	
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

####################################################################################################
# HISTOGRAMS, ET CETERA
####################################################################################################
def band_hist(path: str, band: ndarray, title: str, n_bins: int=625) -> None:
	fig,ax = plt.subplots()
	ax.set_title(title)
	ax.hist(band.flatten(),bins=n_bins)
	plt.savefig(path)
	print("Band plot saved to %s." % path)

def multiband_hist(path: str, img: ndarray, title: str, subtitle: List[str], n_bins: int) -> None:
	colors = ['r','g','b','darkred']
	fig, ax = plt.subplots(nrows=1,ncols=bands.shape[0],sharey=True,tight_layout=True)
	fig.suptitle(title)
	for i in range(img.shape[0]):
		ax[i].hist(img[i].flatten(),bins=n_bins)
		ax[i].set_title(subtitle[i])
	plt.savefig(path)

####################################################################################################
# PROCESSING STUFF
####################################################################################################
def parse_xml(path: str) -> Tuple[str, int, List[int]]:
	'''
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

	Hence, the datastrip is parsed with

	root.find('n1:General_Info',ns)
		.find('Product_Info')
			.find('Product_Organisation')
				.find('Granule_List')
					.find('Granule').attrib['datastripIdentifier']


	The structure of the the band quantification values inside the XML file 
	looks like this

	<Product_Image_Characteristics>
		<QUANTIFICATION_VALUES LIST>
			<BOA_QUANTIFICATION_VALUE>
		<Reflectance_Conversion>
		...

	Hence it is parsed with

	root.find('n1:General_Info',namespaces=ns)
		.find('Product_Image_Characteristics')
			.find('QUANTIFICATION_VALUES_LIST')
				.find('BOA_QUANTIFICATION_VALUE')


	And, the offsets look like this

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
		...

	And so they are parsed, by iterating on the result of

	root.find('n1:General_Info',namespaces=ns)
		.find('Product_Image_Characteristics')
			.find('BOA_ADD_OFFSET_VALUES_LIST')			

	'''

	# fail if wrong path
	assert os.path.isfile(path), "No file found in path %s" % path

	# get datastrip
	root      = ET.parse(path).getroot()
	prod_info = root.find('n1:General_Info',namespaces=ns).find('Product_Info')
	granule   = prod_info.find('Product_Organisation').find('Granule_List').find('Granule')
	# granule   = [e for e in prod_info.iter('Granule')][0]
	datastrip = granule.attrib['datastripIdentifier'].split('_')[-2][1:]

	# quantification values
	prod_char = root.find('n1:General_Info',namespaces=ns).find('Product_Image_Characteristics')
	boa_quant = prod_char.find('QUANTIFICATION_VALUES_LIST').find('BOA_QUANTIFICATION_VALUE')
	quant_val = int(boa_quant.text)

	# offset values
	boa_add = prod_char.find('BOA_ADD_OFFSET_VALUES_LIST') #None | Element 
	if boa_add is not None:
		offsets = [int(e.text) for e in boa_add[1:4]] + [int(boa_add[7].text)]
	else:
		offsets = [0]*4

	return datastrip,quant_val,offsets


def get_gee_id(s2_id: str, datastrip: str) -> str:
	'''
	Read a Sentinel-2 id string, and return the DynamicWorld id.
	'''
	date,tile = s2_id.split('_')[2:6:3]
	gee_id = '_'.join([date,datastrip,tile])
	return gee_id


def get_band_filename(s2_id: str, band: str) -> str:
	'''
	Given a Sentinel-2 product name and band, return the band image file name.
	'''
	# date = s2_id[11:26]
	# tile = s2_id[38:44]
	return '_'.join([s2_id[38:44],s2_id[11:26],band,'10m.jp2'])


def get_band_filenames(s2_img_id: str, bands: [str]) -> [str]:
	'''
	Given a Sentinel-2 product name and bands, return all band image file names.
	'''
	return list(map(get_band_filename,[s2_img_id]*len(bands),bands))


def remove_borders_as_array(dw_array: ndarray) -> ndarray:
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


def remove_borders(src: rio.DatasetReader) -> dict:
	'''
	Take a rasterio DatasetReader for a dynamicworld image and get the indices 
	where non-zeros begin at the top, bottom, left, and right.

	Parameters
	----------
	src: rasterio.DatasetReader
		Dataset reader for a dynamic world array (which has zeroes where S2
		still has data, so checking S2 is redundant).

	Returns
	-------
	dict
		dictionary with indices of first non-zero values at top, left, right, 
		bottom

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


def convert_bounds(dw,s2,dw_ij_dict):
	"""
	Takes DatasetReader objects for S2 and DynamicWorld images and returns a 
	dictionary for the row/col indices of the sentinel image where the non-zero
	data of the DynamicWorld image are.

	Parameters
	----------
	dw: rasterio.io.DatasetReader
		The reader for the DynamicWorld label
	s2: rasterio.io.DatasetReader
		The reader for the Sentinel-2 image
	dw_ij_dict: dict
		A dictionary of four numbers, each corresponding to the first and last 
		indices in both directions of the image containing data (without no
		data, or zeroes).

	Returns
	-------
	dict

	"""
	# s2 indexes:
	# (18, 3391) (10959, 3391) (10959, 10958) (18, 10958) before
	# (19, 3391) (10958, 3391) (10958, 10958) (19, 10958) after removing bounds
	dw_xy_ul = dw.xy(dw_ij_dict['top'],dw_ij_dict['left'],offset='center')
	dw_xy_lr = dw.xy(dw_ij_dict['bottom'],dw_ij_dict['right'],offset='center')
	s2_ij_ul = s2.index(dw_xy_ul[0],dw_xy_ul[1],op=math.floor)
	s2_ij_lr = s2.index(dw_xy_lr[0],dw_xy_lr[1],op=math.floor)

	#return dict of bounds
	return {'top':s2_ij_ul[0],'bottom':s2_ij_lr[0],
		'left':s2_ij_ul[1],'right':s2_ij_lr[1]}


def get_windows(borders: dict) -> [Tuple]:
	'''
	Given a set of starting and stopping boundaries, returns an array list with
	tuples (i,j) for block indices i,j and window objects corresponding to the
	block i,j while taking into consideration only the area of the raster
	within the boundaries defined by the indices in the borders dict. For exam-
	ple, if the array had two rows and a column of no data on the left and top, 
	the blocks would be offset and defined as:

			    left    256     512
				| 0 0 0 0 ..
				| 0 0 0 0 ... 	   
	    top ----+--------+--------+
		     0  |        |        |
		     0  | (0, 0) | (0, 1) |
		     .  |        |        |
		     .  +--------+--------+
		     .  |        |        |
		        | (1, 0) | (1, 1) |
		        |        |        |
		    512 +--------+--------+

	Parameters
	----------
	borders: dict
		The dictionary containing the first and last indices of usable data in
		both directions.
	'''

	# number of rows and cols takin' the boundaries into acct
	n_rows = borders['bottom'] + 1 - borders['top']
	n_cols = borders['right'] + 1 - borders['left']

	#nr of blocks in each direction
	block_rows = n_rows // CHIP_SIZE
	block_cols = n_cols // CHIP_SIZE
	
	#total blocks
	N = block_rows * block_cols

	windows = []

	for k in range(N):
		i = k // block_cols
		j = k % block_cols
		row_start = i * CHIP_SIZE + borders['top']
		# row_stop  = row_start + CHIP_SIZE
		col_start = j * CHIP_SIZE + borders['left']
		# col_stop  = col_start + CHIP_SIZE

		W = Window(col_start,row_start,CHIP_SIZE,CHIP_SIZE)
		# W += [Window.from_slices((row_start,row_stop),(col_start,col_stop))]
		windows += [[(str(i),str(j)),W]]

	return windows


def plot_label_multiclass(path,dw_reader,borders):
	"""
	Plots the whole label raster with nodata borders removed.
	"""
	DW_NAMES_10 = ['masked','water','trees','grass','flooded_vegetation',
		'crops','shrub_and_scrub','built','bare','snow_and_ice'];

	DW_PALETTE_10 = ['000000','419bdf','397d49','88b053','7a87c6','e49635', 
	    'dfc35a','c4281b','a59b8f','b39fe1'];


	# number of rows and cols takin' the boundaries into acct
	h = borders['bottom'] + 1 - borders['top']
	w = borders['right'] + 1 - borders['left']
	read_window = Window(borders['left'],borders['top'],w,h)

	#read, 1-band to 3-band, plot colors
	dw = dw_reader.read(1,window=read_window)
	dw3 = np.stack([dw,np.zeros_like(dw),np.zeros_like(dw)],axis=0).astype(np.uint8)
	for i in range(10):
		r = int(DW_PALETTE_10[i][0:2],16) #hex to int
		g = int(DW_PALETTE_10[i][2:4],16)
		b = int(DW_PALETTE_10[i][4:6],16)
		dw3[:,dw==i] = [[r],[g],[b]]

	# writer parameters
	kwargs = dw_reader.meta.copy()
	kwargs.update({'height':read_window.height,'width':read_window.width,'count':3,
		'compress':'lzw'})

	#write to path
	dst = rio.open(path,'w',**kwargs)
	dst.write(dw3,indexes=[1,2,3])
	dst.close()


def plot_label_multiclass_windowed(path,dw_reader,borders,windows):
	'''
	Plots the whole label raster using windows. Data beyond the windows is not plotted.
	'''
	DW_PALETTE_10 = ['000000','419bdf','397d49','88b053','7a87c6','e49635', 
	    'dfc35a','c4281b','a59b8f','b39fe1'];

	#height
	height = borders['bottom'] + 1 - borders['top']
	width  = borders['right'] + 1 - borders['left']
	height = height - (height % CHIP_SIZE) + 1 #not sure if safe
	width = width - (width % CHIP_SIZE) + 1

	#writer config
	kwargs = dw_reader.meta.copy()
	kwargs.update({'height':height,'width':width,'count':3,'compress':'lzw'})
	dst = rio.open(path,'w',**kwargs)

	for _,w in windows:

		# change array values -- single to 3-band 10-class
		arr1 = dw_reader.read(1,window=w)
		arr3 = np.stack([arr1,np.zeros_like(arr1),np.zeros_like(arr1)],axis=0)
		for c in range(10):
			r = int(DW_PALETTE_10[c][0:2],16)
			g = int(DW_PALETTE_10[c][2:4],16)
			b = int(DW_PALETTE_10[c][4:6],16)
			arr3[:,arr1==c] = [[r],[g],[b]]

		#write window in dst image
		dst.write(arr3,window=w,indexes=[1,2,3])

	dst.close()


def plot_checkerboard(path,dw_reader,borders,windows):
	"""
	Plots whole label raster with overlaying squares corresponding to
	the blocks/windows used as chips. Red squares are chips discarded, yellow
	squares are chips kept.
	"""
	DW_PALETTE_10 = ['000000','419bdf','397d49','88b053','7a87c6','e49635', 
	    'dfc35a','c4281b','a59b8f','b39fe1'];

	#Get lines -- red FF0000, yellow FFFF00
	yellow_line      = np.ones((3,CHIP_SIZE))
	yellow_line[0:2] = 65535
	yellow_line[2]   = 0
	red_line         = np.ones((3,CHIP_SIZE))
	red_line[0]      = 65535
	red_line[1:2]    = 0

	#NEW SIZE OF 
	height = borders['bottom'] - borders['top'] + 1
	width  = borders['right'] - borders['left'] + 1

	#WRITER
	kwargs = dw_reader.meta.copy()
	kwargs.update({
		'count':3,'compress':'lzw','height':height,'width':width,
		'tiled':True,'blockxsize':256,'blockysize':256})
	out_ptr = rio.open(path,'w',**kwargs)

	correct_count = 0

	#FOR EACH WINDOW PLOT SQUARES
	for _,w in windows:
		# single to 3-band 10-class
		arr = dw_reader.read(1,window=w)
		arr3 = np.stack([arr,np.zeros_like(arr),np.zeros_like(arr)],axis=0)
		#plot label colors - hex to int
		for c in range(10):
			r = int(DW_PALETTE_10[c][0:2],16)
			g = int(DW_PALETTE_10[c][2:4],16)
			b = int(DW_PALETTE_10[c][4:6],16)
			arr3[:,arr==c] = [[r],[g],[b]]	

		#LINES
		if (arr==0).sum() > 4:
			for i in range(3):
				arr3[:,i,:]      = red_line #top
				arr3[:,-(i+1),:] = red_line #bottom
				arr3[:,:,i]      = red_line #left
				arr3[:,:,-(i+1)] = red_line #right

		else:			
			n_water = (arr==1).sum()

			if n_water > WATER_MIN  and n_water < WATER_MAX:
				for i in range(3):
					arr3[:,i,:]      = yellow_line #top
					arr3[:,-(i+1),:] = yellow_line #bottom
					arr3[:,:,i]      = yellow_line #left
					arr3[:,:,-(i+1)] = yellow_line #right
				correct_count += 1

			else:
				for i in range(3):
					arr3[:,i,:]      = red_line #top
					arr3[:,-(i+1),:] = red_line #bottom
					arr3[:,:,i]      = red_line #left
					arr3[:,:,-(i+1)] = red_line #right

		#adjust window:pos in reader to pos in writer
		w2 = Window(w.col_off-borders['left'],w.row_off-borders['top'],CHIP_SIZE,CHIP_SIZE)
		
		#WRITE
		out_ptr.write(arr3,window=w2,indexes=[1,2,3])

	print("Nr of good chips in raster: %i" % correct_count)

	#PLOT REMAINING IMAGE BEYOND WINDOWS
	block_rows     = height // CHIP_SIZE
	block_cols     = width // CHIP_SIZE
	remainder_rows = height % CHIP_SIZE
	remainder_cols = width % CHIP_SIZE

	#bottom edge
	reader_window = Window(
		col_off=borders['left'],
		row_off=borders['top']+block_rows*CHIP_SIZE,
		width=block_cols*CHIP_SIZE,
		height=remainder_rows) #reader,starts at border['top'],borders['left']

	writer_window = Window(
		col_off=0,
		row_off=block_rows*CHIP_SIZE,
		width=block_cols*CHIP_SIZE,
		height=remainder_rows) #writer,starts at 0,0

	arr = dw_reader.read(1,window=reader_window)
	arr3 = np.stack([arr,np.zeros_like(arr),np.zeros_like(arr)],axis=0)
	for c in range(10):
		r = int(DW_PALETTE_10[c][0:2],16)
		g = int(DW_PALETTE_10[c][2:4],16)
		b = int(DW_PALETTE_10[c][4:6],16)
		arr3[:,arr==c] = [[r],[g],[b]]
	out_ptr.write(arr3,window=writer_window,indexes=[1,2,3])

	#right edge
	reader_window = Window(
		col_off=borders['left']+block_cols*CHIP_SIZE,
		row_off=borders['top'],
		width = remainder_cols,
		height = height)

	writer_window = Window(
		col_off=block_cols*CHIP_SIZE,
		row_off=0,
		width=remainder_cols,
		height=height) #writer, starts at 0,0
  
	arr = dw_reader.read(1,window=reader_window)
	arr3 = np.stack([arr,np.zeros_like(arr),np.zeros_like(arr)],axis=0)
	for c in range(10):
		r = int(DW_PALETTE_10[c][0:2],16)
		g = int(DW_PALETTE_10[c][2:4],16)
		b = int(DW_PALETTE_10[c][4:6],16)
		arr3[:,arr==c] = [[r],[g],[b]]
	out_ptr.write(arr3,window=writer_window,indexes=[1,2,3])

# CHECK
def folder_check():
	'''
	Do a folder check to remove any folder with bands if that folder does not have a matching .tif
	dynanmic world label.
	'''
	folders = [f for f in os.listdir(DATA_DIR) if f!='dynamicworld' and f[-5:]=='.SAFE']
	for folder in folders:
		# get xml
		xml_name = [f for f in os.listdir(DATA_DIR + id) if f[-4:]=='.xml'][0] #bc MTD, MTD_L2A
		xml_path = '/'.join([DATA_DIR,folder,xml_filename])	

		# get dynarmicworld id
		dstrip, _, _ = parse_xml(xml_path)
		date         = folder[11:26]
		tile         = folder[38:44]		
		dw_id        = '_'.join([date,datastrip,tile])

		#delete all scl's
		scl_file = '_'.join([tile,date,'SCL','20m.jp2'])
		scl_path = '/'.join([DATA_DIR,folder,scl_file])
		if os.path.isfile(scl_path):
			os.remove(scl_path)

		#if dw...
		if os.path.isfile(LABEL_DIR+'/'+dw_id+'.tif'):
			# exists, check files in .SAFE
			n_files = len(os.listdir(folder))
			if n_files != 5:
				print("Folder %s has <5 files." % folder) #in case we actually need to check
		else:
			# d.n.e, remove whole .SAFE dir
			for file in os.listdir(folder):
				os.remove('/'.join([DATA_DIR,folder,file]))
			os.rmdir(folder)

			print("Folder %s removed." % folder)

#TODO
def plot_label_singleclass_windowed():
	pass

#TODO
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

#TODO
def chip_image_gpu(img: ndarray) -> ndarray:
	pass

#TODO
def adjust_label(win_arr: ndarray):
	"""
	Label 1 is water, remove unnecessary labels, equate to one, set rest to zero, nonvals(?),etc.
	"""
	window == 1
	pass

#TODO
def adjust_band(win_arr: ndarray):
	"""
	Do the shift by offset band values and divide by the quantification value,
	clip values if needed
	"""
	pass

#TODO
def check_label_window(arr: ndarray):
	"""
	This checks if there's both water and land in the array.
	"""
	no_data_mask = arr==0
	if no_data_mask.sum() > 4: #if no data
		#red
		return -1
	else:			
		n_water = (arr==1).sum()
		if n_water > WATER_MIN and n_water < WATER_MAX:
			#same as <---- yellow line
			pass
			correct_count += 1


		else:
			#same as <---- red line
			pass

	return True

#TODO
def check_band_window(arr: ndarray):
	if (window == 0).sum() > 0: #no data present
		return False
	return True

#TODO
def process_window(w: Window, ):
	pass

#TODO
def process_product(id: str):
	#1.ID -> BAND PATHS, XML PATH
	band_filename = get_band_filenames(id,['B02','B03','B04','B08'])
	band_paths    = [DATA_DIR+id+'/'+_  for _ in band_filename]

	xml_filename = [f for f in os.listdir(DATA_DIR + id) if f[-4:]=='.xml'][0] #bc MTD, MTD_L2A
	xml_path     = DATA_DIR + '/'.join([id,xml_filename])

	#2.XML -> DW PATH, OFFSETS, QUANTIFICATION VALUE
	datastrip, quantnr, offsets = parse_xml(xml_path)
	gee_id  = get_gee_id(id,datastrip)
	dw_path = DATA_DIR + '/'.join([LABEL_DIR,gee_id]) + '.tif'

	#3.BAND PATHS, DW PATH -> DatasetReader x 5
	label = rio.open(label_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band2 = rio.open(band_paths[0],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band3 = rio.open(band_paths[1],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band4 = rio.open(band_paths[2],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band8 = rio.open(band_paths[3],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)

	#4.DW READER -> BOUNDARIES DW
	label_borders = remove_borders(label)

	#5.BOUNDARIES DW, BAND READER x 4 -> BOUNDARIES S2
	input_borders = convert_bounds(label,band2,label_borders)

	#6.BOUNDARIES S2, BOUNDARIES DW -> WINDOWS DW, WINDOWS S2 
	input_windows = get_windows(input_borders)
	label_windows = get_windows(label_borders)

	#7.ITERATE THROUGH WINDOWS
	pass

#TODO
def plot_tci_windowed(dst_path,bands,borders,windows):

	#need to know size of output
	px_rows = borders['bottom'] - borders['top'] + 1
	px_cols = borders['right'] - borders['left'] + 1
	px_rows_blocks = px_rows - (px_rows % CHIP_SIZE - 1)
	px_cols_blocks = px_cols - (px_cols % CHIP_SIZE -1)



	return True

#TODO
def plot_tci_masks(fname:str, bands:[rio.DatasetReader], quant:int, offsets:[int], borders: dict):
	'''
	Plot whole rgb image with the nodata borders removed.
	'''
	#Full image minus no-data borders
	px_rows = borders['bottom'] + 1 - borders['top']
	px_cols = borders['right'] + 1 - borders['left']
	w = Window(borders['left'],borders['top'],px_cols,px_rows)

	kwargs = bands[0].meta.copy()
	kwargs.update({'count':3,
		'driver':'JP2OpenJPEG','codec':'J2K',
		'height':px_rows,
		'width':px_cols,
		'dtype':'uint8',
		'tiled':True,'blockxsize':512,'blockysize':512,
		})

	#Stack bands
	b,g,r = (_.read(1,window=w) for _ in bands)
	tci   = np.stack([r,g,b],axis=0).astype(np.int_16)

	# nodata
	rgb_zeromask = tci == 0
	and_zeromask = rgb_zeromask.all(axis=0) #AND thru axis 0

	# clipped/saturated data
	esa_highmask = (tci >= 10000).any(axis=0)

	pctiles_99 = [np.percentile(b[~rgb_zeromask[i]],99.9) for i,b in enumerate(tci)]
	pctiles_99 = np.array(pctiles_99).reshape((3,1,1))
	pct_highmask = tci > pctiles_99
	pct_highmask = pct_highmask.any(axis=0)

	# # clip to [1001,11000] and shift
	# if sum(offsets) > 0:
	# 	tci = np.clip(tci,1001,11000) #[1001,11000]
	# 	tci = (tci + np.array(offsets).reshape(3,1,1)) #[1,10000]
	# else:
	# 	tci = np.clip(tci,1,10000)

	# # clip to 1,99 percentile
	# tci = [np.clip(b,np.percentile(b,1),np.percentile(b,99)) for b in tci]
	# tci = minmax_normalize(tci,by_band=True)
	# tci = (tci * 255).astype(np.uint8)

	# AS IS
	tci_esa = tci.copy()
	tci_esa = np.clip(tci_esa,0,10000)
	tci_esa = (minmax_normalize(tci_esa,by_band=True) * 255).astype(np.uint8)

	tci[0] = np.clip(tci[0],0,pctiles_99[0])
	tci[1] = np.clip(tci[1],0,pctiles_99[1])
	tci[2] = np.clip(tci[2],0,pctiles_99[2])
	tci = (minmax_normalize(tci,by_band=True) * 255).astype(np.uint8)

	# 0. TCI - AS IS
	#-----------------------------------------------------------------
	path = './fig/'+fname+'_10000.jp2'
	with rio.open(path,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_esa,indexes=[1,2,3])
	print("TCI written to %s" % path)

	path = './fig/'+fname+'_99.jp2'
	with rio.open(path,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci,indexes=[1,2,3])
	print("TCI written to %s" % path)


	#1.TCI - ZERO PIXELS IN EACH BAND SET TO MAX
	#-----------------------------------------------------------------
	tci_all = tci.copy()
	tci_all[rgb_zeromask] = 255
	path_all = './fig/' + fname + '_ZERO_ALL.jp2'
	with rio.open(path_all,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_all,indexes=[1,2,3])
	print("TCI written to %s" % path_all)

	#2.TCI - AND OF ZERO PIXELS SET TO MAX IN GREEN(correct masking)
	#-----------------------------------------------------------------
	tci_and = tci.copy()
	tci_and[0,and_zeromask] = 255
	tci_and[1,and_zeromask] = 0
	tci_and[2,and_zeromask] = 0
	path_and = './fig/'+fname+'_ZERO_AND.jp2'
	with rio.open(path_and,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_and,indexes=[1,2,3])
	print("TCI written to %s" % path_and)

	#3.TCI - HI (>10k) PIXELS SET TO MAX IN RED
	#-----------------------------------------------------------------	
	tci_hi1 = tci.copy()
	tci_hi1[0,esa_highmask] = 255
	tci_hi1[1,esa_highmask] = 0
	tci_hi1[2,esa_highmask] = 0
	path_hi1 = './fig/'+fname+'_HIGH_ESA.jp2'
	with rio.open(path_hi1,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_hi1)

	print("TCI written to %s" % path_hi1)


	#3.TCI - HI (>99%) PIXELS SET TO MAX IN RED
	#-----------------------------------------------------------------	
	tci_hi2 = tci.copy()
	tci_hi2[0,pct_highmask] = 255
	tci_hi2[1,pct_highmask] = 0
	tci_hi2[2,pct_highmask] = 0
	path_hi2 = './fig/'+fname+'_HIGH_PCT.jp2'
	with rio.open(path_hi2,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_hi2)
	print("TCI written to %s" % path_hi2)

	for f in os.listdir('./fig'):
		if f[-3:] == 'xml':
			os.remove('./fig/'+f)

#TODO
def save_tci_window(path,bands,quant_val,offsets,window):
	return
####################################################################################################
# MAIN
####################################################################################################

if __name__ == '__main__':

	#.SAFE folders in data directory
	folders  = [d for d in os.listdir(DATA_DIR) if d[-5:]=='.SAFE']
	
	#A SINGLE FOLDER
	safe_dir = folders[0]

	#PARSE XML INFO
	xml_file = [f for f in os.listdir(DATA_DIR+'/'+safe_dir) if f[-4:]=='.xml'][0] #bc MTD, MTD_L2A
	xml_path = '/'.join([DATA_DIR,safe_dir,xml_file])
	datastrip,quant_val,offsets = parse_xml(xml_path)

	#GET FILE PATHS
	gee_id = get_gee_id(safe_dir,datastrip)
	label_path = LABEL_DIR + '/' + gee_id + '.tif'
	band_fname = get_band_filenames(safe_dir,['B02','B03','B04','B08'])
	band_path  = ['/'.join([DATA_DIR,safe_dir,_]) for _ in band_fname]

	label = rio.open(label_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band2 = rio.open(band_path[0],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band3 = rio.open(band_path[1],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band4 = rio.open(band_path[2],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band8 = rio.open(band_path[3],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	
	#EQUATE LABEL PIXEL BORDERS
	label_borders = remove_borders(label)
	input_borders = convert_bounds(label,band2,label_borders)

	#WINDOWS TO READ
	input_windows = get_windows(input_borders)
	label_windows = get_windows(label_borders)

	print(safe_dir)
	print("-"*80)
	# plot_tci_masks(gee_id+'_TCI',[band2,band3,band4],quant_val,offsets[0:3],input_borders)
	# plot_checkerboard(gee_id+'_chk.tif',label,label_borders,label_windows)

	# px_rows = input_borders['bottom'] + 1 - input_borders['top']
	# px_cols = input_borders['right'] + 1 - input_borders['left']
	# w = Window(input_borders['left'],input_borders['top'],px_cols,px_rows)
	# band2.read(1,window=w)
	# band_hist(path, , "B02")
