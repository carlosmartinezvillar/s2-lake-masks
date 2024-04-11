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
import multiprocessing as mp

####################################################################################################
# GLOBAL
####################################################################################################
#Function typing
from typing import Tuple, List
ndarray = np.ndarray #quick fix...

#XML namespaces
ns = {
	'n1':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd",
	'other':"http://www.w3.org/2001/XMLSchema-instance",
	'another':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"
	}

#Plots
plt.style.use('fast')

#DIRs and such
DATA_DIR = os.getenv('DATA_DIR')
if DATA_DIR is None:
	DATA_DIR = './dat'
LABEL_DIR = DATA_DIR + '/dynamicworld'
CHIP_DIR  = DATA_DIR + '/chp'

#The important ones
CHIP_SIZE = 256
WATER_MIN = 128*64
WATER_MAX = CHIP_SIZE*CHIP_SIZE-WATER_MIN

####################################################################################################
# BAND ARITHMETIC
####################################################################################################
def minmax_normalize(img: ndarray, by_band: bool=True) -> ndarray:
	"""
	Normalize a set of bands individually, or across all bands, using min and max values.
	
	Parameters
	----------
	img: numpy.ndarray
		A 1-band or 3-band raster image with the first axis being the bands.
	by_band: bool
		Boolean flag. If false, a single pair of min and max values for all bands is used. If true, 
		the min and max values in each band are used to normalize the corresponding bands.

	Returns
	-------
	result: numpy.ndarray
		The normalized image

	""" 	
	if by_band is True and (img.shape[0] in range(1,5)):
		B_n  = img.shape[0] #assuming bands is outermost axis
		mins = np.array([band.min() for band in img]).reshape((B_n,1,1))
		maxs = np.array([band.max() for band in img]).reshape((B_n,1,1))
		return (img - mins) / (maxs - mins)
	else:
		return (img - img.min()) / (img.max() - img.min())

####################################################################################################
# HISTOGRAMS, ET CETERA
####################################################################################################
def band_hist(path: str, band: ndarray, title: str, n_bins: int=4096, color: str='blue') -> None:
	nonzeros = band[band!=0].flatten()
	fig,ax = plt.subplots()
	ax.set_title(title)
	ax.hist(nonzeros,bins=n_bins,histtype='bar',color=color)
	plt.savefig(path)
	print("Band plot saved to %s." % path)
	plt.close()


def multiband_hist(path: str, bands: [ndarray], title: str, n_bins: int=4096) -> None:
	subtitle = ['Red','Green','Blue','NIR']
	color    = ['r','g','b','darkred']
	fig, axs = plt.subplots(nrows=1,ncols=len(bands),sharey=True,tight_layout=True)
	fig.suptitle(title)
	for i in range(len(bands)):
		axs[i].hist(bands[i][bands[i]!=0].flatten(),bins=n_bins,histtype='bar',color=color[i])
		axs[i].set_title(subtitle[i])
	plt.savefig(path)
	print("Multi-band plot saved to %s." % path)

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

	The XML structure of the offsets looks like this

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

	So they are parsed, by iterating on the result of

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
	# boa_quant = prod_char.find('QUANTIFICATION_VALUES_LIST').find('BOA_QUANTIFICATION_VALUE')
	# quant_val = int(boa_quant.text)

	# offset values
	boa_add = prod_char.find('BOA_ADD_OFFSET_VALUES_LIST') #None | Element 
	if boa_add is not None:
		offsets = [int(e.text) for e in boa_add[1:4]] + [int(boa_add[7].text)]
	else:
		offsets = [0]*4

	return datastrip,offsets


def get_gee_id(s2_id: str) -> str:
	xml_name    = [f for f in os.listdir(DATA_DIR+'/'+s2_id) if f[-4:]=='.xml'][0]
	xml_path    = DATA_DIR + '/' + '/'.join([s2_id,xml_name])
	datastrip,_ = parse_xml(xml_path)
	date,tile   = s2_id.split('_')[2:6:3]
	gee_id      = '_'.join([date,datastrip,tile])
	return gee_id


def join_gee_str(s2_id: str, datastrip: str) -> str:
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
	# date = s2_id[11:26];tile = s2_id[38:44]
	return '_'.join([s2_id[38:44],s2_id[11:26],band,'10m.jp2'])


def get_band_filenames(s2_img_id: str, bands: [str]) -> [str]:
	'''
	Given a Sentinel-2 product name and bands, return all band image file names.
	'''
	return list(map(get_band_filename,[s2_img_id]*len(bands),bands))


def remove_label_borders_as_array(dw_array: ndarray) -> ndarray:
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


def remove_label_borders(src: rio.DatasetReader) -> dict:
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
	dw_xy_ul = dw.xy(dw_ij_dict['top'],dw_ij_dict['left'],offset='center')
	dw_xy_lr = dw.xy(dw_ij_dict['bottom'],dw_ij_dict['right'],offset='center')
	s2_ij_ul = s2.index(dw_xy_ul[0],dw_xy_ul[1],op=math.floor)
	s2_ij_lr = s2.index(dw_xy_lr[0],dw_xy_lr[1],op=math.floor)

	#return dict of bounds
	return {'top':s2_ij_ul[0],'bottom':s2_ij_lr[0],
		'left':s2_ij_ul[1],'right':s2_ij_lr[1]}


def fix_bounds(s2_src,dw_src):
	'''
	Match indices and remove borders.
	'''
	dw_ij = {'top':0,'bottom':dw_src.height-1,'left':0,'right':dw_src.width-1}

	dw_xy_ul = dw.xy(dw_ij['top'],dw_ij['left'])
	dw_xy_lr = dw.xy(dw_ij['bottom'],dw_ij['right'])

	s2_ij_t,s2_ij_l = s2.index(dw_xy_ul[0],dw_xy_ul[1],op=math.floor)
	s2_ij_b,s2_ij_r = s2.index(dw_xy_lr[0],dw_xy_lr[1],op=math.floor)

	s2_ij = {'top':s2_ij_t,'bottom':s2_ij_b,'left':s2_ij_l,'right':s2_ij_r}
	
	# REMOVE S2 TILE OVERLAP
	if s2_ij['top'] < 492:
		pass
		#move down to 492

	if s2_ij['bottom'] > 10488:	#10488 is 10980-492
		pass #move up to 10488

	if s2_ij['left'] < 492:
		pass #move right to 492

	if s2_ij['right'] > 10488:
		pass #move loeft to 10488


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


def folder_check():
	'''
	Do a folder check to remove any .SAFE folder if that folder does not have a matching .tif
	dynanmic world label.
	'''
	folders = [f for f in os.listdir(DATA_DIR) if f!='dynamicworld' and f[-5:]=='.SAFE']
	n_input, n_label = 0,0
	removed_folders = []

	for folder in folders:
		# empty folder, remove
		if len(os.listdir(DATA_DIR+'/'+folder)) == 0:
			print("Empty folder %s" % folder)
			# os.rmdir(DATA_DIR+'/'+folder)
			continue

		# get xml, continue if no xml
		try:
			xml_name = [f for f in os.listdir(DATA_DIR+'/'+folder) if f[-4:]=='.xml'][0]
			xml_path = '/'.join([DATA_DIR,folder,xml_name])
		except IndexError:
			print("Index error. No xml file in %s. Skipping." % folder)
			continue
		except Exception as err:
			print('Other error retrieving xml file in %s. Skipping.' % folder)
			print(err)
			continue

		# get dynarmicworld id
		dstrip,_ = parse_xml(xml_path)
		date     = folder[11:26]
		tile     = folder[38:44]
		dw_id    = '_'.join([date,dstrip,tile])

		#delete all scl's
		scl_file = '_'.join([tile,date,'SCL','20m.jp2'])
		scl_path = '/'.join([DATA_DIR,folder,scl_file])
		if os.path.isfile(scl_path):
			print("Deleting %s" % scl_path)
			os.remove(scl_path)

		#if dw...
		if os.path.isfile(LABEL_DIR+'/'+dw_id+'.tif'):
			# exists, check files in .SAFE
			n_files = len(os.listdir(DATA_DIR + '/' + folder))
			if n_files != 5:
				print("Folder %s has <5 files." % folder) #in case we actually need to check
			else:
				print("Folder %s -- OK." % folder)
				n_input += 1
				n_label += 1
		else:
			# d.n.e, remove whole .SAFE dir
			print("--> Removing folder %s" % folder)
			for file in os.listdir(DATA_DIR+'/'+folder):
				os.remove('/'.join([DATA_DIR,folder,file]))
			os.rmdir(DATA_DIR+'/'+folder)
			removed_folders.append(folder)

	print("%i REMOVED:" % len(removed_folders))
	for rf in removed_folders:
		print(rf)


	# if 

#---------------------------------------
def plot_label_multiclass(path,dw_reader,dw_borders):
	"""
	Plots the whole label raster with nodata borders removed.
	"""
	DW_NAMES_10 = ['masked','water','trees','grass','flooded_vegetation',
		'crops','shrub_and_scrub','built','bare','snow_and_ice'];

	DW_PALETTE_10 = ['000000','419bdf','397d49','88b053','7a87c6','e49635', 
	    'dfc35a','c4281b','a59b8f','b39fe1'];


	# number of rows and cols takin' the boundaries into acct
	h = dw_borders['bottom'] + 1 - dw_borders['top']
	w = dw_borders['right'] + 1 - dw_borders['left']
	read_window = Window(borders['left'],borders['top'],w,h)

	#read, 1-band to 3-band, plot colors
	dw1 = dw_reader.read(1,window=read_window)
	dw3 = np.stack([dw1,np.zeros_like(dw1),np.zeros_like(dw1)],axis=0).astype(np.uint8)
	for i in range(10):
		r = int(DW_PALETTE_10[i][0:2],16) #hex to int
		g = int(DW_PALETTE_10[i][2:4],16)
		b = int(DW_PALETTE_10[i][4:6],16)
		dw3[:,dw1==i] = [[r],[g],[b]]

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

#TODO
def plot_label_singleclass_windowed():
	'''
	Plot whole label with 2 classes as black and white. Need for paper print. 
	'''
	pass


def plot_label_checkerboard(path,dw_reader,borders,windows):
	"""
	Plot whole label raster with overlaying squares corresponding to
	to windows and chips. Red squares are chips discarded, yellow
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

#TODO
def plot_rgb_checkerboard(path,s2_id):
	band_fname  = get_band_filenames(s2_id,['B02','B03','B04','B08'])
	band_reader = [rio.open(DATA_DIR+'/'+s2_id+'/'+f,'r') for f in band_fname]

	gee_id    = get_gee_id(s2_id)
	dw_path   = '/'.join([LABEL_DIR,gee_id]) + '.tif'
	dw_reader = rio.open(dw_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)

	dw_borders = remove_label_borders(dw)
	s2_borders = convert_bounds(dw,band_reader[0],dw_borders)

	s2_windows = get_windows(s2_borders)

	kwargs = band_reader[0].meta.copy()
	kwargs.update({
		'count':3,
		'driver':'JP2OpenJPEG','coded':'J2K',
		'tiled':True,'blockxsize':CHIP_SIZE,'blockysize':CHIP_SIZE,
		'dtype':'uint8'
		})
	out_ptr = rio.open(path,'w',**kwargs)

	#plot whole thing 3-bands
	b,g,r = (_.read(1) for _ in bands)
	tci   = np.stack([r,g,b],axis=0).clip(0,32767).astype(np.int16) #int16 [-32767,32767]

	rgb_zeromask = tci == 0
	percentile99 = np.array([np.percentile(b[~rgb_zeromask[i]],99) for i,b in enumerate(tci)])

	#plot windows
	for _,w in windows:
		arr = dw_reader.read(1,window=w)
		arr3 = np.stack([arr,np.zeros_like(arr),np.zeros_like(arr)],axis=0)

#TODO
def plot_tci_masks(fname:str, bands:[rio.DatasetReader], offsets:[int], borders: dict) -> None:
	'''
	Plot whole rgb image with clipped pixels at different thresholds highlighted.
	'''
	#Full image minus no-data borders
	px_rows = borders['bottom'] + 1 - borders['top']
	px_cols = borders['right'] + 1 - borders['left']
	w       = Window(borders['left'],borders['top'],px_cols,px_rows)

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
	tci   = np.stack([r,g,b],axis=0).clip(0,32767).astype(np.int16) #int16 [-32767,32767]

	# nodata
	rgb_zeromask = tci == 0
	and_zeromask = rgb_zeromask.all(axis=0) #AND thru axis 0
	esa_highmask = (tci >= 11000).any(axis=0) #OR thru axis 0

	# add offsets
	# tci        = tci + np.array(offsets).reshape((3,1,1))

	#calculate percentiles and get masks for them
	pctiles_99 = np.array([np.percentile(b[~rgb_zeromask[i]],99.9) for i,b in enumerate(tci)])
	pctiles_01 = np.array([np.percentile(b[~rgb_zeromask[i]],0.01) for i,b in enumerate(tci)])
	pct_highmask = (tci > pctiles_99.reshape((3,1,1))).any(axis=0) #OR mask
	pct_lowmask  = (tci < pctiles_01.reshape((3,1,1))).any(axis=0) #OR mask
	print("Percentiles set to (RGB): %s,%s" % (str(pctiles_99),str(pctiles_01)))


	# 0.TCI
	#-----------------------------------------------------------------
	tci_raw = (minmax_normalize(tci,by_band=True)*255).astype(np.uint8)

	path = './fig/' + fname + '.jp2'
	with rio.open(path,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_raw,indexes=[1,2,3])
	print("TCI written to %s" % path)
	del tci_raw

	# 1.TCI - [0,10000]
	#-----------------------------------------------------------------
	#clip
	tci_esa = tci.copy()
	tci_esa[0] = np.clip(tci_esa[0],0+offsets[0],10000)
	tci_esa[1] = np.clip(tci_esa[1],0+offsets[1],10000)
	tci_esa[2] = np.clip(tci_esa[2],0+offsets[2],10000)

	# normalize
	tci_esa = (minmax_normalize(tci_esa,by_band=True)*255).astype(np.uint8)
	
	# save
	path = './fig/'+fname+'_10000.jp2'
	with rio.open(path,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_esa,indexes=[1,2,3])
	print("TCI written to %s" % path)
	del tci_esa

	# 2.TCI - [0.01%,99.9%]
	#-----------------------------------------------------------------
	# clip
	tci[0] = np.clip(tci[0],pctiles_01[0],pctiles_99[0])
	tci[1] = np.clip(tci[1],pctiles_01[1],pctiles_99[1])
	tci[2] = np.clip(tci[2],pctiles_01[2],pctiles_99[2])
	
	#normalize
	tci = (minmax_normalize(tci,by_band=True)*254+1).astype(np.uint8)

	#save
	path = './fig/'+fname+'_99.jp2'
	with rio.open(path,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci,indexes=[1,2,3])
	print("TCI written to %s" % path)


	# 3.TCI - ZERO PIXELS IN EACH BAND SET TO MAX
	#-----------------------------------------------------------------
	#place values/mask
	tci_rgb = tci.copy()
	tci_rgb[rgb_zeromask] = 255

	#save
	path_rgb = './fig/' + fname + '_ZERO_RGB.jp2'
	with rio.open(path_rgb,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_rgb,indexes=[1,2,3])
	print("TCI written to %s" % path_rgb)
	del tci_rgb

	# 4.TCI - AND OF ZERO PIXELS SET TO MAX IN RED (correct masking)
	#-----------------------------------------------------------------
	#place values/mask
	tci_and = tci.copy()
	tci_and[0,and_zeromask] = 255
	tci_and[1,and_zeromask] = 0
	tci_and[2,and_zeromask] = 0

	#save
	path_and = './fig/'+fname+'_ZERO_AND.jp2'
	with rio.open(path_and,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_and,indexes=[1,2,3])
	print("TCI written to %s" % path_and)
	del tci_and

	# 5.TCI - HI (>10k) PIXELS SET TO MAX IN RED
	#-----------------------------------------------------------------	
	#place values/mask
	tci_hi1 = tci.copy()
	tci_hi1[0,esa_highmask] = 255
	tci_hi1[1,esa_highmask] = 0
	tci_hi1[2,esa_highmask] = 0

	#save
	path_hi1 = './fig/'+fname+'_HI_ESA.jp2'
	with rio.open(path_hi1,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_hi1)
	print("TCI written to %s" % path_hi1)
	del tci_hi1

	# 6.TCI - HI (>99.9%) PIXELS SET TO 255 IN each
	#-----------------------------------------------------------------	
	# place values/mask
	tci_hi2 = tci.copy()
	tci_hi2[0,pct_highmask] = 255
	tci_hi2[1,pct_highmask] = 0
	tci_hi2[2,pct_highmask] = 0
	path_hi2 = './fig/'+fname+'_HI_PCT.jp2'
	with rio.open(path_hi2,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_hi2)
	print("TCI written to %s" % path_hi2)
	del tci_hi2

	# 7.TCI - LOW(<0.01%) PIXELS SET TO 255 IN RED
	#-----------------------------------------------------------------	
	# place values/mask
	tci_low = tci.copy()
	tci_low[0,pct_lowmask] = 255
	tci_low[1,pct_lowmask] = 0
	tci_low[2,pct_lowmask] = 0
	path_low = './fig/'+fname+'_LO_PCT.jp2'
	with rio.open(path_low,'w',photometric='RGB',**kwargs) as dst:
		dst.write(tci_low,indexes=[1,2,3])
	print("TCI written to %s" % path_low)
	del tci_low


	# DELETE UNNECESSARY XML METADATA
	for f in os.listdir('./fig'):
		if f[-3:] == 'xml':
			os.remove('./fig/'+f)

#TODO
def plot_tci_windowed(dst_path,bands,borders,windows):

	#need to know size of output
	px_rows = borders['bottom'] - borders['top'] + 1
	px_cols = borders['right'] - borders['left'] + 1
	px_rows_blocks = px_rows - (px_rows % CHIP_SIZE - 1)
	px_cols_blocks = px_cols - (px_cols % CHIP_SIZE -1)



	return True

#TODO
def save_tci_window(path,bands,offsets,window):
	return
#---------------------------------------
#TODO
def chip_image_cpu(path: str, windows: [Tuple]):

	#split windows into n_cpu arrays

	#init n_cpu file readers
	for i in range(mp.cpu_count):
		readers += [rio.open(path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)]
	return

	#throw a worker to each array

#TODO
def chip_image_cpu_worker(img: ndarray, windows: [Tuple]) -> None:
	return

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


def check_histograms(fname:str, band:rio.DatasetReader, offset:int, borders: dict) -> None:
	#plot params
	n_bins = 2048
	px_plt = 1/plt.rcParams['figure.dpi']
	H,W     = 600*px_plt,800*px_plt

	#Remove no-data borders
	px_rows = borders['bottom'] + 1 - borders['top']
	px_cols = borders['right'] + 1 - borders['left']

	#Read
	w   = Window(borders['left'],borders['top'],px_cols,px_rows)	
	red = band.read(1,window=w).astype(np.int16)

	#Zeroes, percentiles
	zero_mask = red == 0
	red       = red + offset
	pctile_99 = np.percentile(red[~zero_mask],99)
	pctile_01 = np.percentile(red[~zero_mask],1)
	print("Bot percentile: %i" % pctile_01)
	print("Top percentile: %i" % pctile_99)

	#normalize
	nonzero_min = red[~zero_mask].min()
	nonzero_max = red.max()

	print("Min: %i" % nonzero_min)
	print("Max: %i" % nonzero_max)

	# # HIST 0 -- RAW
	hist_path  = '_'.join(['./fig/'+fname,'hist','0.png'])
	fig,ax = plt.subplots(figsize=(W,H))
	ax.set_title("Red band -- original")
	ax.hist(red[~zero_mask],bins=n_bins,histtype='bar',color='red')
	ax.axvline(pctile_99,color='black',linewidth=0.4)
	ax.axvline(pctile_01,color='black',linewidth=0.4)
	ax.axvline(1000+offset,color='blue',linewidth=0.4,linestyle='--')	
	# ax.set_ylim(0,500000)
	plt.savefig(hist_path)
	print("Band histogram saved to %s." % hist_path)
	plt.close()

	# HIST 1 -- normalized to [0,1]*255
	red_normed  = (red[~zero_mask]-(1+offset))/(nonzero_max-(1+offset))
	this_array = (red_normed*255).round().astype(np.uint8)
	hist_path = '_'.join(['./fig/'+fname,'hist','1.png'])
	fig,ax    = plt.subplots(figsize=(W,H))
	ax.set_title("Red band -- scaled to [0,1]*255")
	ax.hist(this_array,bins=255,histtype='bar',color='red')
	# # ax.set_ylim(0,500000)
	ax.axvline(np.percentile(this_array,99),color='black',linewidth=0.4)
	plt.savefig(hist_path)
	print("Band histogram saved to %s." % hist_path)
	plt.close()


	#HIST 2 -- clipped normed binned
	this_array = (np.clip(red[~zero_mask],1,pctile_99)-1)/(pctile_99-1)
	# this_array = (np.clip(red_normed,0,0.99)/(0.99) * 255).round().astype(np.uint8)
	this_array = (this_array * 255).round().astype(np.uint8)
	hist_path  = '_'.join(['./fig/'+fname,'hist','2.png'])
	fig,ax     = plt.subplots(figsize=(W,H))
	ax.set_title("Red band -- shifted, clipped, scaled, and binned")
	ax.hist(this_array,bins=255,histtype='bar',color='red')
	plt.savefig(hist_path)
	print("Band histogram saved to %s." % hist_path)


def load_product(safe_dir:str) -> Tuple:
	'''
	Do all the stuff needed before any step/plots
	'''
	# ID -> BAND READERS, XML PATH
	s2_fnames  = get_band_filenames(safe_dir,['B02','B03','B04','B08'])
	s2_readers = [rio.open(DATA_DIR+'/'+safe_dir+'/'+f,'r') for f in s2_fnames]

	#2.XML -> DW PATH, OFFSETS
	xml_fname  = [f for f in os.listdir(DATA_DIR+'/'+safe_dir) if f[-4:]=='.xml'][0]
	xml_path   = DATA_DIR + '/' + '/'.join([safe_dir,xml_fname])
	datastrip,offsets = parse_xml(xml_path)
	date,tile         = safe_dir.split('_')[2:6:3]
	gee_id            = '_'.join([date,datastrip,tile])
	dw_path           = '/'.join([LABEL_DIR,gee_id]) + '.tif'
	dw_reader         = rio.open(dw_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)

	#4.DW READER -> BOUNDARIES DW
	#5.BOUNDARIES DW, (1) BAND READER -> BOUNDARIES S2
	# dw_borders = remove_label_borders(dw_reader)
	dw_borders = {'top':0,'bottom':dw_reader.height-1,'left':0,'right':dw_reader.width-1}
	s2_borders = convert_bounds(dw_reader,s2_readers[0],dw_borders)

	s2_windows = get_windows(s2_borders)
	dw_windows = get_windows(dw_borders)

	return s2_readers,s2_borders,dw_reader,dw_borders

#TODO
def process_product(safe_dir: str):
	'''
	Process a .SAFE folder.
	'''
	print("\nProcessing %s" % safe_dir)
	print("-"*80)

	#1.ID -> BAND PATHS, XML PATH
	band_filename = get_band_filenames(safe_dir,['B02','B03','B04','B08'])
	band_paths    = [DATA_DIR+'/'+safe_dir+'/'+_  for _ in band_filename]
	xml_filename  = [f for f in os.listdir(DATA_DIR+'/'+safe_dir) if f[-4:]=='.xml'][0]
	xml_path      = DATA_DIR + '/' + '/'.join([safe_dir,xml_filename])

	#2.XML -> DW PATH, OFFSETs
	datastrip,offsets = parse_xml(xml_path)
	gee_id            = join_gee_str(safe_dir,datastrip)
	label_path        = '/'.join([LABEL_DIR,gee_id]) + '.tif'

	#3.BAND PATHS, DW PATH -> DatasetReader x 5
	label = rio.open(label_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band2 = rio.open(band_paths[0],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band3 = rio.open(band_paths[1],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band4 = rio.open(band_paths[2],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	band8 = rio.open(band_paths[3],'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)

	#4.DW READER -> BOUNDARIES DW
	label_borders = remove_label_borders(label)

	#5.BOUNDARIES DW, (1) BAND READER -> BOUNDARIES S2 -- EQUATE S2 & DW BOUNDS
	input_borders = convert_bounds(label,band2,label_borders)

	#6.BOUNDARIES S2, BOUNDARIES DW -> WINDOWS DW, WINDOWS S2
	input_windows = get_windows(input_borders)
	label_windows = get_windows(label_borders)

	#7.DO SOME CHECKs
	# check_histograms(gee_id,band2,offsets[2],input_borders)
	plot_tci_masks(gee_id+'_TCI',[band2,band3,band4],offsets[0:3],input_borders)

	#8.ITERATE THROUGH WINDOWS
	pass	
	
	'''
	FINAL ID should be: 
		DATE_DSTRIP_ROTATION_TILE_ROW_COL
	'''

#TODO
def check_product_overlaps():
	pass

####################################################################################################
# MAIN
####################################################################################################

if __name__ == '__main__':

	#.SAFE folders in data directory
	folders = [d for d in os.listdir(DATA_DIR) if d[-5:]=='.SAFE']

	prods   = [load_product(f) for f in folders]

	print(folders)
	# safe_dir = folders[2]

	# if os.path.isfile('./dat/index_whole.txt'):
	# 	products = []
	# 	with open('./dat/index_whole.txt') as fp:
	# 		for line in fp.readlines():
	# 			col = line.rstrip().split('_')
	# 			products += [(col[2],col[4],col[5].lstrip('T'),col[6].rstrip('.SAFE'))]
	

