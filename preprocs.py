import os
import xml.etree.ElementTree as ET
import rasterio as rio
from rasterio.windows import Window
import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import multiprocessing as mp
import time

# Typing
from typing import Tuple, List
ndarray = np.ndarray #quick fix...

# XML namespace in METADATA.xml
ns = {
	'n1':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd",
	'other':"http://www.w3.org/2001/XMLSchema-instance",
	'another':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"
	}

# Plots
plt.style.use('fast')

#DIRs and such
DATA_DIR = os.getenv('DATA_DIR')
if DATA_DIR is None:DATA_DIR = './dat'
LABEL_DIR = DATA_DIR+'/dynamicworld'
CHIP_DIR  = DATA_DIR+'/chp'

CHIP_SIZE = 256
WATER_MIN = 128*64 #1/8 of the image
WATER_MAX = CHIP_SIZE*CHIP_SIZE-WATER_MIN #balanced for 1/8 land
BAD_PX    = 3276

####################################################################################################
# PLOTS
####################################################################################################
def plot_label_singleclass(path,dw_reader,dw_borders,dw_windows):
	'''
	Plot workable area overlapping windows. Black and white classes.
	'''
	h = dw_borders['bottom'] + 1 - dw_borders['top']
	w = dw_borders['right'] + 1 - dw_borders['left']
	windows_height = h - (h % CHIP_SIZE) + 1 #not sure if safe
	windows_width  = w - (w % CHIP_SIZE) + 1	

	kwargs = dw_reader.meta.copy()
	kwargs.update({'height':windows_height,'width':windows_width,'count':3,'compress':'lzw'})
	out_ptr = rio.open(path,'w',**kwargs)

	for _,w in dw_windows:
		#read
		arr        = dw_reader.read(1,window=w)
		white_mask = arr == 1 #water
		gray_mask  = arr == 0 #nodata
		black_mask = ~(white_mask | gray_mask) #else

		#change
		arr[white_mask] = 255
		arr[black_mask] = 0
		arr[gray_mask]  = 128
		arr_3d = np.repeat(arr[np.newaxis,:,:],repeats=3,axis=0)

		#write
		out_win = Window(w.col_off-dw_borders['left'],w.row_off-dw_borders['top'],CHIP_SIZE,CHIP_SIZE)
		out_ptr.write(arr_3d,window=out_win)

	out_ptr.close()
	print("Label sample written to: %s" % path)

####################################################################################################
# STRINGS+PARSING
####################################################################################################
def parse_xml(path):
	assert os.path.isfile(path), "No file found in path %s" % path

	# get datastrip
	root      = ET.parse(path).getroot()
	prod_info = root.find('n1:General_Info',namespaces=ns).find('Product_Info')
	granule   = prod_info.find('Product_Organisation').find('Granule_List').find('Granule')
	datastrip = granule.attrib['datastripIdentifier'].split('_')[-2][1:]

	return datastrip


def get_gee_id(s2_id: str) -> str:
	# xml_name    = [f for f in os.listdir(DATA_DIR+'/'+s2_id) if f[-4:]=='.xml'][0]
	# xml_path    = DATA_DIR + '/' + '/'.join([s2_id,xml_name])	
	xml_path    = glob.glob(DATA_DIR + '/' + s2_id + '/*.xml')[0]
	datastrip,_ = parse_xml(xml_path)
	date,tile   = s2_id.split('_')[2:6:3]
	gee_id      = '_'.join([date,datastrip,tile])
	return gee_id


def get_band_filenames(s2_id: str) -> [str]:
	tile = s2_id[38:44]
	date = s2_id[11:26]
	return ['_'.join([tile,date,b,'10m.jp2']) for b in ['B02','B03','B04','B08']]


####################################################################################################
# RASTERs
####################################################################################################
def remove_label_borders(src: rio.DatasetReader) -> dict:
	'''
	Take a rasterio DatasetReader for a dynamicworld image and get the indices 
	where non-zeros begin at the top, bottom, left, and right.

	Parameters
	----------
	src: rasterio.DatasetReader
		Dataset reader for a dynamic world array (which has zeroes where S2
		still has data, making it redundant to check for zeroes in the S2 array).

	Returns
	-------
	dict
		dictionary with indices of first non-zero values at top, left, right, 
		bottom

	'''
	top    = 0
	bottom = src.height-1
	left   = 0
	right  = src.width-1

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


def align(s2_src: rio.DatasetReader,dw_src: rio.DatasetReader) -> Tuple:
	'''
	Do everything: match indices and remove borders.
	'''
	# 1. REMOVE DW NO-DATA BORDERS(~1-2px each side)
	dw_ij = remove_label_borders(dw_src)

	# 2. MATCH DW to S2 (DW has ~20px less on each side) 
	# DW ij's (px index) -> DW xy's (coords)
	dw_xy_ul = dw_src.xy(dw_ij['top'],dw_ij['left'],offset='center')
	dw_xy_lr = dw_src.xy(dw_ij['bottom'],dw_ij['right'],offset='center')
	# DW xy's (coords) -> S2 ij's (px index)
	s2_ij = {}
	s2_ij['top'],s2_ij['left']     = s2_src.index(dw_xy_ul[0],dw_xy_ul[1],op=math.floor)
	s2_ij['bottom'],s2_ij['right'] = s2_src.index(dw_xy_lr[0],dw_xy_lr[1],op=math.floor)

	# 3. TRIM S2 -- REMOVE S2 TILE OVERLAP & ADJUST DW
	if s2_ij['top'] < 492: #shift top down
		delta        = 492 - s2_ij['top']
		s2_ij['top'] = 492
		dw_ij['top'] = dw_ij['top'] + delta

	if s2_ij['bottom'] > 10487: #shift bottom up
		delta           = s2_ij['bottom'] - 10487
		s2_ij['bottom'] = 10487	
		dw_ij['bottom'] = dw_ij['bottom'] - delta

	if s2_ij['left'] < 492: #shift left right
		delta         = 492 - s2_ij['left']
		s2_ij['left'] = 492	
		dw_ij['left'] = dw_ij['left'] + delta

	if s2_ij['right'] > 10487: #shift right left
		delta          = s2_ij['right'] - 10487
		s2_ij['right'] = 10487		
		dw_ij['right'] = dw_ij['right'] - delta

	return s2_ij,dw_ij	


def get_windows(borders: dict) -> [Tuple]:
	'''
	Given a dicts of boundaries, returns an array list with tuples (i,j) for block indices i,j and 
	window objects corresponding to the block i,j while considering only the area of the raster
	within the boundaries defined by the indices in the dict. For example, if the array had two rows
	and a column of no data (top and left) the blocks are offseted and defined as:

			    left    256     512
				| 0 0 0 0 ..
				| 0 0 0 0 ... 	   
	    top ----+--------+--------+
		    0 0 |        |        |
		    0 0 | (0, 0) | (0, 1) |
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
		col_start = j * CHIP_SIZE + borders['left']
		W = Window(col_start,row_start,CHIP_SIZE,CHIP_SIZE)
		windows += [[(str(i),str(j)),W]]

	return windows


def folder_check(drop_tiles=True):
	'''
	1.Remove .SAFE/products without a matching dynanmic world label.
	2.Drop two tiles: T11SKD,T11TKE.
	'''
	folders = [f for f in os.listdir(DATA_DIR) if f!='dynamicworld' and f[-5:]=='.SAFE']
	n_input, n_label = 0,0
	removed_folders = []
	removed_tiles   = []

	# 1. FOR EACH .SAFE folder check: empty?,xml?,label?
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
			n_files = len([_ for _ in os.listdir(DATA_DIR +'/'+folder) if _[0]!='.'])
			if n_files < 5:
				print("Folder %s -- <5 files." % folder) #check
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

	if len(removed_folders) == 0:
		print("0 REMOVED.")
	else:
		print("%i REMOVED:" % len(removed_folders))
		for rf in removed_folders:
			print(rf)


def chip_image(path: str, windows: [Tuple], base_id: str):
	print(f'Processing {path}')

	#SPLIT WINDOWS
	n_proc   = mp.cpu_count() - 1
	share    = len(windows) // n_proc
	leftover = len(windows) % n_proc
	start,stop = [],[]
	for i in range(n_proc):
		start += [i*share]
		stop  += [i*share+share]
	stop[-1] += leftover
	work_blocks = [windows[s0:s1] for s0,s1 in zip(start,stop)]

	#THROW WORKERS AT ARRAYS
	for i in range(n_proc):
		p = mp.Process(target=chip_image_worker,args=(path,work_blocks[i],base_id))
		p.start()
	p.join()


#TODO
# ------> THIS FUNCTION IS GONNA NEED TO CREATE 5 READERS within, RGBN + LABEL FOR CHECKING
def chip_image_worker(path: str, windows: [Tuple], base_id: str) -> None:

	b2_path = 
    b3_path = 
    b4_path = 
    b8_path = 
    dw_path = 

	b2_rdr = rio.open(b2_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	b3_rdr = rio.open(b3_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	b4_rdr = rio.open(b4_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	b8_rdr = rio.open(b8_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)
	dw_rdr = rio.open(dw_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)

	print('Done')


def prep_product(s2_id: str) -> Tuple:
	'''
	Do all the stuff needed before any step/plots
	'''
	#1.1 ID -> BAND READERS
	s2_filenames = get_band_filenames(s2_id)
	s2_reader    = [rio.open(DATA_DIR+'/'+s2_id+'/'+s,'r') for f in s2_filenames]

	#1.2 ID -> XML PATH
	#2.XML -> DW PATH
	#3.DW PATH -> DW READER
	xml_path  = glob.glob('/'.join([DATA_DIR,s2_id,'/*.xml']))[0]
	datastrip = parse_xml(xml_path)
	date,tile = s2_id.split('_')[2:6:3]
	gee_id    = '_'.join([date,datastrip,tile])
	dw_path   = '/'.join([LABEL_DIR,gee_id]) + '.tif'
	dw_reader = rio.open(dw_path,'r',tiled=True,blockxsize=CHIP_SIZE,blockysize=CHIP_SIZE)

	#4.DW READER -> BOUNDARIES DW
	#5.BOUNDARIES DW + B2 BAND READER -> BOUNDARIES S2 & BOUNDARIES DW
	s2_borders,dw_borders = align(s2_reader,dw_reader)

	'''
	Simplest chip id: 
		DATE_DSTRIP_TILE_ROTATION_WINROW_WINCOL_B0*.tif

	Label chip id:
		DATE_DSTRIP_TILE_ROTATION_WINROW_WINCOL_LBL.tif
	'''
	rotation = s2_id[33:37]
	base_chip_id = gee_id + '_' + rotation

	return s2_readers,s2_borders,dw_reader,dw_borders,base_chip_id


class Product()
	def __init__(self):


if __name__ == '__main__':

	# folder_check() #<--------------- ADD ARGPARSE ARGV TO STEP HERE

	#.SAFE folders in data directory
	folders = glob.glob('*.SAFE',root_dir=DATA_DIR)
	paths   = glob.glob(DATA_DIR+'/*.SAFE')

	s2_readers,s2_bounds,dw_reader,dw_bounds,out_id = prep_product(folders[0])

	dw_windows = get_windows(dw_bounds)
	s2_windows = get_windows(s2_bounds)

	test = './dat/S2B_MSIL2A_20230131T184619_N0509_R070_T11SKD_20230131T213704.SAFE/T11SKD_20230131T184619_B08_10m.jp2'
	chip_image(test,s2_windows,out_id)
	pass

	# plot_label_singleclass_windows('./fig/'+out_id+'_SINGLE_CLASS.tif',dw_reader,dw_bounds,dw_windows)
	# plot_rgb_windows('./fig/'+out_id+'_RGB.jp2',s2_readers,s2_bounds,s2_windows)