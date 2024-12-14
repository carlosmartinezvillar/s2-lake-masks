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
from PIL import Image
import sys
import argparse
import zipfile

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

parser = argparse.ArgumentParser(
	prog="preprocs.py",
	description="Preprocessing (chipping) of Sentinel-2 and DynamicWorld V1 images.")
parser.add_argument('--data-dir',default='./dat',
	help="Dataset directory")
parser.add_argument('--chip-dir',
	help="Chip directory")
parser.add_argument('--clean',action='store_true',default=False,
	help="Remove unlabeled rasters and unnecessary files",dest='clean')
parser.add_argument('--plots',action='store_true',default=False,
	help='Plot a sample of inputs and labels.',dest='plots')
parser.add_argument('--chips',action='store_true',default=False,
	help='Process .SAFE folders in data directory to create a dataset of raster chips.',dest='chips')
parser.add_argument('--kml',action='store_true',default=False,
	help='Filter original Sentinel-2 tile grid and create two files for AOI. Useful for dataset plots.',dest='kml')

args = parser.parse_args()

#SET DIRS HERE BECAUSE THREAD ACCESS
DATA_DIR  = args.data_dir
LABEL_DIR = DATA_DIR+'/dynamicworld' 
CHIP_DIR  = args.chip_dir
if CHIP_DIR is None:
	CHIP_DIR  = DATA_DIR+'/chips' #Subdir in same place as .SAFE folders

# PIXEL LIMITS -- I suppose these are good here too bc threads/processes
CHIP_SIZE = 256
WATER_MIN = 128*64 #1/8 of the image
WATER_MAX = CHIP_SIZE*CHIP_SIZE-WATER_MIN #balanced for 1/8 land
BAD_PX    = 3276
####################################################################################################
# CLASSES
####################################################################################################
# Tired of keeping track of parameters...
class Product():
	def __init__(self,safe_id):
		self.id    = safe_id
		self.tile  = self.id[38:44]
		self.date  = self.id[11:26]
		self.orbit = self.id[33:37]

		#1.1 ID -> BAND READERS
		self.s2_fnames  = self.get_band_filenames() #sorted
		self.s2_readers = []
		for f in self.s2_fnames:
			self.s2_readers += [rio.open(f'{DATA_DIR}/{safe_id}/{f}','r',tiled=True)]

		#1.2 ID -> XML PATH
		#2.XML -> DW PATH
		#3.DW PATH -> DW READER
		self.gee_id    = self.get_gee_id()
		self.dw_path   = f'{LABEL_DIR}/{self.gee_id}.tif'		
		self.dw_reader = rio.open(self.dw_path,'r',tiled=True)

		#4.DW READER -> BOUNDS DW
		#5.DW READER+BAND2 READER -> BOUNDS S2 & BOUNDS DW
		self.s2_borders, self.dw_borders = align(self.s2_readers[0],self.dw_reader)
	
		#DATE_DSTRIP_TILE_ROTATION_WINROW_WINCOL_B0*.tif
		#DATE_DSTRIP_TILE_ROTATION_WINROW_WINCOL_LBL.tif	
		self.base_chip_id = self.gee_id + '_' + self.orbit

	def get_band_filenames(self):
		return [f'{self.tile}_{self.date}_{b}_10m.jp2' for b in ['B02','B03','B04','B08']]

	def get_gee_id(self):
		xml_path  = glob.glob(f'{DATA_DIR}/{self.id}/*.xml')[0]
		datastrip = parse_xml(xml_path)
		return '_'.join([self.date,datastrip,self.tile])


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


def filter_tile_kml(drop=False):

	# CHECK FILE OR .ZIP OF EXISTS 
	kml_path = 'S2A_OPER_GIP_TILPAR_MPC__20151209T095117_V20150622T000000_21000101T000000_B00.kml'

	if not os.path.isfile(kml_path):
		#unzip
		zip_path = kml_path[0:-4]+'.zip'
		if not os.path.isfile(zip_path):
			print("No KML or ZIP file found for plotting tiles.")
			return
		else:
			try:
				print(f"Extracting {zip_path}")
				with zipfile.ZipFile(zip_path,'r') as zp:
					zp.extractall('./')
			except:
				print("Could not extract KML from .zip file.")
				return

	#LOAD LIST OF TILES IN DATASET
	# products = glob.glob('*.SAFE',root_dir=DATA_DIR)
	# tiles = [p.split('-'[5][1:] for p in products)]
	products = glob.glob('*.tif',root_dir=f'{DATA_DIR}/dynamicworld')
	if drop:
		products = [p.split('_')[2][1:6] for p in products if p!='11TKE' and p!='11SKD']
	else:
		products = [p.split('_')[2][1:6] for p in products]
	tiles_unique,counts = np.unique(products,return_counts=True)
	tiles = list(tiles_unique)

	print("RASTERS PER TILE")
	print("-"*80)
	for t,c in zip(tiles_unique,counts):
		print(f"{t} | {c} | " + "*"*(c//2))

	kml_ns = {
		'':"http://www.opengis.net/kml/2.2",
		'gx':"http://www.opengis.net/kml/2.2",
		'kml':"http://www.opengis.net/kml/2.2",
		'atom':"http://www.w3.org/2005/Atom"
	}

	#PARSE ORIGINAL SENTINEL-2 KML
	source_root = ET.parse(kml_path).getroot() #<kml>, source_root[0] <Document>

	#NAMESPACES GOT SILLY JUST REMOVE THEM
	for e in source_root.iter():
		ns,_ = e.tag.split('}')
		e.tag = _

	source_folder      = source_root[0][5] #<Folder>
	document_keep      = [e for e in source_root[0][0:5]] + [source_root[0][6]]

	# START APPENDING STUFF TO NEW KML
	# <FOLDER>
	# target_folder = ET.Element("{%s}Folder" % kml_ns[''])
	target_folder = ET.Element("Folder")
	target_folder.insert(0,source_folder[0])
	target_folder.insert(1,source_folder[1])

	# for placemark in source_folder.findall('Placemark',kml_ns):
	for placemark in source_folder.findall('Placemark'):
		if placemark[0].text in tiles: #CHECK ID -- placemark[0] is name
			target_folder.append(placemark)

	# <DOCUMENT>
	# target_document = ET.Element("{%s}Document" % kml_ns[''])
	target_document = ET.Element("Document")
	for e in document_keep[0:5]:
		target_document.append(e)
	target_document.text = '\n'
	target_document[0].text = 'filtered'
	target_document.append(target_folder)

	# <XML>
	# target_root = ET.Element("{%s}kml" % kml_ns[''])
	target_root = ET.Element("kml")
	target_root.text = '\n'
	target_root.append(target_document)
	target_root.set('xmlns',kml_ns['']) #not sure why namespace can't be copied as attrib
	target_root.set(f'xmlns:gx',kml_ns['gx'])
	target_root.set(f'xmlns:kml',kml_ns['kml'])
	target_root.set(f'xmlns:atom',kml_ns['atom'])
	target_tree = ET.ElementTree(target_root)	
	target_tree.write('filtered.kml',encoding='UTF-8',xml_declaration=True,default_namespace=None,method='xml')
	print("KML file written to filtered.kml")

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
	xml_path  = glob.glob(DATA_DIR + '/' + s2_id + '/*.xml')[0]
	datastrip = parse_xml(xml_path)
	date,tile = s2_id.split('_')[2:6:3]
	gee_id    = '_'.join([date,datastrip,tile])
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


def folder_check():
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
		dstrip   = parse_xml(xml_path)
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
			print("NO LABEL --> Removing folder %s" % folder)
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


def chip_image(product,index,N):
	print(f'[{index}/{N}] PROCESSING {product.id} ')
	# rgbn_fnames = get_band_filenames(product.safe_id)
	# rgbn_readers = [rio.open(f'{DATA_DIR}/{product.safe_id}/{f}','r') for f in rgbn_fnames]

	# NORMALIZE BANDS
	rgbn = []
	for reader in product.s2_readers:
		# print(f'Loading {reader.name[-34:]}')
		band_array = reader.read(1)
		zero_mask  = band_array == 0
		cutoff     = np.percentile(band_array[~zero_mask],99)
		band_array = np.clip(band_array,0,cutoff)
		band_array = (band_array / cutoff * 255).astype(np.uint8)
		rgbn.append(band_array)

	#SPLIT WINDOWS
	s2_windows = get_windows(product.s2_borders)
	dw_windows = get_windows(product.dw_borders)	
	n_proc   = 8
	share    = len(s2_windows) // n_proc
	leftover = len(s2_windows) % n_proc
	start    = [i*share for i in range(n_proc)]
	stop     = [i*share+share for i in range(n_proc)]
	stop[-1] += leftover
	s2_chunks = [s2_windows[s0:s1] for s0,s1 in zip(start,stop)]
	dw_chunks = [dw_windows[s0:s1] for s0,s1 in zip(start,stop)]	

	lock = mp.Lock()

	#THROW WORKERS AT ARRAYS
	processes = []
	for i in range(n_proc):
		p = mp.Process(
			target=chip_image_worker,
			args=(rgbn,product.dw_path,s2_chunks[i],dw_chunks[i],product.base_chip_id,lock)
			)
		p.start()
		processes.append(p)

	for p in processes:
		p.join()
	print("All workers done.")


def chip_image_worker(rgbn,dw_path,s2_windows,dw_windows,base_id,lock):

	stats = []
	lbl_rdr = rio.open(dw_path,'r',tiled=True)

	for k,(rowcol,w) in enumerate(s2_windows):

		lbl_arr = lbl_rdr.read(1,window=dw_windows[k][1])

		# CHECK LABEL NO DATA
		if (lbl_arr == 0).any():
			continue

		# CHECK WATER/LAND RATIO
		n_water = (lbl_arr==1).sum()
		if n_water < WATER_MIN or n_water > WATER_MAX:
			continue

		# ALL GOOD -- SAVE BANDS
		row = rowcol[0]
		col = rowcol[1]
		for band,name in zip(rgbn,['B02','B03','B04','B08']):
			outfile = f'{CHIP_DIR}/{base_id}_{row:02}_{col:02}_{name}.tif'
			img = Image.fromarray(band[w.row_off:w.row_off+CHIP_SIZE, w.col_off:w.col_off+CHIP_SIZE])
			img.save(outfile)
		
		# ALL GOOD -- SAVE LABEL
		outfile = f'{CHIP_DIR}/{base_id}_{row:02}_{col:02}_LBL.tif'
		lbl_arr[lbl_arr!=1] = 0 #everything else (already checked for zeroes above)
		lbl_arr[lbl_arr==1] = 255 #water
		img = Image.fromarray(lbl_arr)
		img.save(outfile)

		stats.append(f'{outfile.split("/")[-1]}\t{n_water}\n')

	# LOG
	lock.acquire()
	# print(f'Worker {mp.current_process()} done.')	
	with open(f'{CHIP_DIR}/stats.txt','a') as fp:
		for line in stats:
			fp.write(line)
	lock.release()


if __name__ == '__main__':

	if not os.path.isdir(DATA_DIR):
		print("DATA_DIR not found. EXITING.")
		sys.exit()
	print(f"DATA_DIR set to: {DATA_DIR}")	
	if len(glob.glob('*.SAFE',root_dir=DATA_DIR)) == 0 :
		print("EMPTY DATA_DIR")
	print(f"LABEL_DIR set to: {LABEL_DIR}")
	print(f"CHIP_DIR set to: {CHIP_DIR}")

	if args.clean:
		folder_check() 

	if args.chips:
		#.SAFE folders in data directory
		folders = glob.glob('*.SAFE',root_dir=DATA_DIR)
		paths   = glob.glob(DATA_DIR+'/*.SAFE')

		# Check everything is there
		if not os.path.isdir(LABEL_DIR):
			print("LABEL_DIR not found. EXITING.")
			sys.exit()

		#make chip dir if not already there
		# ---> cannot check which chips already there because cannot know chips per raster.
		if not os.path.isdir(CHIP_DIR):
			os.mkdir(CHIP_DIR) 

		#clean log file
		# if os.path.isfile(CHIP_DIR+'/stats.txt'):
			# os.remove(CHIP_DIR+'/stats.txt')

		N = len(folders)
		for i,f in enumerate(folders):
			product = Product(f) #load metadata
			chip_image(product,i,N) #chip

	if args.plots:
		pass

	if args.kml:
		filter_tile_kml()


