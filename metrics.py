'''
THIS script generates some images and plots to evaluate the quality of the products, labels, and
chips produced.
'''
import os
import zipfile
import rasterio as rio
from rasterio.windows import Window
import mapltolib.pyplot as plt
import glob
import time
import argparse
import numpy as np

import preprocs

plt.style.use('fast')
plt.rcParams['font.family'] = 'Courier'
plt.rcParams['font.size'] = 10

def plot_raster_lbl_binary(out_path,P,chip_size=512):
	'''
	Plot workable area overlapping windows. Black and white classes.

	Parameters
	----------
	out_path:
	P: preprocs.Product() class.
	chip_size:

	'''
	h = P.dw_borders['bottom'] + 1 - P.dw_borders['top']
	w = P.dw_borders['right'] + 1 - P.dw_borders['left']
	windows_height = h - (h % chip_size) + 1
	windows_width  = w - (w % chip_size) + 1

	kwargs = P.dw_reader.meta.copy()
	kwargs.update({'height':windows_height,'width':windows_width,'count':3,'compress':'lzw'})
	out_ptr = rio.open(out_path,'w',**kwargs)

	dw_windows = preprocs.get_windows_strided(P.dw_borders,chip_size,stride=chip_size)

	for _,w in dw_windows:
		#read
		arr        = P.dw_reader.read(1,window=w)
		white_mask = arr == 1 #water
		gray_mask  = arr == 0 #nodata
		black_mask = ~(white_mask | gray_mask) #land

		#change
		arr[white_mask] = 255
		arr[black_mask] = 0
		arr[gray_mask]  = 128
		arr_3d = np.repeat(arr[np.newaxis,:,:],repeats=3,axis=0)

		#write
		out_win = Window(w.col_off-P.dw_borders['left'],w.row_off-dw_borders['top'],chip_size,chip_size)
		out_ptr.write(arr_3d,window=out_win)

	out_ptr.close()
	print("IMAGE WRITTEN TO: %s" % out_path)


def plot_raster_lbl_colors():
	DW_PALETTE_10 = ['000000','419bdf','397d49','88b053','7a87c6','e49635', 
	    'dfc35a','c4281b','a59b8f','b39fe1'];

	pass


def plot_raster_lbl_binary_windows(out_path,P,chip_size=512,stride=512):
	'''
	Plot raster with windows marked. Black and white classes.

	Parameters
	----------
	out_path:
	P: preprocs.Product() class.
	chip_size:

	'''
	H = P.dw_reader.height #original raster
	W = P.dw_reader.width
	h = P.dw_borders['bottom'] + 1 - P.dw_borders['top'] #trimmed raster (workable area)
	w = P.dw_borders['right'] + 1 - P.dw_borders['left']
	w_h = h - (h % chip_size) + 1 #raster area overlapping windows
	w_w = w - (w % chip_size) + 1


	#Get lines -- red FF0000, yellow FFFF00, green 00FF00
	# yellow_line    = np.ones((3,CHIP_SIZE))
	# yellow_line[0] = 65535
	# yellow_line[1] = 65535
	# yellow_line[2] = 0
	green_line     = np.ones((3,chip_size))
	green_line[0]  = 0
	green_line[1]  = 65535
	green_line[2]  = 0	
	red_line       = np.ones((3,chip_size))
	red_line[0]    = 65535
	red_line[1]    = 0
	red_line[2]    = 0

	kwargs = P.dw_reader.meta.copy()
	kwargs.update({'height':H,'width':W,'count':3,'compress':'lzw'})
	out_ptr = rio.open(out_path,'w',**kwargs)

	dw_windows = preprocs.get_windows_strided(P.dw_borders,chip_size,stride=chip_size)

	for _,w in dw_windows:
		#read
		arr        = P.dw_reader.read(1,window=w)
		white_mask = arr == 1 #water
		gray_mask  = arr == 0 #nodata
		black_mask = ~(white_mask | gray_mask) #land

		#change
		arr[white_mask] = 255
		arr[black_mask] = 0
		arr[gray_mask]  = 128
		arr_3d = np.repeat(arr[np.newaxis,:,:],repeats=3,axis=0)

		#write
		out_win = Window(w.col_off-P.dw_borders['left'],w.row_off-dw_borders['top'],chip_size,chip_size)
		out_ptr.write(arr_3d,window=out_win)

	out_ptr.close()
	print("IMAGE WRITTEN TO: %s" % out_path)


def plot_raster_rgb_windows():
	pass

def plot_raster_rgb_bounded():
	pass


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


def tile_date_distribution():
	chip_files_tr = glob("*_B0X.tif",root_dir=f"{chip_dir}/training")
	chip_files_va = glob("*_B0X.tif",root_dir=f"{chip_dir}/validation")
	chip_files_te = glob("*_B0X.tif",root_dir=f"{chip_dir}/testing")	

	chip_files = chip_files_tr + chip_files_va + chip_files_te

	pass


def tile_month_distribution():
	pass


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir',default='./dat',
		help="Dataset directory")
	parser.add_argument('--chip-dir',
		help="Chip directory")
	parser.add_argument('--kml',action='store_true',default=False,
		help="Read tile kml and filter.")
	parser.add_argument('--plots',action='store_true',default=False,
		help='Plot a sample of inputs and labels.')
	args = parser.parse_args()
	#check something here..
	return args

if __name__ == '__main__':
	args = parse_args()

	if args.kml:
		filter_tile_kml()

	if args.plots:
		plot_raster_lbl_binary('./fig/raster_lbl_binary.png',sample_product)


