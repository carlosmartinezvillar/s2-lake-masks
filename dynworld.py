import os
import time
import numpy as np
import ee #earth engine Python API
import rasterio as rio
import xml.etree.ElementTree as ET
from PIL import Image
import math
# import pyproj

from typing import Tuple, List
# ee.Authenticate()
# ee.Initialize()

####################################################################################################
# WORKFLOW
####################################################################################################
'''
--> for each .SAFE folder
get_gee_id()
  parse_xml()


'''


####################################################################################################
#GLOBAL VARS, NAMESPACES, ETC.
####################################################################################################

ns = {
	'n1':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd",
	'other':"http://www.w3.org/2001/XMLSchema-instance",
	'another':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"
	}

DATA_DIR = "./dat/"

####################################################################################################
#FUNCTIONS
####################################################################################################
def parse_xml(path: str) -> str:
	"""
	Get the datastrip id from the xml metadata file found in path. The path is assumed to be full.

	Parameters
	----------
	path : str
		The string path to the xml file.

	Returns
	-------
	datastrip_id: str
		The extracted datastrip id

	Following the structure of the xml file:

		root.find('n1:General_Info',ns)
			.find('Product_Info')
				.find('Product_Organisation')
					.find('Granule_List')
						.find('Granule').attrib['datastripIdentifier']
	"""

	assert os.path.isfile(path), "No file found in path %s" % path 	             #fail if wrong path

	root      = ET.parse(path).getroot()
	prod_info = root.find('n1:General_Info',namespaces=ns).find('Product_Info')
	granule   = prod_info.find('Product_Organisation').find('Granule_List').find('Granule')
	# granule   = [e for e in prod_info.iter('Granule')][0]
	datastrip = granule.attrib['datastripIdentifier'].split('_')[-2][1:]
	
	return datastrip


def get_gee_id(s2_img_id: str) -> str:
	'''
	Read a Sentinel-2 image id string and returns the respective DynamicWorld product id.
	'''
	dstrip = parse_xml('./dat/' + s2_img_id + '/MTD.xml')
	date,tile = s2_img_id.split('_')[2:6:3]
	gee_id = '_'.join([date,dstrip,tile])
	return gee_id


def get_band_file(s2_img_id: str, band: str) -> str:
	'''
	Given a Sentinel-2 product name and band, return the band image file name
	'''
	date = s2_img_id[11:26]
	tile = s2_img_id[38:44]
	return '_'.join([tile, date, band, '10m.jp2'])


def get_band_files(s2_img_id: str, bands: [str]) -> [str]:
	'''
	Given a Sentinel-2 product name and band, return the band image file name
	'''
	return list(map(get_band_file,[s2_img_id]*len(bands),bands))


def trim_label_borders(dw_array):
	top,bottom,left,right = 0,dw_array.shape[0],0,dw_array.shape[1]

	print(top,bottom,left,right)

	while(dw_array[top,:].sum() == 0):
		top += 1

	while(dw_array[bottom-1,:].sum() == 0):
		bottom -= 1

	while(dw_array[:,left].sum() == 0):
		left += 1

	while(dw_array[:,right-1].sum() == 0):
		right -= 1

	print(top,bottom,left,right)
	return dw_array[top:bottom,left:right]
########################################################################################### -- TODO:
def download_from_drive():
	pass


def drive_file_check():
	pass


def create_export_task(ee_image: ee.Image, id: str, crs: str, crs_matrix: str) -> ee.batch.Task:

	task = ee.batch.Export.image.toDrive(
		image=ee_image,
		description=id,
		folder="dynamic_world_export",
		scale=10,
		crs=crs,
		crsTransform=crs_matrix,
		format='GEO_TIFF'
		)


	# maybe this with a single dict parameter?
	# task = ee.batch.Export.image.toDrive({
	# 	image: ee_image,
	# 	description:
	# 	folder: "dynamic_world_export",
	# 	scale: 10,
	# 	crs: crs,
	# 	crsTransform: crs_t,
	# 	format:'GEO_TIFF'
	# 	})

	return task

def check_export_task(task):
	task.status()

def reproject_coordinates():
	pass

def open_sentinel_image(path):
	img_ptr = rio.open(path)

def align(s2_reader,dw_reader):
	pass

def test_install():
	# ee.Authenticate()
	ee.Initialize()
	print(ee.Image("NASA/NASADEM_HGT/001").get("title").getInfo())

####################################################################################################
# MAIN
####################################################################################################
# ITERATE AND CALL DL
if __name__ == '__main__':
	# test_install()

	folders = [d for d in os.listdir('./dat/') if os.path.isdir('./dat/' + d)]

	s2prod  = folders[1]
	gee_id  = get_gee_id(s2prod)
	s2path = DATA_DIR + s2prod + '/' +  get_band_file(s2prod,'B02')
	dwpath = DATA_DIR + gee_id + '.tif'

	s2 = rio.open(s2path,'r',tiled=True,blockxsize=256,blockysize=256)
	dw = rio.open(dwpath,'r',tiled=True,blockxsize=256,blockysize=256)
	# align(s2,dw)
	dw_xy_ul = dw.xy(0,0,offset='center')
	dw_xy_ll = dw.xy(dw.height-1,0,offset='center')
	dw_xy_lr = dw.xy(dw.height-1,dw.width-1,offset='center')
	dw_xy_ur = dw.xy(0,dw.width-1,offset='center')

	s2_idx_ul = s2.index(dw_xy_ul[0],dw_xy_ul[1],op=math.floor)
	s2_idx_ll = s2.index(dw_xy_ll[0],dw_xy_ll[1],op=math.floor)
	s2_idx_lr = s2.index(dw_xy_lr[0],dw_xy_lr[1],op=math.floor)
	s2_idx_ur = s2.index(dw_xy_ur[0],dw_xy_ur[1],op=math.floor)

	dwarr = dw.read(1)
	dwarr = trim_label_borders(dwarr)
	pass

	# test_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
	# id_str = "20190101T182741_20190101T182744_T11SQV"
	# lbl = ee.Image('GOOGLE/DYNAMICWORLD/V1/' + id_str).select('label')

	#TODO -- get properties first 
	# lbl_crs = lbl.getInfo()['bands'][0]['crs']
	# lbl_tsf = lbl.getInfo()['bands'][0]['crs_transform']
	# print("CRS: " + gee_crs)
	# url = download(tlbl)
	# print("Got URL: ", url)

	#TODO
	# s2_file_0 = "S2A_MSIL2A_20190101T182741_N0211_R127_T11SQV_20190101T205441.SAFE/GRANULE/"
	# s2_file_1 = "S2B_MSIL2A_20191122T182649_N0213_R127_T11SQA_20191122T204330.SAFE/GRANULE/"

	#TODO -- 

	
