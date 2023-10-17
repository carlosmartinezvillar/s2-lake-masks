import os
import time
import numpy as np
import ee #earth engine Python API
import rasterio as rio
import xml.etree.ElementTree as ET
from PIL import Image
# import pyproj

from typing import Tuple, List
# ee.Authenticate()
# ee.Initialize()

####################################################################################################
# WORKFLOW
####################################################################################################
'''
--> list of folders/product ids
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

####################################################################################################
#FUNCTIONS
####################################################################################################
def parse_xml(path: str) -> str:
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
	"""

	# path check
	assert os.path.isfile(path), "No file found in path %s" % path

	# ----------
	# root.find('n1:General_Info',ns)
	# 	.find('Product_Info')
	# 		.find('Product_Organisation')
	# 			.find('Granule_List')
	# 				.find('Granule').attrib['datastripIdentifier']
	# ----------

	#add try-except-raise:
	root      = ET.parse(path).getroot() #elementree to element
	prod_info = root.find('n1:General_Info',namespaces=ns).find('Product_Info')
	granule   = prod_info.find('Product_Organisation').find('Granule_List').find('Granule')
	# granule   = [e for e in prod_info.iter('Granule')][0]
	datastrip = granule.attrib['datastripIdentifier'].split('_')[-2][1:]
	return datastrip


def download(ee_object,crs,region):
	if isinstance(ee_object, ee.Image):
		print('Downloading single image...')
		url = ee_object.getDownloadUrl({
				'scale':10,
				'crs': crs,
				'format':'GEO_TIFF',
				'region': region
			})

def create_export_task(ee_image: ee.Image, id: str, crs: str, crs_t: str) -> ee.batch.Task:

	task = ee.batch.Export.image.toDrive(
		image=ee_image,
		description=,
		folder="dynamic_world_export",
		scale=10,
		crs=crs,
		crsTransform=
		format='GEO_TIFF'
		)

	# maybe this?...
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


def split_image():
	#open one image/band and get crs

	#calculate region of new image via raster indices
	pass

def build_gee_id(s2_image_id):

	#open metadata to retrieve datastrip id
	pass

def reproject_coordinates():
	pass

def open_sentinel_image(path):
	img_ptr = rio.open(path)

def align_labels_to_input(s2_reader,dw_reader):
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
	dstrip  = parse_xml('./dat/'+s2prod+'/MTD.xml')

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

	
