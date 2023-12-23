import os
import time
import numpy as np
import ee #earth engine Python API
import rasterio as rio
import xml.etree.ElementTree as ET
import sys

from typing import Tuple, List
# ee.Authenticate()
# ee.Initialize()

####################################################################################################
#GLOBAL VARS, NAMESPACES, ETC.
####################################################################################################
ns = {
	'n1':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd",
	'other':"http://www.w3.org/2001/XMLSchema-instance",
	'another':"https://psd-14.sentinel2.eo.esa.int/PSD/User_Product_Level-2A.xsd"
	}

DATA_DIR = os.getenv('DATA_DIR')
if DATA_DIR is None: #local dir, probably empty if cloned
	DATA_DIR = "./dat/"

if not os.path.isdir(DATA_DIR):
	print("DATA_DIR variable not set. No dir %s" % DATA_DIR)
	print("Exiting.")
	sys.exit(1)

DW_PALETTE_10 = [
	'00ff00','419bdf', '397d49', '88b053', '7a87c6',
	'e49635', 'dfc35a', 'c4281b', 'a59b8f', 'b39fe1']

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

	"""

	#fail if wrong path
	assert os.path.isfile(path), "No file found in path %s" % path

	#get root and branchesx
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
	xml_path =  [d for d in os.listdir(DATA_DIR + s2_img_id) if d[-3:]=='xml'][0]
	dstrip = parse_xml(DATA_DIR + s2_img_id + '/' + xml_path)
	date,tile = s2_img_id.split('_')[2:6:3]
	gee_id = '_'.join([date,dstrip,tile])
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


def select_shift_unmask(ee_id):
	dw = ee.Image('GOOGLE/DYNAMICWORLD/V1/' + ee_id)
	lbl_shifted_unmasked = dw.select('label').add(1).unmask().uint8()
	return lbl_shifted_unmasked


########################################################################################### -- TODO:
def create_task(ee_image: ee.Image, ee_id: str, s2_rdr: rio.io.DatasetReader) -> ee.batch.Task:
	'''
	Parameters
	----------
	ee_image : ee.Image
		A google earth engine object pointing to the DynamicWorld file we want
	ee_id : str
		A string with the GEE id of ee_image (following GEE product naming convention for S2)
	s2_rdr: rio.io.DatasetReader
		A rasterio object pointing to a local sentinel 2 image corresponding to the label image

	Returns
	-------
	task : ee.batch.Task
		Task object that can be used to start and check export task to google drive
	'''

	if type(s2_rdr) is rio.io.DatasetReader:
		s2_crs   = s2_rdr.crs.to_string()
		s2_crs_t = str([int(_) for _ in s2_rdr.transform[0:6]]).strip('[]')
	
	if s2_rdr is None:
		s2_rdr   = ee.Image('COPERNICUS/S2_SR_HARMONIZED/' + ee_id).select('B2')
		s2_crs   = s2_rdr.getInfo()['bands'][0]['crs'] #from GEE
		s2_crs_t = str(s2_rdr.getInfo()['bands'][0]['crs_transform']).strip('[]') #from GEE

	task = ee.batch.Export.image.toDrive(
		image=ee_image,
		description=ee_id,
		folder="dynamic_world_export",
		fileNamePrefix=ee_id,
		scale=10,
		crs=s2_crs,
		crsTransform=s2_crs_t,
		maxPixels=1e9,
		fileFormat='GeoTIFF'
		)

	return task


def start_task(task: ee.batch.Task ,id: str):
	print("Launching Drive task %s..." % id)
	task.start()


def check_empty_files(safe_folders):
	for folder in safe_folders:
		for file in os.listdir(DATA_DIR + folder):
			file_size = os.path.getsize(DATA_DIR + folder + '/' + file)
			if file_size <= 92:
				print("%s\t%s\t%i" % (folder,file,file_size))



####################################################################################################
# MAIN
####################################################################################################
# ITERATE AND CALL DOWNLOAD TASK
if __name__ == '__main__':

	print("Running ee.Initialize().")
	ee.Initialize()

	safe_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR + d) and d[-4:]=='SAFE']
	ee_ids,tasks = [],[]

	# check_empty_files(safe_folders)

	# FOR EACH .SAFE -- GET IDs AND CREATE TASKS
	for folder in safe_folders:
		#GET IDs and EE Image
		print("Parsing xml in %s.." % folder)
		ee_id  = get_gee_id(folder)
		ee_ids.append(ee_id)
		ee_img = select_shift_unmask(ee_id)

		#LOAD S2 DATA
		s2_b_path = DATA_DIR + folder + '/' + get_band_file_path(folder,'B02')
		s2_b_read = rio.open(s2_b_path,'r')

		#CREATE TASK
		print("Creating Drive task for product %s..." % folder)		
		task = create_task(ee_img,ee_id,s2_b_read)
		tasks.append(task)

		#CLEAN UP
		s2_b_read.close()

	# LAUNCH TASKS
	for i,t in enumerate(tasks):
		start_task(t,ee_ids[i])
