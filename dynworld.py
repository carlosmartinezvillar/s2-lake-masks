import os
import time
import numpy as np
import ee #earth engine Python API
import rasterio as rio
import xml.etree.ElementTree as ET
import sys
import subprocess sp

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
	try:
		task.start()
	except Exception as e:
		print("Error launching task. Skipping...")


def check_empty_files(safe_folders):
	fp = open('empty_s3_files')

	for folder in safe_folders:
		for file in os.listdir(DATA_DIR + folder):
			file_size = os.path.getsize(DATA_DIR + folder + '/' + file)
			if file_size <= 92:
				# print("%s\t%s\t%i" % (folder,file,file_size))
				fp.write("%s\t%s\t%i\n" % (folder,file,file_size))


def download_empty_files():
	with open('empty_s3_files.tsv') as fp:
		empty_files = fp.readlines()

	prev_safe_folder = ""
	prev_subdir      = ""

	for row in empty_files:

		safe_folder = row.split()[0]
		band_file   = row.split()[1]

		if band_file.split('_')[2] == 'SCL':
			continue

		if prev_safe_folder == safe_folder:
			subdir = prev_subdir
			# print(" "*len(safe_folde)," ",subdir," ",band_file)
		else:
			datetime = safe_folder.split("_")[2]
			y = datetime[0:4]
			m = datetime[4:6]
			d = datetime[6:8]
			src1 = '/'.join(["esa:/EODATA/Sentinel-2/MSI/L2A",y,m,d,safe_folder,"GRANULE"])
			p1 = sp.run(["rclone","lsd",src1],stdout=sp.PIPE)
			subdir = p1.stdout.decode().split()[-1]
			prev_subdir = subdir
			# print(safe_folder," ",subdir," ",band_file)

		src2 = '/'.join([src1,subdir,"IMG_DATA","R10m",band_file])
		dest = DATA_DIR + safe_folder
		p2 = sp.run(["rclone","copy",src2,dest,"-P"])

		prev_safe_folder = safe_folder


def download_errored_files():
	with open('error.tsv') as fp:
		safe_folders = [row.split()[1] for row in fp.readlines()]

	current_folders = [d for d in os.listdir(DATA_DIR) if d[-5:]=='.SAFE']

	for folder in safe_folders:

		# if folder in current_folders:
			# continue
		if len(os.listdir(DATA_DIR+folder)) > 4: #non-empty safe folder
			continue

		datetime = folder.split('_')[2]
		y = datetime[0:4]
		m = datetime[4:6]
		d = datetime[6:8]	
		safe_src = '/'.join(["esa:/EODATA/Sentinel-2/MSI/L2A",y,m,d,folder]) #.SAFE

		xml_src = safe_src + '/' + "MTD_MSIL2A.xml" #XML

		p1       = sp.run(["rclone","lsd",safe_src+"/GRANULE"],stdout=sp.PIPE) #BANDS
		subdir   = p1.stdout.decode().split()[-1]
		tile     = folder.split('_')[5]

		for b in ["B02","B03","B04","B08"]:
			band_file = '_'.join([tile,datetime,b,"10m.jp2"])
			band_src  = '/'.join([safe_src,"GRANULE",subdir,"IMG_DATA","R10m",band_file])
			p2 = sp.run(["rclone","copy",band_src,DATA_DIR+folder,"-P"])
			print("%s/%s written." % (DATA_DIR+folder,band_file))

		p3 = sp.run(["rclone","copy",xml_src,DATA_DIR+folder,"-P"])
		print("%s/%s written." % (DATA_DIR+folder,"MTD_MSIL2A.xml"))


####################################################################################################
# MAIN
####################################################################################################
# ITERATE AND CALL DOWNLOAD TASK
if __name__ == '__main__':

	# download_empty_files()
	# download_errored_files()

	print("Running ee.Initialize().")
	ee.Initialize()

	safe_folders = [d for d in os.listdir(DATA_DIR) if os.path.isdir(DATA_DIR + d) and d[-4:]=='SAFE']
	ee_ids,tasks = [],[]

	# check_empty_files(safe_folders)
	prev_tasks = [i['description'] for i in ee.batch.data.getTaskList()]

	# FOR EACH .SAFE -- GET IDs AND CREATE TASKS
	for i,folder in enumerate(safe_folders):
		#GET IDs and EE Image
		print("[%i/%i] Parsing xml in %s.." % (i,len(safe_folders),folder))
		ee_id  = get_gee_id(folder)

		if ee_id in prev_tasks:
			continue

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

