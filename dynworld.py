import ee #earth engine Python API
import rasterio as rio
from PIL import Image
import os
import pyproj

PX_PER_REGION = 1000

# ee.Authenticate()
ee.Initialize()

####################################################################################################
#FUNCTIONS
####################################################################################################
# PARAMETER SETTING
def download(ee_object,crs,region):
	if isinstance(ee_object, ee.Image):
		print('Downloading single image...')
		url = ee_object.getDownloadUrl({
				'scale':10,
				'crs': crs,
				'format':'GEO_TIFF',
				'region': region
			})

	# if isinstance(ee_object, ee.ImageCollection.ImageCollection):
	# 	print('Downloading image collection...')
	# 	obj_copy = ee_object.mosaic()
	# 	url     = obj_copy.getDownloadUrl({
	# 			'scale':10,
	# 			'crs': 'EPSG:4326',
	# 			'region': region
	# 		})

def region():

	pass

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


####################################################################################################
# MAIN
####################################################################################################
# ITERATE AND CALL DL
if __name__ == '__main__':
	test_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1")
	id_str = "20190101T182741_20190101T182744_T11SQV"
	lbl = ee.Image('GOOGLE/DYNAMICWORLD/V1/' + id_str).select('label')

	#TODO -- get properties first 
	lbl_crs = lbl.getInfo()['bands'][0]['crs']
	lbl_tsf = lbl.getInfo()['bands'][0]['crs_transform']
	# print("CRS: " + gee_crs)
	# url = download(tlbl)
	# print("Got URL: ", url)

	#TODO
	# s2_file_0 = "S2A_MSIL2A_20190101T182741_N0211_R127_T11SQV_20190101T205441.SAFE/GRANULE/"
	# s2_file_1 = "S2B_MSIL2A_20191122T182649_N0213_R127_T11SQA_20191122T204330.SAFE/GRANULE/"

	#TODO -- 

	
# geometry = ee.Geometry.Rectangle([80.058, 26.347, 82.201, 28.447])
# region = geometry.toGeoJSONString()#region must in JSON for
# path = downloader(MyImage,region)#call function


#POLYGON TO COORD
# for cor in polygon_coor_geo:
#     cur_cor_idx = img_RGB.index(cor[0],cor[1])
#     polygon_coor_matrix.append((cur_cor_idx[1],cur_cor_idx[0]))

# polygon_coor_matrix