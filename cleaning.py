import os
import xml.etree.ElementTree as ET
import glob

LABEL_DIR = None
DATA_DIR = None

def parse_xml(path):

	# check path
	assert os.path.isfile(path), "No file found in path %s" % path

	# get datastrip
	root      = ET.parse(path).getroot()
	prod_info = root.find('n1:General_Info',namespaces=ns).find('Product_Info')
	granule   = prod_info.find('Product_Organisation').find('Granule_List').find('Granule')
	datastrip = granule.attrib['datastripIdentifier'].split('_')[-2][1:]

	return datastrip


def folder_check():
	'''
	1.Remove .SAFE folders for products without a matching dynanmic world label.
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
			print("NO LABELS --> Removing folder %s" % folder)
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

if __name__ == '__main__':

	# ARGV
	parser = argparse.ArgumentParser(
		prog="s2-lakes-preprocessing/cleaning.py",
		description="Clean the data directory of Sentinel-2 and DynamicWorld V1 images to remove \
		incomplete products, products without a matching label, and labels without matching \
		products. Also deletes two overlapping tiles (T11SKD and T11TKE).")
	parser.add_argument('--data-dir',default=None,help="Dataset directory")
	args = parser.parse_args()
	DATA_DIR = args.data_dir

	# No argument passed
	if DATA_DIR is None:
		print("No data directory argument given.")
		sys.exit(1)

	# Incorrect path
	if not os.path.isdir(DATA_DIR):
		print("Data directory not found. Exiting.")
		sys.exit(1)
	print(f"Data directory set to:  {DATA_DIR}")

	# Empty directory
	if len(glob.glob('*.SAFE',root_dir=DATA_DIR)) == 0:
		print("EMPTY DATA_DIR")
		sys.exit(1)

	# Run
	folder_check()
