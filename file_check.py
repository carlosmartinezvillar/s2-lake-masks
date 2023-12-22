import subprocess as sp
import os

DATA_DIR = os.getenv('DATA_DIR') #trailing /


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


if __name__ == '__main__':

	# download_empty_files()
	download_errored_files()
