import subprocess as sp
import os

DATA_DIR = os.getenv('DATA_DIR') #trailing /

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