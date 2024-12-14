#!/bin/bash

DATA_DIR=/cache #nrp
N_TRANSFERS=64 #nrp
# DATA_DIR=~/Desktop/lake_chips #local
# N_TRANSFERS=8 #local
DOWNLOAD=0

#--------------------- DEFINE CHUNKS TO PROCESS ----------------------
# Rclone dir to list
string_list=$(rclone lsf nrp:s2-lakes-clean | grep .SAFE | awk '{print substr($0,1,length($0)-1)}')
array=(${string_list})

# Chunk math
chunk_size=25
n_chunks=$(((${#array[@]}) / chunk_size))
remainder=$((${#array[@]} - chunk_size * n_chunks))

# EVEN CHUNKS
for (( i=0; i<n_chunks; i++ )); do
	start=$((i * chunk_size))
	chunk=("${array[@]:$start:$chunk_size}")
	printf "%s/**\n" "${chunk[@]}" > "${DATA_DIR}/chunk_${i}.txt" #To FILE
done

# REMAINDER
start=$((n_chunks * chunk_size))
chunk=("${array[@]:$start:$remainder}")
printf "%s/**\n" "${chunk[@]}" > "${DATA_DIR}/chunk_${n_chunks}.txt"

#--------------------- RUN:COPY AND CHIP -----------------------------
if [ DOWNLOAD == 1 ];then
	# DOWNLOAD LABEL SET IN REMOTE 
	rclone copy nrp:s2-lakes-clean/dynamicworld ${DATA_DIR}/dynamicworld -P --transfers ${N_TRANSFERS}

	for (( i=0; i<n_chunks; i++ )); do

		#.SAFE TO LOCAL CACHE
		rclone copy --include-from ${DATA_DIR}/chunk_${i}.txt nrp:s2-lakes-clean ${DATA_DIR} -P --transfers ${N_TRANSFERS}

		#MAKE CHIPS
		python3 preprocs.py --data-dir ${DATA_DIR} --chip-dir ${DATA_DIR}/chips --chips

		#CLEAN UP -- REMOVE .SAFE
		rm -r ${DATA_DIR}/*.SAFE

		#PUSH CHIPS (SO FAR) TO S3, CLEAN UP
		rclone copy ${DATA_DIR}/chips nrp:lake-chips -P --transfers ${N_TRANSFERS}
		rm  ${DATA_DIR}/chips/*.tif
	done

	#.SAFE TO LOCAL CACHE
	rclone copy --include-from ${DATA_DIR}/chunk_${n_chunks}.txt nrp:s2-lakes-clean ${DATA_DIR} -P	--transfers ${N_TRANSFERS}

	#MAKE CHIPS
	python3 preprocs.py --data-dir ${DATA_DIR} --chip-dir ${DATA_DIR}/chips --chips

	#CLEAN UP -- REMOVE .SAFE
	rm -r ${DATA_DIR}/*.SAFE
	
	#PUSH CHIPS (SO FAR) TO S3, CLEAN UP
	rclone copy ${DATA_DIR}/chips nrp:lake-chips -P --transfers ${N_TRANSFERS}
	rm  ${DATA_DIR}/chips/*.tif
fi