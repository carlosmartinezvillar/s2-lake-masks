#!/bin/bash

DATA_DIR=/cache
N_TRANSFERS=64
DEST_BUCKET=nrp:lake-chips-512
ORIG_BUCKET=nrp:s2-lakes-clean

#==================================================
# SET CHUNKS OF .SAFE FOLDERS TO DOWNLOAD
#==================================================
# Rclone dir to list
string_list=$(rclone lsf ${ORIG_BUCKET} | grep .SAFE | awk '{print substr($0,1,length($0)-1)}')
array=(${string_list})

# Chunk math
chunk_size=64
n_chunks=$(((${#array[@]}) / chunk_size))
remainder=$((${#array[@]} - chunk_size * n_chunks))

#==================================================
# DOWNLOAD WHOLE LABEL SET IN REMOTE 
#==================================================
rclone copy ${ORIG_BUCKET}/dynamicworld ${DATA_DIR}/dynamicworld -P --transfers ${N_TRANSFERS}

#==================================================
# ITERATE--DOWNLOAD, CHIP & PUSH
#==================================================

# EVEN CHUNKS
#--------------------- 
for (( i=0; i<n_chunks; i++ )); do
	echo "BATCH: ${i}"

	#SAVE CHUNK TO TEMP FILE
	start=$((i * chunk_size))
	chunk=("${array[@]:$start:$chunk_size}")
	printf "%s/**\n" "${chunk[@]}" > "${DATA_DIR}/chunk.txt"

	#DOWLOAD/READ CHUNK
	rclone copy --include-from ${DATA_DIR}/chunk.txt ${ORIG_BUCKET} ${DATA_DIR} -P --transfers ${N_TRANSFERS}

	#MAKE CHIPS
	python3 preprocs.py --data-dir ${DATA_DIR} --chip-dir ${DATA_DIR}/chips --chips

	#CLEAN UP .SAFE
	rm -r ${DATA_DIR}/*.SAFE

	#PUSH CHIPS (SO FAR) TO S3
	rclone copy ${DATA_DIR}/chips ${DEST_BUCKET} -P --transfers ${N_TRANSFERS}

	#CLEAN UP TIFFs
	rm  ${DATA_DIR}/chips/*.tif
done

# REMAINDER
#--------------------- 
echo "BATCH: ${n_chunks}"
start=$((n_chunks * chunk_size))
chunk=("${array[@]:$start:$remainder}")
printf "%s/**\n" "${chunk[@]}" > "${DATA_DIR}/chunk.txt"

#.SAFE TO LOCAL CACHE
rclone copy --include-from ${DATA_DIR}/chunk.txt ${ORIG_BUCKET} ${DATA_DIR} -P	--transfers ${N_TRANSFERS}

#CHIPS
python3 preprocs.py --data-dir ${DATA_DIR} --chip-dir ${DATA_DIR}/chips --chips

#CLEAN UP -- REMOVE .SAFE
rm -r ${DATA_DIR}/*.SAFE

#PUSH CHIPS (SO FAR) TO S3
rclone copy ${DATA_DIR}/chips ${DEST_BUCKET} -P --transfers ${N_TRANSFERS}

#CLEAN UP TIFFs
rm  ${DATA_DIR}/chips/*.tif
rm ${DATA_DIR}/chunk.txt
