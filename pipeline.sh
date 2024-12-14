#!/bin/bash

# DATA_DIR=/cache
DATA_DIR=~/Desktop/lake_chips
N_TRANSFERS=8

string_list=$(rclone lsf nrp:s2-lakes-clean | grep .SAFE | awk '{print substr($0,1,length($0)-1)}')
array=(${string_list})

# COPY LABEL SET IN REMOTE TO /${DATA_DIR}/dynamicworld
rclone copy nrp:s2-lakes-clean/dynamicworld ${DATA_DIR}/dynamicworld -P --transfers ${N_TRANSFERS}

chunk_size=25
n_chunks=$(((${#array[@]}) / chunk_size))
remainder=$((${#array[@]} - chunk_size * n_chunks))

for (( i=0; i<n_chunks; i++ )); do
	#SET CHUNK
	start=$((i * chunk_size))
	end=$((start + chunk_size - 1 ))
	chunk=("${array[@]:$start:$chunk_size}")

	#SAVE CHUNK TO TEMP FILE
	# printf "%s/**\n" "${chunk[@]}" > "${DATA_DIR}/chunk_${i}.txt"
	printf "%s/**\n" "${chunk[@]}" > "${DATA_DIR}/chunk.txt"
	cat ${DATA_DIR}/chunk.txt | wc

	# DOWLOAD
	# rclone copy --include-from chunks/chunk_${i}.txt nrp:s2-lakes-clean ${DATA_DIR} -P --transfers 16
	rclone copy --include-from ${DATA_DIR}/chunk.txt nrp:s2-lakes-clean ${DATA_DIR} -P --transfers ${N_TRANSFERS}


	#MAKE CHIPS
	python3 preprocs.py --data-dir ${DATA_DIR} --chip-dir ${DATA_DIR}/chips --chips

	#CLEAN UP -- REMOVE .SAFE USED
	rm -r ${DATA_DIR}/*.SAFE

	#SAVE CHIPS SO FAR, CLEAN UP
	rclone copy ${DATA_DIR}/chips nrp:lake-chips -P --transfers ${N_TRANSFERS}
	rm  ${DATA_DIR}/chips/*.tif
done

# DO THE REMAINDER:
# start=$((n_chunks * chunk_size))
# end=$((start+remainder))
start=$((${#array[@]}-remainder))
end=$((${#array[@]}))
chunk=("${array[@]:$start:$remainder}")

#TEMP CHUNK
printf "%s/**\n" "${chunk[@]}" > "${DATA_DIR}/chunk.txt"

#TRANSFER TO LOCAL CACHE
rclone copy --include-from ${DATA_DIR}/chunk.txt nrp:s2-lakes-clean ${DATA_DIR} -P	--transfers ${N_TRANSFERS}
#CHIPS
python3 preprocs.py --data-dir ${DATA_DIR} --chip-dir ${DATA_DIR}/chips --chips
rm -r ${DATA_DIR}/*.SAFE
#CHIPS IN CACHE TO STORAGE
rclone copy ${DATA_DIR}/chips nrp:lake-chips -P --transfers ${N_TRANSFERS}
rm  ${DATA_DIR}/chips/*.tif

