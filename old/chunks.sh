#!/bin/bash
#ALTERNATIVE TO THE SCRIPT IN 'pipeline.sh'. HERE ALL CHUNKS ARE STORED IN TEMP NUMBERED TXTs.

# DATA_DIR=/cache #nrp
# DATA_DIR=../cache #local
N_TRANSFERS=10 #local
ORIG_BUCKET=nrp:lake-tiles-clean
DOWNLOAD_ALL=0
CHUNK_NR=-1
CREATE=0

while test $# -gt 0; do
	case "$1" in
		-s)
			CREATE=1
			shift
			;;
		-c)
			shift
			if test $# -gt 0; then
				CHUNK_NR=$1
			else
				echo "no chunk number given"
				exit 1
			fi
			shift
			;;
		-o)
			shift
			if test $# -gt 0; then
				DATA_DIR=$1
			else
				echo "no output dir specified"
				exit 1
			fi
			shift
			;;
		-a)
			shift
			DOWNLOAD_ALL=1
			;;
		*)
			break
			;;
	esac
done


#--------------------- DEFINE CHUNKS TO PROCESS ----------------------
if [ $CREATE == 1 ]; then
	# Rclone dir to list
	string_list=$(rclone lsf ${ORIG_BUCKET} | grep .SAFE | awk '{print substr($0,1,length($0)-1)}')
	array=(${string_list})

	# Chunk math
	chunk_size=50
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
fi

#--------------------- RUN:COPY AND CHIP -----------------------------
# Essentially we're running all repeatedly:
# rclone copy --include-from ./chunks/chunk_0.txt nrp:s2-lakes-clean . -P --transfers 32
# python3 ../s2-lake-masks/chipping.py --data-dir . --chip-dir ./chips --chips
# rm -r ./*.SAFE
# rclone copy ./chips/ nrp:lake-chips -P --transfers 32
# rm -r ./chips/*.tif
if [ $DOWNLOAD_ALL == 1 ]; then
	echo "DOWNLOADING ALL PRODUCTS"

	# DOWNLOAD ENTIRE LABEL SET IN REMOTE 
	rclone copy nrp:s2-lakes-clean/dynamicworld ${DATA_DIR}/dynamicworld -P --transfers ${N_TRANSFERS}

	for (( i=0; i<n_chunks; i++ )); do

		#.SAFE TO LOCAL CACHE
		rclone copy --include-from ${DATA_DIR}/chunk_${i}.txt nrp:s2-lakes-clean ${DATA_DIR} -P --transfers ${N_TRANSFERS}

		#MAKE CHIPS
		python3 chipping.py --data-dir ${DATA_DIR} --chip-dir ${DATA_DIR}/chips --chips

		#CLEAN UP -- REMOVE .SAFE
		rm -r ${DATA_DIR}/*.SAFE

		#PUSH CHIPS (SO FAR) TO S3, CLEAN UP
		# rclone copy ${DATA_DIR}/chips nrp:lake-chips -P --transfers ${N_TRANSFERS}
		# rm  ${DATA_DIR}/chips/*.tif
	done

	#.SAFE TO LOCAL CACHE
	rclone copy --include-from ${DATA_DIR}/chunk_${n_chunks}.txt ${ORIG_BUCKET} ${DATA_DIR} -P	--transfers ${N_TRANSFERS}

	#MAKE CHIPS
	python3 chipping.py --data-dir ${DATA_DIR} --chip-dir ${DATA_DIR}/chips --chips

	#CLEAN UP -- REMOVE .SAFE
	rm -r ${DATA_DIR}/*.SAFE
	
	#PUSH CHIPS (SO FAR) TO S3, CLEAN UP
	# rclone copy ${DATA_DIR}/chips nrp:lake-chips -P --transfers ${N_TRANSFERS}
	# rm  ${DATA_DIR}/chips/*.tif
else

	if [ $CHUNK_NR != -1 ]; then

		if [ ! -d "${DATA_DIR}/dynamicworld" ]; then
		  echo "Directory does not exist"
		  # DOWNLOAD ENTIRE LABEL SET IN REMOTE 
		  rclone copy ${ORIG_BUCKET}/dynamicworld ${DATA_DIR}/dynamicworld -P --transfers ${N_TRANSFERS}	  
		fi


		rclone copy --include-from ${DATA_DIR}/chunk_${CHUNK_NR}.txt ${ORIG_BUCKET} ${DATA_DIR} -P --transfers ${N_TRANSFERS}
		python3 chipping.py --data-dir ${DATA_DIR} --chip-dir ${DATA_DIR}/chips --chips
		# rm -r ${DATA_DIR}/*.SAFE

	fi

fi

