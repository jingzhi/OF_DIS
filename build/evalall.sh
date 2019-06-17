#!/bin/bash/
folderPath='/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/'
folders=($(ls $folderPath))
all_flo=$1
for ((i=0; i<${#folders[@]}; i++));do
        imgPath="$folderPath${folders[$i]}"
        gtPath="/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/flow/${folders[$i]}"
        imgs=($(ls $imgPath))
	for ((j=0; j<${#imgs[@]}-1; j++));do
		img1=${imgs[$j]} 
		img2=${imgs[$j+1]} 
		out=${img1//png/flo}
		#echo $gtPath/$out ./all_flo_out/${folders[$i]}/$out
		echo ${folders[$i]}/$out
	        /scratch/lijingz/MasterThesis/OF_DIS/Ideas/evalall $gtPath/$out ./${all_flo}/${folders[$i]}/$out -r=$2
		echo EOF
	done
done

#run_OF_RGB /scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/ambush_4/frame_0001.png /scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/ambush_4/frame_0002.png test.flo
