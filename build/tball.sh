#!/bin/bash/
param_config="5"
folderPath='/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/'
folders=($(ls $folderPath))
rm -f -r all_flo_out
mkdir all_flo_out
for ((i=0; i<${#folders[@]}; i++));do
        imgPath="$folderPath${folders[$i]}"
        gtPath="/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/flow/${folders[$i]}"
        imgs=($(ls $imgPath))
	mkdir all_flo_out/${folders[$i]}
	for ((j=0; j<${#imgs[@]}-1; j++));do
		img1=${imgs[$j]} 
		img2=${imgs[$j+1]} 
		out=${img1//png/flo}
		echo ./run_OF_RGB $imgPath/$img1 $imgPath/$img2 ./all_flo_out/${folders[$i]}/$out $param_config
		./run_OF_RGB $imgPath/$img1 $imgPath/$img2 ./all_flo_out/${folders[$i]}/$out $param_config
	done
done

#run_OF_RGB /scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/ambush_4/frame_0001.png /scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/ambush_4/frame_0002.png test.flo
