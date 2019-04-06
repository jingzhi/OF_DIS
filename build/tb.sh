#!/bin/bash/
param_config="5"

if [ "$2" = "" ];
then
imgFolder="alley_1/"
#imgFolder="cave_2/"
else
imgFolder=$2'/'
fi

if ["$1" = ""];
then
imgName="-all"
else
imgName=$1
fi

imgPath='/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/'$imgFolder
gtPath='/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/flow/'$imgFolder

imgs=($(ls $imgPath))
echo Running in $imgFolder... 
#echo ${imgs[@]}
echo ${imgName}
if [[ $imgName == "-all" ]];then
	for ((i=0; i<${#imgs[@]}-1; i++));do
		img1=${imgs[$i]} 
		img2=${imgs[$i+1]} 
		out=${img1//png/flo}
		echo ./run_OF_RGB $imgPath$img1 $imgPath$img2 $out $param_config
		./run_OF_RGB $imgPath$img1 $imgPath$img2 $out $param_config
	done
else
	img1=${imgs[$imgName]} 
	img2=${imgs[$imgName+1]} 
	out=${img1//png/flo}
	echo ./run_OF_RGB $imgPath$img1 $imgPath$img2 $out $param_config
	./run_OF_RGB $imgPath$img1 $imgPath$img2 $out $param_config
	../Ideas/benchmark $gtPath$out $out -display -r=smalld 

fi


#run_OF_RGB /scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/ambush_4/frame_0001.png /scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/ambush_4/frame_0002.png test.flo
