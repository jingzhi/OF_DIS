#!/bin/bash/
gtPath='/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/flow/alley_1/frame_0021.flo'
#calcPath='/scratch/lijingz/MasterThesis/OF_DIS/build/ref_pt4/'
calcPath='./frame_0021.flo'

./benchmark $gtPath $calcPath -display -r=smalld
//for flo in $(ls $calcPath);do
//	./benchmark $gtPath$flo $calcPath$flo -display -r=smalld
//done


