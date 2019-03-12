import cv2 as cv
import glob
import matplotlib.pyplot as plt
pathToFolder = '/scratch_net/unclemax/lijingz/MasterThesis/MPI-Sintel-complete/training/final/ambush_4/'
images = [cv.imread(file) for file in glob.glob(pathToFolder+'*png')]

#flow= cv.DenseOpticalFlow.calc(images[3],images[4])
flow=cv.calcOpticalFlowFarneback(images[3],images[4],0.5,3,15,3,5,1.2,0)
print(flow)
