import cv2
import os
import numpy as np
from scipy.misc import imread, imresize
from kmeans import TFKMeansCluster

K = 4
#DATA_DIR = "../../data/ImagesTrain_Sorted/Outdoors"
DATA_DIR = "../../data/Outdoors_subset"


def main():

	vectors = getData()

	print("Entering K-Means")
	TFKMeansCluster(vectors, K)




# Get list of vectors
def getData():
	vectors = []

	for jpg in os.listdir(DATA_DIR):
		# print("JPG: " + jpg)
		image = imread(DATA_DIR + "/" + jpg)

		#print("img shape: " + str(image.shape))

		image = image.astype(np.float64)
		#cv2.imshow("a", image)
		#cv2.waitKey()
		vectors.append(image);

	# print(vectors)
	return vectors

if __name__ == "__main__":
	main()