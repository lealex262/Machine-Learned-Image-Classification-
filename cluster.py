import cv2
import os
import numpy as np
from scipy.misc import imread, imresize
from kmeans import TFKMeansCluster

K = 4
#DATA_DIR = "../../data/ImagesTrain_Sorted/Outdoors"
DATA_DIR = "../../data/Outdoors_subset"
Results = "../../data/KMeans_results"


def main():

	vectors = getData()

	#print(vectors[0][0][0][0])

	print("Entering K-Means")
	centroids, assignments, cluster_assigns = TFKMeansCluster(vectors, K)

	f = open(Results, 'w')
	f.write("Centroids: \n" + str(centroids))
	f.write("Assignments: \n" + str(assignments))
	f.write("Cluster_assigns: \n" + str(cluster_assigns))







# Get list of vectors
def getData():
	vectors = []

	for jpg in os.listdir(DATA_DIR):

		image_path = DATA_DIR + "/" + jpg
		image = imread(image_path)

		image = image.astype(np.float64)

		if len(image.shape) == 3:
			vectors.append(tuple([image, image_path]))


	# print(vectors)
	return vectors

if __name__ == "__main__":
	main()