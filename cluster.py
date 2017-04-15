import cv2
import os
import numpy as np
from scipy.misc import imread, imresize
from kmeans import TFKMeansCluster

K = 4
#DATA_DIR = "../../data/ImagesTrain_Sorted/Outdoors"
DATA_DIR = "../../data/Outdoors_subset"
IMAGE_PREFIX = "../../data/Outdoors_Clustered"
Results = "../../data/KMeans_results.txt"


def main():

	vectors = getData()


	print("Entering K-Means")

	centroids, assignments, cluster_assigns = TFKMeansCluster(vectors, K)

	f = open(Results, 'w')
	f.write("Cluster_assigns: \n" + str(cluster_assigns))

	for cluster in range(len(cluster_assigns)):
		for image_title in cluster_assigns[cluster]:
			image = imread(DATA_DIR + "/" + image_title)
			cv2.imwrite(IMAGE_PREFIX + "/" + str(cluster) + "/" + image_title, image)


# Get list of vectors
def getData():
	vectors = []

	for jpg in os.listdir(DATA_DIR):

		image_path = DATA_DIR + "/" + jpg
		image = imread(image_path)

		image = image.astype(np.float64)

		if len(image.shape) == 3:
			vectors.append(tuple([image, jpg]))


	# print(vectors)
	return vectors

if __name__ == "__main__":
	main()