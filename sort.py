import cv2 as cv
import os

label_file = "C:/Users/ButlerJacob/GeorgiaTech/Spring2017/HD-Hackathon/data/categoriesTrain.txt"
image_dir = "C:/Users/ButlerJacob/GeorgiaTech/Spring2017/HD-Hackathon/data/ImagesTrain"
dst_dir = "C:/Users/ButlerJacob/GeorgiaTech/Spring2017/HD-Hackathon/data/ImagesTrain_Sorted"

with open(label_file) as f:
	count = 0
	for line in f:
		category = line.split("|")[1]
		image_file = line.split("|")[2].strip()

		#print("Cat: " + str(category))
		#print("Image name: " + image_file)

		for image_subset in os.listdir(image_dir):

			#print("Image_subset: " + image_subset)

			for image in os.listdir(image_dir + "/" + image_subset):
				
				#if (image_file[:20] == image[:20]):
				#	print("image_file: " + image_file)
				#	print("image: " + image)
				
				image = str(image)

				if (image_file[:] == image[:]):
					src = image_dir + "/" + image_subset + "/" + image
					dst = dst_dir + "/" + category + "/" + image

					if not os.path.isfile(dst):
						os.rename(src, dst)
						print("Moved " + str(count))
						count += 1



print("Done!")

#a05740ca-7851-4d2f-8b78-afa7a76a8c4d_65.jpg
#a05740ca-7851-4d2f-8b78-afa7a76a8c4d_65.jpg