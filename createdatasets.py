import os;
from random import shuffle


def create_data_sets():
    imagesPath = "ImagesTrain_Sorted/"
    textFilesPath = ""
    classifiers = {}
    classifierIndex = 0
    images_per_folder = []
    for folderName in os.listdir(imagesPath):
        classifiers.update({folderName: classifierIndex})
        classifierIndex += 1
        images_per_folder.append(len(os.listdir(imagesPath+"//"+folderName)))

    images_per_class = min(images_per_folder)
    percent_testing = .2
    testingNum = percent_testing * images_per_class

    trainingList = []
    testingList = []
    for folderName in os.listdir(imagesPath):
        currPath = imagesPath.__add__("/%s" % folderName)
        imageslist = os.listdir(currPath)
        shuffle(imageslist)
        for x in range(0, images_per_class):
            entry = imagesPath + folderName + "/" + imageslist[x] + "," + str(classifiers[folderName])
            if (x < testingNum):
                testingList.append(entry)
            else:
                trainingList.append(entry)
    shuffle(trainingList)
    shuffle(testingList)

    # Creates/overwrites existing text files for training and testing
    training = open(textFilesPath + "train.txt", "w+")
    testing = open(textFilesPath + "test.txt", "w+")
    # writes to training and testing text files
    for entry in trainingList:
        training.write(entry + "\n")
    for entry in testingList:
        testing.write(entry + "\n")

    # Closes the text files
    training.close()
    testing.close()


