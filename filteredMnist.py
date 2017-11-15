from sklearn import datasets
from sklearn.datasets import fetch_mldata

def filteredMnist():
	print("Fetching the dataset...")
	digits = fetch_mldata('MNIST original', data_home=".\\")
	
	# contains many 2 dimensional array of pixel values which represents digits
	images = digits.data

	# contains labels for the pixel values
	labels = digits.target

	# this array will store the indexes for which the the values 1, 2, 3, 4 and 5 occur for labels
	indexValues = []

	# we populate the indexValues array with only values which represent the indexes of label array
	# for which the value is less than 6
	for i in range(len(labels)):
		# labels need to be greater than 0 and less than 6, both of the expression need to be true
		# for the entire expression to be true
		if labels[i] == 1 or labels[i] == 5:
			indexValues.append(i)

	# will hold the array of images of 1 to 5 only
	filteredImages = []
	# will hold the numbers 1 to 5 only
	filteredLabels = []

	for i in range(len(indexValues)):
		filteredImages.append(images[indexValues[i]])
		filteredLabels.append(labels[indexValues[i]])

	print("Images and labels filtered for values only 1 and 5...")


	return (filteredImages, filteredLabels)
