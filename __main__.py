from SOM import SOM
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from filteredMnist import filteredMnist


def displayImages(gridSize, gridData):
	row = 0
	col = 0

	for i in range(1, gridSize * 2 + 1):
		plt.subplot(gridSize, gridSize, i)
		fig = plt.imshow(gridData[row][col])
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)

	plt.show()


def main():
	images, labels = filteredMnist()

	som = SOM(20, len(images), len(images[0]), 0.1)

	# # plt.imshow(som.getWeights(), interpolation='none')
	# # plt.savefig("initialImage.png")

	# for i in tqdm(range(len(images))):
	# 	som.train(images, i)

	# # plt.imshow(som.getWeights(), interpolation='none')
	# # plt.savefig("finalImage.png")




if __name__ == "__main__":
	main()