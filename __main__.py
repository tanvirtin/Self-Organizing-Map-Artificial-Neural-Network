from SOM import SOM
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from filteredMnist import filteredMnist


def displayImages(gridSize, gridData):
	row = 0
	col = 0

	for i in range(1, gridSize * gridSize + 1):
		plt.subplot(gridSize, gridSize, i)
		fig = plt.imshow(gridData[row][col])
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)
		# we increment column index each time in the loop
		col += 1
		# if the column index equals the gridSize we know
		# we reached our limit, we set col to 0 when col hits gridSize
		# which is the maximum index in the grid on the x axis
		if (col == gridSize):
			col = 0
			row += 1
			# when we hit maximum gridSize we know we go down the y axis
			if (row == gridSize):
				row = 0
	plt.show()


def main():
	images, labels = filteredMnist()

	gridSize = 20
	epsilon = 0.1
	trainingDataSize = len(images)
	trainingDataDimension = len(images[0])

	som = SOM(gridSize, trainingDataSize, trainingDataDimension, epsilon)

	for i in tqdm(range(len(images))):
		som.train(images, i)

	displayImages(gridSize, som.getPlottingData())


if __name__ == "__main__":
	main()