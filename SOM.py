import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
import math

# because 42 is the answer to life, the universe
np.random.seed(42)

class SOM(object):
	# numData is the number of data to be organized
	# dimension is the number of dimensions the data has, which can be pixel dimensions
	def __init__(self, mapSize, numData, dimension, epsilon):
		self.weights = self.initWeights(mapSize, dimension)
		self.learningRate = epsilon
		self.numData = numData
		self.mapSize = mapSize
		self.dimension = dimension

	def initWeights(self, n, d):
		table = []
		# number of rows times
		for i in range(n):
			row = []
			# number of columns
			for j in range(n):
				# represents one point in the table which is an array of dimension length
				row.append(np.random.rand(d))

			table.append(np.array(row))

		return np.array(table)

	def getWeights(self):
		return self.weights

	def train(self, inputData, timeStep):
		# compute the best matching unit for that node
		bmu = self.BMU(inputData[timeStep])

		coordinates = np.indices((self.mapSize, self.mapSize)).swapaxes(0, 2).swapaxes(0, 1)

		distance = coordinates - bmu

		distance = np.array([[np.linalg.norm(distance[i][j])] * self.dimension for i in range(len(distance)) for j in range(len(distance[i]))])

		distance = distance.reshape(self.mapSize, self.mapSize, self.dimension)

		lRatio = self.learningRatio(timeStep)

		lRadius = self.learningRadius(timeStep, distance)

		self.weights += lRatio * lRadius * (inputData[timeStep] - self.weights)


	# from all the nodes finds the best matching unit by calculating
	# the euclidian distance
	def BMU(self, inputData):
		# convert the inputData to numpy array so that we can use numpy methods on it
		inputData = np.array(inputData)

		# to make shallow copy and avoid the same variable pointing to the same variable
		bmu = copy.copy(self.weights)

		# following the formula for euclidian distance
		bmu -= inputData
		bmu = np.square(bmu)

		minimumVal = float("inf")

		for i in range(len(bmu)):
			for j in range(len(bmu[i])):
				sumVal = np.sum(bmu[i][j])
				if sumVal < minimumVal:
					minimumVal = sumVal
					index = (i, j)

		# argmin returns just flatten, serial index, 
		# so convert it using unravel_index
		return index

	def learningRatio(self, iterationNum):
		return self.learningRate * math.exp((1 - iterationNum) / float(self.numData / 4))

	def learningRadius(self, iterationNum, distance):
		return np.exp(-distance**2 / (2 * math.pow((float(self.mapSize / 2) * math.exp((1 - iterationNum) / float(self.numData / 4))), 2)))

def main():
	randomPixels = np.random.rand(1000,3)

	som = SOM(20, 1000, 3, 0.1)

	plt.imshow(som.getWeights(), interpolation='none')
	plt.savefig("init.png")

	for i in tqdm(range(1000)):
		som.train(teachers, i)

	plt.imshow(som.getWeights(), interpolation='none')
	plt.savefig("final.png")



if __name__ == "__main__":
	main()