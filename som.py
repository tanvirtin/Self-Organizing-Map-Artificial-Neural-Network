import numpy as np
import random
import copy
import math
import matplotlib.pyplot as plt

# because 42 is the answer to life, the universe
np.random.seed(42)

class SOM(object):
	# mapSize is the grid size
	# numData is the number of data to be organized
	# dimension is the number of dimensions the data has, which can be pixel dimensions
	# epsilon is the learning rate
	def __init__(self, mapSize, numData, dimension, epsilon):
		self.weights = self.initWeights(mapSize, dimension)
		self.learningRate = epsilon
		self.numData = numData
		self.mapSize = mapSize
		self.dimension = dimension

	# initializes the weight matrix
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

	# a getter which returns the weight matrix
	def getWeights(self):
		return self.weights

	# trains the weight by updating on each iteration
	# the inputData needs to be the entire dataset and timestep is the number of iteration that the training is on
	def train(self, inputData, timeStep):
		# compute the best matching unit for that node
		bmu = self.BMU(inputData[timeStep])

		coordinates = np.indices((self.mapSize, self.mapSize)).swapaxes(0, 2).swapaxes(0, 1)

		distance = coordinates - bmu

		distance = np.array([[np.linalg.norm(distance[i][j])] * self.dimension for i in range(len(distance)) for j in range(len(distance[i]))])

		distance = distance.reshape(self.mapSize, self.mapSize, self.dimension)

		lRatio = self.learningRatio(timeStep)

		lRadius = self.learningRadius(timeStep, distance)

		# the weight now is updated
		self.weights += lRatio * lRadius * (inputData[timeStep] - self.weights)


	# from all the nodes finds the best matching unit by calculating
	# the euclidian distance
	def BMU(self, inputData):
		# convert the inputData to numpy array so that we can use numpy methods on it
		inputData = np.array(inputData)

		# to make shallow copy and avoid the same variable pointing to the same memory address
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

	def getPlottingData(self):
		table = []
		reshapeSize = int(np.sqrt(len(self.weights[0][0])))
		for i in range(self.mapSize):
			row = []
			for j in range(self.mapSize):
				row.append(self.weights[i][j].reshape((reshapeSize, reshapeSize)))
			table.append(row)
		return table
