from SOM import SOM
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def main():
	randomPixels = np.random.rand(1000,3)

	som = SOM(20, 1000, 3, 0.1)

	plt.imshow(som.getWeights(), interpolation='none')
	plt.savefig("initialImage.png")

	for i in tqdm(range(1000)):
		som.train(randomPixels, i)

	plt.imshow(som.getWeights(), interpolation='none')
	plt.savefig("finalImage.png")



if __name__ == "__main__":
	main()