import numpy as np

def MNIST_load(num_patterns):
	"""
	Randomly draws a number of examples
	from the mnist dataset.
	Readily converts the pattern arrays
	into binary ∈ {-1, 1},
	as Hopfield neurons are bipolar binary.
	If array is composed ∈ {0, 1}, energy
	function won't work properly.
	"""
	from keras.datasets import mnist
	(X, y), (_, _ ) = mnist.load_data()
	idx = np.random.randint(len(X), size = num_patterns)
	X = X[idx]
	y = y[idx]
	X = (X > 128).astype(int)
	X = np.where(X == 0, -1, 1).astype(int)
	X = X.reshape(-1, 784)
	y = y.astype(np.int32)

	return X, y