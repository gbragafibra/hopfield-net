import numpy as np

def add_noise(X, noise_level=0.2):
	#Make noisy patterns
    X_noisy = X.copy()
    #num of pixels to change
    n_pixels_change = int(X.shape[1] * noise_level)
    for i in range(X.shape[0]):
    	#pick rnd idxs to change
        rnd_idxs = np.random.randint(X.shape[1], size=n_pixels_change)
        X_noisy[i, rnd_idxs] = -X_noisy[i, rnd_idxs] #Flip bits
    return X_noisy