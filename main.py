from MNIST_ import *
from hopfield_net import *
from utils import *

num_patterns = 2
X, y = MNIST_load(num_patterns)

X_noisy = add_noise(X) #Make noisy patterns

net = Hopfield_Net(X, y) #train
preds = net.inference(X_noisy, 20) #inference
net.reconstruction()
net.original_patterns()