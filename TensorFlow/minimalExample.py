import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



observations = 1000
# create some fake data
xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(-10, 10, (observations, 1))

# stack the observations
inputs = np.column_stack((xs, zs))

# check the shape of the inputs
# print(inputs.shape)

# create targets with noise to mix 
# this sh*t up a little
# AI will learn to cancel the noise out
noise = np.random.uniform(-1, 1, (observations, 1))
targets = 2*xs - 3*zs + 5 + noise
targets = targets.reshape(observations,)

# check shape again
# print(targets.shape)

# plot data
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(xs, zs, targets)
# ax.set_xlabel("xs")
# ax.set_ylabel("zs")
# ax.view_init(azim=100)
# plt.show()
targets = targets.reshape(observations,1)


# set the initial weights and biases
init_range = 0.1
weights = np.random.uniform(low=-init_range, high=init_range, size=(2, 1))
biases = np.random.uniform(low=-init_range, high=init_range, size=1)


print(weights)
print(biases)
learningRate = 0.04


# train the model
for i in range(10000):
	outs = np.dot(inputs, weights) + biases
	deltas = outs - targets
	loss = np.sum(deltas ** 2) / 2 / observations
	print(loss)
	print(i)
	scaledDeltas = deltas / observations
	weights = weights - learningRate * np.dot(inputs.T, scaledDeltas)
	biases = biases - learningRate * np.sum(scaledDeltas)


print(weights)
print(biases)

plt.plot(outs, targets)
plt.xlabel = ('outputs')
plt.ylabel = ('targets')
plt.show()