import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


observations = 100000
# create some fake data
xs = np.random.uniform(low=-10, high=10, size=(observations, 1))
zs = np.random.uniform(-10, 10, (observations, 1))

# stack the observations
generatedInputs = np.column_stack((xs, zs))

# check the shape of the inputs
# print(inputs.shape)

# create targets with noise to mix 
# this sh*t up a little
# AI will learn to cancel the noise out
noise = np.random.uniform(-1, 1, (observations, 1))
generatedTargets = 2*xs - 3*zs + 5 + noise
np.savez('TF_intro', inputs=generatedInputs, targets=generatedTargets)

inputSize = 2
outputSize = 1

# add "data"
inputs = tf.compat.v1.placeholder(tf.float32, [None, inputSize])
targets = tf.compat.v1.placeholder(tf.float32, [None, outputSize])

# create the model
weights = tf.Variable(tf.random.uniform([inputSize, outputSize], -0.1, 0.1))
biases = tf.Variable(tf.random.uniform([outputSize], -0.1, 0.1))

# create outputs
outputs = tf.matmul(inputs, weights) + biases

# regular loss function
# mean_loss = tf.losses.mean_squared_error(labels=targets, predictions=outputs) / 2
# huber loss function
mean_loss = tf.losses.huber_loss(labels=targets, predictions=outputs) / 2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(mean_loss)

# prepare to execute the little f***

session = tf.InteractiveSession()

initializer = tf.global_variables_initializer()

session.run(initializer)

# load data
trainingData = np.load('TF_intro.npz')

# learn you little f***
for e in range(1000):
	_, currLoss = session.run([optimizer, mean_loss], feed_dict={inputs: trainingData['inputs'], targets: trainingData['targets']})
	print(currLoss)


outs = session.run([outputs], feed_dict={inputs: trainingData['inputs'], targets: trainingData['targets']})
plt.plot(np.squeeze(outs), np.squeeze(trainingData['targets']))
plt.xlabel = ('outputs')
plt.ylabel = ('targets')
# plt.show()