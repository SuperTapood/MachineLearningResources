# this is the hello world of ML

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# read the data set and normalize it
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# outline the model
# 784 x 50 x 50 x 10
inputSize = 784
outputSize = 10
hiddenLayerSize = 100

# reset all variables in memory
tf.compat.v1.reset_default_graph()

# declare placeholders
inputs = tf.placeholder(tf.float32, [None, inputSize])
targets = tf.placeholder(tf.float32, [None, outputSize])


# declare weights and biases
weights1 = tf.get_variable("weights1", [inputSize, hiddenLayerSize])
biases1 = tf.get_variable("biases1", [hiddenLayerSize])

# get the outputs
# tf.nn holds the activation functions
outputs1 = tf.nn.tanh(tf.matmul(inputs, weights1) + biases1)

weights2 = tf.get_variable("weights2", [hiddenLayerSize, hiddenLayerSize])
biases2 = tf.get_variable("biases2", [hiddenLayerSize])

outputs2 = tf.nn.tanh(tf.matmul(outputs1, weights2) + biases2)

weights3 = tf.get_variable("weights3", [hiddenLayerSize, outputSize])
biases3 = tf.get_variable("biases3", [outputSize])


outputs = tf.matmul(outputs2, weights3) + biases3


loss = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=targets)

# find the mean of the loss
meanLoss = tf.reduce_mean(loss)

# declare optimization function
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(meanLoss)

# get accuracy (bool vec)
outEqualsTarget = tf.equal(tf.argmax(outputs, 1), tf.argmax(targets, 1))

accuracy = tf.reduce_mean(tf.cast(outEqualsTarget, tf.float32))


sess = tf.InteractiveSession()

initializer = tf.global_variables_initializer()

sess.run(initializer)

batchSize = 100

batchesNumber = mnist.train._num_examples // batchSize

maxEpochs = 15

prevValidationLoss = 99999999999

# make it learn BABYYYY

for ecount in range(maxEpochs):
	currEpochLoss = 0
	for bcount in range(batchesNumber):
		# for each batch
		inputBatch, targetBatch = mnist.train.next_batch(batchSize)
		_, batchLoss = sess.run([optimizer, meanLoss], feed_dict={inputs: inputBatch, targets: targetBatch})
		currEpochLoss += batchLoss
	currEpochLoss /= batchesNumber
	inputBatch, targetBatch = mnist.validation.next_batch(mnist.validation._num_examples)
	validationLoss, validationAccuracy = sess.run([meanLoss, accuracy], feed_dict={inputs: inputBatch, targets:targetBatch})
	print(f"Epoch {str(ecount + 1)}")
	print(f"training loss: {currEpochLoss}")
	print(f"validation loss: {validationLoss}")
	print(f"validation accuracy: {validationAccuracy * 100}%")
	if validationLoss > prevValidationLoss:
		break
	prevValidationLoss = validationLoss

# test our beautiful boi
inputBatch, targetBatch = mnist.test.next_batch(mnist.test._num_examples)
testAccuracy = sess.run([accuracy], feed_dict={inputs: inputBatch, targets: targetBatch})
print(f"test accuracy: {testAccuracy[0] * 100}%")
