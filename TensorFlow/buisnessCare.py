import numpy as np
from sklearn import preprocessing
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


class AudioBooksDataReader:
	def __init__(self, dataset, batchSize=None):
		npz = np.load(f'Audiobooks_data_{dataset}.npz')
		self.inputs, self.targets = npz['inputs'].astype(np.float), npz['targets'].astype(np.int)
		if batchSize is None:
			self.batchSize = self.inputs.shape[0]
		else:
			self.batchSize = batchSize
		self.currBatch = 0
		self.batchCount = self.inputs.shape[0] // self.batchSize
		return

	def __next__(self):
		if self.currBatch >= self.batchCount:
			self.currBatch = 0
			raise StopIteration
		batchSlice = slice(self.currBatch * self.batchSize, (self.currBatch + 1) * self.batchSize)
		inputsBatch = self.inputs[batchSlice]
		targetsBatch = self.targets[batchSlice]
		self.currBatch += 1
		classesNum = 2
		targetsOneHot = np.zeros((targetsBatch.shape[0], classesNum))
		targetsOneHot[range(targetsBatch.shape[0]), targetsBatch] = 1
		return inputsBatch, targetsOneHot

	def __iter__(self):
		return self
	pass



# preprocess that THICC data
rawCSVData = np.loadtxt("Audiobooks_data.csv", delimiter=",")

unscaledInputAll = rawCSVData[:, 1:-1]
targetsAll = rawCSVData[:, -1]

# balance the datty set
numOneTargets = int(np.sum(targetsAll))
zeroCount = 0
indicesToRemove = []
for i in range(targetsAll.shape[0]):
	if targetsAll[i] == 0:
		zeroCount += 1
		if zeroCount > numOneTargets:
			indicesToRemove.append(i)

unscaledInputWithEqualPriors = np.delete(unscaledInputAll, indicesToRemove, axis=0)
targetsEqualPriors = np.delete(targetsAll, indicesToRemove, axis=0)

# scale inputs
scaledInputs = preprocessing.scale(unscaledInputWithEqualPriors)

# shuffle and truffle
# SHUFFLE EM REAL GOOD

shuffledIndices = np.arange(scaledInputs.shape[0])
np.random.shuffle(shuffledIndices)

shuffledInputs = scaledInputs[shuffledIndices]
shuffledTargets = targetsEqualPriors[shuffledIndices]

# split to train valid and test

sampleCount = shuffledInputs.shape[0]

trainSampleCount = int(0.8 * sampleCount)
validationSampleCount = int(0.1* sampleCount)
testSampleCount = sampleCount - trainSampleCount - validationSampleCount

trainInputs = shuffledInputs[:trainSampleCount]
trainTargets = shuffledTargets[:trainSampleCount]

validationInputs = shuffledInputs[trainSampleCount:trainSampleCount+ validationSampleCount]
validationTargets = shuffledTargets[trainSampleCount:trainSampleCount+ validationSampleCount]

testInputs = shuffledInputs[trainSampleCount+ validationSampleCount:]
testTargets = shuffledTargets[trainSampleCount+validationSampleCount:]

np.savez('Audiobooks_data_train', inputs=trainInputs, targets=trainTargets)
np.savez('Audiobooks_data_validation', inputs=validationInputs, targets=validationTargets)
np.savez('Audiobooks_data_test', inputs=testInputs, targets=testTargets)


# 10 x 50 x 50 x 2
inputSize = 10
outputSize = 2
hiddenLayerSize = 50

# reset all variables in memory
tf.compat.v1.reset_default_graph()

# declare placeholders
inputs = tf.placeholder(tf.float32, [None, inputSize])
targets = tf.placeholder(tf.int32, [None, outputSize])


# declare weights and biases
weights1 = tf.get_variable("weights1", [inputSize, hiddenLayerSize])
biases1 = tf.get_variable("biases1", [hiddenLayerSize])

# get the outputs
# tf.nn holds the activation functions
outputs1 = tf.nn.relu(tf.matmul(inputs, weights1) + biases1)

weights2 = tf.get_variable("weights2", [hiddenLayerSize, hiddenLayerSize])
biases2 = tf.get_variable("biases2", [hiddenLayerSize])

outputs2 = tf.nn.relu(tf.matmul(outputs1, weights2) + biases2)

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

maxEpochs = 50

prevValidationLoss = 99999999999

trainData = AudioBooksDataReader('train', batchSize)
validationData = AudioBooksDataReader('validation')

# train the model
for epochCounter in range(maxEpochs):
	currEpochLoss = 0
	for inputBatch, targetBatch in trainData:
		_, batchLoss = sess.run([optimizer, meanLoss],
		feed_dict={inputs: inputBatch, targets: targetBatch})
		currEpochLoss += batchLoss

	currEpochLoss /= trainData.batchCount
	validationLoss = 0
	validationAccuracy = 0
	for inputBatch, targetBatch in validationData:
		validationLoss, validationAccuracy = sess.run([meanLoss, accuracy],
		feed_dict={inputs: inputBatch, targets: targetBatch})
	print(f"Epoch {str(epochCounter + 1)}")
	print(f"training loss: {currEpochLoss}")
	print(f"validation loss: {validationLoss}")
	print(f"validation accuracy: {validationAccuracy * 100}%")
	if validationLoss > prevValidationLoss:
		break
	prevValidationLoss = validationLoss
print("End of training")

# test the model
testData = AudioBooksDataReader('test')
for inputBatch, targetBatch in trainData:
		testAccuracy = sess.run([accuracy],
		feed_dict={inputs: inputBatch, targets: targetBatch})

print(f"test accuracy: {testAccuracy[0] * 100}%")