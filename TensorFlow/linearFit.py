import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

"""# set seed for tf and np to verify results
np.random.seed(101)
tf.set_random_seed(101)
randA = np.random.uniform(0, 100, (5, 5))
# print(randA)
randB = np.random.uniform(0, 100, (5, 1))
# print(randB)

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
addOp = a + b
mulOp = a * b
with tf.Session() as sess:
	addResult = sess.run(addOp, {a:randA, b:randB})
	# print(addResult, end="\n\n")
	mulResult = sess.run(mulOp, {a:randA, b: randB})
	# print(mulResult)"""
"""
nFeatures = 10
# 1 layer of 3 neurons
nDenseNeurons = 3
x = tf.placeholder(tf.float32, (None, nFeatures))
W = tf.Variable(tf.random_normal([nFeatures, nDenseNeurons]))
b = tf.Variable(tf.ones([nDenseNeurons]))
xW = tf.matmul(x, W)
z = tf.add(xW, b)
a = tf.nn.sigmoid(z)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	layerOut = sess.run(a, {x:np.random.random([1, nFeatures])})
print(layerOut)
"""
xData = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
print(xData)
yLabels = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
print(yLabels)
plt.plot(xData, yLabels, "*")
# plt.show()
# y = mx + b
m = tf.Variable(np.random.randn(1)[0])
b = tf.Variable(np.random.randn(1)[0])
error = 0
for x, y in zip(xData, yLabels):
	yHat = m * x + b
	error += (y - yHat) ** 2
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	trainingSteps = 100
	for i in range(trainingSteps):
		print(i)
		sess.run(train)
	finalSlope, finalIntercept = sess.run([m, b])
xTest = np.linspace(-1, 11, 10)
yPred = finalSlope * xTest + finalIntercept
plt.plot(xTest, yPred, 'r')
plt.show()