import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

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
saver = tf.train.Saver()
with tf.Session() as sess:
	sess.run(init)
	trainingSteps = 10000
	for i in range(trainingSteps):
		print(i)
		sess.run(train)
	finalSlope, finalIntercept = sess.run([m, b])
	saver.save(sess, 'models/model.ckpt')
with tf.Session() as sess:
	saver.restore(sess, 'models/model.ckpt')
	restoredSlope, restoredIntercept = sess.run([m, b])
xTest = np.linspace(-1, 11, 10)
yPred = restoredSlope * xTest + restoredIntercept
plt.plot(xTest, yPred, 'r')
plt.plot(xData, yLabels, '*')
plt.show()