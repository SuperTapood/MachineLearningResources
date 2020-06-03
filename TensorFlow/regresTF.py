import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

xData = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(xData))

# y = mx + b
# b = 5

yTrue = (0.5) * xData + 5 + noise
xDF = pd.DataFrame(data=xData, columns=['X Data'])
yDF = pd.DataFrame(data = yTrue, columns=['Y'])
myData = pd.concat([xDF, yDF], axis=1)
batchSize = 8
m = tf.Variable(np.random.randn(1)[0])
b = tf.Variable(np.random.randn(1)[0])

xph = tf.placeholder(tf.float64, [batchSize])
yph = tf.placeholder(tf.float64, [batchSize])

yModel = m*xph+b
error = tf.reduce_sum(tf.square(yph-yModel))

optimizer = tf.train.GradientDescentOptimizer(0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	batches = 10000
	for i in range(batches):
		print(i)
		randInd = np.random.randint(len(xData), size=batchSize)
		feed = {xph: xData[randInd], yph: yTrue[randInd]}
		sess.run(train, feed_dict=feed)
	modelM, modelB = sess.run([m, b])
print(modelM)
print(modelB)
yHat = xData * modelM + modelB
myData.sample(250).plot(kind='scatter', x='X Data', y="Y")
plt.plot(xData, yHat)
plt.show()