import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

xData = np.linspace(0.0, 10.0, 1000000)
noise = np.random.randn(len(xData))
yTrue = (0.5) * xData + 5 + noise
xDF = pd.DataFrame(data=xData, columns=['X Data'])
yDF = pd.DataFrame(data = yTrue, columns=['Y'])
myData = pd.concat([xDF, yDF], axis=1)

featCols = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=featCols)

xTrain, xEval, yTrain, yEval = train_test_split(xData, yTrue, test_size=0.3, random_state=101)

inputFunc = tf.compat.v1.estimator.inputs.numpy_input_fn({'x': xTrain}, yTrain, batch_size=8, num_epochs=None, shuffle=True)
trainInputFunc = tf.compat.v1.estimator.inputs.numpy_input_fn({'x': xTrain}, yTrain, batch_size=8, num_epochs=1000, shuffle=False)
evalInputFunc = tf.compat.v1.estimator.inputs.numpy_input_fn({'x': xEval}, yEval, batch_size=8, num_epochs=1000, shuffle=False)

estimator.train(input_fn=inputFunc, steps=1000)
trainMetrics = estimator.evaluate(input_fn=trainInputFunc, steps=1000)
evalMetrics = estimator.evaluate(input_fn=evalInputFunc, steps=1000)
print("TRAINING DATA METRICS")
print(trainMetrics)
print("EVAL METRICS")
print(evalMetrics)

brandNewData = np.linspace(0, 10, 10)
inputFNPredict = tf.estimator.inputs.numpy_input_fn({'x': brandNewData}, shuffle=False)
print(list(estimator.predict(inputFNPredict)))
predictions = []
for pred in estimator.predict(inputFNPredict):
	predictions.append(pred['predictions'])
print(predictions)
myData.sample(250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(brandNewData, predictions, 'r*')
plt.show()