from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.contrib.layers import fully_connected


wine_data = load_wine()

feat_data = wine_data['data']
labels = wine_data['target']

X_train, X_test, y_train, y_test = train_test_split(feat_data, labels, test_size=0.3, random_state=101)

scaler = MinMaxScaler()

scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

onehot_y_train = pd.get_dummies(y_train).values
onehot_y_test = pd.get_dummies(y_test).values

num_feat = 13
num_hidden1 = 13
num_hidden2 = 13
num_outputs = 3
learning_rate = 0.01


X = tf.placeholder(tf.float32, shape=[None, num_feat])
y_true = tf.placeholder(tf.float32, shape=[None, 3])

actf = tf.nn.relu

hidden1 = fully_connected(X, num_hidden1, actf)
hidden2 = fully_connected(hidden1, num_hidden2, actf)
output = fully_connected(hidden2, num_outputs)
loss = tf.losses.softmax_cross_entropy(y_true, output)
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

training_steps = 1000
with tf.Session() as sess:
	sess.run(init)
	for i in range(training_steps):
		print(i)
		sess.run(train, {X: scaled_x_train, y_true:onehot_y_train})
	logits = output.eval({X: scaled_x_test})
	preds = tf.argmax(logits, 1)
	results = preds.eval()
print(classification_report(results, y_test))