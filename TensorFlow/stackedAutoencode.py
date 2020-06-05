import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("../03-Convolutional-Neural-Networks/MNIST_data/", one_hot=True)

tf.reset_default_graph()

# 784
# 392
# 196
# 392
# 784

num_inputs = 784
neurons_hid1 = 392
neurons_hid2 = 196
neurons_hid3 = neurons_hid1
num_outputs = num_inputs
learning_rate = 0.01
actf = tf.nn.relu
X = tf.placeholder(tf.float32, [None, num_inputs])
initalizer = tf.variance_scaling_initializer()

w1 = tf.Variable(initalizer([num_inputs, neurons_hid1]), dtype=tf.float32)
w2 = tf.Variable(initalizer([neurons_hid1, neurons_hid2]), dtype=tf.float32)
w3 = tf.Variable(initalizer([neurons_hid2, neurons_hid3]), dtype=tf.float32)
w4 = tf.Variable(initalizer([neurons_hid3, num_outputs]), dtype=tf.float32)

b1 = tf.Variable(tf.zeros(neurons_hid1))
b2 = tf.Variable(tf.zeros(neurons_hid2))
b3 = tf.Variable(tf.zeros(neurons_hid3))
b4 = tf.Variable(tf.zeros(num_outputs))

hid_layer1 = actf(tf.matmul(X, w1) + b1)
hid_layer2 = actf(tf.matmul(hid_layer1, w2) + b2)
hid_layer3 = actf(tf.matmul(hid_layer2, w3) + b3)
output_layer = actf(tf.matmul(hid_layer3, w4) + b4)

loss = tf.reduce_mean(tf.square(output_layer - X))
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

num_epochs = 5
batch_size = 150

with tf.Session() as sess:
	sess.run(init)
	for e in range(num_epochs):
		num_batches = mnist.train.num_examples // batch_size
		for i in range(num_batches):
			X_batch, y_batch = mnist.train.next_batch(batch_size)
			feed_dict = {X: X_batch}
			sess.run(train, feed_dict)
		training_loss = loss.eval(feed_dict=feed_dict)
		print(f"EPOCH {e}: LOSS {training_loss}")
	saver.save(sess, "./example_stacked_autoencoder.ckpt")

num_test_images = 10
with tf.Session() as sess:
	saver.restore(sess, "./example_stacked_autoencoder.ckpt")
	results = output_layer.eval({X:mnist.test.images[:num_test_images]})
f, a = plt.subplots(2, 10, figsize=(20, 4))
for i in range(num_test_images):
	a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
	a[1][i].imshow(np.reshape(results[i], (28, 28)))
plt.show()