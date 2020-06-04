import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

## helper
def init_weights(shape):
	init_random_distribution = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(init_random_distribution)

def init_biases(shape):
	init_bias_vals = tf.constant(0.1, shape=shape)
	return tf.Variable(init_bias_vals)

def conv2d(x, W):
	# x -> [batch,H,W,Channels]
	# w -> [filter H, filter W, Channels IN, Channels OUT]
	return tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')

def max_pool_2x2(x):
	# x same as in conv2d
	return tf.nn.max_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], "SAME")

def convolutional_layer(input_x, shape):
	W = init_weights(shape)
	b = init_biases([shape[3]])
	return tf.nn.relu(conv2d(input_x, W) + b)

def normal_full_layer(input_layer, size):
	input_size = int(input_layer.get_shape()[1])
	W = init_weights([input_size, size])
	b = init_biases([size])
	return tf.matmul(input_layer, W) + b

# PLACEHOLDERS
x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])
# LAYERS
x_image = tf.reshape(x, [-1, 28, 28, 1])

convo_1 = convolutional_layer(x_image, [5, 5, 1, 32])
convo_1_pooling = max_pool_2x2(convo_1)

convo_2 = convolutional_layer(convo_1_pooling, [5, 5, 32, 64])
convo_2_pooling = max_pool_2x2(convo_2)

convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7*7*64])
full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

# DROPOUT
hold_prob = tf.placeholder(tf.float32)
full_one_dropout = tf.nn.dropout(full_layer_one, hold_prob)

y_pred = normal_full_layer(full_one_dropout, 10)

# LOSS FUNCTION

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(cross_entropy)

init = tf.global_variables_initializer()

steps = 50000000
with tf.Session() as sess:
	sess.run(init)
	for i in range(steps):
		print(i)
		batch_X, batch_Y = mnist.train.next_batch(50)
		sess.run(train, {x: batch_X, y_true: batch_Y, hold_prob: 0.5})
		if i % 100 == 0:
			print(f"ON STEP: {i}")
			matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
			acc = tf.reduce_mean(tf.cast(matches, tf.float32))
			print(f"ACCURACY: {sess.run(acc, {x:mnist.test.images, y_true:mnist.test.labels, hold_prob:1.0}) * 100}%")
			print()