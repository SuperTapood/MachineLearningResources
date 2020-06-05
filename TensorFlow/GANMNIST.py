import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from time import time

def convert_t(t):
	if t < 60:
		return f"00:00:{int(t)}"
	elif t < 3600:
		s = t % 60
		m = t // 60
		return f"00:{int(m)}:{int(s)}"
	else:
		h = t // 3600
		r = t % 3600
		m = r // 60
		s = r % 60
		return f"{int(h)}:{int(m)}:{int(s)}"
	return

mnist = input_data.read_data_sets("../mnist", one_hot=True)

def generator(z, reuse=None):
	with tf.variable_scope('gen', reuse=reuse):
		hidden1 = tf.layers.dense(z, 128)
		alpha = 0.01
		hidden1 = tf.nn.leaky_relu(hidden1, alpha)
		# hidden1 = tf.maximum(alpha*hidden1, hidden1)
		hidden2 = tf.layers.dense(hidden1, 128)
		hidden2 = tf.nn.leaky_relu(hidden2, alpha)
		# hidden2 = tf.maximum(alpha*hidden2, hidden2)
		output = tf.layers.dense(hidden2, 784, activation=tf.nn.tanh)
		return output
	return

def discriminator(X, reuse=None):
	with tf.variable_scope('dis', reuse=reuse):
		hidden1 = tf.layers.dense(X, 128)
		alpha = 0.01
		hidden1 = tf.nn.leaky_relu(hidden1, alpha)
		# hidden1 = tf.maximum(alpha*hidden1, hidden1)
		hidden2 = tf.layers.dense(hidden1, 128)
		hidden2 = tf.nn.leaky_relu(hidden2, alpha)
		# hidden2 = tf.maximum(alpha*hidden2, hidden2)
		logits = tf.layers.dense(hidden2, 1)
		output = tf.sigmoid(logits)
		return output, logits
	return

real_images = tf.placeholder(tf.float32, [None, 784])
z = tf.placeholder(tf.float32, [None, 100])

G = generator(z)

D_output_real, D_logits_real = discriminator(real_images)

D_output_fake, D_logits_fake = discriminator(G, reuse=True)

# LOSSES
def loss_func(logits, labels):
	return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

D_real_loss = loss_func(D_logits_real, tf.ones_like(D_logits_real)*0.9)
D_fake_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_fake)*0.9)

D_loss = D_real_loss + D_fake_loss

G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))

learning_rate = 0.001

tvars = tf.trainable_variables()

d_vars = [v for v in tvars if 'dis' in v.name]
g_vars = [v for v in tvars if 'gen' in v.name]

D_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(D_loss, var_list=d_vars)
G_trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(G_loss, var_list=g_vars)

batch_size = 100
epochs = 16000
init = tf.global_variables_initializer()
samples = []
with tf.Session() as sess:
	sess.run(init)

	for e in range(epochs):
		s = time()
		num_batches = mnist.train.num_examples // batch_size

		for i in range(num_batches):
			batch = mnist.train.next_batch(batch_size)
			batch_images = batch[0].reshape((batch_size, 784))
			batch_images = batch_images * 2 - 1
			batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
			_ = sess.run(D_trainer, feed_dict={real_images: batch_images, z: batch_z})
			_ = sess.run(G_trainer, feed_dict={z: batch_z})
		en = time() - s
		print(f"EPOCH {e}. ETA {convert_t(en * (epochs - e))}. {int(100 * (100 * (e / epochs))) / 100}%")
		sample_z = np.random.uniform(-1, 1, size=(1, 100))
		gen_sample = sess.run(generator(z, True), feed_dict={z: sample_z})
		samples.append(gen_sample)


fig = plt.figure(figsize=(15, 5))
r = 1
c = 6
for img, i in zip(samples[len(samples)-5:], range(1, r*c, 1)):
	fig.add_subplot(r, c, i)
	plt.imshow(img.reshape(28, 28))
plt.show()