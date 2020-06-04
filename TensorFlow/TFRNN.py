import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TimeSeriesData:
	def __init__(self, num_points, xmin, xmax):
		self.xmin = xmin
		self.xmax = xmax
		self.num_points = num_points
		self.resolution = (xmax-xmin) / num_points
		self.X_data = np.linspace(xmin, xmax, num_points)
		self.Y_true = np.sin(self.X_data)
		return

	def ret_true(self, x_series):
		return np.sin(x_series)

	def next_batch(self, batch_size, steps, return_batch_ts=False):
		# grab a random starting point
		rand_start = np.random.rand(batch_size, 1)

		# convert to be on the actual time series
		ts_start = rand_start * (self.xmax-self.xmin - (steps * self.resolution))

		# create a batch time series on the x axis
		batch_ts = ts_start + np.arange(0.0, steps + 1) * self.resolution

		# create y data for the time series
		y_batch = self.ret_true(batch_ts)

		# format for the RNN
		if return_batch_ts:
			return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1), batch_ts
		else:
			return y_batch[:, :-1].reshape(-1, steps, 1), y_batch[:, 1:].reshape(-1, steps, 1)
		return
	pass

ts_data = TimeSeriesData(250, 0, 10)
num_time_steps = 30
# y1, y2, ts = ts_data.next_batch(1, num_time_steps, True)
# plt.plot(ts.flatten()[1:], y2.flatten(), '*')
# plt.plot(ts_data.X_data, ts_data.Y_true, label="sin(t)")
# plt.legend()
# plt.show()

# TRAINING DATA
train_inst = np.linspace(5, 5 + ts_data.resolution * (num_time_steps + 1), num_time_steps + 1)
# print(train_inst)
# plt.plot(train_inst[:-1], ts_data.ret_true(train_inst[:-1]), 'bo', markersize=15, alpha=0.5, label='INSTANCE')
# plt.plot(train_inst[1:], ts_data.ret_true(train_inst[1:]), 'ko', markersize=7, label='TARGET')
# plt.legend()
# plt.show()

tf.reset_default_graph()

# CONSTANTS
num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.001
num_train_iterations = 2000
batch_size = 1

# PLACEHOLDERS
X = tf.placeholder(tf.float32, [None, num_time_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_time_steps, num_outputs])


# RNN LAYER CELL
cell = tf.contrib.rnn.GRUCell(num_neurons, tf.nn.relu)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, num_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# Mean Squared Error
loss = tf.reduce_mean(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)
init = tf.global_variables_initializer()

# for GPU processing
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)

saver = tf.train.Saver()

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.run(init)
	for iteration in range(num_train_iterations):
		x_batch, y_batch = ts_data.next_batch(batch_size, num_time_steps)
		sess.run(train, {X: x_batch, y: y_batch})

		if iteration % 100 == 0:
			mse = loss.eval({X: x_batch, y: y_batch})
			print(iteration, "\tMSE", mse)
	saver.save(sess, "./rnn_time_series_model")

with tf.Session() as sess:
	saver.restore(sess, "./rnn_time_series_model")

	X_new = np.sin(np.array(train_inst[:-1].reshape(-1, num_time_steps, num_inputs)))
	y_pred = sess.run(outputs, {X: X_new})

# plt.title("MODEL TEST")
# plt.plot(train_inst[:-1], np.sin(train_inst[:-1]), "bo", markersize=15, alpha=0.5, label='TRAINING INST')
# plt.plot(train_inst[1:], np.sin(train_inst[1:]), "ko", markersize=10, label='target')
# plt.plot(train_inst[1:], y_pred[0, :, 0], "r.", markersize=10, label='PREDICTIONS')
# plt.legend()
# plt.show()



with tf.Session() as sess:
	saver.restore(sess, "./rnn_time_series_model")
	zero_seq_seed = [0 for i in range(num_time_steps)]
	for iteration in range(len(ts_data.X_data) - num_time_steps):
		x_batch = np.array(zero_seq_seed[-num_time_steps:]).reshape(1, num_time_steps, 1)
		y_pred = sess.run(outputs, {X: x_batch})
		zero_seq_seed.append(y_pred[0, -1, 0])
# plt.plot(ts_data.X_data, zero_seq_seed, "b-")
# plt.plot(ts_data.X_data[:num_time_steps], zero_seq_seed[:num_time_steps], "r", linewidth=3)
# plt.show()

with tf.Session() as sess:
	saver.restore(sess, "./rnn_time_series_model")
	training_instance = list(ts_data.Y_true[:30])
	for iteration in range(len(training_instance) - num_time_steps):
		x_batch = np.array(training_instance[-num_time_steps:]).reshape(1, num_time_steps, 1)
		y_pred = sess.run(outputs, {X: x_batch})
		training_instance.append(y_pred[0, -1, 0])

plt.plot(ts_data.X_data, ts_data.Y_true, "b-")
plt.plot(ts_data.X_data[:num_time_steps], training_instance[:num_time_steps], "r", linewidth=3)
plt.show()