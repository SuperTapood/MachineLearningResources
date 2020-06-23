import collections
import math
import os
import errno
import random
import zipfile
import numpy as np
from six.moves import urllib
from six.moves import xrange 
import tensorflow.compat.v1 as tf
from collections import Counter
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from time import time

tf.disable_eager_execution()
data_url = 'http://mattmahoney.net/dc/text8.zip'
data_dir = "word2vec_data/words"

def convert_t(t):
	if t < 60:
		return t
	elif t < 3600:
		s = t % 60
		m = t // 60
		return f"{int(m)}:{int(s)}"
	else:
		h = t // 3600
		r = t % 3600
		m = r // 60
		s = r % 60
		return f"{int(h)}:{int(m)}:{int(s)}"
	return

def fetch_words_data(url=data_url, words_data=data_dir):
    
    # Make the Dir if it does not exist
    os.makedirs(words_data, exist_ok=True)
    
    # Path to zip file 
    zip_path = os.path.join(words_data, "words.zip")
    
    # If the zip file isn't there, download it from the data url
    if not os.path.exists(zip_path):
        urllib.request.urlretrieve(url, zip_path)
        
    # Now that the zip file is there, get the data from it
    with zipfile.ZipFile(zip_path) as f:
        data = f.read(f.namelist()[0])
    
    # Return a list of all the words in the data source.
    return data.decode("ascii").split()

words = fetch_words_data()
def create_counts(vocab_size=50000):
	vocab = [] + Counter(words).most_common(vocab_size)
	vocab = np.array([word for word, _ in vocab])
	dictionary = {word: code for code, word in enumerate(vocab)}
	data = np.array([dictionary.get(word, 0) for word in words])
	return data, vocab

data, vocabulary = create_counts()

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
    if data_index == len(data):
        buffer[:] = data[:span]
        data_index = span
    else:
        buffer.append(data[data_index])
        data_index += 1
  # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

# CONSTANTS
batch_size = 128
embedding_size = 150
skip_window = 1
num_skips = 2
valid_size = 16
valid_window = 100
valid_examples = np.random.choice(valid_window, valid_size, False)
num_sampled = 64
learning_rate = 0.01
vocabulary_size = 50000

tf.reset_default_graph()
train_inputs = tf.placeholder(tf.int32, [None])
train_labels = tf.placeholder(tf.int32, [batch_size, 1])
valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

init_embeds = tf.random_uniform([vocabulary_size, embedding_size], -1, 1)
embeddings = tf.Variable(init_embeds)
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1 / np.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, train_labels, embed, num_sampled, vocabulary_size))
optimizer = tf.train.AdamOptimizer(learning_rate=1.0)
trainer = optimizer.minimize(loss)

# Compute the cosine similarity between minibatch examples and all embeddings.
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), axis=1, keep_dims=True))
normalized_embeddings = embeddings / norm
valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

data_index = 0

init = tf.global_variables_initializer()
saver = tf.train.Saver()

num_steps = 500000
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.run(init)
	average_loss = 0
	for step in range(num_steps):
		s = time()
		batch_input, batch_labels = generate_batch(batch_size, num_skips, skip_window)
		feed_dict = {train_inputs: batch_input, train_labels: batch_labels}
		_, loss_val = sess.run([trainer, loss], feed_dict)
		average_loss += loss_val
		remain = num_steps - step
		e = time() - s
		print(f"current step {step}. ETA {convert_t(e * remain)}")
		if step % 1000 == 0:
			if step > 0:
				average_loss = average_loss / 1000
			print(f"Average Loss at step {step} is: {average_loss}")
			average_loss = 0
		final_embeddings = normalized_embeddings.eval()
	saver.save(sess, "word2vecModel/500kstepsmodel")

tsne = TSNE(init='pca', n_iter=5000)
plot_only = 5000
low_dim_embed = tsne.fit_transform(final_embeddings[:plot_only, :])
labels = [vocabulary[i] for i in range(plot_only)]

def plot_with_labels(low_dim_embs, labels):
    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
    plt.figure(figsize=(18, 18))  #in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
plot_with_labels(low_dim_embed, labels)
plt.show()
