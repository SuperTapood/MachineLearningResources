from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
# example data
# path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# custom data.txt:
path_to_file = 'smith.txt'

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
# print ('Length of text: {} characters'.format(len(text)))

vocab = sorted(set(text))
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
  return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

# print(f"Text: {text[:17]}")
# print(f"Encoded: {text_to_int(text[:17])}")

def int_to_text(ints):
  try:
    ints = ints.numpy()
  except:
    pass
  return ''.join(idx2char[ints])

# print(int_to_text(text_as_int[:17]))

seq_len = 200
examples_per_epoch = len(text) // (seq_len + 1)
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_len + 1, drop_remainder=True)

def split_input_target(chunk):
  input_text = chunk[:-1]
  target_text = chunk[1:]
  return input_text, target_text

dataset = sequences.map(split_input_target)

# for x, y in dataset.take(2):
#   print("\n\n EXAMPLE \n")
#   print("INPUT")
#   print(int_to_text(x))
#   print("\n OUTPUT")
#   print(int_to_text(y))

BATCH_SIZE = 128
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 256
RNN_UNITS = 2048
BUFFER_SIZE = 10000
data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'), 
    tf.keras.layers.Dense(vocab_size)])
  return model


garbaginator = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
# garbaginator.summary()

for input_example_batch, target_example_batch in data.take(1):
  example_batch_predictions = garbaginator(input_example_batch)

pred = example_batch_predictions[0]
sampled_indices = tf.random.categorical(pred, 1)

sampled_indices = np.reshape(sampled_indices, (1, -1))[0]

predicted_chars = int_to_text(sampled_indices)

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

garbaginator.compile(optimizer='adam', loss=loss)

checkpoint_dir = './training_checkpoints/'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}.ckpt")
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
  filepath=checkpoint_prefix, 
  save_weights_only=True)

history = garbaginator.fit(data, epochs=100, callbacks=[checkpoint_callback])

garbaginator_v2 = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, 1)

garbaginator_v2.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
garbaginator_v2.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
  num_generate = 500
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)
  text_generated = []
  temperature = 1.0
  model.reset_states()

  for i in range(num_generate):
    predictions = model(input_eval)
    prediction = tf.squeeze(predictions, 0)
    predictions = prediction / temperature
    predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()
    input_eval = tf.expand_dims([predicted_id], 0)
    text_generated.append(idx2char[predicted_id])
  return (start_string + ''.join(text_generated))

print(generate_text(garbaginator_v2, "Hello"))