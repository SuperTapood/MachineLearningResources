from __future__ import unicode_literals, print_function, division, absolute_import
import numpy as np
import pandas as pd
from six.moves import urllib
import tensorflow.compat.v2.feature_column as fc
import tensorflow as tf
import matplotlib.pyplot as plt


# load data
dftrain = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/train.csv')
dfeval = pd.read_csv('https://storage.googleapis.com/tf-datasets/titanic/eval.csv')
y_train = dftrain.pop('survived')
y_eval = dfeval.pop('survived')
# plot the histogram of the passenger ages
# dftrain.age.hist(bins=20)
# plt.show()

# plot the barh of the passenger sex
# dftrain.sex.value_counts().plot(kind='barh')
# plt.show()

# plot the barh of the passenger class
# dftrain['class'].value_counts().plot(kind='barh')
# plt.show()

# survival rate by sex
# pd.concat([dftrain, y_train], axis=1).groupby('sex').survived.mean().plot(kind='barh').set_xlabel('survival rate')
# plt.show()

CATEGORICAL_COLUMNS = ['sex', 'n_siblings_spouses', 'parch', 'class', 'deck',
                       'embark_town', 'alone']
NUMERIC_COLUMNS = ['age', 'fare']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
  vocabulary = dftrain[feature_name].unique()  # gets a list of all unique values from given feature column
  feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, vocabulary))

for feature_name in NUMERIC_COLUMNS:
  feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))

def make_input_fn(data_df, label_df, num_epochs=10, shuffle=True, batch_size=32):
  def input_function():  # inner function, this will be returned
    ds = tf.data.Dataset.from_tensor_slices((dict(data_df), label_df))  # create tf.data.Dataset object with data and its label
    if shuffle:
      ds = ds.shuffle(1000)  # randomize order of data
    ds = ds.batch(batch_size).repeat(num_epochs)  # split dataset into batches of 32 and repeat process for number of epochs
    return ds  # return a batch of the dataset
  return input_function  # return a function object for use

train_input_fn = make_input_fn(dftrain, y_train)  # here we will call the input_function that was returned to us to get a dataset object we can feed to the model
eval_input_fn = make_input_fn(dfeval, y_eval, num_epochs=1, shuffle=False)

# create a linear estimator (a Tensorflow API)
linear_est = tf.estimator.LinearClassifier(feature_columns=feature_columns)

# train the estimator
linear_est.train(train_input_fn)
# get the data of the model on a dataset it hasn't seen
result = linear_est.evaluate(eval_input_fn)

print(result['accuracy'])

# result is a dict of data
# print(result)

# make some predictions
# predictions are a generator object
result = list(linear_est.predict(eval_input_fn))
# print all of the predictions for all of the passengers
print(dfeval.loc[2])
print(y_eval.loc[3])
print(result[2]['probabilities'][1])