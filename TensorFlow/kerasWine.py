from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.contrib.keras import models
from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import losses, optimizers, metrics, activations

wine_data = load_wine()

feat_data = wine_data['data']
labels = wine_data['target']

X_train, X_test, y_train, y_test = train_test_split(feat_data, labels, test_size=0.3, random_state=101)

scaler = MinMaxScaler()

scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

dnn_keras_model = models.Sequential()
dnn_keras_model.add(layers.Dense(13, input_dim=13, activation='relu'))
dnn_keras_model.add(layers.Dense(13, 'relu'))
dnn_keras_model.add(layers.Dense(13, 'relu'))
dnn_keras_model.add(layers.Dense(3, 'softmax'))

dnn_keras_model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])

dnn_keras_model.fit(scaled_x_train, y_train, epochs=500)

predictions = dnn_keras_model.predict_classes(scaled_x_test)
print(classification_report(predictions, y_test))