from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import estimator
from sklearn.metrics import classification_report, confusion_matrix

wine_data = load_wine()

feat_data = wine_data['data']
labels = wine_data['target']

X_train, X_test, y_train, y_test = train_test_split(feat_data, labels, test_size=0.3, random_state=101)

scaler = MinMaxScaler()

scaled_x_train = scaler.fit_transform(X_train)
scaled_x_test = scaler.transform(X_test)

feat_cols = [tf.feature_column.numeric_column('x', shape=[13])]

deep_model = estimator.DNNClassifier(hidden_units=[13, 13, 13], feature_columns=feat_cols, n_classes=3, optimizer=tf.train.GradientDescentOptimizer(0.01))

input_fn = estimator.inputs.numpy_input_fn(x={'x': scaled_x_train}, y=y_train, shuffle=True, batch_size=10, num_epochs=5)

deep_model.train(input_fn=input_fn, steps=500)

input_fn_eval = tf.estimator.inputs.numpy_input_fn(x={'x': scaled_x_test}, shuffle=False)
preds = list(deep_model.predict(input_fn_eval))
predictions = [p['class_ids'][0] for p in preds]
print(classification_report(y_test, predictions))