import tensorflow as tf

n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1 + n2
with tf.Session() as sess:
	result = sess.run(n3)
# print(result)
# print(n3)
# print(tf.get_default_graph())
g = tf.Graph()
# print(g)
g1 = tf.get_default_graph()
g2 = tf.Graph()
# print(g)
# print(g1)
# print(g2)
with g2.as_default():
	pass
	# print(g2 is tf.get_default_graph())
# print(g2 is tf.get_default_graph())

myTensor = tf.random_uniform((4,4), 0, 1)
print(myTensor)
myVar = tf.Variable(myTensor)
print(myVar)
init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print(sess.run(myVar))
ph = tf.placeholder(tf.float32)
