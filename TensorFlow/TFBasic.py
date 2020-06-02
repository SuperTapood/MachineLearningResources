import tensorflow as tf


hello = tf.constant("Hello ")
world = tf.constant("World")
# print(hello)
# print(type(hello))
with tf.Session() as sess:
	result = sess.run(hello+world)

# print(result)

a = tf.constant(10)
b = tf.constant(20)
# print(type(a))
# print(a + b)
with tf.Session() as sess:
	result = sess.run(a+b)
# print(result)

# const = tf.constant(10)
# fillMat = tf.fill((4, 4), 10)
# myZeroes = tf.zeros((4, 4))
# myOnees = tf.ones((4, 4))
# myrandn = tf.random_normal((4, 4), mean=0, stddev=1.0)
# myrandu = tf.random_uniform((4, 4), minval=0, maxval=1)
# myOps = [const,fillMat, myZeroes, myOnees,myrandn,myrandu]
# sess = tf.InteractiveSession()
# for op in myOps:
# 	print(sess.run(op))
# 	print()

sess = tf.InteractiveSession()
a = tf.constant([[1, 2], [3, 4]])
print(a.get_shape())
b = tf.constant([[10], [100]])
print(b.get_shape())
result = tf.matmul(a, b)
print(sess.run(result))
print(result.eval())