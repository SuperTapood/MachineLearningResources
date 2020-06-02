import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class Opertaion():
	def __init__(self, inputNodes=[]):
		self.inputNodes = inputNodes
		self.outputNodes = []
		for node in self.inputNodes:
			node.outputNodes.append(self)
		_defaultGraph.operations.append(self)
		return

	def compute(self):
		pass
	pass

class add(Opertaion):
	def __init__(self, x, y):
		super().__init__([x, y])
		return

	def compute(self, a, b):
		self.inputs = [a, b]
		return a + b
	pass

class multiply(Opertaion):
	def __init__(self, x, y):
		super().__init__([x, y])
		return

	def compute(self, a, b):
		self.inputs = [a, b]
		return a * b
	pass

class matmul(Opertaion):
	def __init__(self, x, y):
		super().__init__([x, y])
		return

	def compute(self, a, b):
		self.inputs = [a, b]
		return a.dot(b)
	pass

class Placeholder():
	def __init__(self):
		self.outputNodes = []
		_defaultGraph.placeholders.append(self)
		return
	pass

class Variable():
	def __init__(self, initalValue=None):
		self.value = initalValue
		self.outputNodes = []
		_defaultGraph.variables.append(self)
		return
	pass

class Graph():
	def __init__(self):
		self.operations = []
		self.placeholders = []
		self.variables = []
		return

	def setAsDefault(self):
		global _defaultGraph
		_defaultGraph = self
		return
	pass

def traversePostorder(opertaion):
	nodesPostorder = []
	def recurse(node):
		if isinstance(node, Opertaion):
			for inputNode in node.inputNodes:
				recurse(inputNode)
		nodesPostorder.append(node)
	recurse(opertaion)
	return nodesPostorder

# z = Ax + b
# A = 10
# b = 1
# z = 10x + 1
# x is a placeholder

# g = Graph()
# g.setAsDefault()
# A = Variable(10)
# b = Variable(1)
# x = Placeholder()
# y = multiply(A, x)
# z = add(y, b)


class Session:
	def run(self, operation, feed_dict={}):
		nodesPostorder = traversePostorder(operation)
		for node in nodesPostorder:
			if type(node) == Placeholder:
				node.output = feed_dict[node]
			elif type(node) == Variable:
				node.output = node.value
			else:
				# OPERATION
				node.inputs = [inputNode.output for inputNode in node.inputNodes]
				node.output = node.compute(*node.inputs) #*args
			if type(node.output) == list:
				node.output = np.array(node.output)
		return operation.output

# sess = Session()
# result = sess.run(opertaion=z, feed_dict={x:10})


# g = Graph()
# g.setAsDefault()
# A = Variable([[10, 20], [30, 40]])
# b = Variable([1, 1])
# x = Placeholder()
# y = matmul(A, x)
# z = add(y, b)
# session = Session()
# result = session.run(z, {x:10})


# classification
## activation function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# sampleZ = np.linspace(-10, 10, 100)
# sampleA = sigmoid(sampleZ)

# plt.plot(sampleZ, sampleA)
# # plt.show()

class Sigmoid(Opertaion):
	def __init__(self, x):
		super().__init__([x])
		return

	def compute(self, xVal):
		return sigmoid(xVal)
	pass

data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
features = data[0]
labels = data[1]

plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')
# plt.show()

x = np.linspace(0, 11, 10)
y = -x + 5
plt.plot(x, y)
#plt.show()

# (1, 1) * f - 5 = 0
# f.shape = (2, 1)

arr = np.array([1,1]).dot(np.array([[8], [10]])) - 5
print(arr[0])
arr = np.array([1,1]).dot(np.array([[2], [-10]])) - 5
print(arr[0])

g = Graph()
g.setAsDefault()

x = Placeholder()
w = Variable([1, 1])
b = Variable(-5)
z = add(matmul(w, x), b)
a = Sigmoid(z)
sess = Session()
result = sess.run(operation=a, feed_dict={x:[8, 10]})
print(result) # <- 0.999 = very Fing sure its 1 class
result = sess.run(operation=a, feed_dict={x:[2, -10]})
print(result) # <- 2.26 * 10 ** -6 = very Fing sure its 0 class