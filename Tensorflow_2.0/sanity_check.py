## SANITY CHECK FOR THE ENTIRE FOLDER ##


missing_modules = []

## tensorflow ##
try:
	import tensorflow as tf
	assert int(tf.__version__[0]) == 2
except:
	missing_modules.append("Tensorflow>=2.0")
try:
	import six
except:
	missing_modules.append("six")
try:
	import matplotlib
except:
	missing_modules.append("matplotlib")
try:
	import pandas
except:
	missing_modules.append("pandas")
try:
	import numpy
except:
	missing_modules.append("numpy")

## output the missing modules, if any ##
if missing_modules == []:
	print("You have all of the needed modules")
else:
	print("the following modules are missing and are required:")
	for m in missing_modules:
		print(m)

