## SANITY CHECK FOR THE ENTIRE FOLDER ##
## TODO ##
# tensorflow_probability
# tensorflow_datasets


missing_modules = []

## tensorflow ##
try:
	import tensorflow as tf
	assert int(tf.__version__[0]) == 2
except:
	missing_modules.append("Tensorflow>=2.0")

## six ##
try:
	import six
except:
	missing_modules.append("six")

## matplotlib ## 
try:
	import matplotlib
except:
	missing_modules.append("matplotlib")

## pandas ##
try:
	import pandas
except:
	missing_modules.append("pandas")

## numpy ##
try:
	import numpy
except:
	missing_modules.append("numpy")

## tensorflow_probabilities
try:
	import tensorflow_probability
except:
	missing_modules.append("tensorflow_probability")

## output the missing modules, if any ##
if missing_modules == []:
	print("You have all of the needed modules")
else:
	print("the following modules are missing and are required:")
	for m in missing_modules:
		print(m)

