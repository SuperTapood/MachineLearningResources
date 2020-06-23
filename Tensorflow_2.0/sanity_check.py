## SANITY CHECK FOR THE ENTIRE FOLDER ##
missing_modules = []

## tensorflow ##
try:
	import tensorflow as tf
	assert int(tf.__version__[0]) == 2
except:
	missing_modules.append("Tensorflow>=2.0")

if missing_modules == []:
	print("You have all of the needed modules")
else:
	print("the following modules are missing and are required:")
	for m in missing_modules:
		print(m)

