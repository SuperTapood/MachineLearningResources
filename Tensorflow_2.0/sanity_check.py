## SANITY CHECK FOR THE ENTIRE FOLDER ##
## TODO ##
# numpy
# pandas
# matplotlib
# six


missing_modules = []

## tensorflow ##
try:
	import tensorflow as tf
	assert int(tf.__version__[0]) == 2
except:
	missing_modules.append("Tensorflow>=2.0")


## output the missing modules, if any ##
if missing_modules == []:
	print("You have all of the needed modules")
else:
	print("the following modules are missing and are required:")
	for m in missing_modules:
		print(m)

