Experimental environment: Ubuntu 16.04; caffe 

1. Image pre-processing:
Verify the directory in the sensor folder:

sensor:

	imagepatches 
  
	lmdb

Change the path in ‘preprocess3pf/1.cutpatch.py’ ,’preprocess3qf/2.createtxt.sh’, and ‘preprocess3qf/3.createlmdb.sh’ and run ‘preprocess3pf/runall.sh’

2 train and test 
Change the path in ‘train.prototxt’ and ‘test.prototxt’, and run them.
