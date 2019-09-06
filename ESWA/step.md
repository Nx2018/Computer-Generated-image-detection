Experimental environment: Ubuntu 16.04; Python 3; Tensorflow

https://github.com/kratzert/finetune_alexnet_with_tensorflow/

1. Image pre-process
Confirm your directory in the datasets folder.

datasets:

	CG
  
	PG 
Output:

	CGG
  
	Real

Run 'picpatch’ to cut the image.
Then, run ’create_txt.py’ to get ’train.txt’ and ’text.txt’ .

2. Train and test 
Change the path in ‘trainval_alexnet.py’ and run it in Python3.
