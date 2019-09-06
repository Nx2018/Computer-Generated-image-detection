Experimental environment: Ubuntu 16.04; python3; Tensorflow

https://github.com/NicoRahm/CGvsPhoto

Installing
Running in terminal : pip3 install CGvsPhoto

Database format
Your database must follow this organization :

Database:

	test:
  
		Real
    
		CGG
    
	train:
  
		Real
    
		CGG
    
	validation:
  
		Real
    
		CGG

Then, change the path in run ’create_DB.py’ and ‘create_patches_splicing.py’. Run them in python3.
For training and testing, change the path in ‘config.ini’ and ‘train.py’. Run them in python3.
