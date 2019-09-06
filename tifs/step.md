Experimental environment: Ubuntu16.04; MATLAB 2014; Python 2.6; Keras-2.2.4

1. image pre-processing
Confirm your directory in the datasets folder.

datasets:

    full:
    
    	PG
      
    	CG
      
    patches:
    
	    train:
  
		    PG
    
		    CG
    
	    test:
  
		   PG
    
		   CG
    
	   valid:
  
		   PG
    
		   CG
    
Change the path in 'utils/makePatches.m’ and run it in MATLAB.

2 train and test 

For training the model: Change the path in 'src/model.py’ and run it in python2.
For testing the model: run patchesTestAcc.py
