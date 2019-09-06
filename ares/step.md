Experimental environment: Ubuntu 16.04; Python 2; keras 2.2.4.

1 image pre-processing 

Since the original paper did not give a specific cutting method, we cut each original image into 12 224X224 image patches.
You can run ‘cutpic.py’ to cut images.

2 train and test
Change the path in ‘ares.py’ and run it for traing.
Once you have the model, you can run 'test.py' to test it.
