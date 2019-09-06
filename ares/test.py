
from __future__ import print_function
import keras
from keras import backend as K
from keras.preprocessing.image import *
from keras.models import Sequential, load_model
import numpy as np 
from keras.utils import multi_gpu_model
model = load_model('./vgg19_fineturning.h5')
batch_size = 64
test_datagen = ImageDataGenerator()
test_generator = test_datagen.flow_from_directory(
        './cut_12per/test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')
#print (test_generator)

#score = model.evaluate_generator(test_generator, verbose=1)
predict_label = model.predict_generator(test_generator,verbose = 1 )
datas,ori_labels = test_generator.next()
print(ori_labels.shape)
np.savetxt("ori_labels.txt", ori_labels)

'''
score: [0.012795081801579572, 0.9981481481481481]
'''
