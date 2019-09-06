import os
import os.path
import glob
import numpy as np
np.random.seed(1)
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from keras.preprocessing import image
from keras.applications.vgg19 import VGG19
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
from keras.layers import Conv2D,BatchNormalization,Dropout,Flatten,Activation,Dense,concatenate,Lambda
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import multi_gpu_model
import keras.backend as K
import tensorflow as tf 


#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory('./cut_12per/train', 
                                                    target_size = (224, 224), 
                                                    batch_size = 64,
                                                    class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        './cut_12per/valid',
        target_size=(224, 224),
        batch_size=64,
        class_mode='categorical')
print(train_generator.samples)

# Creating base_model (VGG19)
                                                                                                                                                                                                                                                                                          
base_model = VGG19(weights='imagenet')


#VGG19features = base_model.predict(X)

feature_1 = base_model.get_layer('block1_conv2').output
feature_2 = base_model.get_layer('block2_conv2').output
feature_3 = base_model.get_layer('block3_conv4').output

# feature 1 
feature_1 = Conv2D(128,(3,3),strides=(1,1))(feature_1)
feature_1 = BatchNormalization()(feature_1)
feature_1 = Activation('relu')(feature_1)

feature_1 = Conv2D(64,(3,3),strides=(1,1))(feature_1)
feature_1 = BatchNormalization()(feature_1)
feature_1 = Activation('relu')(feature_1)
#mean_1 = K.mean(feature_1,(1,2))
#var_1 = K.var(feature_1,axis=(1,2))
mean_1=Lambda(lambda x: K.mean(x,(1,2)))(feature_1)
var_1=Lambda(lambda x: K.var(x,(1,2)))(feature_1)


#feature 2 

feature_2 = Conv2D(128,(3,3),strides=(1,1))(feature_2)
feature_2 = BatchNormalization()(feature_2)
feature_2 = Activation('relu')(feature_2)

feature_2 = Conv2D(64,(3,3),strides=(1,1))(feature_2)
feature_2 = BatchNormalization()(feature_2)
feature_2 = Activation('relu')(feature_2)

#mean_2 = K.mean(feature_2,(1,2))
#var_2 = K.var(feature_2,axis=(1,2))
mean_2=Lambda(lambda x: K.mean(x,(1,2)))(feature_2)
var_2=Lambda(lambda x: K.var(x,(1,2)))(feature_2)

#feature_2 = Flatten()(feature_2)

# feature 3
feature_3 = Conv2D(128,(3,3),strides=(1,1))(feature_3)
feature_3 = BatchNormalization()(feature_3)
feature_3 = Activation('relu')(feature_3)

feature_3 = Conv2D(64,(3,3),strides=(1,1))(feature_3)
feature_3 = BatchNormalization()(feature_3)
feature_3 = Activation('relu')(feature_3)

#mean_3 = K.mean(feature_3,(1,2))
#var_3 = K.var(feature_3,axis=(1,2))
mean_3=Lambda(lambda x:  K.mean(x,(1,2)))(feature_3)
var_3=Lambda(lambda x: K.var(x,(1,2)))(feature_3)

#feature ALL

feature_all = concatenate([mean_1,mean_2,mean_3,var_1,var_2,var_3])
#print(feature_all)
#feature_all = Lambda(lambda x: x[:])(feature_all)
#feature_all=Lambda(lambda x: K.round(x))(feature_all)
#print(K.is_keras_tensor(feature_all))
#feature_all = Flatten()(feature_all)#

predictions = Dense(384)(feature_all)
predictions = Dropout(1/3)(predictions)
predictions = Dense(512)(predictions)
output_ = Dense(2,activation="softmax")(predictions)
#output_ = Lambda(lambda x: K.round(x))(prediction)

model = Model(inputs=base_model.input, outputs=output_)
model = multi_gpu_model(model, gpus=3)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_generator,steps_per_epoch = train_generator.samples ,
                    epochs = 1,verbose = 1 ,
                    validation_data = validation_generator, 
                    validation_steps = validation_generator.samples)
model.save('./vgg19_fineturning.h5') 

print("DONE!")
