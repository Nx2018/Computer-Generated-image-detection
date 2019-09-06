from __future__ import print_function
import keras
import math
from keras import initializers
from keras import backend as K
from keras.preprocessing.image import *
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import optimizers, regularizers
from time import time
from keras.callbacks import TensorBoard, ReduceLROnPlateau, ModelCheckpoint,LearningRateScheduler
from keras.utils import multi_gpu_model

import os

# visible device
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# subclassing TensorBoard to show LR
class LRTensorBoard(TensorBoard):
    def __init__(self, log_dir):  
        super().__init__(log_dir=log_dir)

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'lr': K.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)

def step_decay(epoch):
    initial_lrate = 1e-3
    drop = 0.9
    epochs_drop = 40.0
    lrate = initial_lrate * math.pow(drop,math.floor((1+epoch)/epochs_drop))
    return lrate
def scheduler(epoch):
    
    if epoch % 40 == 0 and epoch != 0:
        lr = K.get_value(model.optimizer.lr)
        K.set_value(model.optimizer.lr, lr * 0.1)
        print("lr changed to {}".format(lr * 0.1))
    return K.get_value(model.optimizer.lr)


# building the model
model = Sequential()

# convLayer
model.add(Conv2D(32, (7, 7), input_shape=(233, 233, 3), 
                             kernel_initializer= initializers.RandomNormal(stddev=0.01),
                             kernel_regularizer=regularizers.l1(1e-4),
                             bias_initializer=initializers.constant(0)))
#model.add(Conv2D(32, (7, 7), input_shape=(233, 233, 3), kernel_regularizer=regularizers.l1(0.0015)))

# C1
model.add(Conv2D(64, (7, 7),strides=(2,2), 
                            kernel_initializer= initializers.RandomNormal(stddev=0.01),
                            kernel_regularizer=regularizers.l1(1e-4),
                            bias_initializer=initializers.constant(0)))
#model.add(Conv2D(64, (7, 7)))
model.add(BatchNormalization(scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# C2
model.add(Conv2D(48, (5, 5), kernel_initializer= initializers.RandomNormal(stddev=0.01),
                             kernel_regularizer=regularizers.l1(1e-4),
                             bias_initializer=initializers.constant(1)))
model.add(BatchNormalization(scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# C3
model.add(Conv2D(64, (3, 3),kernel_initializer= initializers.RandomNormal(stddev=0.01), 
                            kernel_regularizer=regularizers.l1(1e-4),
                            bias_initializer=initializers.constant(1)))
model.add(BatchNormalization(scale=True))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=2))

# FC4 (Dense)
model.add(Flatten())
model.add(Dense(4096, kernel_initializer= initializers.RandomNormal(stddev=0.005),
                      activation='relu',
                      bias_initializer=initializers.constant(1)))
model.add(Dropout(0.5))

# FC5
model.add(Dense(4096,kernel_initializer= initializers.RandomNormal(stddev=0.005), 
                     activation='relu',
                     bias_initializer=initializers.constant(1)))
model.add(Dropout(0.5))

# Output
model.add(Dense(2, kernel_initializer= initializers.RandomNormal(stddev=0.01),
                   activation='softmax',
                   bias_initializer=initializers.constant(0)))

# optimizer

#adam = optimizers.Adam(lr=1e-6)
sgd = optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)

# multi-gpu
# model = multi_gpu_model(model, gpus=4)

# loss function is binary crossentropy (for binary classification)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# make the data generators for image data
batch_size = 128

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
        '../datasets/patches/train', 
        target_size=(233, 233),  # input size as in paper
        batch_size=batch_size,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        '../datasets/patches/valid',
        target_size=(233, 233),
        batch_size=batch_size,
        class_mode='binary')

# make callbacks to use while training
#lrate = LearningRateScheduler(step_decay)
tensorboard = LRTensorBoard(log_dir="../logs/{}".format(time()))
#reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, min_lr=0)
reduce_lr = LearningRateScheduler(scheduler)
checkpoint = ModelCheckpoint("../checkpoints/NIvsCG_sgd_lr-1e-3_WITH_REDUCE-LR.{epoch:02d}-{val_loss:.2f}.h5", monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)

# start training
model.fit_generator(
        train_generator,
        steps_per_epoch=None,
        epochs=180,
        validation_data=validation_generator,
        validation_steps=None,
        callbacks=[tensorboard, reduce_lr, checkpoint])

# save the model if ever finish
model.save('../models/NIvsCG_model_150epochs_None-NoneStep.h5') 
