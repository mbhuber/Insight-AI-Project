#!/usr/bin/env python
import os, shutil
import numpy as np
import time
import matplotlib.pyplot as plt

from keras import applications
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Lambda, Reshape, UpSampling2D
#from keras.layers.convolutional import Conv2DTranspose
from keras.layers.pooling import AveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K

from keras.callbacks import TensorBoard

K.set_image_dim_ordering('th')

def setDataParameters():
    imaDim   = 128
    numClass = 6
    return (imaDim,numClass)

def setModelParameters():
    batchSize   = 32

    return (batchSize)

def setDataPaths():
    pipePath = '../../data/imdb-wiki_crop_clean_align128_kerasPipe3/'
    trainDir = pipePath + "train/"
    testDir  = pipePath + "valid/"
    return (trainDir, testDir)

def createImageGenerator(imageDir,batchSize,targetSize):
    return ImageDataGenerator().flow_from_directory(
        imageDir,  # this is the target director
        target_size=targetSize,  # all images will be resized to 150x150
        batch_size=batchSize,
        class_mode='categorical')  # since we use binary_crossentropy loss, we need binary labels

def getModelArch(imaDim=128, numClass=6):
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5-0.5, input_shape=(3, imaDim, imaDim)))

    model.add(Conv2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(BatchNormalization(axis=1))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(numClass))
    model.add(Activation('softmax'))
    return model

if __name__ == '__main__':
    ct = time.time()

    # parameters
    imaDim, numClass =setDataParameters()
    batchSize       =setModelParameters()

    # data
    trainDir, testDir = setDataPaths()
    print(trainDir)
    print(testDir)
    train_generator = createImageGenerator(trainDir,batchSize,(imaDim,imaDim))
    valid_generator  = createImageGenerator(testDir,batchSize,(imaDim,imaDim))

    # parameters and model
    model = getModelArch(imaDim,numClass)

    # compile
    model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

    mhist= model.fit_generator(
        generator=train_generator,
        samples_per_epoch=10* batchSize,
        nb_epoch=10,
        validation_data= valid_generator,
        nb_val_samples= valid_generator.n/100,
        callbacks=[TensorBoard(log_dir='/tmp/ageClass_v1')])

    model.save("ageClass_model.h5")
    print("time= {} min".format((time.time()-ct)/60.))
