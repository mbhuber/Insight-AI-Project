#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train an Wasserstein Auxiliary Classifier Generative Adversarial Network (WACGAN) on the MNIST dataset.
See https://arxiv.org/abs/1610.09585 for more details about ACGAN.
See https://arxiv.org/abs/1701.07875 for more details about WACGAN.
You should start to see reasonable images after ~3 epochs.
According to the paper, the performance is highly related to the discriminator loss.
You should use a GPU, as the convolution-heavy operations are very slow on the CPU.
Prefer the TensorFlow backend if you plan on iterating, as the compilation time can be a blocker using Theano.
Timings:
Hardware           | Backend | Time / Epoch
-------------------------------------------
 CPU               | TF      | 3 hrs
 Titan X (maxwell) | TF      | 4 min
 Titan X (maxwell) | TH      | 7 min
Consult https://github.com/bobchennan/Wasserstein-GAN-Keras for more information and
example output
The original ACGAN implementation can be found in https://github.com/lukedeo/keras-acgan
More tricks to train GAN can be found in https://github.com/soumith/ganhacks
"""
from __future__ import print_function

import time

from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle
from PIL import Image

from six.moves import range

import keras.backend as K
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Embedding, merge, Dropout, BatchNormalization
from keras.layers.pooling import AveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Convolution2D
from keras.layers.noise import GaussianNoise
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras.backend.common import _EPSILON
from keras.utils.generic_utils import Progbar

from keras.preprocessing.image import ImageDataGenerator

import numpy as np

np.random.seed(1331)

K.set_image_dim_ordering('th')

imaChan = 3 # image channels
imageDim = 128 # image size

# batch and latent size taken from the paper
nb_epochs = 20
batch_size = 64
test_batch_size = 256
latent_size = 100

# Adam parameters suggested in https://arxiv.org/abs/1511.06434
adam_lr = 0.0002
adam_beta_1 = 0.5

pipePath = '../../data/imdb-wiki_crop_clean_align128_kerasPipe3/'

def createImageGenerator(imageDir,batchSize,targetSize):
    return ImageDataGenerator().flow_from_directory(
        imageDir,  # this is the target directory
        target_size = targetSize,
        batch_size=batchSize,
        seed = 1234,
        class_mode='sparse')

def preprocessImageBatch(imaArr):
    #imaArr = imaArr.transpose((0,3,1,2))
    imaArr = (imaArr.astype(np.float32) - 127.5) / 127.5
    return imaArr

def modified_binary_crossentropy(target, output):
    #output = K.clip(output, _EPSILON, 1.0 - _EPSILON)
    #return -(target * output + (1.0 - target) * (1.0 - output))
    return K.mean(target*output)

def build_generator(latent_size):
    # we will map a pair of (z, L), where z is a latent vector and L is a
    # label drawn from P_c, to image space (..., 1, 28, 28)

    idim = imageDim

    cnn = Sequential()

    cnn.add(Dense(1024, input_dim=latent_size))
    cnn.add(LeakyReLU())
    cnn.add(Dense(128 * idim/4 * idim/4))
    cnn.add(LeakyReLU())
    #cnn.add(BatchNormalization())
    cnn.add(Reshape((128, idim/4, idim/4)))

    # upsample to (..., idim/2, idim/2)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(256, 5, 5, border_mode='same',
                          init='glorot_uniform'))
    cnn.add(LeakyReLU())

    # upsample to (..., idim, idim)
    cnn.add(UpSampling2D(size=(2, 2)))
    cnn.add(Convolution2D(128, 5, 5, border_mode='same',
                          init='glorot_uniform'))
    cnn.add(LeakyReLU())

    # take a channel axis reduction
    cnn.add(Convolution2D(imaChan, 2, 2, border_mode='same',
                          activation='tanh', init='glorot_uniform'))

    # this is the z space commonly refered to in GAN papers
    latent = Input(shape=(latent_size, ))

    # this will be our label
    image_class = Input(shape=(1,), dtype='int32')

    # 10 classes in MNIST
    cls = Flatten()(Embedding(numClass, latent_size,
                              init='glorot_uniform')(image_class))

    # hadamard product between z-space and a class conditional embedding
    h = merge([latent, cls], mode='mul')

    fake_image = cnn(h)

    return Model(input=[latent, image_class], output=fake_image)

def build_discriminator():
    # build a relatively standard conv net, with LeakyReLUs as suggested in
    # the reference paper
    cnn = Sequential()
    #cnn.add(GaussianNoise(0.2, input_shape=(1, 28, 28)))
    #cnn.add(Convolution2D(32, 3, 3, border_mode='same', subsample=(2, 2),
    #                      input_shape=(imaChan, imageDim, imageDim)))
    cnn.add(Convolution2D(32, 3, 3, border_mode='same',
                          input_shape=(imaChan, imageDim, imageDim)))
    cnn.add(LeakyReLU())
    cnn.add(AveragePooling2D(pool_size=(2,2)))
    cnn.add(Dropout(0.3))

#    cnn.add(Convolution2D(64, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(Convolution2D(64, 3, 3, border_mode='same'))
    cnn.add(LeakyReLU())
    #cnn.add(AveragePooling2D(pool_size=(2,2)))
    cnn.add(Dropout(0.3))

    #cnn.add(Convolution2D(128, 3, 3, border_mode='same', subsample=(2, 2)))
    cnn.add(Convolution2D(128, 3, 3, border_mode='same'))
    cnn.add(LeakyReLU())
    cnn.add(AveragePooling2D(pool_size=(2,2)))
    cnn.add(Dropout(0.3))

    #cnn.add(Convolution2D(256, 3, 3, border_mode='same', subsample=(1, 1)))
    cnn.add(Convolution2D(256, 3, 3, border_mode='same'))
    cnn.add(LeakyReLU())
    #cnn.AveragePooling2D(pool_size=(2,2))
    cnn.add(Dropout(0.3))

    cnn.add(Flatten())

    image = Input(shape=(imaChan, imageDim, imageDim))

    features = cnn(image)

    # first output (name=generation) is whether or not the discriminator
    # thinks the image that is being shown is fake, and the second output
    # (name=auxiliary) is the class that the discriminator thinks the image
    # belongs to.
    fake = Dense(1, activation='linear', name='generation')(features)
    aux = Dense(numClass, activation='softmax', name='auxiliary')(features)

    return Model(input=image, output=[fake, aux])

if __name__ == '__main__':
    # for create image generators
    trainGenerator = createImageGenerator(pipePath+'train/',batch_size,(imageDim,imageDim))
    testGenerator  = createImageGenerator(pipePath+'test/' ,test_batch_size,(imageDim,imageDim))

    nb_train, nb_test = trainGenerator.n, testGenerator.n
    print("# train: {}, # test: {}".format(nb_train,nb_test))

    # class info
    numClass= len(trainGenerator.class_indices.keys())
    assert(numClass == len(testGenerator.class_indices.keys()))
    print("# classes: {}".format(numClass))

    # build the discriminator
    discriminator = build_discriminator()
    discriminator.compile(
        optimizer=SGD(clipvalue=0.01),#Adam(lr=adam_lr, beta_1=adam_beta_1),
        loss=[modified_binary_crossentropy, 'sparse_categorical_crossentropy']
    )

    # build the generator
    generator = build_generator(latent_size)
    generator.compile(optimizer=Adam(lr=adam_lr, beta_1=adam_beta_1),
                      loss='binary_crossentropy')

    latent = Input(shape=(latent_size, ))
    image_class = Input(shape=(1,), dtype='int32')

    # get a fake image
    fake = generator([latent, image_class])

    # we only want to be able to train generation for the combined model
    discriminator.trainable = False
    fake, aux = discriminator(fake)
    combined = Model(input=[latent, image_class], output=[fake, aux])

    combined.compile(
        optimizer='RMSprop',
        loss=[modified_binary_crossentropy, 'sparse_categorical_crossentropy']
    )

    train_history = defaultdict(list)
    test_history = defaultdict(list)

    # fix the shown examples
    noiseExamples = np.random.normal(0, 1, (10*numClass, latent_size))

    ct = time.time()
    for epoch in range(nb_epochs):
        print('Epoch {} of {}'.format(epoch + 1, nb_epochs))

        nb_batches = int(nb_train / batch_size)
        progress_bar = Progbar(target=nb_batches)

        epoch_gen_loss = []
        epoch_disc_loss = []

        trainGenerator.reset() # enforce shuffling in each epoch
        testGenerator.reset()

        for index in range(nb_batches):
            if len(epoch_gen_loss) + len(epoch_disc_loss) > 1:
                progress_bar.update(index, values=[('disc_loss',np.mean(np.array(epoch_disc_loss),axis=0)[0]), ('gen_loss', np.mean(np.array(epoch_gen_loss),axis=0)[0])])
            else:
                progress_bar.update(index)
            # generate a new batch of noise
            #noise = np.random.uniform(-1, 1, (batch_size, latent_size))
            noise = np.random.normal(0, 1, (batch_size, latent_size))

            # get a batch of real images
            image_batch, label_batch = trainGenerator.next()
            image_batch = preprocessImageBatch(image_batch)

            # sample some labels from p_c
            sampled_labels = np.random.randint(0, numClass, batch_size)

            # generate a batch of fake images, using the generated labels as a
            # conditioner. We reshape the sampled labels to be
            # (batch_size, 1) so that we can feed them into the embedding
            # layer as a length one sequence
            generated_images = generator.predict(
                [noise, sampled_labels.reshape((-1, 1))], verbose=0)

            X = np.concatenate((image_batch, generated_images))
            y = np.array([-1] * batch_size + [1] * batch_size)
            aux_y = np.concatenate((label_batch, sampled_labels), axis=0)

            # see if the discriminator can figure itself out...
            epoch_disc_loss.append(discriminator.train_on_batch(X, [y, aux_y]))

            # make new noise. we generate 2 * batch size here such that we have
            # the generator optimize over an identical number of images as the
            # discriminator
            #noise = np.random.uniform(-1, 1, (2 * batch_size, latent_size))
            noise = np.random.normal(0, 1, (2 * batch_size, latent_size))
            sampled_labels = np.random.randint(0, numClass, 2 * batch_size)

            # we want to train the genrator to trick the discriminator
            # For the generator, we want all the {fake, not-fake} labels to say
            # not-fake
            trick = -np.ones(2 * batch_size)

            epoch_gen_loss.append(combined.train_on_batch(
                [noise, sampled_labels.reshape((-1, 1))], [trick, sampled_labels]))

        print('\nTesting for epoch {}:'.format(epoch + 1))

        # evaluate the testing loss here
        # get a batch of real images
        X_test, y_test = testGenerator.next()
        X_test = preprocessImageBatch(X_test)

        # overwrite nb_test to number of test batches
        nb_test = testGenerator.batch_size

        # generate a new batch of noise
        #noise = np.random.uniform(-1, 1, (nb_test, latent_size))
        noise = np.random.normal(0, 1, (nb_test, latent_size))

        # sample some labels from p_c and generate images from them
        sampled_labels = np.random.randint(0, numClass, nb_test)
        generated_images = generator.predict(
            [noise, sampled_labels.reshape((-1, 1))], verbose=False)

        X = np.concatenate((X_test, generated_images))
        y = np.array([1] * nb_test + [0] * nb_test)
        aux_y = np.concatenate((y_test, sampled_labels), axis=0)

        # debug
        #print([X_test.shape,X.shape,y.shape,aux_y.shape])

        # see if the discriminator can figure itself out...
        discriminator_test_loss = discriminator.evaluate(
            X, [y, aux_y], verbose=False)

        discriminator_train_loss = np.mean(np.array(epoch_disc_loss), axis=0)

        # make new noise
        #noise = np.random.uniform(-1, 1, (2 * nb_test, latent_size))
        noise = np.random.normal(0, 1, (2 * nb_test, latent_size))
        sampled_labels = np.random.randint(0, numClass, 2 * nb_test)

        trick = np.ones(2 * nb_test)

        generator_test_loss = combined.evaluate(
            [noise, sampled_labels.reshape((-1, 1))],
            [trick, sampled_labels], verbose=False)

        generator_train_loss = np.mean(np.array(epoch_gen_loss), axis=0)

        # generate an epoch report on performance
        train_history['generator'].append(generator_train_loss)
        train_history['discriminator'].append(discriminator_train_loss)

        test_history['generator'].append(generator_test_loss)
        test_history['discriminator'].append(discriminator_test_loss)

        print('{0:<22s} | {1:4s} | {2:15s} | {3:5s}'.format(
            'component', *discriminator.metrics_names))
        print('-' * 65)

        ROW_FMT = '{0:<22s} | {1:<4.2f} | {2:<15.2f} | {3:<5.2f}'
        print(ROW_FMT.format('generator (train)',
                             *train_history['generator'][-1]))
        print(ROW_FMT.format('generator (test)',
                             *test_history['generator'][-1]))
        print(ROW_FMT.format('discriminator (train)',
                             *train_history['discriminator'][-1]))
        print(ROW_FMT.format('discriminator (test)',
                             *test_history['discriminator'][-1]))
        # save weights every epoch
        generator.save_weights(
            'params_generator_epoch_{0:03d}.hdf5'.format(epoch), True)
        discriminator.save_weights(
            'params_discriminator_epoch_{0:03d}.hdf5'.format(epoch), True)

        # generate some digits to display
        #noise = np.random.uniform(-1, 1, (100, latent_size))
        noise = noiseExamples

        sampled_labels = np.array([
            [i] * 10 for i in range(numClass)
        ]).reshape(-1, 1)

        # get a batch to display
        generated_images = generator.predict(
            [noise, sampled_labels], verbose=0)

        # arrange them into a grid
        if imaChan==1:
            img = (np.concatenate([r.reshape(-1, imageDim)
                                   for r in np.split(generated_images, numClass)
                                   ], axis=-1) * 127.5 + 127.5).astype(np.uint8)
        else:
            img = (np.concatenate([r.transpose(0,2,3,1).reshape(-1, imageDim,3)
                               for r in np.split(generated_images, numClass)
                               ], axis=1) * 127.5 + 127.5).astype(np.uint8)

        Image.fromarray(img).save(
            'plot_epoch_{0:03d}_generated.png'.format(epoch))

        pickle.dump({'train': train_history, 'test': test_history},
                open('acgan-history.pkl', 'wb'))

    print("Elapsed time= {} min".format((time.time()-ct)/60.))
