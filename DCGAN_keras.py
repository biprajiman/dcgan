from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, pdb, traceback
from termcolor import colored
import tensorflow as tf
import numpy as np
import time
from pprint import pprint
# pylint: disable=missing-docstring
# pylint: disable=line-too-long
# pylint: disable=invalid-name

# Adding local Keras
KERAS_PATH = '/media/manish/Data/keras-master/'
sys.path.insert(0, KERAS_PATH)
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras'))
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras', 'layers'))
import keras
from keras.layers import Reshape, Input, Conv2D, Convolution2D, Dropout, Conv2DTranspose, Dense, Flatten, UpSampling2D
from keras.layers import LeakyReLU, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import *
from keras.models import Model
K.set_learning_phase(1)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from visdom import Visdom as V

print(colored("Tensorflow version: {}".format(tf.__version__), 'red'))
print(colored("Keras version: {}".format(keras.__version__), 'red'))

class DCGANK(object):
    def __init__(self, sess, ndf=32, ngf=32, batch_size=16, nrows=28, ncols=28, nch=3, zdim=100, train_data_dir=None,
                test_data_dir=None, checkpoint_dir=None, learning_rate=1e-03):
        self.batch_size = batch_size
        self.nrows = nrows #Height
        self.ncols = ncols #Width
        self.nch = nch #channels

        self.z_dim = zdim #number of neurons for the generator input
        self.ndf = ndf # number of discriminator filters
        self.ngf = ngf # number of generator filters

        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.lr = learning_rate

        # all the variables that will be initialized later at some point
        self.inputs = None
        self.noise = None

        self.D = None
        self.G = None

        self.saver = None
        self.traindata_generator = None
        self.testdata_generator = None
        self.opt = None
        self.dopt = None
        self.GAN = None

        self.build_model()

    def build_model(self):

        self.inputs = Input(shape=[self.nrows, self.ncols, self.nch])
        self.noise = Input(shape=[self.z_dim])
        self.opt = Adam(lr=self.lr/10)
        self.dopt = Adam(lr=self.lr)

        inputs = self.inputs
        noise = self.noise
        self.D = self.discriminator(inputs)
        self.G = self.generator(noise)

        # Freeze weights in the discriminator for stacked training
        def make_trainable(net, val):
            net.trainable = val
            for l in net.layers:
                l.trainable = val
        make_trainable(self.D, False)

        # Build stacked GAN model
        H = self.G(noise)
        gan_V = self.D(H)
        self.GAN = Model(noise, gan_V)
        self.GAN.compile(loss='binary_crossentropy', optimizer=self.opt)
        self.GAN.summary()

        # create the train datagenerator
        self.datagenerator()

    def discriminator(self, image):
        # input image NXHXWXC -- tensorflow format
        '''conv1 = Conv2D(self.ndf, (4, 4), strides=(2, 2), padding="same", name="d_conv1")(image)
        lrlu1 = LeakyReLU(alpha=0.2, name="d_lrelu1")(conv1)
        print(colored(" >> After conv1: {}".format(lrlu1.shape), 'red'))

        conv2 = Conv2D(self.ndf*2, (4, 4), strides=(2, 2), padding="same", name="d_conv2")(lrlu1)
        bnorm2 = BatchNormalization(momentum=0.3, name="d_bn2")(conv2)
        lrlu2 = LeakyReLU(alpha=0.2, name="d_lrelu2")(bnorm2)
        print(colored(" >> After conv2: {}".format(lrlu2.shape), 'red'))

        conv3 = Conv2D(self.ndf*4, (4, 4), strides=(2, 2), padding="same", name="d_conv3")(lrlu2)
        bnorm3 = BatchNormalization(momentum=0.3, name="d_bn3")(conv3)
        lrlu3 = LeakyReLU(alpha=0.2, name="d_lrelu3")(bnorm3)
        print(colored(" >> After conv3: {}".format(lrlu3.shape), 'red'))

        conv4 = Conv2D(self.ndf*8, (4, 4), strides=(2, 2), padding="same", name="d_conv4")(lrlu3)
        bnorm4 = BatchNormalization(momentum=0.3, name="d_bn4")(conv4)
        lrlu4 = LeakyReLU(alpha=0.2, name="d_lrelu4")(bnorm4)
        print(colored(" >> After conv1: {}".format(lrlu4.shape), 'red'))

        logit = Flatten()(lrlu4)
        logit = Dense(1, name="d_linear")(logit)
        output = Activation('sigmoid', name="d_sigmoid")(logit)

        discriminator = Model(image, output)
        discriminator.compile(loss='binary_crossentropy', optimizer=self.dopt)
        discriminator.summary()'''
        dropout_rate = 0.3
        H = Convolution2D(256, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(image)
        H = LeakyReLU(0.2)(H)
        H = Dropout(dropout_rate)(H)
        H = Convolution2D(512, 5, 5, subsample=(2, 2), border_mode='same', activation='relu')(H)
        H = LeakyReLU(0.2)(H)
        H = Dropout(dropout_rate)(H)
        H = Flatten()(H)
        H = Dense(256)(H)
        H = LeakyReLU(0.2)(H)
        H = Dropout(dropout_rate)(H)
        d_V = Dense(1, activation='sigmoid')(H)
        discriminator = Model(image, d_V)
        discriminator.compile(loss='binary_crossentropy', optimizer=self.dopt)
        discriminator.summary()

        return discriminator

    def generator(self, noise):
        '''deconv1 = Conv2DTranspose(self.ngf*8, (4, 4), padding="valid", name="g_deconv1")(noise)
        bnorm1 = BatchNormalization(momentum=0.3, name="g_bnorm1")(deconv1)
        rlu1 = LeakyReLU(alpha=0.2, name="g_rlu1")(bnorm1)

        deconv2 = Conv2DTranspose(self.ngf*4, (4, 4), strides=(2, 2), padding="same", name="g_deconv2")(rlu1)
        bnorm2 = BatchNormalization(momentum=0.3, name="g_bnorm2")(deconv2)
        rlu2 = LeakyReLU(alpha=0.2, name="g_rlu2")(bnorm2)

        deconv3 = Conv2DTranspose(self.ngf*2, (4, 4), strides=(2, 2), padding="same", name="g_deconv3")(rlu2)
        bnorm3 = BatchNormalization(momentum=0.3, name="g_bnorm3")(deconv3)
        rlu3 = LeakyReLU(alpha=0.2, name="g_rlu3")(bnorm3)

        deconv4 = Conv2DTranspose(self.ngf, (4, 4), strides=(2, 2), padding="same", name="g_deconv4")(rlu3)
        bnorm4 = BatchNormalization(momentum=0.3, name="g_bnorm4")(deconv4)
        rlu4 = LeakyReLU(alpha=0.2, name="g_rlu4")(bnorm4)

        logit = Conv2DTranspose(self.nch, (4, 4), strides=(2, 2), padding="same", name="g_output")(rlu4)
        output = Activation('tanh', name="g_tanh")(logit)'''
        nch = int(200) #self.ngf*8
        H = Dense(nch*14*14, init='glorot_normal')(noise)
        H = BatchNormalization()(H)
        H = Activation('relu')(H)
        H = Reshape([14, 14, nch])(H)
        H = UpSampling2D(size=(2, 2))(H)
        H = Convolution2D(int(nch/2), 3, 3, border_mode='same', init='glorot_uniform')(H)
        H = BatchNormalization()(H)
        H = Activation('relu')(H)
        H = Convolution2D(int(nch/4), 3, 3, border_mode='same', init='glorot_uniform')(H)
        H = BatchNormalization()(H)
        H = Activation('relu')(H)
        H = Convolution2D(self.nch, 1, 1, border_mode='same', init='glorot_uniform')(H)
        g_V = Activation('sigmoid')(H)
        generator = Model(noise, g_V)
        generator.compile(loss='binary_crossentropy', optimizer=self.opt)
        generator.summary()

        #-- output will be of shape HxWxC same as input image
        return generator

    def datagenerator(self):
        seed = 1234

        # Augmentation parameter for the training
        # rescale=1. / 255
        train_datagen = ImageDataGenerator()

        self.traindata_generator = train_datagen.flow_from_directory(
                                    self.train_data_dir,
                                    class_mode="sparse",
                                    target_size=(self.nrows, self.ncols),
                                    batch_size=self.batch_size,
                                    seed=seed)

    def train(self, config):

        # use fixed sample to see the progress for the same noise
        sample_z = np.random.uniform(0, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        # Training Begins
        tot_iter = config.number_per_epoch * config.epoch

        with self.sess.as_default():
            print(" >> Training Begin:")

            start_step = 0
            start_time = time.time()

            vis = V()
            for ibatch in range(start_step+1, start_step+tot_iter+1):
                batch_images, _ = self.traindata_generator.next()
                batch_images = batch_images/255
                real = np.transpose(batch_images,  (0, 3, 1, 2))
                #vis.images(real, opts=dict(title='Real Images', caption='Should be numbers.'), win="real_images", env="model")

                curr_batch_size = batch_images.shape[0]
                z_batch = np.random.uniform(0, 1, [curr_batch_size, self.z_dim]).astype(np.float32)

                generated_images = self.G.predict(z_batch)

                # Train discriminator on real and generated images
                X = np.concatenate((batch_images, generated_images))
                y = np.zeros([2*curr_batch_size])
                y[0:curr_batch_size] = 1

                errD  = self.D.train_on_batch(X, y)

                # train Generator-Discriminator stack on input noise to non-generated output class
                noise_tr = np.random.uniform(-1, 1, [curr_batch_size, self.z_dim]).astype(np.float32)
                y2 = np.ones([curr_batch_size])

                errG = self.GAN.train_on_batch(noise_tr, y2)

                print(" >> Epoch/Batch: [%2d]/[%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (int(ibatch/config.number_per_epoch), int(ibatch%config.number_per_epoch),
                    time.time() - start_time, errD, errG))

                if np.mod(ibatch, 10) == 0:
                    samples = self.G.predict(sample_z)
                    samples = np.array(samples)
                    samples = np.transpose(samples, (0, 3, 1, 2))
                    real = np.transpose(batch_images,  (0, 3, 1, 2))
                    print(colored(" >> Generated samples shape: {}".format(samples.shape), 'red'))
                    vis.images(samples, opts=dict(title='Generated Images', caption='Should be dogs.'), win="generated_images", env="model")
                    vis.images(real, opts=dict(title='Real Images', caption='Should be dogs.'), win="real_images", env="model")