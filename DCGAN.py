from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, pdb, traceback
from termcolor import colored
import tensorflow as tf
import numpy as np
import time
# pylint: disable=missing-docstring
# pylint: disable=line-too-long
# pylint: disable=invalid-name

# Adding local Keras
KERAS_PATH = '/media/manish/Data/keras-master/'
sys.path.insert(0, KERAS_PATH)
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras'))
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras', 'layers'))
import keras
from keras.layers import Conv2D, Conv2DTranspose, Dense
from keras.layers import LeakyReLU, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
K.set_learning_phase(1)

import matplotlib.pyplot as plt
import matplotlib.animation as animation

print(colored("Tensorflow version: {}".format(tf.__version__), 'red'))
print(colored("Keras version: {}".format(keras.__version__), 'red'))

class DCGAN(object):
    def __init__(self, sess, ndf=32, ngf=32, batch_size=4, nrows=64, ncols=64, nch=3, zdim=100, train_data_dir=None, test_data_dir=None, checkpoint_dir=None):
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

        # all the variables that will be initialized later at some point
        self.inputs = None
        self.noise = None
        self.real_labels = None
        self.fake_labels = None
        self.D_real = None
        self.D_logits_real = None
        self.G = None
        self.D_fake = None
        self.D_logits_fake = None
        self.d_real_loss = None
        self.d_fake_loss = None
        self.g_loss = None
        self.d_total_loss = None
        self.saver = None
        self.traindata_generator = None
        self.testdata_generator = None

        self.build_model()

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.nrows, self.ncols, self.nch), name="real_images")
        self.noise = tf.placeholder(tf.float32, shape=(self.batch_size, self.z_dim), name="noise")

        self.real_labels = tf.ones(self.batch_size, name="real_labels")
        self.fake_labels = tf.zeros(self.batch_size)

        inputs = self.inputs
        noise = self.noise
        self.D_real, self.D_logits_real = self.discriminator(inputs, reuse=False)
        self.G = self.generator(noise) # generates the fake images
        self.D_fake, self.D_logits_fake = self.discriminator(self.G, reuse=True)

        self.d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_real, labels=tf.ones_like(self.D_real)))
        self.d_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake, labels=tf.zeros_like(self.D_fake)))

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_fake, labels=tf.ones_like(self.D_fake)))

        self.d_total_loss = self.d_real_loss + self.d_fake_loss # log(D(x)) + log(1 - D(G(z)))

        self.summary_d_loss = tf.summary.scalar("d_loss", self.d_total_loss)
        self.summary_g_loss = tf.summary.scalar("g_loss", self.g_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver()
        
        # create the train datagenerator
        self.datagenerator()

    def discriminator(self, image, reuse=False):
        #TODO: return model and linear logits
        with tf.variable_scope("discriminator") as scope:
            if(reuse):
                scope.reuse_variables()

            # input image NXHXWXC -- tensorflow format
            conv1 = Conv2D(self.ndf, (4, 4), strides=(2, 2), padding="same", name="d_conv1")(image)
            bnorm1 = BatchNormalization()(conv1)
            lrlu1 = LeakyReLU(alpha=0.2)(bnorm1)
            print(colored(" >> After conv1: {}".format(lrlu1.shape), 'red'))

            conv2 = Conv2D(self.ndf*2, (4, 4), strides=(2, 2), padding="same", name="d_conv2")(lrlu1)
            bnorm2 = BatchNormalization()(conv2)
            lrlu2 = LeakyReLU(alpha=0.2)(bnorm2)
            print(colored(" >> After conv2: {}".format(lrlu2.shape), 'red'))

            conv3 = Conv2D(self.ndf*4, (4, 4), strides=(2, 2), padding="same", name="d_conv3")(lrlu2)
            bnorm3 = BatchNormalization()(conv3)
            lrlu3 = LeakyReLU(alpha=0.2)(bnorm3)
            print(colored(" >> After conv3: {}".format(lrlu3.shape), 'red'))

            conv4 = Conv2D(self.ndf*8, (4, 4), strides=(2, 2), padding="same", name="d_conv4")(lrlu3)
            bnorm4 = BatchNormalization()(conv4)
            lrlu4 = LeakyReLU(alpha=0.2)(bnorm4)
            print(colored(" >> After conv1: {}".format(lrlu4.shape), 'red'))

            logit = Conv2D(1, (4, 4), padding="valid", name="d_linear")(lrlu4)
            output = Activation('sigmoid')(logit)

            #-- output will be of shape batchsizeX1X1
            print(colored(" >> Discriminator ouput shape: {}".format(output.shape), 'red'))
            return output, logit

    def generator(self, noise):
         #TODO: return fake image with same size as input
        with tf.variable_scope("generator") as scope:
            print(colored(" >> Before deconv1: {}".format(noise.shape), 'red'))

            #deconv1 = Dense(self.ngf*8*4*4, name="g_deconv1")(noise)#Conv2DTranspose(self.ngf*8, (4, 4), padding="valid", name="g_deconv1")(noise)
            #deconv1 = tf.reshape(deconv1, [-1, 4, 4, self.ngf*8])
            noise = tf.reshape(noise, [-1, 1, 1, self.z_dim])
            deconv1 = Conv2DTranspose(self.ngf*8, (4, 4), padding="valid", name="g_deconv1")(noise)
            bnorm1 = BatchNormalization()(deconv1)
            rlu1 = Activation('relu')(bnorm1)
            print(colored(" >> After deconv1: {}".format(rlu1.shape), 'red'))

            deconv2 = Conv2DTranspose(self.ngf*4, (4, 4), strides=(2, 2), padding="same", name="g_deconv2")(rlu1)
            print(colored(" >> After deconv2: {}".format(deconv2.shape), 'red'))
            bnorm2 = BatchNormalization()(deconv2)
            rlu2 = Activation('relu')(bnorm2)

            deconv3 = Conv2DTranspose(self.ngf*2, (4, 4), strides=(2, 2), padding="same", name="g_deconv3")(rlu2)
            bnorm3 = BatchNormalization()(deconv3)
            rlu3 = Activation('relu')(bnorm3)
            print(colored(" >> After deconv3: {}".format(rlu3.shape), 'red'))

            deconv4 = Conv2DTranspose(self.ngf, (4, 4), strides=(2, 2), padding="same", name="g_deconv4")(rlu3)
            bnorm4 = BatchNormalization()(deconv4)
            rlu4 = Activation('relu')(bnorm4)
            print(colored(" >> After deconv4: {}".format(rlu4.shape), 'red'))

            logit = Conv2DTranspose(self.nch, (4, 4), strides=(2, 2), padding="same", name="g_logit")(rlu4)
            output = Activation('tanh')(logit)
            print(colored(" >> Discriminator ouput shape: {}".format(output.shape), 'red'))

            #-- output will be of shape HxWxC same as input image
            return output

    def datagenerator(self):
        seed = 1234
        train_data_gen_args = dict()
        self.traindata_generator = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                    self.train_data_dir,
                                    class_mode="sparse",
                                    target_size=(self.nrows, self.ncols),
                                    batch_size=self.batch_size,
                                    seed=seed)

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_total_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # use fixed sample to see the progress for the same noise
        sample_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Training Begins
        tot_iter = config.number_per_epoch * config.epoch
        self.sess.run(tf.global_variables_initializer())

        with self.sess.as_default():
            # restore from a checkpoint if exists
            '''if load_model(self.sess, self.saver, self.checkpoint_dir):
                print(" >> Load previous model success.")
            else:
                print(" >> No previous model found.")'''

            print(" >> Training Begin:")

            start_step = global_step.eval()
            start_time = time.time()
            plt.ion()
            for ibatch in range(start_step+1, start_step+tot_iter+1):

                #TODO: use the datagenerator to get the batch data in each run
                batch_images, _ =  self.traindata_generator.next()
                #print(colored(" >> Batch Images shape: {}".format(batch_images.shape), 'red'))

                z_batch = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
                #print(colored(" >> Random noise shape: {}".format(z_batch.shape), 'red'))

                # Update the discriminator network
                d_s = self.sess.run([d_optim],
                    feed_dict = {
                        self.inputs: batch_images,
                        self.noise: z_batch
                    })

                #Update the generator network
                g_s = self.sess.run([g_optim],
                    feed_dict = {
                        self.noise: z_batch
                    })

                errD_fake = self.d_fake_loss.eval({self.noise: z_batch})
                errD_real = self.d_real_loss.eval({self.inputs: batch_images})
                errG = self.g_loss.eval({self.noise: z_batch})

                def gallery(array, ncols=4):
                    nindex, height, width, intensity = array.shape
                    
                    if nindex < ncols:
                        ncols = nindex

                    nrows = nindex//ncols
                    assert nindex == nrows*ncols
                    result = (array.reshape((nrows, ncols, height, width, intensity))
                            .swapaxes(1, 2)
                            .reshape((height*nrows, width*ncols, intensity)))
                    plt.gca().clear()
                    plt.imshow(np.divide(result, 255))

                global_step.assign(ibatch).eval()

                print(" >> Epoch/Batch: [%2d]/[%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (int(ibatch/config.number_per_epoch), int(ibatch%config.number_per_epoch),
                    time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(ibatch, 100) == 0:
                    samples = self.sess.run(
                                            [self.G],
                                            feed_dict={
                                                self.noise: sample_z
                                            })
                    samples = np.squeeze(np.array(samples), axis=0)
                    #print(colored(" >> Generated samples shape: {}".format(np.shape(samples)), 'red'))
                    gallery(samples)
                    time.sleep(1)
                    plt.show()