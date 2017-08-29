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
from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten
from keras.layers import LeakyReLU, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
K.set_learning_phase(1)

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from visdom import Visdom as V

print(colored("Tensorflow version: {}".format(tf.__version__), 'red'))
print(colored("Keras version: {}".format(keras.__version__), 'red'))

class DCGAN(object):
    def __init__(self, sess, ndf=32, ngf=32, batch_size=16, nrows=64, ncols=64, nch=3, zdim=100, train_data_dir=None, test_data_dir=None, checkpoint_dir=None):
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
        self.g_loss = None
        self.d_total_loss = None
        self.saver = None
        self.traindata_generator = None
        self.testdata_generator = None

        self.build_model()

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, shape=(None, self.nrows, self.ncols, self.nch), name="real_images")
        self.noise = tf.placeholder(tf.float32, shape=(None, 1, 1, self.z_dim), name="noise")

        inputs = self.inputs
        noise = self.noise
        self.D_real, self.D_logits_real = self.discriminator(inputs, reuse=False)
        self.G = self.generator(noise) # generates the fake images output, rlu1, rlu2, rlu3, rlu4
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
        self.d_vars = [var for var in t_vars if 'discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'generator' in var.name]

        self.saver = tf.train.Saver()
        
        # create the train datagenerator
        self.datagenerator()

    def discriminator(self, image, reuse=False):
        #TODO: return model and linear logits
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # input image NXHXWXC -- tensorflow format
            conv1 = Conv2D(self.ndf, (4, 4), strides=(2, 2), padding="same", name="d_conv1")(image)
            lrlu1 = LeakyReLU(alpha=0.2, name="d_lrelu1")(conv1)
            print(colored(" >> After conv1: {}".format(lrlu1.shape), 'red'))

            conv2 = Conv2D(self.ndf*2, (4, 4), strides=(2, 2), padding="same", name="d_conv2")(lrlu1)
            bnorm2 = BatchNormalization(momentum=0.3, name="d_bn2")(conv2)
            lrlu2 = LeakyReLU(alpha=0.2, name="d_lrelu1")(bnorm2)
            print(colored(" >> After conv2: {}".format(lrlu2.shape), 'red'))

            conv3 = Conv2D(self.ndf*4, (4, 4), strides=(2, 2), padding="same", name="d_conv3")(lrlu2)
            bnorm3 = BatchNormalization(momentum=0.3, name="d_bn3")(conv3)
            lrlu3 = LeakyReLU(alpha=0.2, name="d_lrelu1")(bnorm3)
            print(colored(" >> After conv3: {}".format(lrlu3.shape), 'red'))

            conv4 = Conv2D(self.ndf*8, (4, 4), strides=(2, 2), padding="same", name="d_conv4")(lrlu3)
            bnorm4 = BatchNormalization(momentum=0.3, name="d_bn4")(conv4)
            lrlu4 = LeakyReLU(alpha=0.2, name="d_lrelu1")(bnorm4)
            print(colored(" >> After conv1: {}".format(lrlu4.shape), 'red'))

            logit = Flatten()(lrlu4)
            logit = Dense(1, name="d_linear")(logit)
            output = Activation('sigmoid', name="d_sigmoid")(logit)

            #-- output will be of shape batchsizeX1X1
            print(colored(" >> Discriminator ouput shape: {}".format(output.shape), 'red'))
            return output, logit

    def generator(self, noise):
         #TODO: return fake image with same size as input
        with tf.variable_scope("generator") as scope:

            print(colored(" >> Before deconv1: {}".format(noise.shape), 'red'))

            deconv1 = Conv2DTranspose(self.ngf*8, (4, 4), padding="valid", name="g_deconv1")(noise)
            bnorm1 = BatchNormalization(momentum=0.3, name="g_bnorm1")(deconv1)
            rlu1 = LeakyReLU(alpha=0.2, name="g_rlu1")(bnorm1)
            print(colored(" >> After deconv1: {}".format(rlu1.shape), 'red'))

            deconv2 = Conv2DTranspose(self.ngf*4, (4, 4), strides=(2, 2), padding="same", name="g_deconv2")(rlu1)
            bnorm2 = BatchNormalization(momentum=0.3, name="g_bnorm2")(deconv2)
            rlu2 = LeakyReLU(alpha=0.2, name="g_rlu2")(bnorm2)
            print(colored(" >> After deconv2: {}".format(rlu2.shape), 'red'))

            deconv3 = Conv2DTranspose(self.ngf*2, (4, 4), strides=(2, 2), padding="same", name="g_deconv3")(rlu2)
            bnorm3 = BatchNormalization(momentum=0.3, name="g_bnorm3")(deconv3)
            rlu3 = LeakyReLU(alpha=0.2, name="g_rlu3")(bnorm3)
            print(colored(" >> After deconv3: {}".format(rlu3.shape), 'red'))

            deconv4 = Conv2DTranspose(self.ngf, (4, 4), strides=(2, 2), padding="same", name="g_deconv4")(rlu3)
            bnorm4 = BatchNormalization(momentum=0.3, name="g_bnorm4")(deconv4)
            rlu4 = LeakyReLU(alpha=0.2, name="g_rlu4")(bnorm4)
            print(colored(" >> After deconv4: {}".format(rlu4.shape), 'red'))

            logit = Conv2DTranspose(self.nch, (4, 4), strides=(2, 2), padding="same", name="g_output")(rlu4)
            logit = tf.reshape(logit, [-1, self.nrows, self.ncols, self.nch])
            output = Activation('tanh', name="g_tanh")(logit)
            print(colored(" >> Generator ouput shape: {}".format(output.shape), 'red'))

            #-- output will be of shape HxWxC same as input image
            return output

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
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_total_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        # use fixed sample to see the progress for the same noise
        sample_z = np.random.uniform(-1, 1, [self.batch_size, 1, 1, self.z_dim]).astype(np.float32)

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

            vis = V()
                
            for ibatch in range(start_step+1, start_step+tot_iter+1):

                #TODO: use the datagenerator to get the batch data in each run
                batch_images, _ =  self.traindata_generator.next()
                batch_images = (batch_images-127.5)/127.5
                #print(colored(" >> Batch Images shape: {}".format(batch_images.shape), 'red'))

                z_batch = np.random.uniform(-1, 1, [batch_images.shape[0], 1, 1, self.z_dim]).astype(np.float32)
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
                    
                    return np.array(result)

                global_step.assign(ibatch).eval()

                print(" >> Epoch/Batch: [%2d]/[%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (int(ibatch/config.number_per_epoch), int(ibatch%config.number_per_epoch),
                    time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(ibatch, 10) == 0:
                    samples = self.sess.run(
                                            [self.G],
                                            feed_dict={
                                                self.noise: sample_z
                                            })
                    samples = np.array(samples[0])
                    samples = np.transpose(samples, (0, 3, 1, 2))
                    real = np.transpose(batch_images,  (0, 3, 1, 2))
                    #samples = np.squeeze(np.array(samples), axis=0)
                    #print(colored(" >> Generated samples shape: {}".format(tf.shape(rlu1)), 'red'))
                    #print(colored(" >> Generated samples shape: {}".format(tf.shape(rlu2)), 'red'))
                    #print(colored(" >> Generated samples shape: {}".format(tf.shape(rlu3)), 'red'))
                    #print(colored(" >> Generated samples shape: {}".format(tf.shape(rlu4)), 'red'))
                    print(colored(" >> Generated samples shape: {}".format(samples.shape), 'red'))
                    vis.images(samples, opts=dict(title='Generated Images', caption='Should be dogs.'), win="generated_images", env="model")
                    vis.images(real, opts=dict(title='Real Images', caption='Should be dogs.'), win="real_images", env="model")