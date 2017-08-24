from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys, pdb, traceback
from termcolor import colored
import tensorflow as tf
# pylint: disable=missing-docstring
# pylint: disable=line-too-long
# pylint: disable=invalid-name

# Adding local Keras
KERAS_PATH = '/media/manish/Data/keras-master/'
sys.path.insert(0, KERAS_PATH)
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras'))
sys.path.insert(0, os.path.join(KERAS_PATH, 'keras', 'layers'))
import keras
from keras import backend as K
import matplotlib.pyplot as plt

class DCGAN(object):
    def __init__(self, sess, ndf=32, ngf=32, batch_size=32, nrows=64, ncols=64, nch=1, zdim=100, train_data_dir, test_data_dir, checkpoint_dir):
        self.batch_size = batch_size
        self.nrows = nr #Height
        self.ncols = nc #Width
        self.nch = nch #channels

        self.zdim = zdim #number of neurons for the generator input
        self.ndf = ndf # number of discriminator filters
        self.ngf = ngf # number of generator filters

        self.sess = sess
        self.checkpoint_dir = checkpoint_dir
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir 

        self.build_model()

    def build_model(self):

        self.inputs = tf.placeholder(tf.float32, shape=(self.batch_size, self.nrows, self.ncols, self.nch), name="real_images")
        self.noise = tf.placeholder(tf.float32, shape=(self.batch_size, self.zdim), name="noise")

        self.real_labels = tf.ones(self.batch_size, name="real_labels")
        self.fake_labels = tf.zeros(self.batch_size)

        inputs = self.inputs
        self.D_real, self.D_logits_real= self.discriminator(inputs, reuse=False)
        self.G = self.generator(self.noise) # generates the fake images
        self.D_fake, self.D_logits_fake = self.discrimator(self.G, reuse=True)

        self.d_real_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_real, tf.ones_like(self.D_real)))
        self.d_fake_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_fake, tf.zeros_like(self.D_fake)))

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_fake, tf.ones_like(self.D_fake)))

        self.d_total_loss = self.d_real_loss + self.d_fake_loss # log(D(x)) + log(1 - D(G(z)))

        self.summary_d_loss = tf.summary.scalar("d_loss", self.d_total_loss)
        self.summary_g_loss = tf.summary.scalar("g_loss", self.g_loss)

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
        self.saver = tf.train.Saver()
        
    def discriminator(self, image, reuse=False):
        #TODO: return model and linear logits
        with tf.variable_scope("discriminator") as scope:
            if(reuse):
                scope.reuse_variables()
            
            # input image NXHXWXC -- tensorflow format
            conv1 = Convolution2D(self.ndf, (4, 4), stride=(2, 2), W_regularizer=l2(0.0), b_regularizer=l2(0.0), name="d_conv1")(image)
            bnorm1 = BatchNormalization()(conv1)
            lrlu1 = LeakyReLU(alpha=0.2)(bnorm1)

            conv2 = Convolution2D(self.ndf*2, (4, 4), stride=(2, 2), W_regularizer=l2(0.0), b_regularizer=l2(0.0), name="d_conv2")(lrlu1)
            bnorm2 = BatchNormalization()(conv2)
            lrlu2 = LeakyReLU(alpha=0.2)(bnorm2)

            conv3 = Convolution2D(self.ndf*4, (4, 4), stride=(2, 2), W_regularizer=l2(0.0), b_regularizer=l2(0.0), name="d_conv3")(lrlu2)
            bnorm3 = BatchNormalization()(conv3)
            lrlu3 = LeakyReLU(alpha=0.2)(bnorm3)

            conv4 = Convolution2D(self.ndf*8, (4, 4), stride=(2, 2), W_regularizer=l2(0.0), b_regularizer=l2(0.0), name="d_conv4")(lrlu3)
            bnorm4 = BatchNormalization()(conv4)
            lrlu4 = LeakyReLU(alpha=0.2)(bnorm4)

            logit = Convolution2D(self.ndf*8, (1, 1), W_regularizer=l2(0.0), b_regularizer=l2(0.0), name="d_linear")(lrlu4)
            output = Activation('sigmoid')(logit)
            #-- output will be of shape batchsizeX1X1

            return output, logit

    def generator(self, noise):
         #TODO: return fake image with same size as input
        with tf.variable_scope("generator") as scope:
            deconv1 = Conv2DTranspose(self.ngf*8, (4, 4), name="g_deconv1")(noise)
            bnorm1 = BatchNormalization()(deconv1)
            rlu1 = Activation('relu')(bnorm1)

            deconv2 = Conv2DTranspose(self.ngf*4, (4, 4), stride=(2, 2), name="g_deconv2")(rlu1)
            bnorm2 = BatchNormalization()(deconv2)
            rlu2 = Activation('relu')(bnorm2)

            deconv3 = Conv2DTranspose(self.ngf*2, (4, 4), stride=(2, 2), name="g_deconv3")(rlu2)
            bnorm3 = BatchNormalization()(deconv3)
            rlu3 = Activation('relu')(bnorm3)

            deconv4 = Conv2DTranspose(self.ngf, (4, 4), stride=(2, 2), name="g_deconv4")(rlu3)
            bnorm4 = BatchNormalization()(deconv4)
            rlu4 = Activation('relu')(bnorm4)   

            logit = Conv2DTranspose(self.nch, (4, 4), stride=(2, 2), name="g_logit")(rlu4)
            output = Activation('tanh')(logit)

            #-- output will be of shape HxWxC same as input image
            return output

    def traingenerator(self):
        seed = 1234
        train_image_datagen = ImageDataGenerator(**train_data_gen_args).flow_from_directory(
                                    self.train_data_dir,
                                    class_mode="sparse",
                                    target_size=(self.nrows, self.ncols),
                                    batch_size=self.batch_size,
                                    seed=seed)
        return train_image_datagen

    def train(self, config):
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.d_total_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
              .minimize(self.g_loss, var_list=self.g_vars)
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        
        # use this to see the progress for the same noise
        sample_z = np.random.unitform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

        counter = 1
        start_time = time.time()
        global_step = tf.Variable(0, name='global_step', trainable=False)

        # Training Begins
        tot_iter = self.batch_size * config.numepoch
        self.sess.run(tf.global_variables_initializer())

        with self.sess.as_default():
            # restore from a checkpoint if exists
            if load_model(self.sess, self.saver, self.checkpoint_dir):
                print(" >> Load previous model success.")
            else:
                print(" >> No previous model found.")

            print(" >> Training Begin:")
            
            start_step = global_step.eval()
            start_t = time.time()
            for ibatch in range(start_step+1, start_step+tot_iter+1):
        
                #TODO: use the datagenerator to get the batch data in each run
                train_generator = self.traingenerator()

                batch_images =  train_generator.next()

                z_batch = np.random.unitform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)

                # Update the discriminator network
                d_s = self.sess.run([d_optim],
                    feed_dict = {
                        self.inputs = batch_images,
                        self.noise = z_batch
                    })
                
                #Update the generator network
                g_s = self.sess.run([g_optim],
                    feed_dict = {
                        self.noise = z_batch
                    })
                
                errD_fake = self.d_fake_loss({self.noise: z_batch})
                errD_real = self.d_real_loss({self.inputs: batch_images})
                errG = self.g_loss({self.noise: z_batch})

                def gallery(array, ncols=4):
                    nindex, height, width, intensity = array.shape
                    nrows = nindex//ncols
                    assert nindex == nrows*ncols
                    result = (array.reshape((nrows, ncols, height, width, intensity))
                            .swapaxes(1,2)
                            .reshape((height*nrows, width*ncols, intensity)))
                    return result
                
                
                global_step.assign(ibatch).eval()

                print(" >> Epoch/Batch: [%2d]/[%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                % (int(ibatch/self.batch_size), int(ibatch%self.batch_size),
                    time.time() - start_time, errD_fake+errD_real, errG))

                if np.mod(ibatch, 100) == 1:
                    samples = self.sess.run(
                                            [self.G],
                                            feed_dict={
                                                self.noise: sample_z
                                            })
                    samples = gallery(samples)
                    plt.imshow(samples)
                    plt.show()