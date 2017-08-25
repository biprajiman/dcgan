import os
import scipy.misc
import numpy as np

from DCGAN import DCGAN
import pprint as pp
import tensorflow as tf

flags = tf.app.flags
flags.DEFINE_integer("epoch", 25, "Epoch to train [25]")
flags.DEFINE_integer("number_per_epoch", 20580, "Samples to see per epoch to train")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate for adam [0.0002]")
flags.DEFINE_float("beta1", 0.2, "Momentum term of adam [0.5]")
flags.DEFINE_integer("batch_size", 32, "The size of batch images [32]")
flags.DEFINE_integer("input_height", 64, "The size of image to use (will be center cropped). [64]")
flags.DEFINE_integer("input_width", None, "The size of image to use (will be center cropped). If None, same value as input_height [None]")
flags.DEFINE_string("train_data_dir", "traindata", "Directory name to read the traindata from [traindata]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", True, "True for training, False for testing [False]")
FLAGS = flags.FLAGS

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.input_width is None:
    FLAGS.input_width = FLAGS.input_height

  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

  #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
  run_config = tf.ConfigProto()
  run_config.gpu_options.allow_growth=True

  with tf.Session(config=run_config) as sess:
      #sess, ndf=32, ngf=32, batch_size=32, nrows=64, ncols=64, nch=1, zdim=100, train_data_dir, test_data_dir, checkpoint_dir
      dcgan = DCGAN(
          sess,
          batch_size=FLAGS.batch_size,
          train_data_dir=FLAGS.train_data_dir,
          checkpoint_dir=FLAGS.checkpoint_dir)

      #if FLAGS.train:
      dcgan.train(FLAGS)
      #else:
        #if not dcgan.load(FLAGS.checkpoint_dir)[0]:
          #raise Exception("[!] Train a model first, then run test mode")

if __name__ == '__main__':
  tf.app.run()