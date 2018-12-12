from model import R_FSRCNN

import numpy as np
import tensorflow as tf

import pprint
import os

flags = tf.app.flags
flags.DEFINE_boolean("fast", False, "Use the fast model (FSRCNN-s) [False]")
flags.DEFINE_integer("epoch", 15000, "Number of epochs [10]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [128]")
flags.DEFINE_float("learning_rate", 1e-3, "The learning rate of gradient descent algorithm [1e-3]")
flags.DEFINE_float("momentum", 0.9, "The momentum value for the momentum SGD [0.9]")
flags.DEFINE_integer("c_dim", 1, "Dimension of image color [1]")
flags.DEFINE_integer("scale", 3, "The size of scale factor for preprocessing input image [3]")
flags.DEFINE_integer("stride", 4, "The size of stride to apply to input image [4]")
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Name of checkpoint directory [checkpoint]")
flags.DEFINE_string("output_dir", "result", "Name of test output directory [result]")
flags.DEFINE_string("data_dir", "General-100-aug", "Name of data directory to train on [FastTrain]")
flags.DEFINE_boolean("train", False, "True for training, false for testing [True]")
flags.DEFINE_integer("threads", 1, "Number of processes to pre-process data with [1]")
flags.DEFINE_boolean("params", False, "Save weight and bias parameters [False]")
flags.DEFINE_integer("pic_index", 2, "Index of image based on lexicographic order in data folder")

FLAGS = flags.FLAGS

pp = pprint.PrettyPrinter()

def main(_):
  pp.pprint(flags.FLAGS.__flags)

  if FLAGS.fast:
    FLAGS.checkpoint_dir = 'fast_{}'.format(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
  if not os.path.exists(FLAGS.output_dir):
    os.makedirs(FLAGS.output_dir)


  with tf.Session() as sess:
    R_fsrcnn = R_FSRCNN(sess, config=FLAGS)
    R_fsrcnn.run()
    writer = tf.summary.FileWriter(".\log", tf.get_default_graph())
    writer.close()
    
if __name__ == '__main__':
  tf.app.run()
