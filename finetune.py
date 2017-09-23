import tensorflow as tf
import numpy as np
from vgg19net import VGGNet


x = tf.placeholder(tf.float32, [1, 224, 224, 3])
keep_prob = tf.placeholder(tf.float32)
skip_layer  = ['fc6', 'fc7', 'fc8']
vgg = VGGNet(x, keep_prob, 3, skip_layer)
