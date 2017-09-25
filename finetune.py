import tensorflow as tf
import numpy as np
from vgg19net import VGGNet
from scipy.misc import imread, imsave, imresize


epoch = 10
batch_size = 128


x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
y = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
skip_layer  = ['fc6', 'fc7', 'fc8']
vgg = VGGNet(x, keep_prob, 2, skip_layer)

prediction = vgg.net['fc8']


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
optimizer = tf.AdamOptimizer().minimize(cost)


def get_batch(start_index, batch_size):
	out = [np.zeros() for i in range(batch_size)]
	
