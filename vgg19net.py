"""

program for finetuning vggnet


"""



import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize
from caffe_classes import class_names

class VGGNet(object):

	def __init__(self, x, keep_prob, num_classes, skip_layer, ns_layer=0,
			weights_path = 'DEFAULT'):
		"""
		x should be [224, 224]

		"""


		self.X = x
		self.NUM_CLASSES = num_classes
		self.KEEP_PROB = keep_prob
		self.SKIP_LAYER = skip_layer
		self.VGG19_LAYERS = (
				'conv1_1', 'conv1_2', 'pool1',

				'conv2_1', 'conv2_2', 'pool2',

				'conv3_1', 'conv3_2', 'conv3_3', 'conv3_4', 'pool3',

				'conv4_1', 'conv4_2', 'conv4_3', 'conv4_4', 'pool4',

				'conv5_1', 'conv5_2', 'conv5_3', 'conv5_4', 'pool5',

				'fc6', 'fc7', 'fc8'
				)

		self.net = {}

		if weights_path == 'DEFAULT':
			self.WEIGHTS_PATH = 'vgg19.npy'
		else:
			self.WEIGHTS_PATH = weights_path

		self.create()

	def create(self):
		current = self.X

		for layer in self.VGG19_LAYERS[:-3]:
			kind = layer[:4]

			if kind == 'conv':
				if layer[4] == '1':
					filter_num = 64
				elif layer[4] == '2':
					filter_num = 128
				elif layer[4] == '3':
					filter_num = 256
				else:
					filter_num = 512
				
				current = conv(current, 3, filter_num, 1, layer)
			elif kind == 'pool':
				current = max_pool(current, 2, 2, layer)

			self.net[layer] = current


		current = tf.reshape(current, [-1, 7*7*512])

		current = fc(current, 7*7*512, 4096, 'fc6')
		self.net['fc6'] = current
		current = fc(current, 4096, 4096, 'fc7')
		self.net['fc7'] = current
		current = fc(current, 4096, self.NUM_CLASSES, 'fc8')
		self.net['fc8'] = current


	def load_initial_weights(self, session):
		
		weights_dict = np.load(self.WEIGHTS_PATH, encoding='bytes').item()

		
		for op_name in weights_dict:
			if op_name not in self.SKIP_LAYER:
		
				with tf.variable_scope(op_name, reuse=True):
					for data in weights_dict[op_name]:
					
						if len(data.shape) == 1:
							var = tf.get_variable('biases', trainable=False)
							session.run(var.assign(data))

						else:
							var = tf.get_variable('weights', trainable=False)
							session.run(var.assign(data))

		return 
							
					

	

def conv(x, filter_size, filter_num, stride, name, padding="SAME"):
	
	input_channels = int(x.get_shape()[-1])

	convolve = lambda i, k: tf.nn.conv2d(i, k,
										strides=[1, stride, stride, 1], 
										padding=padding)

	with tf.variable_scope(name) as scope:
		weights = tf.get_variable('weights', shape=[filter_size, filter_size,
													input_channels,
													filter_num])
		biases = tf.get_variable('biases', shape=[filter_num])

		conv = convolve(x, weights)
		bias = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape().as_list())
		relu = tf.nn.relu(bias)

	return relu

def max_pool(x, filter_size, stride, name, padding="SAME"):

	return tf.nn.max_pool(x, ksize=[1, filter_size, filter_size, 1],
							strides=[1, stride, stride, 1],
							padding=padding, name=name)

def fc(x, num_in, num_out, name):
	with tf.variable_scope(name) as scope:
		weights = tf.get_variable('weights', shape=[num_in, num_out])
		biases = tf.get_variable('biases', [num_out])

		act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

		
	return tf.nn.relu(act)





def test_forward():

	x = tf.Variable(tf.truncated_normal([1, 224, 224, 3]))
	kp = tf.placeholder(tf.float32)
	
	v = VGGNet(x, kp, 1000, [])
	
	with tf.Session() as sess:
		print 'loading initial weights'
		v.load_initial_weights(sess)
		print 'weights loaded'
	
		img = imread('../alexnet/images/honey')
		img = imresize(img, [224, 224])
		img = np.reshape(img, [1, 224, 224, 3])
		sess.run(tf.assign(x, img))
	
		out = sess.run(v.net['fc8'], feed_dict={kp:1})
	
		print class_names[out.argmax()]
	
	
	
