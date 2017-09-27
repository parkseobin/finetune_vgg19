from vgg19net import VGGNet
import tensorflow as tf
import numpy as np
from scipy.misc import imread, imresize


batch_size = 64
train_data_size = 12500

def get_batch(start_index, batch_size):
	directory = '/home/park/programming/machine_learning/datasets/cat_vs_dog/train/'
	out = []

	for i in range(batch_size/2):
		img = imread(directory+'dog.{0}.jpg'.format(i+start_index))
		img = imresize(img, (224, 224))
		out.append(np.array(img))

	for i in range(batch_size/2):
		img = imread(directory+'cat.{0}.jpg'.format(i+start_index))
		img = imresize(img, (224, 224))
		out.append(np.array(img))

	out = np.reshape(out, [batch_size, 224, 224, 3])

	return out



x = tf.placeholder(tf.float32, [batch_size, 224, 224, 3])
learning_rate = tf.placeholder(tf.float32)
vgg = VGGNet(x, 1, 2, ['fc7', 'fc8']) 



with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print 'initializing weights'
	vgg.load_initial_weights(sess)
	print 'weights initialized'

	y = [[1, 0]]*(batch_size/2) + [[0, 1]]*(batch_size/2)
	start_index = 0
	for i in range(train_data_size/batch_size):
		out = get_batch(start_index, batch_size)
		ext = sess.run(vgg.net['fc6'], feed_dict={x:out})
		start_index += batch_size
		np.save('data/ext.%d.npy'%i, ext)
		print i


