import tensorflow as tf
import numpy as np
from vgg19net import VGGNet
from scipy.misc import imread, imsave, imresize
from time import time
from datetime import datetime


lr = 0.1
epoch = 10
batch_size = 16
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
y = tf.placeholder(tf.float32)
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)
skip_layer  = ['fc6', 'fc7', 'fc8']
vgg = VGGNet(x, keep_prob, 2, skip_layer)
var_list = [v for v in tf.trainable_variables() if v.name.split('/')[0] in skip_layer]

prediction = vgg.net['fc8']


cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
gradients = tf.gradients(cost, var_list)
gradients = list(zip(gradients, var_list))
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
train_op = optimizer.apply_gradients(grads_and_vars=gradients)



gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
	sess.run(tf.global_variables_initializer())
	vgg.load_initial_weights(sess)


	y_ = [[1, 0]]*(batch_size/2) 
	y_ += [[0, 1]]*(batch_size/2)
	y_ = np.array(y_, dtype=np.float32)
	# one hot!

	start_time = time()
	last_epoch_loss = 0
	epoch_loss = 0
	for e in range(epoch):

		last_epoch_loss = epoch_loss
		epoch_loss = 0
		start_index = 0
		for i in range(train_data_size/batch_size):
			x_ = get_batch(start_index, batch_size)
			
			_, c = sess.run([train_op, cost], feed_dict={x:x_, y:y_, keep_prob:0.5, learning_rate:lr})
			epoch_loss += c
			start_index += batch_size

		if(last_epoch_loss <= epoch_loss):
			lr *= 0.1
			print 'learning rate decreased! [ > %f ]' % lr
				
			
		print "\n###!  epoch %d >> loss: %f   [%s]\n" % (e+1, epoch_loss, datetime.now())


	print 'end of training!!'
	print 'elapsed time : %d' % (time()-start_time)
	
	#make dictionmary!
	finetuned_dict = {}
	for layer in vgg.VGG19_LAYERS:
		if(layer[:4] == 'pool'):
			continue
		finetuned_dict[layer] = {}
		with tf.variable_scope(layer, reuse=True):
			finetuned_dict[layer][0] = sess.run(tf.get_variable('weights'))
			finetuned_dict[layer][1] = sess.run(tf.get_variable('biases'))

	np.save('weights.npy', finetuned_dict)




