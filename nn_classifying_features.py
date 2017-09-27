import tensorflow as tf
import numpy as np
from datetime import datetime


lr = 0.0001
epoch = 2000
batch_size = 64
train_data_size = 12500

n_nodes_hl1 = 4096
n_nodes_hl2 = 2

x = tf.placeholder(tf.float32, [batch_size, 4096])
y = tf.placeholder(tf.float32, [batch_size, 2])
learning_rate = tf.placeholder(tf.float32)
keep_prob = tf.placeholder(tf.float32)



hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([4096, n_nodes_hl1])),
				'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([4096, n_nodes_hl2])),
				'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

l1 = tf.nn.xw_plus_b(x, hidden_layer_1['weights'], hidden_layer_1['biases'])
l1 = tf.nn.relu(l1)
#l1 = tf.nn.dropout(l1, keep_prob)

l2 = tf.nn.xw_plus_b(l1, hidden_layer_2['weights'], hidden_layer_2['biases'])
l2 = tf.nn.relu(l2)


pred = l2
y_ = [[1, 0]]*(batch_size/2) + [[0, 1]]*(batch_size/2)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)


with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	epoch_loss = 0
	prev_loss = 0
	try:
		for e in range(epoch):
	
			prev_loss = epoch_loss
			epoch_loss = 0
			start_index = 0
			for i in range(train_data_size/batch_size):
				ext = np.load('data/ext.%d.npy'%(i+start_index))
				_, c = sess.run([optimizer, cost], feed_dict={x:ext, y:y_, learning_rate:lr})
				epoch_loss += c
				
			if(prev_loss <= epoch_loss):
				lr *= 0.1
				print 'learning rate decreased'
	
			print "%d epoch >> %f [%s]" % (e, epoch_loss, datetime.now())
	except(KeyboardInterrupt):
		print 'interrupted'
		pass


	finetuned_dict = np.load('vgg19.npy').item()
	finetuned_dict['fc7'] = sess.run([hidden_layer_1['weights'], hidden_layer_1['biases']])
	finetuned_dict['fc8'] = sess.run([hidden_layer_2['weights'], hidden_layer_2['biases']])
	np.save('weights.npy', finetuned_dict)
	print 'weights saved'




