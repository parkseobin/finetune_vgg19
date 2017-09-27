# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io

VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)


VGG19_LAYERS2 = (
    'conv1_1','conv1_2',

    'conv2_1','conv2_2',

    'conv3_1','conv3_2', 'conv3_3',
    'conv3_4', 

    'conv4_1', 'conv4_2', 'conv4_3',
    'conv4_4', 

    'conv5_1', 'conv5_2', 'conv5_3',
    'conv5_4',

	'fc6', 'fc7', 'fc8',
	)

def load_net2(data_path):
	data = np.load(data_path).item()
	return data

def net_preloaded2(weights, input_image, pooling):
	net = {}
	current = input_image
	for name in VGG19_LAYERS2:
		kernels = weights[name][0]
		bias = weights[name][1]
		kernels = np.array(kernels, dtype=np.float32)
		bias = np.array(bias, dtype=np.float32)

		if name[:4] == 'conv':
			current = _conv_layer(current, kernels, bias)
			current = tf.nn.relu(current)
			po = name[4:7]
			if po == '1_2' or po == '2_2' or po == '3_4' or po == '4_4' or po == '5_4':
				current = _pool_layer(current, pooling)
		elif name[:2] == 'fc':
			if(name[2] == '6'):
				current = tf.reshape(current, [-1, 7*7*512])
			current = _fc(current, kernels, bias)


		net[name] = current

	return net


def net_preloaded(weights, input_image, pooling):
    net = {}
    current = input_image
    for i, name in enumerate(VGG19_LAYERS):
        kind = name[:4]
        if kind == 'conv':
            kernels, bias = weights[i][0][0][0][0]
            # matconvnet: weights are [width, height, in_channels, out_channels]
            # tensorflow: weights are [height, width, in_channels, out_channels]
            kernels = np.transpose(kernels, (1, 0, 2, 3))
            bias = bias.reshape(-1)
            current = _conv_layer(current, kernels, bias)
        elif kind == 'relu':
            current = tf.nn.relu(current)
        elif kind == 'pool':
            current = _pool_layer(current, pooling)
        net[name] = current

    assert len(net) == len(VGG19_LAYERS)
    return net

def _conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, weights, strides=(1, 1, 1, 1),
            padding='SAME')
    return tf.nn.bias_add(conv, bias)


def _pool_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),
                padding='SAME')


def _fc(input, weights, bias):
	out = tf.nn.xw_plus_b(input, weights, bias)
	return out

def preprocess(image, mean_pixel):
    return image - mean_pixel


def unprocess(image, mean_pixel):
    return image + mean_pixel
