execfile("simple_vgg.py")

from scipy.misc import imread, imresize

w = load_net2('weights.npy')

input_image = imread('dog.jpg')
input_image = np.array(imresize(input_image, (224, 224)), dtype=np.float32)
input_image = np.reshape(input_image, [1, 224, 224, 3])
net = net_preloaded2(w, input_image, 'max')

with tf.Session() as sess:
	out = sess.run(net['fc8'])

	print out

