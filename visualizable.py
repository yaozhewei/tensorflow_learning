import tensorflow as tf 
import numpy as np 

def add_layer(inputs, in_size, out_size, activiatoin_func=None):

	with tf.name_scope('layer'):
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
		with tf.name_scope('biase'):
			biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
		with tf.name_scope('Wx_b'):
			Wx_b = tf.matmul(inputs, Weights) + biases
		if activiatoin_func is None:
			output = Wx_b
		else:
			output = activiatoin_func(Wx_b)

		return output


x_data = np.linspace(-1,1,300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32,[None,1], name='x_input')
	ys = tf.placeholder(tf.float32,[None,1], name='y_input')


l1 = add_layer(xs, 1, 10, activiatoin_func=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activiatoin_func = None)

with tf.name_scope('loss'):
	loss =tf.reduce_mean(tf.reduce_sum( tf.square(ys - prediction), reduction_indices = [1]))

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

sess = tf.Session()

writer = tf.train.SummaryWriter("log/",sess.graph)
sess.run(init)

for i in range(1000):
	sess.run(train_step, feed_dict={xs:x_data, ys:y_data})
	if i%50 == 0:
		print(sess.run(loss, feed_dict={xs:x_data, ys:y_data}))