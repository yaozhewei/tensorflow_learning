import tensorflow as tf 
import numpy as np 
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot= True)

def add_layer(inputs, in_size, out_size, activiatoin_func=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_b = tf.matmul(inputs, Weights) + biases
	if activiatoin_func is None:
		output = Wx_b
	else:
		output = activiatoin_func(Wx_b)

	return output

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs:v_xs})
	correct_prection = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prection, tf.float32))
	result = sess.run(accuracy,feed_dict={xs:v_xs, ys:v_ys})
	return result


# define placeholder for inputs to networks
xs = tf.placeholder(tf.float32, [None, 784])
ys = tf.placeholder(tf.float32, [None, 10])

# add one hidden layer
#l1 = add_layer(xs, 784, 50, activiatoin_func=tf.nn.relu)


# add output layer
prediction = add_layer(xs, 784, 10, activiatoin_func=tf.nn.softmax)


# error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))


train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs:batch_xs, ys:batch_ys})
	if i%50==0:
		print(sess.run(cross_entropy, feed_dict={xs:batch_xs, ys:batch_ys}))
		#print(compute_accuracy(mnist.test.images, mnist.test.labels))

