import tensorflow as tf 
import numpy as np 

def add_layer(inputs, in_size, out_size, activiatoin_func=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_b = tf.matmul(inputs, Weights) + biases
	if activiatoin_func is None:
		output = Wx_b
	else:
		output = activiatoin_func(Wx_b)

	return output