import tensorflow as tf 
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = load_digits()
X = digits.data 
Y = digits.target
y=LabelBinarizer().fit_transform(Y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)

def add_layer(inputs, in_size, out_size, activiatoin_func=None):
	Weights = tf.Variable(tf.random_normal([in_size, out_size]))
	biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
	Wx_b = tf.matmul(inputs, Weights) + biases
	Wx_b = tf.nn.dropout(Wx_b, keep_prob)
	if activiatoin_func is None:
		output = Wx_b
	else:
		output = activiatoin_func(Wx_b)

	return output

def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction, feed_dict={xs:v_xs, keep_prob:1.})
	correct_prection = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prection, tf.float32))
	result = sess.run(accuracy,feed_dict={xs:v_xs, ys:v_ys})
	return result

# define placeholder for inputs to networks
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

# add one hidden layer
l1 = add_layer(xs, 64, 50, activiatoin_func=tf.nn.tanh)
prediction = add_layer(l1, 50, 10, activiatoin_func=tf.nn.softmax)



# error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))


train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
	#batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={xs:X_train, ys:y_train, keep_prob: 0.5})
	if i%50==0:
		print(compute_accuracy(X_train, y_train))
		print(compute_accuracy(X_test, y_test))
		print('**********')

