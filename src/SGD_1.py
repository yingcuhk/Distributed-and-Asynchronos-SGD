import os
import tensorflow as tf



def SGD(N = 100000, M = 10):

	K = 1
	graph = tf.Graph()
	with graph.as_default()
		theta_old = tf.placeholder(tf.float32, shape = (M,K))# beta is the optimization variable 
		X_train = tf.placeholder(tf.float32, shape = (None,M)) # training samples
		Y_train = tf.placeholder(tf.float32, shape = (None,K)) # training labels
		exp = tf.exp(tf.matmul(X,theta_old))
		h_theta_x = tf.log(tf.div(exp,exp + 1))
		cost = tf.reduce_sum(tf.mul(Y,h_theta_x) + tf.mul(tf.sub(1,Y), tf.sub(1,h_theta_x)))
		grad = tf.gradients(cost, theta_old)

		theta_new = tf.Variable(tf.sub(theta_old, tf.mul(stepsize, grad)))# beta is the optimization variable 

		X_array = np.random.randn(5000,M)
		Y_array = 
		X_test = tf.Constant
		Y_test = 

		predict = tf.matmul()
		# linear fitting 

