import os
import tensorflow as tf
import numpy as np
from math import sqrt

def SGD():

	M = 30
	K = 10
	N_test = 1000
	Beta = np.random.randn(M+1,K)
	X_test = np.random.randn(N_test,M)
	add_1s = np.ones((N_test,1))
	X_test = np.concatenate((X_test,add_1s),axis = 1)
	temp = np.dot(X_test, Beta) + 0.1 * np.random.randn(N_test,K)
	Y_test = np.argmax(temp, axis = 1) 
	N_train = 50000
	X_train= np.random.randn(N_train,M)
	add_1s = np.ones((N_train,1))
	X_train = np.concatenate((X_train,add_1s),axis = 1)
	temp = np.dot(X_train, Beta) + 0.01 * np.random.randn(N_train,K)
	Y_train = np.argmax(temp, axis = 1) 
	
	y_val = Y_train[0]
	y_sample = np.zeros(K, dtype = np.float32)
	y_sample[y_val] = 1
	graph = tf.Graph()
	with graph.as_default():
		beta = tf.placeholder(tf.float32, shape = (M+1,K))# beta is the optimization variable 
		Train_F = tf.placeholder(tf.float32, shape = (None,M+1)) # training samples
		Train_L = tf.placeholder(tf.float32, shape = (None,K)) # training labels
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(Train_F,beta),Train_L))
	
		grad = tf.gradients(cost, beta)
		Test_F = tf.constant(X_test,dtype = tf.float32)
		Test_L = tf.constant(Y_test, dtype = tf.float32)
		predict = tf.argmax(tf.matmul(Test_F, beta),1)	

	with tf.Session(graph = graph) as sess:
		sess.run(tf.initialize_all_variables())	
		beta_cur = np.zeros((M+1,K))
		for t in xrange(N_train):
			x_sample = X_train[t:t+1,:]
			y_val = Y_train[t]
			y_sample = np.zeros((1,K), dtype = np.float32)
			y_sample[0,y_val] = 1.0
			Dict = {Train_F:x_sample,beta:beta_cur, Train_L:y_sample}
			g = sess.run(grad, feed_dict = Dict)
			g = g[0]
			beta_cur = beta_cur - 1.0/sqrt(t+1.0) * g
			Dict = {Train_F:x_sample,beta:beta_cur, Train_L:y_sample}
			pre_cur = sess.run(predict, Dict)	
			if t % 100 == 0:
				print "error rate: ", np.mean(pre_cur != Y_test)

		#print beta_cur
		#	print type(g)
		#	print np.array(g,dtype = np.float32).shape, beta_cur.shape
		# linear fitting 


if __name__== "__main__":
	SGD()
