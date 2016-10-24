import os
import tensorflow as tf
import numpy as np
from math import sqrt

def SGD():

	M = 50
	K = 10
	N_test = 1000
	Beta_real = np.random.randn(M+1,K)
	Beta = Beta_real + 0.01*np.random.randn(M+1,K)
	
	# test dataset
	X_test = np.random.randn(N_test,M)
	add_1s = np.ones((N_test,1))
	X_test = np.concatenate((X_test,add_1s),axis = 1)
	temp = np.dot(X_test, Beta) 
	Y_test = np.argmax(temp, axis = 1) 
	
	# train dataset
	N_train = 5000
	X_train= np.random.randn(N_train,M)
	add_1s = np.ones((N_train,1))
	X_train = np.concatenate((X_train,add_1s),axis = 1)
	temp = np.dot(X_train, Beta)
	Y_train = np.argmax(temp, axis = 1) 
	
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
	
		errorRate = 0.5	

		L = 5 # size of each batch
		for k in xrange(N_train*10):
			t = k % N_train
			x_sample = X_train[t:t+L,:]
			y_val = Y_train[t:t+L]
			y_sample = np.zeros((L,K), dtype = np.float32)
			print y_val
			for l, y_val_l in enumerate(y_val):
				y_sample[l,y_val_l] = 1.0
			print y_sample
			Dict = {Train_F:x_sample,beta:beta_cur, Train_L:y_sample}
			g = sess.run(grad, feed_dict = Dict)
			g = g[0]
			#stepsize = 1.0/sqrt(k+1.0)
			#stepsize = min(1.0/(k/10+1.0),0.1)
			stepsize = 0.001
			#stepsize = 1.0/sqrt(k+1.0)
			#stepsize = 1.0/(k+1.0)
			beta_cur = beta_cur - stepsize * g
			if t % 100 == 0:
				Dict = {Train_F:x_sample,beta:beta_cur, Train_L:y_sample}
				pre_cur = sess.run(predict, Dict)	
				errorRate = np.mean(pre_cur != Y_test)
				print "error rate: ", errorRate, "distance to optima: ", np.linalg.norm(beta_cur - Beta_real)

		#print beta_cur
		#	print type(g)
		#	print np.array(g,dtype = np.float32).shape, beta_cur.shape
		# linear fitting 


if __name__== "__main__":
	SGD()
