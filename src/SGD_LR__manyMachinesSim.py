import os
import tensorflow as tf
import numpy as np
from math import sqrt, pow
import plotly.plotly as py
import plotly.graph_objs as go

from subFunc import RandomGraph, RegularGraph

def plot_sgd(x,y, name = "SGD"):
	#data = []
	data = [go.Scatter(x = x, y = y, name = name)]
	layout = go.Layout(xaxis = dict(type = 'log', autorange = True),yaxis = dict(autorange=True))
	fig = go.Figure(data = data, layout = layout)

	py.plot(fig,filename = name)	

def SGD(Num_Nodes = 9, Num_Neigh = 3,verbose = False):
	
	if Num_Neigh >= Num_Nodes:
		print "too many neighbours required!"
		return 
	#G = RandomGraph(Num_Nodes, Num_Neigh*Num_Nodes/2)
	G = RegularGraph(Num_Nodes, Num_Neigh*Num_Nodes/2)
	np.savetxt('adjacencyMatrix_'+str(Num_Nodes)+'_'+str(Num_Neigh)+'.txt', G, delimiter = ',', fmt = '%1.1d')

	Noise = 2
	M = 70
	K = 10
	Beta_real = np.random.randn(M+1,K)
	
	# test dataset
	N_test = 2000
	X_test = np.random.randn(Num_Nodes,N_test,M)
	Y_test = np.zeros((Num_Nodes,N_test))

	add_1s = np.ones((Num_Nodes,N_test,1))
	X_test = np.concatenate((X_test,add_1s),axis = 2)

	Beta_All = np.random.randn(Num_Nodes,M+1,K)
	for node in xrange(Num_Nodes):
		Beta_All[node] = Beta_real + Noise*node/(0.0+Num_Nodes)
	for node in xrange(Num_Nodes):
		Beta = Beta_All[node] + Noise*np.random.randn(M+1,K)
		temp = np.dot(X_test[node], Beta)
		temp = np.argmax(temp, axis = 1) 
		Y_test[node] = temp

	X_test_All = X_test.reshape(N_test*Num_Nodes, M+1)
	Y_test_All = Y_test.reshape(N_test*Num_Nodes)	
	# train dataset
	N_train = 2000
	
	graph = tf.Graph()
	Mile = 1
	S = 1

	with graph.as_default():
		beta = tf.placeholder(tf.float32, shape = (M+1,K))# beta is the optimization variable 
		Train_F = tf.placeholder(tf.float32, shape = (None,M+1)) # training samples
		Train_L = tf.placeholder(tf.float32, shape = (None,K)) # training labels
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(tf.matmul(Train_F,beta),Train_L))
	
		grad = tf.gradients(cost, beta)
		#Test_F = tf.constant(X_test,dtype = tf.float32)
		#Test_L = tf.constant(Y_test, dtype = tf.float32)
		Test_F = tf.placeholder(tf.float32, shape = (None,M+1)) # training samples
		predict = tf.argmax(tf.matmul(Test_F, beta),1)	

	with tf.Session(graph = graph) as sess:
		sess.run(tf.initialize_all_variables())	
	
		Beta_cur_All = np.zeros((Num_Nodes,M+1,K))
		Beta_cur_All_ave = Beta_cur_All # time average of Beta_cur_All
		#Beta_cur_All = np.random.randn(Num_Nodes,M+1,K)
	
		X_step = []
		Dist = []
		Dist_consensus = []
		Error = []
		L = 1 # size of each batch
		k = 0
		while k <= N_train*Num_Nodes:
		#for k in xrange(N_train):
			#Select_One_Nodes
			node = np.random.randint(Num_Nodes)
			node_neighbors = []
			for nei, val in enumerate(list(G[node])):
				if k == node:
					continue
				if val == 1:
					node_neighbors.append(nei)
			"""
			if Num_Nodes > 1:
				#node_neighbors = [(node+i)%Num_Nodes for i in xrange(Num_Nodes // 3)]
				for nei in xrange(Num_Neigh):
					node_neighbors.append((node+nei+1)%Num_Nodes)
				#node_neighbors = [(node+1)%Num_Nodes, (node-1)%Num_Nodes]
			"""
			if np.random.rand() < 0.5:
				beta_cur = Beta_cur_All[node]
				y_sample = np.zeros((L,K), dtype = np.float32)
				x_sample = np.random.randn(L,M)
				add_1s = np.ones((L,1))
				x_sample = np.concatenate((x_sample,add_1s),axis = 1)
				#Beta = Beta_real + Noise*np.random.randn(M+1,K)
				Beta = Beta_All[node] + Noise*np.random.randn(M+1,K)
				temp = np.dot(x_sample, Beta)  
				y_val = np.argmax(temp, axis = 1) 
				for l, y_val_l in enumerate(y_val):
					y_sample[l,y_val_l] = 1.0
				Dict = {Train_F:x_sample, beta:beta_cur, Train_L:y_sample}
				g = sess.run(grad, feed_dict = Dict)
				g = g[0]
				#stepsize = 1.0/sqrt(k+1.0)
				#stepsize = min(1.0/(k/10+1.0),0.1)
				#stepsize = 1e-7
				#stepsize = 1.0/sqrt(k+1.0)
				stepsize = 1.0/(pow((k+1.0)/Num_Nodes, 0.51))
				beta_cur = beta_cur - stepsize * g
				Beta_cur_All[node] = beta_cur
				#Beta_cur_All = (1.0-0.1)*Beta_cur_All + 0.1*Beta_cur_All0
				Beta_cur_All_ave = (1.0-1.0/(k+1))*Beta_cur_All_ave + 1.0/(k+1)*Beta_cur_All
				k += 1
			else:
				# compute the average
				sum_beta = Beta_cur_All[node]
				for nei in node_neighbors:
					sum_beta += Beta_cur_All[nei]
				ave_beta = sum_beta / (1.0+len(node_neighbors))
				Beta_cur_All[node] = ave_beta
				for nei in node_neighbors:
					Beta_cur_All[nei] = ave_beta
				
			if k >= Mile:
				# to evaluate the performance of current average
				Mile += S
				if Mile % (S*10) == 0:
					S = S*10
				ErrorSum = 0
				DistSum = 0
				count = 0.0
				beta_cur = np.mean(Beta_cur_All_ave,axis = 0)
				dist_consensus = 0
				for node in xrange(Num_Nodes):
					#Dict = {Test_F:X_test[node],beta:Beta_cur_All[node]}
					#pre_cur = sess.run(predict, Dict)	
					dist_consensus += np.linalg.norm(beta_cur - Beta_cur_All[node])
					Dict = {Test_F:X_test[node],beta:beta_cur}
					pre_cur = sess.run(predict, Dict)	
					ErrorSum += np.sum(pre_cur != Y_test[node])
					#print "#", node, " : ", np.mean(pre_cur != Y_test[node])
					count += len(Y_test[node])
				#print k, " error rate: ", errorRate, "distance to optima: ", dist2Optima
				errorRate = ErrorSum/count
				
				if verbose:
					print errorRate, N_test, count, dist_consensus	
				X_step.append(k)
				Error.append(errorRate)
				Dist_consensus.append(dist_consensus/Num_Nodes)
		"""
		plot_sgd(X_step, Error, name = "error rate")
		plot_sgd(X_step, Dist, name = "distance to optima")
		"""
	return Error[-1], Error, X_step
	#return Error[-1], Dist_consensus, X_step
		#print beta_cur
		#	print type(g)
		#	print np.array(g,dtype = np.float32).shape, beta_cur.shape
		# linear fitting 


if __name__== "__main__":

	

	error0, ErrorRate0, X_step = SGD(Num_Nodes = 10, Num_Neigh = 4,verbose = False)
	name = "10 Nodes, 4 neighbours"
	trace0 = go.Scatter(x = X_step, y = ErrorRate0, name = name)
	error2, ErrorRate2, X_step = SGD(Num_Nodes = 20, Num_Neigh = 4,verbose = False)
	name = "20 Nodes, 4 neighbours"
	trace2 = go.Scatter(x = X_step, y = ErrorRate2, name = name)
	error3, ErrorRate3, X_step = SGD(Num_Nodes = 30, Num_Neigh = 4,verbose = False)
	name = "30 Nodes, 4 neighbours"
	trace3 = go.Scatter(x = X_step, y = ErrorRate3, name = name)

	print error0,error2,error3

	data = [trace0, trace2, trace3]
	layout = go.Layout(title = 'ToyExample', xaxis = dict(type = 'linear',autorange = True),yaxis = dict(type = 'linear',autorange=True))
	fig = go.Figure(data = data, layout = layout)
	py.plot(fig,filename = "different number of neighbours, convergence speed")	

	"""
	k = 5
	Error = []
	Num_Nodes = []	
	while k <= 30:
		Num_Nodes.append(k)
		error, dist_consensus = SGD(Num_Nodes = k,verbose = False)
		print k, error, dist_consensus
		Error.append(error)
		k += 1		
	
	name = "toy example: Prediction Error vs Number of Machines"
	data = [go.Scatter(x = Num_Nodes, y = Error, name = name )]
	layout = go.Layout(xaxis = dict(autorange = True),yaxis = dict(autorange=True))
	fig = go.Figure(data = data, layout = layout)
	py.plot(fig,filename = name)	
	"""
