import os
import tensorflow as tf
import numpy as np
from math import sqrt, pow
import plotly.plotly as py
import plotly.graph_objs as go

from subFunc import RandomGraph, RegularGraph

from GenerateToySamples import ToySamples

from six.moves import cPickle as pickle


def plot_sgd(x,y, xtype, ytype, name = "SGD"):
	#data = []
	data = [go.Scatter(x = x, y = y, name = name)]
	layout = go.Layout(xaxis = dict(type = xtype, autorange = True),yaxis = dict(type = ytype, autorange=True))
	fig = go.Figure(data = data, layout = layout)
	py.plot(fig,filename = name)	

def SGD(N_train = 500, Num_Nodes = 30,verbose = False):


	#G = RandomGraph(Num_Nodes, Num_Neighbors)
	G0 = RegularGraph(Num_Nodes, 5*Num_Nodes/2)
	G1 = RegularGraph(Num_Nodes, 15*Num_Nodes/2)
	with open('ToyTest.pickle', 'rb') as f:
		Data = pickle.load(f)
		X_test = Data['X']
		Y_test = Data['Y']

		f.close()	
	
	(N_test, temp) = np.shape(X_test)
	with open('ToyTrain.pickle', 'rb') as f:
		Data = pickle.load(f)
		X_train = Data['X']
		Y_train = Data['Y']
		f.close()
	graph = tf.Graph()
	Mile = 1
	S = 1
	(temp, NumSamples, M) = np.shape(X_train)
	M = M-1
	K = len(list(np.unique(Y_train[0])))
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
		#Beta_cur_All = np.zeros((Num_Nodes,M+1,K))
		#Beta_cur_All = np.random.randn(Num_Nodes,M+1,K)
		Beta_cur_All0 = np.zeros((Num_Nodes,M+1,K))
		Beta_cur_All_ave0 = Beta_cur_All0	
		Beta_cur_All1 = np.zeros((Num_Nodes,M+1,K))
		Beta_cur_All_ave1 = Beta_cur_All1
		X_step = []
		Dist_consensus = []
		Dist = []
		Error0 = []
		Error1 = []
		L = 1 # size of each batch
		k = 0
		while k <= N_train*Num_Nodes:
		#for k in xrange(N_train):
			#Select_One_Nodes
			node = np.random.randint(Num_Nodes)
			node_neighbors0 = []
			node_neighbors1 = []
			for nei, val in enumerate(G0[node]):
				if val > 0.5:
					node_neighbors0.append(nei)
			for nei, val in enumerate(G1[node]):
				if val > 0.5:
					node_neighbors1.append(nei)
			if np.random.rand() < 0.5:
				pos = np.random.randint(NumSamples) 
				x_sample = X_train[node][pos:pos+L]
				y_val = Y_train[node][pos:pos+L] 
				y_sample = np.zeros((L,K), dtype = np.float32)
				for l, y_val_l in enumerate(y_val):
					y_sample[l,int(y_val_l)] = 1.0
				
				#stepsize = 1.0/sqrt(k+1.0)
				#stepsize = min(1.0/(k/10+1.0),0.1)
				#stepsize = 0.001
				#stepsize = 1.0/sqrt(k+1.0)
				stepsize = 1.0/(pow((k+1.0), 0.6))
				beta_cur = Beta_cur_All0[node]
				Dict = {Train_F:x_sample, beta:beta_cur, Train_L:y_sample}
				g = sess.run(grad, feed_dict = Dict)
				g = g[0]
				beta_cur = beta_cur - stepsize * g * (1+len(node_neighbors0))
				Beta_cur_All0[node] = beta_cur
							
				beta_cur = Beta_cur_All1[node]
				Dict = {Train_F:x_sample, beta:beta_cur, Train_L:y_sample}
				g = sess.run(grad, feed_dict = Dict)
				g = g[0]
				beta_cur = beta_cur - stepsize * g * (1+len(node_neighbors1))
				Beta_cur_All1[node] = beta_cur
				k += 1
				Beta_cur_All_ave0 = (1-1/(k+1.0))*Beta_cur_All_ave0 + 1/(k+1.0)*Beta_cur_All0
				Beta_cur_All_ave1 = (1-1/(k+1.0))*Beta_cur_All_ave1 + 1/(k+1.0)*Beta_cur_All1
			else:

				# compute the average and make projections 
				def LocalAverage(Beta_cur_All, node, node_neighbors):
					sum_beta = Beta_cur_All[node]
					for nei in node_neighbors:
						sum_beta += Beta_cur_All[nei]
					ave_beta = sum_beta / (1+len(node_neighbors))
					Beta_cur_All[node] = ave_beta
					for nei in node_neighbors:
						Beta_cur_All[nei] = ave_beta
					return Beta_cur_All	
				Beta_cur_All0 = LocalAverage(Beta_cur_All0,node, node_neighbors0)	
				Beta_cur_All1 = LocalAverage(Beta_cur_All1,node, node_neighbors1)	
			if k >= Mile:
				Mile += S
				if Mile % (S*10) == 0:
					S = S*10
				beta_cur = np.mean(Beta_cur_All_ave0,axis = 0)
				Dict = {Test_F:X_test,beta:beta_cur}
				pre_cur = sess.run(predict, Dict)	
				ErrorSum = np.sum(pre_cur != Y_test)
				errorRate0 = ErrorSum/(N_test+0.0)
				
				beta_cur = np.mean(Beta_cur_All_ave1,axis = 0)
				Dict = {Test_F:X_test,beta:beta_cur}
				pre_cur = sess.run(predict, Dict)	
				ErrorSum = np.sum(pre_cur != Y_test)
				errorRate1 = ErrorSum/(N_test+0.0)
				Error0.append(errorRate0)
				Error1.append(errorRate1)
				print k, errorRate0, errorRate1
	if 0: 
		flag = str(Num_Nodes)+ "_" + str(Num_Neighbors) + ", "
		plot_sgd(X_step, Dist_consensus, 'log', 'linear', name = flag + "distance to global consensus")
	
		plot_sgd(X_step, Error, 'log', 'linear',name = flag + "error rate")
		plot_sgd(X_step, Dist,'log', 'linear', name = flag + "distance to optima")
	return Error0, Error1, X_step


if __name__== "__main__":

	N_Nodes = 30	
	N_train = 2000
	#ToySamples(Noise = 0.5, Num_Nodes = N_Nodes,M = 200, K =10)	

	Xstep, Error0, Error1 = SGD(N_train = N_train,Num_Nodes = N_Nodes,verbose = True)
	trace0 = go.Scatter(x = Xstep, y = Error0, name = '30_5')

	trace1 = go.Scatter(x = Xstep, y = Error1, name = '30_15')

	data = [trace0, trace1]	
	layout = go.Layout(xaxis = dict(type = 'log', autorange = True),yaxis = dict(type = 'log', autorange=True))
	fig = go.Figure(data = data, layout = layout)
	#py.plot(fig,filename = 'Different Num of Neighbors')	

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

	
	name = "Prediction Error vs Number of Machines"
	data = [go.Scatter(x = Num_Nodes, y = Error, name = name )]
	layout = go.Layout(xaxis = dict(autorange = True),yaxis = dict(autorange=True))
	fig = go.Figure(data = data, layout = layout)

	py.plot(fig,filename = name)	
	
	"""	

