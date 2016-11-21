import numpy as np


def RegularGraph(N,K):

	M = 2*K/N
	G = np.zeros((N,N))
	MM = M/2
	
	for i in xrange(N):
		for j in range(i+1, i + MM + 1):
			j = j%N
			G[i][j] = 1
			G[j][i] = 1
		
	if M % 2 == 1:
		for i in xrange(N/2):
			G[i][i+N/2] = 1
			G[i+N/2][i] = 1

	return G



def RandomGraph(N,K):
	
	# to radnomly generate a connected graph with N ndoes and K edges
	G = np.zeros((N,N))
	e = K-N+1

	G[0][0] = 0

 	for k in xrange(N):
		G[k][k] = 0
		if k >= 1:
			j = np.random.randint(k)
			G[k][j] = 1
			G[j][k] = 1

	while e >0:
		i = np.random.randint(N)
		if i == 0:
			continue
		j = np.random.randint(i)
		if G[i][j] == 0:
			G[i][j] = 1
			G[j][i] = 1
			e -= 1

	return G
