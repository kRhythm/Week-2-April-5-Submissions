import numpy as np
def compute_cost_function (m,t0,t1,x,y):
	return 1/2/m*sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])
def gradient_descent(alpha,x,y,max_iter):
	converged=False
	iter=0
	m=x.shape[0]
	t0=0
	t1=0
	J=compute_cost_function(m,t0,t1,x,y)
	while not converged:
		grad0=1/m*sum[(t0+t1*np.asarray(x[i])-y[i]for i in range(m))]  
		grad1=1/m*sum([(t0 + t1*np.asarray([x[i]]) - y[i])*np.asarray([x[i]]) for i in range(m)])	
		temp0=t0-alpha*grad0
		temp1=t1-alpha*grad1
		t0=temp0
		t1=temp1
		e=compute_cost_function(m,t0,t1,x,y)
		J=e
		iter+=1
		if iter==max_iter:
			print('maximum iterations occured')
			converged=True
	return (t0,t1)
data=np.genfromtxt('http://cs229.stanford.edu/ps/ps1/logistic_x.txt',delimiter=' ')
x=data[:,:]
data=np.genfromtxt('http://cs229.stanford.edu/ps/ps1/logistic_y.txt',delimiter=' ')    
y=data[:,:]
gradient_descent(0.1,x,y,1500)
