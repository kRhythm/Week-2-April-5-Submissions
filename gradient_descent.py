import numpy as np

def compute_cost_function (m,t0,t1,x,y):
	return 1/2/m*sum([(t0 + t1* np.asarray([x[i]]) - y[i])**2 for i in range(m)])
def gradient_descent(alpha,x,y,max_iter=1500)
	converged=False
	iter=0
	m=x.shape[0]
	t0=0
	t1=0
	J=compute_cost_function(m,t0,t1,x,y)
	while not converged:
		grad0=1/m*sum[(t0+t1*np.asarray(x[i])-y[i])] for in range(m) 
		grad1=1/m*sum[(t0+t1*np.asarray(x[i])-y[i])]*np.asarray(x[i]) for i in range(m)
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
return t0,t1
             
