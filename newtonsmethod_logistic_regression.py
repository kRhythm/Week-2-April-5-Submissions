import numpy as np
def sigmoid(x1,x2,t1,t2,t0):
	z=(t0+t1*x1+t2*x2).astype("float_")
	return 1.0/(1.0+np.exp(-z))

def log_likelihood (x1,x2,y,t1,t2,t0):
	sigmoid_probs=sigmoid(x1,x2,t1,t2,t0)
	return np.sum(y*np.log(sigmoid_probs)+(1-y)*log(1-sigmoid_probs))

def gradient(x1,x2,y,t1,t2,t0):                                                         
    sigmoid_probs = sigmoid(x1,x2,t1,t2,t0)                                        
    return np.array([[np.sum((y - sigmoid_probs) * x1),np.sum((y - sigmoid_probs) * x2),                          
                     np.sum((y - sigmoid_probs) * 1)]])                         

def hessian(x1,x2,t1,t2,t0):                                                          
    sigmoid_probs = sigmoid(x1,x2,t1,t2,t0)                                        
    d1 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x1* x1)                  
    d2 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x1 * 1)
    d3 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * 1 * 1) 
    d4 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x1* x2)
    d5 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x2* x2)                  
    d6 = np.sum((sigmoid_probs * (1 - sigmoid_probs)) * x2 * 1)                                                     
    H = np.array([[d1, d4 ,d2],[d4, d5 ,d6],[d2, d6 ,d3]])                                           
    return H

def newtons_method(x1,x2,y):                                                                                                                              
  	t1=0
 	t2=0
 	t0=0
 	k=np.Infinity                                                                
    l = log_likelihood(x1,x2,y,t1,t2,t0)                                                                 
    # Convergence Conditions                                                        
    δ = .0000000001                                                                 
    max_iterations = 15                                                            
    i = 0                                                                           
    while abs(k) > δ and i < max_iterations:                                       
        i += 1                                                                      
        g = gradient(x1,x2,y,t1,t2,t0)                                                      
        hess = hessian(x1,x2,y,t1,t2,t0)                                                
        H_inv = np.linalg.inv(hess)                                                 
        # @ is syntactic sugar for np.dot(H_inv, g.T)¹
        Δ = H_inv @ g.T                                                             
        Δt1 = Δ[0][0]                                                              
        Δt2 = Δ[1][0]
        Δt0 = Δ[2][0]                                                                                                                             
        # Perform our update step                                                    
        t1 += Δt1                                                                 
        t2 += Δt2
        t0 += Δt0                                                                         
                                                                                    
        # Update the log-likelihood at each iteration                                     
        l_new = log_likelihood(x1,x2,y,t1,t2,t0)                                                      
        k = l - l_new                                                           
        l = l_new                                                                
    return np.array([t1,t2,t0])
