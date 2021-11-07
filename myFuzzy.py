'''
This library is for creating fuzzy sets with trapezoidal, triangular, &
gaussian
'''

import numpy as np

def TrapMf(interval, point):
	'''
	This function is to create a fuzzy set with a trapezoidal membership function.

	Parameter:
		Interval: the set of initial continuous values for which you want to find the degree of membership
		points: the points A,B,C,D needed to create a trapezoidal membership function
	
	Returns:
		The degree of membership of each point on the interval
	'''	
	A = point[0]
	B = point[1]
	C = point[2]
	D = point[3]
	
	membership_degree=[]
	
	def TMF(x):
		if (x<A):
			return 0
		elif (A<=x<B):
			return (x-A)/(B-A)
		elif (B<=x<=C):
			return 1
		elif (C<x<=D):
			return (D-x)/(D-C)
		else:
			return 0
			
	for x in interval:
		membership_degree.append(TMF(x))
		
	return membership_degree
	
	
def TriMf(interval, point):
	'''
	This function is to create a fuzzy set with a triangular membership function.
	
	Parameter:
		Interval: the set of initial continuous values for which you want to find the degree of membership
		point: the points A,B,C needed to create a triangular membership function
	
	Returns:
		The degree of membership of each point on the interval
	'''	
	A = point[0]
	B = point[1]
	C = point[2]    
    
	membership_degree=[]    
    
	def TrMF(x):
		if (x<A or x>C):
			return 0
		elif (A<=x<=B):
			return (x-A)/(B-A)
		elif (B<x<=C):
			return (C-x)/(C-B)        		
        
	for x in interval:
		membership_degree.append(TrMF(x))   

	return membership_degree

def L_Mf(interval, point): 
	'''
	This function is to make the fuzzy set open to the left.

	Parameter:
		Interval: the set of initial continuous values for which you want to find the degree of membership
		point: the points A,B needed to make the function open to the left
	
	Returns:
		The degree of membership of each point on the interval
	'''
	
	A = point[0]
	B = point[1]        
    
	membership_degree=[]    
    
	def LMF(x):
		if (x<=A):
			return 1
		elif (A<=x<=B):
			return (B-x)/(B-A)
		else:
			return 0        
        
	for x in interval:
		membership_degree.append(LMF(x))   

	return membership_degree

def R_Mf(interval, point): 
	'''
	This function is to make the fuzzy set open to the right.

	Parameter:
		Interval: the set of initial continuous values for which you want to find the degree of membership
		point: the points A,B needed to make the function open to the right
	
	Returns:
		The degree of membership of each point on the interval
	'''
	
	A = point[0]
	B = point[1]        
    
	membership_degree=[]    
    
	def RMF(x):
		if (x<A):
			return 0
		elif (A<=x<=B):
			return (x-A)/(B-A)
		else:
			return 1
        
	for x in interval:
		membership_degree.append(RMF(x))   

	return membership_degree


def GaussMf(interval, point):
	'''
	This function is to create a fuzzy set with a Gaussian membership function.

	Parameter:
		Interval: the set of initial continuous values for which you want to find the degree of membership
		point: the values of sigma and c needed to create a Gaussian membership function.

	Returns:
		The degree of membership of each point on the interval
	'''
	
	sigma = point[0]
	c = point[1]        
    
	membership_degree=[]
    
	def GaussMF(x):
		return np.exp(-(x-c)**2/2*sigma**2) 
        
	for x in interval:
		membership_degree.append(GaussMF(x))

	return membership_degree