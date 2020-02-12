###############################################################
#MATH 123 - Journal
#
#PDA.py (PrincipalDirectionAnalysis)
#
#Functions to calculate and plot Principal Direction Analysis
#
#Created by: Matthew Holsten
#		on: Feb 12 2020
###############################################################


""" LIBRARIES"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt



""" DEBUGGING HELPER FUNCTIONS """
#Function: caller
#Parameters: i = int depth of function caller
#Returns: function who called
#Does: returns the name of the function this function was
#		ran inside, with i being the depth
def caller(i = 1):
	import sys
	return sys._getframe(i).f_code.co_name



#Function: throwTypeError
#Parameters: param_name = name of parameter that caused error
#			 good_type = type param should hvae been
#			 bad_type = type param was
#Returns: None.
#Does: exits program with error message
def throwTypeError(param_name, good_type, bad_type):
	good_type = "<class '"+good_type+"'>"
	exit_str = "TypeError: [{}] must have [{}] param of type [{}]. \
Type [{}] was inputted.".format(caller(i=2), param_name, good_type, bad_type)
	exit(exit_str)



#Function: throwTypeWarning
#Parameters: param_name = name of parameter that caused error
#			 good_type = type param should hvae been
#			 bad_type = type param was
#Returns: None.
#Does: displays non-fatal warning message without exiting program
def throwTypeWarninig(param_name, good_type, bad_type):
	good_type = "<class '"+good_type+"'>"
	warning_str = "TypeWarning: [{}] expected [{}] param of type [{}]. \
Succesfully reconciled inputted [{}] to [{}].".format(caller(i=2), param_name, 
												good_type,bad_type, good_type)
	print(warning_str)



#Function: throwIndexError
#Parameters: param_name = name of parameter that caused error
#			 bad_val = value of param that caused error
#			 low = lower end of index range
#			 high = upper end of index range
#			 [low_bracket] = lower end of index range inclusivity bracket type
#							 default = '['
#			 [high_bracket] = upper end of index range inclusivity bracket type
#							 default = ']'
#Returns: None.
#Does: exits program with error message					
def throwIndexError(param_name, bad_val, low, high, low_bracket='[', 
													high_bracket=']'):
	exit_str = "IndexError: [{}] must have [{}] param of value within range \
{}{},{}{}. Index [{}] was inputted.".format(caller(i=2), param_name, 
											low_bracket, low, high, 
											high_bracket, bad_val)
	exit(exit_str)



#Function: throwError
#Parameters: error_msg = message to display
#Returns: None.
#Does: exits program with error message
def throwError(error_msg):
	exit_str = "Error (in {}): {}".format(caller(i=2),error_msg)
	exit(exit_str)

#Function: throwWarning
#Parameters: warning_msg = message to display
#Returns: None.
#Does: prints non-fatal error message without exiting program
def throwWarning(warning_msg):
	warning_str = "Warning (in {}): {}".format(caller(i=2), warning_msg)
	print(warning_str)


""" DEBUGGING HELPER FUNCTIONS """
#Function: toMatrix
#Parameters: data_list = list
#Returns: Instance of numpy.matrix
#Does: converts a list to a numpy.matrix
def toMatrix(data_list):
	try:
		return np.asmatrix(data_list)
	except:
		throwTypeError('data_list', 'list', type(data_list))
		


#Function: toList
#Parameters: mtrx = numpy.matrix instance
#Returns: Instance of list
#Does: converts a numpy.array/numpy.matrix to a list
def toList(mtrx):	
	try:
		return mtrx.tolist()
	except:
		throwTypeError('mtrx', 'numpy.matrix', type(mtrx))



#Function: centerData
#Parameters: mtrx = numpy.matrix instance
#Returns: numpy.matrix
#Does: converts a non-centered matrix of data to a centered matrix of data
def centerData(mtrx):
	
	#CHECKING INPUTS
	if not isinstance(mtrx, (np.matrix, np.generic)):
		try:
			input_type = type(mtrx)
			mtrx = toMatrix(mtrx)
			throwTypeWarninig('mtrx', 'numpy.matrix', input_type)
		except:
			throwTypeError('mtrx', 'numpy.matrix', type(mtrx))
	
	#LOGIC
	avg_mtrx = (1/mtrx.shape[0])*np.sum(mtrx, axis=0)
	cntrd_mtrx = np.zeros(mtrx.shape)
	
	for i in range(mtrx.shape[0]):
		for j in range(mtrx.shape[1]):
			cntrd_mtrx[i,j] = (mtrx[i,j] - avg_mtrx[0,j])
	
	return np.asmatrix(cntrd_mtrx)



#Function: isCentered
#Parameters: mtrx = numpy.matrix instance
#Returns: Boolean
#Does: returns the truth value of if the inputted matrix is centered or not
def isCentered(mtrx):
	
	#CHECKING INPUTS
	if not isinstance(mtrx, (np.matrix, np.generic)):
		try:
			input_type = type(mtrx)
			mtrx = toMatrix(mtrx)
			throwTypeWarninig('mtrx', 'numpy.matrix', input_type)
		except:
			throwTypeError('mtrx', 'numpy.matrix', type(mtrx))
	
	#LOGIC
	if np.allclose(mtrx, centerData(mtrx)):
		return True
	else:
		return False
		
		
		
#Function: randMatrix
#Parameters: m = number of columns in random matrix
#			 n = number of rows in random matrix
#			 [scalar] = value to scale all cells of matrix by, default = 1
#Returns: numpy.matrix of random numbers [0,1)
#Does: Produces a random mxn matrix
def randMatrix(m,n, scalar=1):
	return toMatrix(np.random.rand(m,n))*scalar
	
	
	
""" PDA FUNCTIONS """
#Function: varianceMatrix
#Parameters: mtrx =  NxN numpy.matrix instance
#Returns: NxN numpy.matrix instance: variance matrix
#Does: finds the variance matrix (Sigma) of inputted numpy.matrix
def varianceMatrix(mtrx):
	
	#CHECKING INPUTS
	if not isinstance(mtrx, (np.matrix, np.generic)):
		try:
			input_type = type(mtrx)
			mtrx = toMatrix(mtrx)
			throwTypeWarninig('mtrx', 'numpy.matrix', input_type)
		except:
			throwTypeError('mtrx', 'numpy.matrix', type(mtrx))
			
	if not isCentered(mtrx):
		throwWarning("Inputted data matrix is not centered.\
Resulting variance matrix may be inaccurate to expectations..")
		
	#LOGIC
	sigma = np.zeros([mtrx.shape[1],mtrx.shape[1]])

	for i in range(mtrx.shape[0]):
		sigma = np.add(sigma, np.multiply(mtrx[i],mtrx[i].T))
				
	sigma *= 1/mtrx.shape[0]
	return sigma



#Function: principalDirection
#Parameters: mtrx =  numpy.matrix instance
#			 [dir_degree] = the nth highest direction of most variance
#							default = 1				  
#Returns: 1xN numpy.matrix instance representing unit vector
#Does: finds unit vector in direction of most variance in a variance matrix
def principalDirection(mtrx, dir_degree=1):
	
	#CHECKING INPUTS
	if not isinstance(mtrx, (np.matrix, np.generic)):
		try:
			input_type = type(mtrx)
			mtrx = toMatrix(mtrx)
			throwTypeWarninig('mtrx', 'numpy.matrix', input_type)
		except:
			throwTypeError('mtrx', 'numpy.matrix', type(mtrx))
			
	if dir_degree < 1 or dir_degree > mtrx.shape[1]:
		throwIndexError('mtrx', dir_degree, 1, mtrx.shape[1])
		
	if not isCentered(mtrx):
		nc_str = 'Inputted data matrix not centered. '
		try:
			mtrx = centerData(mtrx)		
			throwWarning(nc_str+'Able to successfully autocenter data.')
		except:
			throwError(nc_str+'Unable to successfully autocenter data.')
	
	#LOGIC
	sigma = varianceMatrix(mtrx)
	e_vals, e_vects = LA.eig(sigma)
	e_dict 			= dict(zip(e_vals,e_vects))
	
	#getting nth highest e-vector
	for i in range(dir_degree-1): 
		del e_dict[max(e_dict)]
	
	return e_dict[max(e_dict)]


""" MATPLOTLIB VISUALIZATION """
#Function: plotVector_R2
#Parameters: dx = length of vector in x direction
#			 dy = length of vector in y direction
#			 [x] = starting x coordinate, default = 0
#			 [y] = starting y coordinate, defualt = 0
#			 [color] = color of vector, default = blue ('b')
#			 [alpha] = opacity of vector [0,1], default = 1
#Returns: None.
#Does: draws unit vector on current plot in R2
def plotVector_R2(dx,dy, x=0, y=0, color='b', alpha=1):
	ax = plt.gca()
	ax.annotate("", xy=(x+dx,y+dy), xytext=(x, y), 
				arrowprops=dict(arrowstyle="->", color=color, alpha=alpha))



#Function: plotLine_R2
#Parameters: x = x coordinates of two points in R2
#			 y = y coordinates of two points in R2
#			 [min_dim] = dimension which the line will draw to, default = 10000
#			 [color] = color of line, default = black
#			 [alpha] = opacity of line [0,1], default = 1
#Returns: None.
#Does: draws line on current plot R2
def plotLine_R2(x,y, min_dim=10000, color='black', alpha=1, label=''):
	try:
		slope = (y[1]-y[0])/(x[1]-x[0])
		x[0] -= min_dim
		y[0] -= min_dim*slope
		x[1] += min_dim
		y[1] += min_dim*slope
		
	#if vertical line, avoid DivideByZeroError
	except: 
		y[0] -= min_dim
		y[1] += min_dim
		
	ax = plt.gca()
	ax.plot(x, y, color=color, alpha=alpha, label=label)



#Function: plotAxes_R2
#Parameters: [min_dim] = dimension which the line will draw to, default = 10000
#			 [color] = color of line, default = black
#Returns: None.
#Does: draws in the x and y axes in R2
def plotAxes_R2(min_dim=10000,color='black'):
	plotLine_R2([-1,1], [0,0], min_dim=min_dim, color=color)
	plotLine_R2([0,0], [-1,1], min_dim=min_dim, color=color)


	
#Function: maxDim
#Parameters: data = list of data points
#Returns: value of max dimension
#Does: gets the maximum value of any single dimension of any data point in
#	   order to allow for the graph to contain all of the data points
def maxDim(data):
	
	#CHECKING INPUTS
	if not isinstance(data, list):
		try:
			input_type = type(data)
			data = data.tolist()
			throwTypeWarninig('data', 'list', input_type)
		except:
			throwTypeError('data', 'list', type(data))
	
	#LOGIC
	vals_list = []
	for i in range(toMatrix(data).shape[0]):
		for j in range(toMatrix(data).shape[1]):
			vals_list.append(abs(data[i][j]))
			
	return max(vals_list)



#Function: plotScatterPlot_R2
#Parameters: data = list of 2D data
#			 [title] = title of graph
#			 [x_label] = label on x-axis, default = blank
#			 [y_label] = label on x-axis, default = blank
#Returns: None.
#Does: draws graph of data on current plot
def plotScatterPlot_R2(data, title='', x_label='', y_label=''):
	
	#CHECKING INPUTS
	if not isinstance(data, list):
		try:
			input_type = type(data)
			data = data.tolist()
			throwTypeWarninig('data', 'list', input_type)
		except:
			throwTypeError('data', 'list', type(data))
	
	#LOGIC
	x, y = [], []
	for i in range(toMatrix(data).shape[0]):
		x.append(data[i][0])
		y.append(data[i][1])
	
	plt.scatter(x, y)
	plt.xlabel(x_label)
	plt.ylabel(y_label)
	plt.title(title)
	plt.grid(True)
	plotAxes_R2()
	max_dim = maxDim(data)
	xy_dim = max_dim + (5-max_dim%5)
	plt.xlim(-xy_dim,xy_dim)
	plt.ylim(-xy_dim, xy_dim)

	plt.legend()
	plt.show()



#Function: plotVectorLine_R2
#Parameters: unit_vect = unit vector
#			 [show_vect] = boolean to draw vector, defualt = True
#			 [show_line] = boolean to draw span of vector, defualt = True
#			 [min_dim] = size which the span of the vector will draw to,
#						 default = 10000
#			 [o] = origin of vector, default = [0,0]
#			 [vect_color] = color of vector, defualt = 'red'
#			 [line_color] = color of span of vector, defualt = 'red'
#			 [lbl] = label of span of vector, default = blank
#Returns: None.
#Does: draws unit vector and span vector on current plot
def plotVectorLine_R2(unit_vect, show_vect=True, show_line=True, o=[0,0], 
					min_dim=10000, vect_color='r', line_color='r', lbl=''):
	u = unit_vect.tolist()[0]
	if show_line:
		if show_vect:
			plotLine_R2([0, u[0]],[0, u[1]], color=line_color, alpha=0.5, 
						label = lbl)
		else:
			plotLine_R2([0, u[0]],[0, u[1]], color=line_color, label=lbl)
	if show_vect:
		plotVector_R2(u[0], u[1], color=vect_color)



#Function: displayPDA_R2
#Parameters: data = list of data points
#			 [title] = title of graph
#Returns: None.
#Does: Encorperates all other methods to display the centered data points,
#	   the first and second principal direction unit vector and span, and the
#	   value of the principal component vectors
def displayPDA_R2(data, title=''):
	
	#CHECKING INPUT
	if not isinstance(data, list):
		try:
			input_type = type(data)
			data = toList(data)
			throwTypeWarninig('data', 'list', input_type)
		except:
			throwTypeError('data', 'list', type(data))
					
	#LOGIC
	centered = centerData(toMatrix(data))
	
	pc1 = principalDirection(centered, dir_degree=1)
	pc2 = principalDirection(centered, dir_degree=2)
	
	plotVectorLine_R2(pc1,
		lbl='PC1: u='+str(principalDirection(toMatrix(centered))),
		vect_color= '#990000')
	plotVectorLine_R2(pc2, vect_color = '#006600', line_color = 'green', 
		lbl='PC2: u='+str(principalDirection(toMatrix(centered), dir_degree=2)))
	
	
	plotScatterPlot_R2(toList(centered), title=title)