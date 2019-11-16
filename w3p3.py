
# coding: utf-8

# # Week 2 Problem 3
# 
# If you are not using the `Assignments` tab on the course JupyterHub server to read this notebook, read [Activating the assignments tab](https://github.com/lcdm-uiuc/info490-sp17/blob/master/help/act_assign_tab.md).
# 
# A few things you should keep in mind when working on assignments:
# 
# 1. Make sure you fill in any place that says `YOUR CODE HERE`. Do **not** write your answer in anywhere else other than where it says `YOUR CODE HERE`. Anything you write anywhere else will be removed or overwritten by the autograder.
# 
# 2. Before you submit your assignment, make sure everything runs as expected. Go to menubar, select _Kernel_, and restart the kernel and run all cells (_Restart & Run all_).
# 
# 3. Do not change the title (i.e. file name) of this notebook.
# 
# 4. Make sure that you save your work (in the menubar, select _File_  → _Save and CheckPoint_)
# 
# 5. You are allowed to submit an assignment multiple times, but only the most recent submission will be graded.

# # Problem 3.3. Supervised Learning: Naive Bayes
# 
# In this problem, we will implement Naive Bayes from scratch using **Numpy**. You are free to use numpy functions or explicit math functions for this problem.

# In[4]:

import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal


# ## Separate Data by Class
# 
# Write a function that takes two Numpy arrays, separates data by class value, and returns the result as a dictionary. The keys of the dictionary are class values, and the dictionary values are the rows in `X` that correspond to each class value.

# In[5]:

def separate_by_class(X, y):
    '''
    Separate the training set ("X") by class value ("y")
    so that we can calculate statistics for each class.
    
    Parameters
    ----------
    X: A 2d numpy array
    y: A 1d numpy array
    Returns
    -------
    A dictionary of 2d numpy arrays
    '''
    #Separates data by class
    a=X[np.where(y==1)]
    b=X[np.where(y==0)]
    #Creates dictionary
    separated={0:b, 1:a}

    return separated


# In[6]:

X_t = np.array( [[2, 21], [1, 20], [3, 22]] )
y_t = np.array( [1, 0, 1] )
separated_t = separate_by_class(X_t, y_t)
assert_array_equal(separated_t[0], np.array( [ [1, 20] ] ))
assert_array_equal(separated_t[1], np.array( [ [2, 21], [3, 22] ] ))


# ## Calculate Mean
# 
# We calculate the mean and use it as the middle of our gaussian distribution when calculating probabilities. If the input array is a 2d array (i.e., a matrix), you should take the mean of each **column**.

# In[7]:

def calculate_mean(array):
    '''
    Calculates the mean of each column, i.e. each attribute.
    
    Parameters
    ----------
    A 1d or 2d numpy array
    
    Returns
    -------
    A 1d or 2d numpy array
    '''
    
    #This is the formula for calculating mean
    mean=np.mean(array,axis=0)
    
    return mean


# In[8]:

array_t = np.array( [ [1, 4, 7], [2, 5, 6], [3, 6, 8] ] )
mean_t = calculate_mean(array_t)
assert_array_equal(mean_t, np.array( [2., 5., 7.] ))


# ## Calculate Standard Deviation
# 
# Write a function that calculates the standard deviation of each **column** using the **N-1** method. The input array can be a 2d array.

# In[9]:

def calculate_stdev(array):
    '''
    Calculates the standard deviation (N-1 method) of each column, i.e. each attribute.

    Parameters
    ----------
    A 1d or 2d numpy array
    
    Returns
    -------
    A 1d or 2d numpy array
    '''
    #This is the formula for calculating std
    stdev=np.std(array, axis=0, ddof=1)
    
    return stdev


# In[10]:

array_t = np.array( [ [1, 20, 14], [2, 21, 15], [3, 22, 16] ] )
stdev_t = calculate_stdev(array_t)
assert_array_equal(stdev_t, np.array( [1., 1., 1.] ))


# ## Summarize Data Set
# 
# For a given list of instances (for a class value), we calculate the mean and the standard deviation for each attribute. The output is a numpy array of tuples of (mean, standard deviation) pairs for each attribute.

# In[11]:

def summarize(X):
    '''
    For a given list of instances (for a class value),
    calculates the mean and the standard deviation for each attribute.
    
    Parameters
    ----------
    A 2d numpy array
    
    Returns
    -------
    A 2d numpy array
    '''
    #Creates summary statistics for the attributes
    summary=[ (calculate_mean(X)[i],calculate_stdev(X)[i]) for i in range(len(calculate_mean(X)))]
    
    return summary


# In[12]:

X_t = np.array( [ [1, 20], [2, 21], [3, 22] ] )
summary_t = summarize(X_t)
assert_array_equal(summary_t, np.array( [ (2.0, 1.0), (21.0, 1.0) ] ))


# ## Summarize Attributes By Class
# 
# We calculate the summaries for each attribute.

# In[13]:

def summarize_by_class(X, y):
    '''
    Separates a training set into instances grouped by class.
    It then calculates the summaries for each attribute.
    
    Parameters
    ----------
    X: A 2d numpy array. Represents training attributes.
    y: A 1d numpy array. Represents class labels.
    Returns
    -------
    A dictionary of 2d numpy arrays. Uses each class label as keys
    and summary for each class label as values.
    '''
    #Calls on the previous function
    dic = separate_by_class(X,y)
    #Initialize the dicttionary
    summaries={}
    #loop through all items in the dictionary
    for classvalue, instances in dic.items():
        #Calculate the summary statistics based off of the class
        summaries[classvalue]=summarize(instances)
    

    
    return summaries


# In[14]:

X_t = np.array( [ [1, 20], [2, 21], [3, 22], [4, 22] ] )
y_t = np.array( [1, 0, 1, 0] )
summaries_t = summarize_by_class(X_t, y_t)
assert_array_almost_equal(summaries_t[0], np.array( [ (3., 1.41421356), (21.5, 0.70710678) ] ))
assert_array_almost_equal(summaries_t[1], np.array( [ (2., 1.41421356), (21.0, 1.41421356) ] ))


# ## Calculate Log of Gaussian Probability Density Function
# 
# Calculate the **log** of a Gaussian Probability Density Function. The conditional probabilities for each class given an attribute value are small. When they are multiplied together they result in very small values, which can lead to floating point underflow (numbers too small to represent in Python). A common fix for this is to combine the log of the probabilities together. If the input arguments are 1d arrays, the output should be a 1d array as well, and the n-th element in the output array is the log probability calculated using n-th elements of the input arrays.

# In[15]:

def calculate_log_probability(x, mean, stdev):
    '''
    Calculates log of Gaussian function to estimate
    the log probability of a given attribute value.
    Assume x, mean, stdev have the same length.
    
    Parameters
    ----------
    x: A float or 1d numpy array
    mean: A float or 1d numpy array
    stdev: A float or 1d numpy array
    
    Returns
    -------
    A float or 1d numpy array
    '''
    #The result from taking the log of the gaussian
    log_probability= np.log(1/(np.sqrt(2*np.pi)*stdev))-((x-mean)**2)/(2*(stdev)**2)
    

    return log_probability


# In[16]:

array_t = calculate_log_probability(np.array( [71.5] ), np.array( [73] ), np.array( [6.2] ))
assert_array_almost_equal(array_t, np.array( [ -2.7727542144336588 ] ))

array_t2 = calculate_log_probability(np.array( [1, 2] ), np.array( [3, 4] ), np.array( [5, 6] ))
assert_array_almost_equal(array_t2, np.array( [-2.60837645, -2.76625356] ))


# ## Calculate Class Probabilities
# 
# Remember that you calculated **log** of probabilities. Therefore, instead of combine probabilities together by multiplying them, you should **add** the log of probabilities. If the input array has more than one instance, for each instance you should have a summed log probability of attributes for each class value. 

# In[52]:

def calculate_class_log_probabilities(summaries, input_array):
    '''
    Combines the probabilities of all of the attribute values for a data instance
    and comes up with a probability of the entire data instance belonging to the class.

    Parameters
    ----------
    summaries: A dictionary of 2d numpy arrays
    input_array: A numpy array of instances; each instance is a numpy array of attributes
    
    Returns
    -------
    A dictionary of 1d numpy arrays of summed log probabilities
    '''
    
   
    #calculates the log probabilities based off of class. Loops through the keys and values of summaries. Note that you have to flatten 
    log_probabilities={key: calculate_log_probability(input_array, value[0,0], value[0,1]).flatten()
                       for key,value in summaries.items()}
    
    
    return log_probabilities


# In[53]:

summaries_t = {0: np.array( [ (1, 0.5) ]), 1: np.array( [ (20, 5.0) ] )}
input_t = np.array( [[1.1]] )
log_probabilities = calculate_class_log_probabilities(summaries_t, input_t)
assert_array_almost_equal(log_probabilities[0], np.array( [-0.24579135264472743] ))
assert_array_almost_equal(log_probabilities[1], np.array( [-9.6725764456387715] ))

input_t2 = np.array( [[4], [.9], [0]] )
log_probabilities2 = calculate_class_log_probabilities(summaries_t, input_t2)
assert_array_almost_equal(log_probabilities2[0], np.array( [-18.225791352644727, -0.24579135264472729, -2.2257913526447273] ))
assert_array_almost_equal(log_probabilities2[1], np.array( [-7.6483764456387728, -9.8245764456387743, -10.528376445638774] ))


# ## Make Predictions
# 
# Calculate the probability of a data instance belonging to each class value, and then we can look for the largest probability and return the associated class.

# In[76]:

def predict(summaries, input_array):
    '''
    Calculates the probability of each data instance belonging to each class value,
    looks for the largest probability, and return the associated class.
    
    Parameters
    ----------
    summaries: A dictionary of numpy arrays
    input_array: A numpy array of instances; each instance is a numpy array of attributes
    
    Returns
    -------
    A 1d numpy array
    '''
    probs=calculate_class_log_probabilities(summaries, input_array)
    hotdogs=np.argmax(probs)
    #It failed :(
    best_class=np.concatenate((probs, hotdogs), axis=1)
    
    

    return best_class


# In[77]:

summaries_t = {0: np.array( [ (1, 0.5) ] ), 1: np.array( [ (20, 5.0) ] )}
input_t1 = np.array( [[1.1]] )
result_t1 = predict(summaries_t, input_t1)
assert_array_equal(result_t1, np.array( [0.] ))

test_set_t2 = np.array( [[1.1], [19.1]] )
result_t2 = predict(summaries_t, test_set_t2)
assert_array_equal(result_t2, np.array( [0., 1.] ))

test_set_t3 = np.array( [[4], [.9], [0]] )
result_t3 = predict(summaries_t, test_set_t3)
assert_array_equal(result_t3, np.array( [1., 0., 0.] ))


# If you used Numpy correctly, it shouldn't be necessary to iterate over each test instance and make predictions. In other words, you don't have to use the `for` loop for `predict()`.
