
# coding: utf-8

# # Week 14 Problem 3
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
# -----
# # Problem 14.3. Spark MLlib
# In this problem, we will use Spark MLlib to perform a logistic regression on the flight data to determine whether a flight would be delayed or not.

# In[1]:

import pyspark
from pyspark import SparkContext
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.classification import LogisticRegressionWithLBFGS

from nose.tools import (
    assert_equal, assert_is_instance,
    assert_true, assert_almost_equal
    )


# We run Spark in [local mode](http://spark.apache.org/docs/latest/programming-guide.html#local-vs-cluster-modes) from within our Docker container.

# In[2]:

sc = SparkContext('local[*]')


# We use code similar to the RDD code from the [Introduction to Spark](../notebooks/intro2spark.ipynb) notebook to import two columns: `ArrDealy` and `DepDelay`.

# In[3]:

text_file = sc.textFile('/home/data_scientist/data/2001/2001-12.csv')

data = (
    text_file
    .map(lambda line: line.split(","))
    # 14: ArrDelay, 15: DepDelay
    .map(lambda p: (p[14], p[15]))
    .filter(lambda line: 'ArrDelay' not in line)
    .filter(lambda line: 'NA' not in line)
    .map(lambda p: (int(p[0]), int(p[1])))
    )

len_data = data.count()
assert_equal(len_data, 462433)
assert_equal(
    data.take(5),
    [(27, 24), 
     (-18, -10), 
     (-8, -5), 
     (24, -3), 
     (8, -5)])


# ## Function: to_binary
# - Write a function that transforms the `ArrDelay` column into binary labels that indicate whether a flight arrived late or not. We define a flight to be delayed if its arrival delay is **15 minutes or more**, the same definition used by the FAA (source: [Wikipedia](https://en.wikipedia.org/wiki/Flight_cancellation_and_delay)).
# 
# - The `DepDelay` column should remain unchanged.

# In[4]:

def to_binary(rdd):
    '''
    Transforms the "ArrDelay" column into binary labels
    that indicate whether a flight arrived late (1) or not (0).
    
    Parameters
    ----------
    rdd: A pyspark.rdd.RDD instance.
    
    Returns
    -------
    A pyspark.rdd.PipelinedRDD instance.
    '''
    #Transforms ArrDelay column into binary labels on whether or not it arrived. Delayed if delay >15 minutes
    rdd = rdd.map(lambda delay: (int(delay[0] >= 15), delay[1]))
    
    return rdd


# In[5]:

binary_labels = to_binary(data)
print(binary_labels.take(5))


# In[6]:

assert_is_instance(binary_labels, pyspark.rdd.PipelinedRDD)
assert_equal(binary_labels.count(), len_data)
assert_equal(
    binary_labels.take(5),
    [(1, 24), 
     (0, -10), 
     (0, -5), 
     (1, -3), 
     (0, -5)])
assert_equal(to_binary(sc.parallelize([(15.0, 120.0)])).first(), (1, 120.0))
assert_equal(to_binary(sc.parallelize([(14.9, 450.0)])).first(), (0, 450.0))


# ## Function: to_labeled_point
# Our data must be in a Spark specific data structure called [LabeledPoint](https://spark.apache.org/docs/latest/mllib-data-types.html#labeled-point). So: 
# 
# - Write a function that turns a Spark sequence of tuples into a sequence containing LabeledPoint values for each row. 
# - The arrival delay should be the label, and the departure delay should be the feature.

# In[7]:

def to_labeled_point(rdd):
    '''
    Transforms a Spark sequence of tuples into
    a sequence containing LabeledPoint values for each row.
    
    The arrival delay is the label.
    The departure delay is the feature.
    
    Parameters
    ----------
    rdd: A pyspark.rdd.RDD instance.
    
    Returns
    -------
    A pyspark.rdd.PipelinedRDD instance.
    '''
    #Transforms data structure to labeledPoint
    rdd = rdd.map(lambda tuples: LabeledPoint(tuples[0], [tuples[1]]))
    
    return rdd


# In[8]:

labeled_point = to_labeled_point(binary_labels)
print(labeled_point.take(5))


# In[9]:

assert_is_instance(labeled_point, pyspark.rdd.PipelinedRDD)
assert_equal(labeled_point.count(), len_data)
assert_true(all(isinstance(p, LabeledPoint) for p in labeled_point.take(5)))
assert_equal([p.label for p in labeled_point.take(5)], [1.0, 0.0, 0.0, 1.0, 0.0])
assert_true(all(
    isinstance(p.features, pyspark.mllib.linalg.DenseVector)
    for p
    in labeled_point.take(5)
    ))
assert_equal(
    [p.label for p in labeled_point.take(5)],
    [1.0,
     0.0,
     0.0,
     1.0,
     0.0]
    )
assert_equal(
    [p.features.values.tolist() for p in labeled_point.take(5)],
    [[24.0],
     [-10.0],
     [-5.0],
     [-3.0],
     [-5.0]]
    )


# ## Function: fit_and_predict
# - Use [LogisticRegressionWithLBFGS](http://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#pyspark.mllib.classification.LogisticRegressionWithLBFGS) to train a [logistic regression](http://spark.apache.org/docs/latest/mllib-linear-methods.html#logistic-regression) model. 
# - Use 10 iterations. Use default parameters for all other parameters other than `iterations`.
# - Use the resulting logistic regression model to make predictions on the entire data, and return an RDD of (label, prediction) pairs.

# In[14]:

def fit_and_predict(rdd):
    '''
    Fits a logistic regression model.
    
    Parameters
    ----------
    rdd: A pyspark.rdd.RDD instance.
    
    Returns
    -------
    An RDD of (label, prediction) pairs.
    '''
    #Creates logistical regression model with 10 iterations that predicts on entire data
    model=LogisticRegressionWithLBFGS.train(rdd, iterations=10)
    #makes RDD with label and predictions
    rdd=rdd.map(lambda x: (x.label, float(model.predict(x.features))))
    
    return rdd


# In[15]:

labels_and_preds = fit_and_predict(labeled_point)
print(labels_and_preds.take(5))


# In[16]:

assert_is_instance(labels_and_preds, pyspark.rdd.PipelinedRDD)
assert_equal(labels_and_preds.count(), len_data)
assert_equal(
    labels_and_preds.take(5),
    [(1.0, 1.0),
     (0.0, 0.0),
     (0.0, 0.0),
     (1.0, 0.0),
     (0.0, 0.0)]
    )


# ## Function: get_accuracy
# - Write a function that computes the accuracy from a Spark sequence of (label, prediction) pairs.
# - Accuracy is defined as the total number of correctly classified instances out of the total number of instances.

# In[19]:

def get_accuracy(rdd):
    '''
    Computes accuracy.
    
    Parameters
    ----------
    rdd: A pyspark.rdd.RDD instance.
    
    Returns
    -------
    A float.
    '''
    #Total count
    total = rdd.count()
    #Number of correct predictions
    correct = rdd.filter(lambda pair: pair[0] == pair[1]).count()
    #Computes accuracy
    accuracy = correct/total
    
    return accuracy


# In[20]:

accuracy = get_accuracy(labels_and_preds)
print(accuracy)


# In[21]:

assert_is_instance(accuracy, float)
assert_almost_equal(get_accuracy(sc.parallelize([(0.0, 1.0), (1.0, 0.0)])), 0.0)
assert_almost_equal(get_accuracy(sc.parallelize([(0.0, 1.0), (0.0, 0.0)])), 0.5)
assert_almost_equal(get_accuracy(sc.parallelize([(0.0, 0.0), (1.0, 0.0)])), 0.5)
assert_almost_equal(get_accuracy(sc.parallelize([(0.0, 0.0), (1.0, 1.0)])), 1.0)
assert_almost_equal(get_accuracy(sc.parallelize([(1.0, 0.0), (0.0, 1.0), (0.0, 1.0)])), 0.0)
assert_almost_equal(get_accuracy(sc.parallelize([(1.0, 1.0), (0.0, 1.0), (0.0, 1.0)])), 1/3)
assert_almost_equal(get_accuracy(sc.parallelize([(1.0, 1.0), (0.0, 0.0), (0.0, 1.0)])), 2/3)
assert_almost_equal(get_accuracy(sc.parallelize([(1.0, 1.0), (0.0, 0.0), (1.0, 1.0)])), 1.0)
assert_almost_equal(accuracy, 0.7193388015128678)


# ## Cleanup
# 
# We must stop the SparkContext in order to release the spark resources before existing this Notebook.

# In[22]:

sc.stop()


# In[ ]:



