
# coding: utf-8

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
# 4. Make sure that you save your work (in the menubar, select _File_ → _Save and CheckPoint_)
# 
# 5. You are allowed to submit an assignment multiple times, but only the most recent submission will be graded.

# # Problem 7.2. Text Classification.
# 
# In this problem, we perform text classificatoin tasks by using the scikit learn machine learning libraries.

# In[1]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import scipy as sp
import re
import requests

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import check_random_state
from sklearn.cross_validation import StratifiedShuffleSplit


from nose.tools import (
    assert_equal,
    assert_is_instance,
    assert_almost_equal,
    assert_true
)
from numpy.testing import assert_array_equal


# We will be using a data set borrowed from [here](https://github.com/jacoxu/StackOverflow) that has been made available by Kaggle. It contains 20,000 instances of StackOverFlow post titles accompanied by labels in a separate file. For the purposes of this assignment, I have combined them in one file.
# Firstly, we load the contents of the file into a Pandas DataFrame.

# In[2]:

file_path = "/home/data_scientist/data/misc/combined_StackOverFlow.txt"
sof_df = pd.read_table(file_path, header=None, names=["Label","Text"])
sof_df.head()


# ## Splitting data set for training and testing
# 
# We shall be making use of [StratifiedShuffleSplit](http://scikit-learn.org/0.17/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html) to split the data set into train and test sets. 

# In[3]:

def train_test_split(X, y, test_size, random_state):
    '''
    Creates a training set and a test from the data set.
    
    Parameters
    ----------
    X: pd.core.series.Series object.
    y: pd.core.series.Series object.
    fileids: A list of strings.
    categories: A list of strings.
    
    Returns
    -------
    A 4-tuple (X_train, X_test, y_train, y_test)
    All four elements in the tuple are pd.core.series.Series.
    '''
    #Creates Stratified ShuffleSplit cross validation iterator
    sss=StratifiedShuffleSplit(y,test_size=test_size, random_state=random_state )
    #Creates X_train, X_test, y_train, y_test
    for train_index, test_index in sss:
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    return X_train, X_test, y_train, y_test
    


# In[4]:

X_train, X_test, y_train, y_test= train_test_split(sof_df['Text'], sof_df['Label'], 0.25, check_random_state(0))


# In[5]:

assert_is_instance(X_train, pd.core.series.Series)
assert_is_instance(X_test, pd.core.series.Series)
assert_is_instance(y_train, pd.core.series.Series)
assert_is_instance(y_test, pd.core.series.Series)

assert_true(all(isinstance(elem, str) for elem in X_train))
assert_true(all(isinstance(elem, str) for elem in X_test))
assert_true(all(isinstance(elem, str) for elem in y_train))
assert_true(all(isinstance(elem, str) for elem in y_test))

assert_equal(len(X_train), 15000)
assert_equal(len(X_test), 5000)
assert_equal(len(X_train), len(y_train))
assert_equal(len(X_test), len(y_test))

assert_equal(X_train[0][:20], 'How do I fill a Data')
assert_equal(y_train[0], 'linq')
assert_equal(X_test.iloc[0][:20], 'Can MacOS be run in ')
assert_equal(y_test.iloc[0], 'osx')

assert_equal(X_train[2][:20], 'Best Subversion clie')
assert_equal(y_train[2], 'svn')
assert_equal(X_test.iloc[2][:20], 'How to format a inpu')
assert_equal(y_test.iloc[2], 'matlab')


# ## Logistic Regression (no pipeline, no stop words)
# Use `CountVectorizer` to create a document term matrix for the titles, and apply the Logistic Regression algorithm to classify which label the title belongs to. Do not use pipeline (yet). Do not use stop words (yet). Use default parameters for both `CountVectorizer` and `LogisticRegression`.

# In[6]:

def cv_lr(X_train, y_train, X_test, random_state):
    '''
    Creates a document term matrix and uses LR classifier to make text classifications.
    
    Parameters
    ----------
    X_train: A pd.core.Core. Series object.
    y_train: A pd.core.Core. Series object.
    X_test: A pd.core.Core. Series object.
    random_state: A np.random.RandomState instance.
    
    Returns
    -------
    A tuple of (cv, lr, y_pred)
    cv: A CountVectorizer instance.
    lr: A LogisticRegression instance.
    y_pred: A numpy array.
    '''
    #Creates matrix of token counts and fits X_train to it
    cv = CountVectorizer()
    cv.fit(X_train)
    #Creats model to fit and predict
    data_to_train = cv.transform(X_train)
    data_to_test = cv.transform(X_test)
    lr = LogisticRegression(random_state=random_state)
    lr.fit(data_to_train, y_train)
    y_pred = lr.predict(data_to_test)
    return cv, lr, y_pred


# In[7]:

cv1, lr1, y_pred1 = cv_lr(X_train, y_train, X_test, random_state=check_random_state(0))
score1 = accuracy_score(y_pred1, y_test)
print("LR prediction accuracy = {0:3.1f}%".format(100.0 * score1))


# In[8]:

assert_is_instance(cv1, CountVectorizer)
assert_is_instance(lr1, LogisticRegression)
assert_is_instance(y_pred1, np.ndarray)
assert_equal(cv1.stop_words, None)
assert_equal(len(y_pred1), len(y_test))
assert_array_equal(y_pred1[:5], ['osx', 'ajax', 'matlab', 'qt', 'matlab'])
assert_array_equal(y_pred1[-5:], ['haskell', 'svn', 'drupal', 'cocoa', 'scala'])
assert_almost_equal(score1, 0.871)


# ## Logistic Regression (Pipeline, no stop words)
# 
# - Build a pipeline by using `CountVectorizer` and `LogisticRegression`. Name the first step `cv` and the second step `lr`. Do not use stop words (yet). Use default parameters for both `CountVectorizer` and `LogisticRegression`.

# In[9]:

def cv_lr_pipe(X_train, y_train, X_test, random_state):
    '''
    Creates a document term matrix and uses LR classifier to make text classifications.

    Parameters
    ----------
    X_train: A pd.core.Core. Series object.
    y_train: A pd.core.Core. Series object.
    X_test: A pd.core.Core. Series object.
    random_state: A np.random.RandomState instance.


    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A numpy array.
    '''
    #Creates a pipeline
    tools = [('cv', CountVectorizer()), ('lr', LogisticRegression(random_state=random_state))]
    clf = Pipeline(tools)

    #Fits and predicts to the model
    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)

    return clf,predicted


# In[10]:

clf2, y_pred2 = cv_lr_pipe(X_train, y_train, X_test, random_state=check_random_state(0))
score2 = accuracy_score(y_pred2, y_test)
print("LR prediction accuracy = {0:3.1f}%".format(100.0 * score2))


# In[11]:

assert_is_instance(clf2, Pipeline)
assert_is_instance(y_pred2, np.ndarray)
cv2 = clf2.named_steps['cv']
assert_is_instance(cv2, CountVectorizer)
assert_is_instance(clf2.named_steps['lr'], LogisticRegression)
assert_equal(cv2.stop_words, None)
assert_equal(len(y_pred2), len(y_test))
assert_array_equal(y_pred1, y_pred2)
assert_array_equal(y_pred1, y_pred2)
assert_almost_equal(score1, score2)


# ## Logistic Regression (Pipeline and stop words)
# 
# - Build a pipeline by using `CountVectorizer` and `LogisticRegression`. Name the first step `cv` and the second step `lr`. Use English stop words. Use default parameters for both `CountVectorizer` and `LogisticRegression`.

# In[12]:

def cv_lr_pipe_sw(X_train, y_train, X_test, random_state):
    '''
    Creates a document term matrix and uses LR classifier to make document classifications.
    Uses English stop words.
    
    Parameters
    ----------
    X_train: A pd.core.Core. Series object.
    y_train: A pd.core.Core. Series object.
    X_test: A pd.core.Core. Series object.
    random_state: A np.random.RandomState instance.
    
    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A numpy array.
    '''
    #Same as before but with stop words
    tools = [('cv', CountVectorizer()), ('lr',  LogisticRegression(random_state=random_state))]
    clf = Pipeline(tools)
    
    clf.set_params(cv__stop_words = 'english')

    
    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return clf, predicted


# In[13]:

clf3, y_pred3 = cv_lr_pipe_sw(X_train, y_train, X_test, random_state=check_random_state(0))
score3 = accuracy_score(y_pred3, y_test)
print("LR prediction accuracy = {0:3.1f}%".format(100.0 * score3))


# In[14]:

assert_is_instance(clf3, Pipeline)
assert_is_instance(y_pred3, np.ndarray)
cv3 = clf3.named_steps['cv']
assert_is_instance(cv3, CountVectorizer)
assert_is_instance(clf3.named_steps['lr'], LogisticRegression)
assert_equal(cv3.stop_words, 'english')
assert_equal(len(y_pred3), len(y_test))
assert_array_equal(y_pred3[:5], ['osx', 'ajax', 'matlab', 'cocoa', 'matlab'])
assert_array_equal(y_pred3[-5:], ['haskell', 'svn', 'drupal', 'cocoa', 'scala'])
assert_almost_equal(score3, 0.87239999999999995)


# ## Pipeline of TF-IDF and Logistic Regression with stop words
# 
# - Build a pipeline by using `TfidfVectorizer` and `LogisticsRegression`. Name the first step `tf` and the second step `lr`. Use English stop words. Use default parameters for both `TfidfVectorizer` and `LogisticsRegression`.

# In[21]:

def tfidf_lr(X_train, y_train, X_test, random_state):
    '''
    Creates a document term matrix and uses Logistic Regression classifier to make text classifications.
    Uses English stop words.
    
    Parameters
    ----------
    X_train: A pd.core.Core. Series object.
    y_train: A pd.core.Core. Series object.
    X_test: A pd.core.Core. Series object.
    random_state: A np.random.RandomState instance.
    
    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A numpy array.
    '''
    #Same as before but with TfidfVectorizer
    tools = [('tf', TfidfVectorizer()), ('lr',  LogisticRegression(random_state=random_state))]
    clf = Pipeline(tools)
    
    clf.set_params(tf__stop_words = 'english')

    
    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return clf, predicted


# In[22]:

clf4, y_pred4 = tfidf_lr(X_train, y_train, X_test, random_state=check_random_state(0))
score4 = accuracy_score(y_pred4, y_test)
print("LR prediction accuracy = {0:5.1f}%".format(100.0 * score4))


# In[23]:

assert_is_instance(clf4, Pipeline)
assert_is_instance(y_pred4, np.ndarray)
tf4 = clf4.named_steps['tf']
assert_is_instance(tf4, TfidfVectorizer)
assert_is_instance(clf4.named_steps['lr'], LogisticRegression)
assert_equal(tf4.stop_words, 'english')
assert_equal(len(y_pred4), len(y_test))
assert_array_equal(y_pred4[:5], ['osx', 'ajax', 'matlab', 'cocoa', 'matlab'])
assert_array_equal(y_pred4[-5:], ['haskell', 'svn', 'drupal', 'cocoa', 'scala'])
assert_almost_equal(score4, 0.872)


# In[ ]:



