
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

# # Problem 3.2. Supervised Learning: Support Vector Machine
# 
# In this problem, we will use Support Vector Machine to see if we can use machine learning techniques to predict departure delays at the O'Hare airport (ORD).

# In[1]:

get_ipython().magic('matplotlib inline')

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import svm, preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score

from nose.tools import assert_equal, assert_in, assert_is_not
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
from pandas.util.testing import assert_frame_equal, assert_index_equal

sns.set(style="white")


# We use the 2001 on-time airline performance data set. We import the following columns:
# 
# - Column 5: CRSDepTime, scheduled departure time (local, hhmm)
# - Column 8: UniqueCarrier, unique carrier code
# - Column 15: DepDelay, departure delay, in minutes
# - Column 16: Origin, origin IATA airport code
# - Column 18: Distance, in miles

# In[2]:

df = pd.read_csv(
    '/home/data_scientist/data/2001.csv',
    encoding='latin-1',
    usecols=(5, 8, 15, 16, 18)
    )


# In this problem, we use only `AA` flights (American Airlines, the largest airline using the O'Hare airport). Recall that we had an unbalanced data set by using `DepDelay > 15` in Problem 3.1. In this problem, we use `DepDelay > 0` to make our data set more balanced.

# In[3]:

local = df[(df['Origin'] == 'ORD') & (df['UniqueCarrier'] == 'AA')]
local = local.drop(['UniqueCarrier', 'Origin'], axis=1) # we don't need the Month and Origin columns anymore.
local['Delayed'] = (local['DepDelay'] > 0).astype(np.int) # 1 if a flight was delayed, 0 if not.
local = local.drop('DepDelay', axis=1).dropna() # we don't need the DepDelay column.


# Let's compare the number of delayed flights to non-delayed flights and check if we have a balanced data set.
# 
# ```python
# >>> print('Delayed: {}\nNot delayed: {}'.format(
#     (local.Delayed == 0).sum(),
#     (local.Delayed == 1).sum()
#     ))
# ```
# 
# ```
# Delayed: 61932
# Not delayed: 44006
# ```

# In[4]:

print('Delayed: {}\nNot delayed: {}'.format(
    (local.Delayed == 0).sum(),
    (local.Delayed == 1).sum()
    ))


# Let's print the first few columns and see what we'll be working with.
# 
# ```python
# >>> print(local.head(5))
# ```
# 
# ```
#         CRSDepTime  Distance  Delayed
# 398444        1905      1846        1
# 398445        1905      1846        1
# 398446        1905      1846        1
# 398447        1905      1846        0
# 398448        1905      1846        1
# ```

# In[5]:

print(local.head(5))


# ## Split
# 
# Write a function named `split()` that takes a DataFrame as its first argument. The second argument `test_column` specifies which column should be used as a label (`Delayed` in our case). All remaining columns except `test_column` should be used for training. In other words, the returned DataFrames `y_train` and `y_test` both have only one column, `test_column`, and `X_train` and `X_test` contain all columns in `df` minus `test_column`.
# 
# This function is (almost) the same function we wrote in Problem 3.1. You could simply cut and paste the answer. But cut-and-pasting is boring (and it's easier to work with Numpy arrays in this problem), so let's modify the function a little bit and **return Numpy arrays** this time. Pay close attention to the shape of `y_train` and `y_test` arrays: they should be **row vectors**, not column vectors.
# 
# ```python
# >>> print(local[['Delayed']].values[:5]) # column vector
# ```
# 
# ```
# [[0]
#  [0]
#  [0]
#  [1]
#  [0]]
# ```
# 
# ```python
# >>> print(local[['Delayed']].values.shape) # column vector
# ```
# 
# ```
# (341284, 1)
# ```
# 
# ```python
# >>> print(local_delayed_as_a_row_vector[:5])
# ```
# 
# ```
# [0 0 0 1 0]
# ```
# 
# ```python
# >>> print(local_deayed_as_a_row_vector.shape)
# (341284,)
# ```
# 
# Don't forget that we have to pass an instance of `check_random_state()` to the `train_test_split()` function for reproducibility.

# In[6]:

def split(df, test_column, test_size, random_state):
    '''
    Uses sklearn.train_test_split to split "df" into a testing set and a test set.
    The "test_columns" lists the column that we are trying to predict.
    All columns in "df" except "test_columns" will be used for training.
    The "test_size" should be between 0.0 and 1.0 and represents the proportion of the
    dataset to include in the test split.
    The "random_state" parameter is used in sklearn.train_test_split.
    
    Parameters
    ----------
    df: A pandas.DataFrame
    test_columns: A list of strings
    test_size: A float
    random_state: A numpy.random.RandomState instance
    
    Returns
    -------
    A 4-tuple of numpy.ndarrays
    '''
    #create copy of df without the test_column
    df2=df.drop(test_column, axis=1)
    #Creates testing and training sets from the dataframe with the proportionality of test_size
    X_train, X_test, y_train, y_test = train_test_split(df2, df[test_column], 
                                                        test_size=test_size, random_state=random_state)
    #Flattens y_train and converts it to an array
    y_train=np.array(y_train).flatten()
    #Flattens y_test and converts it to an array
    y_test=np.array(y_test).flatten()
    return X_train, X_test, y_train, y_test


# Now we will split `local` into a training set and a test set. We won't use a validation set this time, because training SVM is more computationally expensive than training a $k$-Nearest Neighbors. However, keep in mind that SVM also has hyperparameters that need to be tuned, e.g., `kernel`, `C`, or `gamma` values. In practice, you should create a validation set, or preferably perform a cross-validation.

# In[7]:

X_train, X_test, y_train, y_test = split(
    df=local,
    test_column=['Delayed'],
    test_size=0.4,
    random_state=check_random_state(0)
    )


# Tests.

# In[8]:

n_samples_train, n_features_train = X_train.shape
n_samples_test, n_features_test = X_test.shape

assert_equal(n_features_train, 2)
assert_equal(n_features_test, 2)
n_features = n_features_train

assert_equal(np.abs(n_samples_train - np.round(len(local) * 0.6)) <= 1, True)
assert_equal(np.abs(n_samples_test - np.round(len(local) * 0.4)) <= 1, True)

assert_array_equal(X_train[:5],
    np.array(
        [[ 1500.,  1846.],
         [ 1415.,   802.],
         [ 1138.,   409.],
         [ 1649.,   723.],
         [ 1835.,   678.]]
        ))
assert_array_equal(X_test[:5],
    np.array(
        [[  645.,  1745.],
         [  620.,   622.],
         [  645.,  1745.],
         [ 2040.,   678.],
         [  835.,   268.]]
        ))

assert_array_equal(y_train[:10], np.array([1, 0, 1, 0, 1, 1, 1, 1, 0, 1]))
assert_array_equal(y_test[:10], np.array([1, 1, 0, 0, 0, 0, 0, 1, 1, 1]))


# ## Scale
# 
# In Problem 3.1, we saw that the columns we want to use for training have different scales, so we scaled each column to the [0, 1] range. For SVM, we will scale features to be in [-1, 1] range.

# In[9]:

def standardize(df):
    '''
    Takes a dataframe and normlizes features to be in range [0, 1].
    
    Parameters
    ----------
    df: A pandas.DataFrame
    
    Returns
    -------
    A pandas.DataFrame
    '''
    #Standardized the df on the range [-1,1] by multiplying the previous normalized by 2 and subtracting 1
    scaled= 2*(df -df.min())/(df.max() - df.min()) -1
    return scaled


# In[10]:

df0 = pd.DataFrame({
    'a': [0, 1, 2, 3, 4],
    'b': [-50, -20, 10, 45, 50],
    'c': [-200, 450, 100, 500, -500]
    })
test1 = standardize(df0)
answer1 = pd.DataFrame({
        'a': [ -1., -0.5, 0., 0.5, 1.],
        'b': [ -1., -0.4, 0.2, 0.9, 1.],
        'c': [ -0.4, 0.9, 0.2, 1., -1.]
    })
assert_frame_equal(test1, answer1)


# In[11]:

X_train_scaled, X_test_scaled = map(standardize, [X_train, X_test])


# ## Train SVM
# 
# Now that we have standardized the training set, we are ready to apply the SVM algorithm.
# 
# Write a function named `fit_and_predict()`. It should return a tuple of `(svm.SVC, np.ndarray)`.

# In[12]:

def fit_and_predict(X_train, y_train, X_test, kernel):
    '''
    Fits a Support Vector Machine on the training data on "X_train" and "y_train".
    Returns the predicted values on "X_test".
    
    Parameters
    ----------
    X_train: A numpy.ndarray
    y_train: A numpy.ndarray
    X_test: A numpy.ndarray
    kernel: A string that specifies kernel to be used in SVM
    
    Returns
    -------
    model: An svm.SVC instance trained on "X_train" and "y_train"
    y_pred: A numpy array. Values predicted by "model" on "X_test"
    '''
    #Creates SVM model
    model= svm.SVC(kernel=kernel, C=1)
    #Fits the model
    model.fit(X_train, y_train)
    #Creates prediction for the model
    y_pred=model.predict(X_test)

    return model, y_pred


# Training SVM on `X_train` and `y_train` will take a while, so let's first make sure that the function works correctly.

# In[13]:

# toy sample
X_train_t = [[-2, -1], [-1, -1], [-1, -2], [1, 1], [1, 2], [2, 1]]
y_train_t = [1, 1, 1, 2, 2, 2]
X_test_t = [[-1, -1], [2, 2], [3, 2]]
y_test_t = [1, 2, 2]

model1, pred1 = fit_and_predict(X_train_t, y_train_t, X_test_t, 'linear')
assert_equal(isinstance(model1, svm.SVC), True)
assert_equal(model1.kernel, 'linear')
assert_array_equal(pred1, y_test_t)

model2, pred2 = fit_and_predict(X_train_t, y_train_t, X_test_t, 'rbf')
assert_equal(model2.kernel, 'rbf')
assert_array_equal(pred2, y_test_t)


# Run the following cell to actually train on the flights data. It might take a while.

# In[14]:

clf, y_pred = fit_and_predict(X_train_scaled, y_train, X_test_scaled, 'rbf')


# Let's see what the accuracy score is.

# In[15]:

print(accuracy_score(y_test, y_pred))


# It looks like using only two features `CRSDepTime` and `Distance` really hurt our performance (and predicting flight delays is a hard problem). To improve performance, we probably need to include more features, and not just the features provided in `2001.csv`, but data from other sources. For example, I would guess that weather has a significant impact on flights delays. One possibility is to find historical weather data (which will likely require using techniques you learned in the previous course) and include weather as a feature.
# 
# ## Confusion matrix
# 
# Plot a a colored heatmap that displays the relationship between predicted and actual types. The `plot_confusion()` function must return a `maplotlib.axes.Axes` object. Use `numpy.histogram2d()` and `seaborn.heatmap()` as demonstrated in lesson 1. Here's an exmaple:
# 
# ![](https://raw.githubusercontent.com/UI-DataScience/info490-sp16/master/Week3/assignments/images/svm_confusion.png)

# In[25]:

def plot_confusion():
    '''
    Plots a confusion matrix using numpy.histogram2d() and seaborn.heatmap().
    Returns a maptlotlib.axes.Axes instance.
    '''
    #Calls the previously defined dataframes
    a=y_test
    b=y_pred
    #Defines the location for the axes labels
    r=np.arange(0.5,2.5,1)
    #Histograms the data in a 2x2 bin. Only want the first element
    data=np.histogram2d(a,b, [2,2])[0]
    #Creates a heatmap with the parameters that were found with experimentation
    ax=sns.heatmap(data, vmin=4000, vmax=20000, annot=True, fmt='g')
    #Sets the tite and labels
    plt.title('Confusion matrix for SVM')
    plt.yticks(r,['Delayed','Not delayed'])
    plt.xticks(r,['Not delayed','Delayed'])
    return ax

    


# In[26]:

ax = plot_confusion()


# In[ ]:

assert_equal(isinstance(ax, mpl.axes.Axes), True, msg="Your function should return a matplotlib.axes.Axes object.")

texts = [t.get_text() for t in ax.texts]
assert_equal(texts, ['14971', '2712', '22450', '2243'])
             
x_tick_labels = [l.get_text() for l in ax.get_xticklabels()]
y_tick_labels = [l.get_text() for l in ax.get_yticklabels()]
assert_equal(y_tick_labels, ['Delayed', 'Not delayed'])

assert_is_not(len(ax.title.get_text()), 0, msg="Your plot doesn't have a title.")

