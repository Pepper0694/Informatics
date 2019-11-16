
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

# ----
# 
# ## Problem 2.3. Logistic Regression
# 
# In this problem, we will fit a logistic regression model on day of the week and air carriers to predict whether a flight is delayed or not.

# In[1]:

get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
    
import statsmodels
import statsmodels.api as sm

from nose.tools import assert_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal
from pandas.util.testing import assert_frame_equal

sns.set(style="white", font_scale=2.0)


# We will use the columns `DayOfWeek` and `UniqueCarrier` as attributes and `DepDelay` as the target prediction. For simplicity, we will only use the flights that departed from O'Hare.

# In[2]:

filename = '/home/data_scientist/data/2001.csv'

usecols = (3, 8, 15, 17)
columns = ['DayOfWeek', 'UniqueCarrier', 'DepDelay', 'Origin']

all_data = pd.read_csv(filename, header=0, na_values=['NA'], usecols=usecols, names=columns).dropna()

local = all_data.loc[all_data['Origin'] == 'ORD'].dropna()


# Let's print print out the first few columns.
# 
# ```python
# >>> print(local.head())
# ```
# 
# ```
#       DayOfWeek UniqueCarrier  DepDelay Origin
# 1855          2            US        -1    ORD
# 1856          3            US        -4    ORD
# 1857          4            US        -3    ORD
# 1858          5            US        -3    ORD
# 1859          6            US        -4    ORD
# ```

# In[3]:

print(local.head())


# We will using logistic regression on the `DayOfWeek` and `UniqueCarrier` columns to predict whether a flight is delayed or not. However, logistic regression is for predicting **binary** outcomes and `DepDelay` is not binary. So our first task will be to convert this column into binary numbers.
# 
# ## Convert DepDelay to binary
# 
# - Write a function named `convert_to_binary()` that converts a specific column of a DataFrame into 0's or 1's using the `cutoff` parameter. See the function doctsring for more.
# - It actually does not matter whether 0's or 1's are integers or floats, but to pass assertion tests, make sure that your 0's and 1's are **integers** for all parts of this notebook unless otherwise required.

# In[4]:

def convert_to_binary(df, column, cutoff):
    '''
    Converts one column in Pandas.DataFrame "df" into binary
    as a new column, and returns the new DataFrame ("df" plus new binary column).
    Note that "df" should NOT be altered.
    
    The returned DataFrame has one more column than "df".
    The name of this column is in the form "column_binary".
    For example, if "column" is "DepDelay", the name of the extra column
    in the returned DataFrame is "DepDelay_binary".
    
    We assume that df[column] contains only ints or floats.
    If df[column] < cutoff, df[column_binary] is 0.
    If df[column] >= cutoff, df[column_binary] is 1.
    
    Parameters
    ----------
    df: A Pandas.DataFrame.
    column: A string.
    cutoff: An int.
    
    Returns
    -------
    A Pandas.DataFrame.
    '''
    #Creates a copy of df so it doesn't get altered
    result=df.copy()
    #Changes type of column to int
    result[column]=result[column].astype(int)
    #If entry is less than cutoff, then the entry in the new column is 0. Else, it is 1.
    result[column+'_binary']=(result[column]>=cutoff).astype(int)
    
    
    return result


# We will define a flight to be late if its departure delay is more than or equal to 5 minutes, and on-time if its departure delay is less than 5 minutes.
# 
# ```python
# >>> local = convert_to_binary(local, 'DepDelay', 5)
# >>> print(local.tail(10))
# ```
# 
# ```
#          DayOfWeek UniqueCarrier  DepDelay Origin  DepDelay_binary
# 5960735          6            DL         4    ORD                0
# 5960736          7            DL         7    ORD                1
# 5960737          1            DL        -2    ORD                0
# 5960738          2            DL        -3    ORD                0
# 5960739          3            DL         0    ORD                0
# 5960740          4            DL        58    ORD                1
# 5960741          5            DL         1    ORD                0
# 5960742          6            DL         0    ORD                0
# 5960743          7            DL        -8    ORD                0
# 5960744          1            DL        -3    ORD                0
# ```

# In[5]:

local = convert_to_binary(local, 'DepDelay', 5)
print(local.tail(10))


# Let's use some simple unit tests to see if the function works as expected.

# In[6]:

df0 = pd.DataFrame({
    'a': list(range(-5, 5)),
    'b': list(range(10))
    })

test1 = convert_to_binary(df0, 'a', 0)
answer1 = df0.join(pd.DataFrame({'a_binary': [0] * 5 + [1] * 5}))
assert_frame_equal(test1, answer1)

test2 = convert_to_binary(df0, 'b', 4)
answer2 = df0.join(pd.DataFrame({'b_binary': [0] * 4 + [1] * 6}))
assert_frame_equal(test2, answer2)


# ## Convert categorical variables to dummy indicator variables
# 
# `DayOfWeek` and `UniqueCarrier` are categorical variables, while we need binary indicator variables to perform logistic regression.
# 
# - Use [pandas.get_dummies()](http://pandas.pydata.org/pandas-docs/stable/generated/pandas.get_dummies.html) to write a function named `convert_to_dummy()` that transforms categorical variables into binary indicator variables.

# In[7]:

def convert_to_dummy(df, dummy_columns, keep_columns):
    '''
    Transforms categorical variables of dummy_columns into binary indicator variables.
    
    Parameters
    ----------
    df: A pandas.DataFrame
    dummy_columns: A list of strings. Columns of df that are converted to dummies.
    keep_columns: A list of strings. Columns of df that are kept in the result.
    
    
    Returns
    -------
    A pandas.DataFrame
    '''
    #Dummifies the dummy columns and converts its type to int
    result=pd.get_dummies(df[dummy_columns],columns=dummy_columns).astype(int)
    #Joins the keep columns from the old dataframe
    result=df[keep_columns].join(result)

    return result


# Now we should have only binary indicators in all columns.
# 
# ```python
# >>> data = add_dummy(local, add_columns=['DayOfWeek', 'UniqueCarrier'], keep_columns=['DepDelay_binary'])
# >>> print(data.head())
# ```
# 
# ```
#       DepDelay_binary  DayOfWeek_1  DayOfWeek_2  DayOfWeek_3  DayOfWeek_4  \
# 1855                0            0            1            0            0   
# 1856                0            0            0            1            0   
# 1857                0            0            0            0            1   
# 1858                0            0            0            0            0   
# 1859                0            0            0            0            0   
# 
#       DayOfWeek_5  DayOfWeek_6  DayOfWeek_7  UniqueCarrier_AA  \
# 1855            0            0            0                 0   
# 1856            0            0            0                 0   
# 1857            0            0            0                 0   
# 1858            1            0            0                 0   
# 1859            0            1            0                 0   
# 
#       UniqueCarrier_AS  UniqueCarrier_CO  UniqueCarrier_DL  UniqueCarrier_HP  \
# 1855                 0                 0                 0                 0   
# 1856                 0                 0                 0                 0   
# 1857                 0                 0                 0                 0   
# 1858                 0                 0                 0                 0   
# 1859                 0                 0                 0                 0   
# 
#       UniqueCarrier_MQ  UniqueCarrier_NW  UniqueCarrier_TW  UniqueCarrier_UA  \
# 1855                 0                 0                 0                 0   
# 1856                 0                 0                 0                 0   
# 1857                 0                 0                 0                 0   
# 1858                 0                 0                 0                 0   
# 1859                 0                 0                 0                 0   
# 
#       UniqueCarrier_US  
# 1855                 1  
# 1856                 1  
# 1857                 1  
# 1858                 1  
# 1859                 1  
# ```

# In[8]:

data = convert_to_dummy(local, dummy_columns=['DayOfWeek', 'UniqueCarrier'], keep_columns=['DepDelay_binary'])
print(data.head())


# In[9]:

df0 = pd.DataFrame({
    'a': ['a'] * 3,
    'b': [1] * 3,
    'c': [c for c in 'abc'],
    'd': list(range(3))
    })

test1 = convert_to_dummy(df0, dummy_columns=['c'], keep_columns=['a'])
answer1 = pd.DataFrame({
    'a': ['a'] * 3,
    'c_a': [1, 0, 0], 'c_b': [0, 1, 0], 'c_c': [0, 0, 1]
    })
assert_frame_equal(test1, answer1)

test2 = convert_to_dummy(df0, dummy_columns=['c', 'd'], keep_columns=['b'])
answer2 = pd.DataFrame({
    'b': [1] * 3,
    'c_a': [1, 0, 0], 'c_b': [0, 1, 0], 'c_c': [0, 0, 1],
    'd_0': [1, 0, 0], 'd_1': [0, 1, 0], 'd_2': [0, 0, 1]
    })

assert_frame_equal(test2, answer2)


# ## Add intercept
# 
# The [Logit()](http://statsmodels.sourceforge.net/0.6.0/generated/statsmodels.discrete.discrete_model.Logit.html) function doesn't include intercept by default and we have to manualy add the intercept.
# 
# - Write a function named `add_intercept()` that adds an extra column named `Intercept` with all 1's.

# In[10]:

def add_intercept(df):
    '''
    Appends to "df" an "Intercept" column whose values are all 1.0.
    Note that "df" should NOT be altered.
    
    Parameters
    ----------
    df: A pandas.DataFrame
    
    Returns
    -------
    A pandas.DataFrame
    '''
    #Creates a copy of df so it doesn't get altered
    result=df.copy()
    #Sets the intercept to 1 
    result['Intercept']=1

    return result


# Let's check if there is now an `Intercept` column.
# 
# ```python
# >>> data = add_intercept(data)
# >>> print(data['Intercept'].head())
# ```
# 
# ```
# 1855    1
# 1856    1
# 1857    1
# 1858    1
# 1859    1
# Name: Intercept, dtype: int64
# ```

# In[11]:

data = add_intercept(data)
print(data['Intercept'].head())


# In[12]:

df0 = pd.DataFrame({'a': [c for c in 'abcde']})

test1 = add_intercept(df0)
answer1 = df0.join(pd.DataFrame({'Intercept': [1] * 5}))

assert_frame_equal(test1, answer1)


# ## Function: fit\_logistic()
# 
# - Use statsmodels [Logit()](http://blog.yhat.com/posts/logistic-regression-and-python.html) to fit a logistic regression model to the columns in `train_columns`. Use (non-regularized) maximum likelihood with the default parameters (no optional parameters).

# In[13]:

def fit_logitistic(df, train_columns, test_column):
    '''
    Fits a logistic regression model on "train_columns" to predict "test_column".
    
    The function returns a tuple of (model ,result).
    "model" is an instance of Logit(). "result" is the result of Logit.fit() method.
    
    Parameters
    ----------
    train_columns: A list of strings
    test_column: A string
    
    Returns
    -------
    A tuple of (model, result)
    model: An object of type statsmodels.discrete.discrete_model.Logit
    result: An object of type statsmodels.discrete.discrete_model.BinaryResultsWrapper
    '''
    #Creates a logistical model 
    model=sm.Logit(df[test_column], df[train_columns])
    #Creates the fit for the model
    result=model.fit()

    return model, result


# Note that we exclude `DayOfWeek_1` and `UniqueCarrier_AA` from our fit to prevent [multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity#Remedies_for_multicollinearity).
# 
# ```python
# >>> model, result = fit_logitistic(data, train_columns=train_columns, test_column='DepDelay_binary')
# ```
# 
# ```
# Optimization terminated successfully.
#          Current function value: 0.589094
#          Iterations 5
# ```
# 
# ```python
# >>> print(result.summary())
# ```
# 
# ```
#                            Logit Regression Results                           
# ==============================================================================
# Dep. Variable:        DepDelay_binary   No. Observations:               321227
# Model:                          Logit   Df Residuals:                   321211
# Method:                           MLE   Df Model:                           15
# Date:                Thu, 21 Jan 2016   Pseudo R-squ.:                0.005735
# Time:                        22:05:38   Log-Likelihood:            -1.8923e+05
# converged:                       True   LL-Null:                   -1.9032e+05
#                                         LLR p-value:                     0.000
# ====================================================================================
#                        coef    std err          z      P>|z|      [95.0% Conf. Int.]
# ------------------------------------------------------------------------------------
# DayOfWeek_2         -0.1574      0.015    -10.479      0.000        -0.187    -0.128
# DayOfWeek_3          0.0164      0.015      1.113      0.266        -0.012     0.045
# DayOfWeek_4          0.2148      0.014     14.911      0.000         0.187     0.243
# DayOfWeek_5          0.2059      0.014     14.274      0.000         0.178     0.234
# DayOfWeek_6          0.0229      0.015      1.514      0.130        -0.007     0.053
# DayOfWeek_7          0.1085      0.015      7.397      0.000         0.080     0.137
# UniqueCarrier_AS    -0.3596      0.134     -2.679      0.007        -0.623    -0.096
# UniqueCarrier_CO    -0.0101      0.030     -0.339      0.735        -0.069     0.048
# UniqueCarrier_DL     0.5507      0.024     22.889      0.000         0.504     0.598
# UniqueCarrier_HP     0.8619      0.039     22.121      0.000         0.786     0.938
# UniqueCarrier_MQ     0.0906      0.012      7.502      0.000         0.067     0.114
# UniqueCarrier_NW     0.2597      0.025     10.572      0.000         0.212     0.308
# UniqueCarrier_TW     0.3749      0.036     10.343      0.000         0.304     0.446
# UniqueCarrier_UA     0.1901      0.010     19.987      0.000         0.172     0.209
# UniqueCarrier_US     0.2573      0.027      9.632      0.000         0.205     0.310
# Intercept           -1.1426      0.012    -94.960      0.000        -1.166    -1.119
# ====================================================================================
# ```

# In[14]:

train_columns = [ ### 'DayofWeek_1' # do not include this
        'DayOfWeek_2', 'DayOfWeek_3', 'DayOfWeek_4',
        'DayOfWeek_5', 'DayOfWeek_6', 'DayOfWeek_7',
        ### 'UniqueCarrierAA' # do not include this
        'UniqueCarrier_AS', 'UniqueCarrier_CO', 'UniqueCarrier_DL',
        'UniqueCarrier_HP', 'UniqueCarrier_MQ', 'UniqueCarrier_NW',
        'UniqueCarrier_TW', 'UniqueCarrier_UA', 'UniqueCarrier_US',
        'Intercept'
        ]

model, result = fit_logitistic(data, train_columns=train_columns, test_column='DepDelay_binary')


# In[15]:

print(result.summary())


# In[16]:

assert_equal(isinstance(model, statsmodels.discrete.discrete_model.Logit), True)
assert_equal(isinstance(result, statsmodels.discrete.discrete_model.BinaryResultsWrapper), True)

assert_equal(model.exog_names, train_columns)
assert_equal(model.endog_names, 'DepDelay_binary')

assert_array_equal(model.exog, data[train_columns].values)
assert_array_equal(model.endog, data['DepDelay_binary'].values)

test_conf_int = result.conf_int()
answer_conf_int = pd.DataFrame(
    index=train_columns,
    data={
        0: np.array([
            -0.18681953, -0.01247828,  0.18652782,  0.17760447, -0.00675086,
            0.07974488, -0.6227236 , -0.06873794,  0.50352299,  0.78551841,
            0.06694527,  0.21153022,  0.30383117,  0.17150234,  0.20497387,
            -1.166157  ]),
        1: np.array([
            -0.12794527,  0.04524193,  0.24298324,  0.23413964,  0.05254801,
            0.13724129, -0.09649653,  0.04848345,  0.59783265,  0.93824414,
            0.11429806,  0.30780938,  0.44591082,  0.20879553,  0.30969833,
            -1.11899193])
        }
    )
assert_frame_equal(test_conf_int, answer_conf_int)


# We see that the probability of flights being delayed is higher on Thursdays (`DayOfWeek_4`) and Fridays(`DayOfWeek_5`). In terms of carriers, `HP` an `MQ` airlines are more likely to be delayed than others.
# 
# Does this result make sense? Let calculate the mean of `DepDelay` for each day of the week. We see that Thursday and Friday have the highest mean values.
# 
# ```python
# >>> print(local.groupby('DayOfWeek').mean().sort_values(by='DepDelay', ascending=False))
# ```
# 
# ```
#             DepDelay  DepDelay_binary
# DayOfWeek                            
# 4          11.419251         0.311135
# 5          11.306297         0.309324
# 7          10.244282         0.288786
# ```

# In[17]:

print(local.groupby('DayOfWeek').mean().sort_values(by='DepDelay', ascending=False))


# We can do the same for `UniqueCarrier`, and HP and DL airline indeed have the highest mean departure delay.
# 
# ```python
# >>> print(local.groupby('UniqueCarrier').mean().sort_values(by='DepDelay', ascending=False))
# ```
# 
# ```
#                DayOfWeek   DepDelay  DepDelay_binary
# UniqueCarrier                                       
# HP              3.973684  18.245494         0.444845
# DL              3.972141  11.719453         0.370235
# UA              3.953615  11.027225         0.291036
# ```

# In[18]:

print(local.groupby('UniqueCarrier').mean().sort_values(by='DepDelay', ascending=False))


# In[ ]:



