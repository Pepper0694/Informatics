
# coding: utf-8

# # Week 4 Problem 1
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

# # Decision Trees
# 
# In this problem, we will use the Decision Trees algorithm to see if we can use machine learning techniques to predict departure delays at the O'Hare airport (ORD).
# 
# A bit of introduction before we begin. You will see that this problem is not really about decision trees but data preparation. However, it is what practical data science is really about; the actual machine learning, especially with packages like scikit-learn, is a line of `fit` and `predict`. The rest is data wrangling.

# In[1]:

import numpy as np
import pandas as pd

import os
import time
import requests
import json
from pprint import pprint

from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn.utils import check_random_state
from sklearn.metrics import accuracy_score

from nose.tools import assert_equal, assert_is_not, assert_is_instance
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal
from pandas.util.testing import assert_frame_equal


# Suppose you want to include weather as training features. `2001.csv` doesn't have weather information, so we have to gather the data ourselves. There are various weather APIs available, one of which is [Weather Underground](http://www.wunderground.com/). Their terms of service says I have to display their logo, so here it is:
# 
# ![](http://www.wunderground.com/logos/images/wundergroundLogo_4c.jpg)
# 
# After you sign up for an account and generate an API token, you can issue HTTP requests such as:
# 
# ```
# http://api.wunderground.com/api/<token number>/history_20010101/conditions/q/KORD.json
# ```
# 
# The above example will return a JSON with historical weather information on January 1, 2001 (20010101) at O'Hare (KORD). To save you the trouble of dealing with the Weather Underground API, I saved the JSON responses as `.json` files in `/home/data_scientist/data/weather`.
# 
# ```shell
# $ ls /home/data_scientist/data/weather | head
# ```
# 
# ```
# weather_kord_2001_0101.json
# weather_kord_2001_0102.json
# weather_kord_2001_0103.json
# weather_kord_2001_0104.json
# weather_kord_2001_0105.json
# weather_kord_2001_0106.json
# weather_kord_2001_0107.json
# weather_kord_2001_0108.json
# weather_kord_2001_0109.json
# weather_kord_2001_0110.json
# ```

# In[2]:

get_ipython().system('ls /home/data_scientist/data/weather | head')


# Each file contains exactly the same response you would get from the Weather Underground API, because I simply dumped the JSON responses to files. Here is the full code that generated these files:
# 
# ```python
# def get_2001_json(date, year=2001):
#     url = 'http://api.wunderground.com/api/e693d410bdf457e2/history_{0}{1}/conditions/q/KORD.json'.format(year, date)
#     resp = requests.get(url)
#     resp_json = resp.json()
#     return resp_json
# 
# def save_2001_json(date, dir_name='data', filename='weather_kord_2001_{}.json'):
#     data = get_2001_json(date)
#     path = os.path.join(dir_name, filename.format(date))
#     with open(path, 'w') as f:
#         json.dump(data, f)
# 
# dates = ['{0:0>2}{1:0>2}'.format(m, d) for m in [1, 3, 5, 7, 8, 10, 12] for d in range(1, 32)]
# dates.extend(['{0:0>2}{1:0>2}'.format(m, d) for m in [4, 6, 9, 11] for d in range(1, 31)])
# dates.extend(['02{0:0>2}'.format(d) for d in range(1, 29)])
# 
# if not os.path.exists('data'):
#     os.mkdir('data')
# 
# for d in dates:
#     save_2001_json(d)
#     time.sleep(6) # free plan limit: 10 calls/min
# ```
# 
# Do not run this code to generate these files. We will use the files in `/home/data_scientist/data/weather` instead.
# 
# ## Load JSON files
# 
# - Write a function named `from_json_to_dict()` that takes a string in the format `MMDD` (M = Month, D = Day of month) and returns a dictoinary.

# In[4]:

def from_json_to_dict(date, path='/home/data_scientist/data/weather/', prefix='weather_kord_2001_'):
    '''
    Takes a string in the format MMDD where M = month, D = day of month.
    Read a json file at "path" + "prefix" + "date".
    Returns the JSON dictionary.
    
    Parameters
    ----------
    date: A string.
    
    Optional
    --------
    path: A string.
    prefix: A string.
    
    Returns
    -------
    A dict.
    '''
    with open(path+prefix+date+'.json') as file:    
        data = json.load(file)
    
    
    return data


# Tests for `from_json_to_dict()`:

# In[5]:

test_0101_dict = from_json_to_dict('0101')
assert_is_instance(test_0101_dict, dict)
assert_equal('history' in test_0101_dict, True)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
assert_equal('observations' in test_0101_dict['history'], True)
assert_is_instance(test_0101_dict['history']['observations'], list)

test_0103_dict = from_json_to_dict('0103')
assert_is_instance(test_0103_dict, dict)
assert_equal('history' in test_0103_dict, True)
assert_equal('observations' in test_0103_dict['history'], True)
assert_is_instance(test_0103_dict['history']['observations'], list)


# ## Parse time and visibility from JSON
# 
# - Write a function named `from_dict_to_visibility()` that takes a dictionary and returns a tuple of `(Month, Day, Hour, Minute, Visibility)`.
# 
# We covered the json format in the previous course, so you know how to do this. Let's say you created a dictionary called `data` by reading the json file `weather_kord_2001_0101.json`.
# 
# ```python
# >>> data = from_json_to_dict('0101')
# >>> print(data.keys()
# ```
# 
# ```
# dict_keys(['response', 'current_observation', 'history'])
# ```
# 
# You can peek into `response` and `current_observation` but they are not important for our purposes, so we look at `history`:
# 
# ```python
# >>> print(data['history'].keys())
# ```
# 
# ```
# dict_keys(['observations', 'date', 'dailysummary', 'utcdate'])
# ```
# 
# Here, `observations` is a list.
# 
# ```python
# >>> print(type(data['history']['observations']))
# ```
# 
# ```
# <class 'list'>
# ```
# 
# The first element looks like as follows:
# 
# ```python
# >>> from pprint import pprint
# >>> pprint(data['history']['observations'][0])
# ```
# 
# ```
# {'conds': 'Overcast',
#  'date': {'hour': '00',
#           'mday': '01',
#           'min': '56',
#           'mon': '01',
#           'pretty': '12:56 AM CST on January 01, 2001',
#           'tzname': 'America/Chicago',
#           'year': '2001'},
#  'dewpti': '10.9',
#  'dewptm': '-11.7',
#  'fog': '0',
#  'hail': '0',
#  'heatindexi': '-9999',
#  'heatindexm': '-9999',
#  'hum': '92',
#  'icon': 'cloudy',
#  'metar': 'METAR KORD 010656Z 36004KT 9SM BKN055 OVC095 M11/M12 A3034 RMK '
#           'AO2 SLP285 T11061117 $',
#  'precipi': '-9999.00',
#  'precipm': '-9999.00',
#  'pressurei': '30.38',
#  'pressurem': '1028.5',
#  'rain': '0',
#  'snow': '0',
#  'tempi': '12.9',
#  'tempm': '-10.6',
#  'thunder': '0',
#  'tornado': '0',
#  'utcdate': {'hour': '06',
#              'mday': '01',
#              'min': '56',
#              'mon': '01',
#              'pretty': '6:56 AM GMT on January 01, 2001',
#              'tzname': 'UTC',
#              'year': '2001'},
#  'visi': '9.0',
#  'vism': '14.5',
#  'wdird': '360',
#  'wdire': 'North',
#  'wgusti': '-9999.0',
#  'wgustm': '-9999.0',
#  'windchilli': '5.2',
#  'windchillm': '-14.9',
#  'wspdi': '4.6',
#  'wspdm': '7.4'}
# ```

# In[6]:

def from_dict_to_visibility(json_data):
    '''
    Takes a dictionary and returns a tuple of (Month, Day, Hour, Minute, Visibility).
    
    Parameters
    ----------
    json_data: A dict.
    
    Returns
    -------
    A 5-tuple (str, str, str, str, str)
    '''
    
    # Just defining where we are looking at the data
    data = json_data['history']['observations']
    # Define result as an empty list
    result = []
    # Run data through a for loop appending each tuple of data to result
    for i in range(0,len(data)):
        result.append(tuple((data[i]['date']['mon'], data[i]['date']['mday'], data[i]['date']['hour'], data[i]['date']['min'], data[i]['visi'])))
    
    return result


# Tests for `from_dict_to_visibility()`:

# In[7]:

test_0101_visi = from_dict_to_visibility(test_0101_dict)
assert_is_instance(test_0101_visi, list)
assert_equal(len(test_0101_visi), 24)
for item in test_0101_visi:
    assert_is_instance(item, tuple)
    assert_equal(len(item), 5) # month, day, hour, minute, visibility
    assert_equal(item[0], '01')
    assert_equal(item[1], '01')
    
test_0103_visi = from_dict_to_visibility(test_0103_dict)
assert_is_instance(test_0103_visi, list)
assert_equal(len(test_0103_visi), 34) # some days have more than one measurement per hour
for item in test_0103_visi:
    assert_is_instance(item, tuple)
    assert_equal(len(item), 5)
    assert_equal(item[0], '01')
    assert_equal(item[1], '03')


# ## Process all 365 files
# 
# We will use the functions `from_json_to_dict()` and `from_dict_to_visibility()` (in a loop) for all 365 days of the year. Let's first generate a list of dates in sequential order.

# In[8]:

dates = ['{0:0>2}{1:0>2}'.format(m, d + 1) for m in [1, 3, 5, 7, 8, 10, 12] for d in range(31)]
dates.extend(['{0:0>2}{1:0>2}'.format(m, d + 1) for m in [4, 6, 9, 11] for d in range(30)])
dates.extend(['02{0:0>2}'.format(d + 1) for d in range(28)])
dates.sort()

assert_equal(len(dates), 365)

print("The first five elements are {}".format(dates[:5]))
print("The last five elements are {}".format(dates[-5:]))


# - Write a function named `collect_365_days()` that takes a list of strings, iterates through the list, and uses `from_json_to_dict()` and `from_dict_to_visibility()` to return a list of 5-tuples `(month, day, hour, minute, visibility)`.
# 
# Here's the output you should get:
# 
# ```python
# >>> visibilities = collect_365_days(dates)
# >>> print("The length of visibilities is {}.".format(len(visibilities)))
# >>> print("The first five elements of visibilities are {}".format(visibilities[:5]))
# ```
# 
# ```
# The length of visibilities is 10159.
# The first five elements of visibilities are [('01', '01', '00', '56', '9.0'), ('01', '01', '01', '56', '7.0'), ('01', '01', '02', '56', '10.0'), ('01', '01', '03', '56', '10.0'), ('01', '01', '04', '56', '9.0')]
# ```

# In[9]:

def collect_365_days(dates):
    '''
    Uses from_json_to_dict() and from_dict_to_visiblility() to
    generate a list of tuples of the form
    (Month, Day, Hour, Minute, Visibility)
    
    Parameters
    ----------
    dates: A list of strings "MMDD"
    
    Returns
    -------
    A list of 5-tuples (str, str, str, str, str)
    '''
    
    # Initialize visibilities
    visibilities = []
    # Run through a for loop to get the whole years visibilities
    for date in dates:
        data = from_json_to_dict(date)
        vis = from_dict_to_visibility(data)
        visibilities += vis
        
    return visibilities


# In[10]:

visibilities = collect_365_days(dates)

print("The length of visibilities is {}.".format(len(visibilities)))
print("The first five elements of visibilities are {}".format(visibilities[:5]))


# In[11]:

assert_is_instance(visibilities, list)
assert_equal(len(visibilities), 10159)
assert_equal(visibilities[:5],
    [('01', '01', '00', '56', '9.0'),
     ('01', '01', '01', '56', '7.0'),
     ('01', '01', '02', '56', '10.0'),
     ('01', '01', '03', '56', '10.0'),
     ('01', '01', '04', '56', '9.0')]
    )
assert_equal(visibilities[-5:],
    [('12', '31', '19', '56', '10.0'),
     ('12', '31', '20', '56', '10.0'),
     ('12', '31', '21', '56', '10.0'),
     ('12', '31', '22', '56', '10.0'),
     ('12', '31', '23', '56', '10.0')]
    )


# Now we will combine the weather data with our flights data. We import the following columns of `2001.csv`:
# 
# - Column 1: Month, 1-12
# - Column 2: DayofMonth, 1-31
# - Column 5: CRSDepTime, scheduled departure time (local, hhmm)
# - Column 8: UniqueCarrier, unique carrier code
# - Column 15: DepDelay, departure delay, in minutes
# - Column 16: Origin, origin IATA airport code

# In[12]:

df = pd.read_csv(
    '/home/data_scientist/data/2001.csv',
    encoding='latin-1',
    usecols=(1, 2, 5, 8, 15, 16)
    )


# We use only AA flights that departed from ORD (American Airlines is the largest airline using the O'Hare airport). We define a flight to be delayed if its departure delay is 15 minutes or more, the same definition used by the FAA (source: [Wikipedia](https://en.wikipedia.org/wiki/Flight_cancellation_and_delay)).

# In[14]:

local = df[(df['Origin'] == 'ORD') & (df['UniqueCarrier'] == 'AA')]
local = local.drop(['UniqueCarrier', 'Origin'], axis=1) # we don't need the Month and Origin columns anymore.
local['Delayed'] = (local['DepDelay'] > 15).astype(np.int) # 1 if a flight was delayed, 0 if not.
local = local.drop('DepDelay', axis=1).dropna() # we don't need the DepDelay column.


# Let's print the first few columns and see what we'll be working with.
# 
# ```python
# >>> print(local.head(5))
# ```
# 
# ```
#       Month  DayofMonth  CRSDepTime  Delayed
#         Month  DayofMonth  CRSDepTime  Delayed
# 398444      1           1        1905        1
# 398445      1           2        1905        1
# 398446      1           3        1905        1
# 398447      1           4        1905        0
# 398448      1           5        1905        1
# ```

# In[15]:

print(local.head(5))


# ## Convert strings to numbers
# 
# Now we want to match the `Month` and `DayofMonth` columns in `local` with the corresponding entries in `visibilities` and find the time in `visibilities` that is closes to the `CRSDepTime`. What would be the best way to about matching the times?
# 
# Rahter than comparing three columns, I think it's better to combine the three numbers into one long number and compare just one column. Recall that we had a tuple of strings, while the data types in `local` is integer.
# 
# ```python
# >>> print(local.CRSDepTime.dtype)
# ```
# 
# ```
# int64
# ```
# 
# So let's convert the strings into integers in the form `mmddHHMM`, where `m` is month, `d` is day of month, `H` is hour, and `M` is minute. Let's create a data frame from tuple while we are at it so our function can do:
# 
# ```python
# >>> print(visibilities[:3])
# ```
# 
# ```
# [('01', '01', '00', '56', '9.0'), ('01', '01', '01', '56', '7.0'), ('01', '01', '02', '56', '10.0')]
# ```
# 
# ```python
# >>> time_visi = from_string_to_numbers(visibilities)
# >>> print(time_visi.head(3))
# ```
# 
# ```
#       Time  Visibility
# 0  1010056           9
# 1  1010156           7
# 2  1010256          10
# ```

# In[16]:

def from_string_to_numbers(visibilities):
    '''
    Takes a list of 5-tuples of strings.
    Convert the strings into integers in the form `mmddHHMM`,
    where `m` is month, `d` is day of month, `H` is hour, and `M` is minute.
    Returns a pandas.DataFrame with two columns "Time" and "Visibility".
    
    Parameters
    ----------
    visibilities: A list of 5-tuple of strings.
    
    Returns
    -------
    A pandas.DataFrame
    '''
    
    Time = []
    Visibility = []
    result = pd.DataFrame()
    # for loop converting everything to the proper format
    for i in range(len(visibilities)):
        temp = visibilities[i][0]+visibilities[i][1]+visibilities[i][2]+visibilities[i][3]
        Time.append(int(temp))
        temp2 = visibilities[i][4]
        Visibility.append(float(temp2))
    # Assignment of variables 
    result["Time"]=Time
    result['Visibility']=Visibility
    
    return result


# In[17]:

time_visi = from_string_to_numbers(visibilities)
print(time_visi.head(3))


# In[18]:

visi0 = [
    ('01', '01', '06', '00', '1.0'),
    ('02', '31', '08', '00', '2.0'),
    ('10', '05', '07', '00', '3.0'),
    ('12', '29', '09', '00', '4.0'),
    ('09', '30', '23', '00', '5.0'),
    ('07', '04', '12', '00', '6.0'),
    ('05', '12', '15', '00', '7.0'),
    ('11', '11', '18', '00', '8.0')
]

visi_answer = pd.DataFrame({
    'Time': [1010600, 2310800, 10050700, 12290900,
             9302300, 7041200, 5121500, 11111800],
    'Visibility': [1., 2., 3., 4., 5., 6., 7., 8.]
    })

assert_frame_equal(from_string_to_numbers(visi0), visi_answer)


# ## Create a Time column
# 
# - Do the same for the `local` data frame. Put the result into a column named `Time` so we have
# 
# ```python
# >>> time_delayed = combine_time(local)
# >>> print(time_delayed.head())
# ```
# 
# ```
#         Month  DayofMonth  CRSDepTime  Delayed     Time
# 398444      1           1        1905        1  1011905
# 398445      1           2        1905        1  1021905
# 398446      1           3        1905        1  1031905
# 398447      1           4        1905        0  1041905
# 398448      1           5        1905        1  1051905
# ```

# In[19]:

def combine_time(df):
    '''
    Combines "Month", "DayofMonth", and "CRSDepTime" in the form mmddHHMM.
    Creates a new column named "Time".
    
    Parameters
    ----------
    df: A pandas.DataFrame
    
    Returns
    -------
    A pandas.DataFrame
    '''
    # Create a copy of the dataframe
    result = df.copy()
    # Compact the df
    result['Time'] = result['Month'].astype(str) + result['DayofMonth'].map("{:02}".format).astype(str)+result['CRSDepTime'].map("{:04}".format).astype(str)
    # Convert int
    result=result.applymap(int)
    
    return result


# In[20]:

time_delayed = combine_time(local)
print(time_delayed.head())


# In[21]:

df0 = pd.DataFrame({
    'Month':      [  1,   2,  10,   12,   9,     7,    5,   11],
    'DayofMonth': [  1,  31,   5,   29,  30,     4,   12,   11],
    'CRSDepTime': [600, 800, 700,  900, 2300, 1200, 1500, 1800]
    })

df_answer = df0.join(pd.DataFrame({
    'Time': [1010600, 2310800, 10050700, 12290900, 9302300, 7041200, 5121500, 11111800]
    }))

assert_is_not(combine_time(df0), df0)
assert_frame_equal(combine_time(df0), df_answer)


# Now we find the time closest to the departure time. The following code cell will take a few minutes because we are using `iterrows()`, which is essentially a `for` loop. When you are doing numerical operations with big data in Python, you should avoid for loops as much as possible, and this is why. It's slow. Maybe there's a clever way to do this in a vectorized way, but I couldn't figure it out.
# 
# You don't have to write the `match_visibility()` function, but you should understand what it's doing.
# 
# ```python
# >>> local_visi = match_visibility(time_delayed, time_visi)
# >>> print(local_visi.head())
# ```
# 
# ```
#         Month  DayofMonth  CRSDepTime  Delayed     Time  Visibility
# 398444      1           1        1905        1  1011905          10
# 398445      1           2        1905        1  1021905           9
# 398446      1           3        1905        1  1031905           5
# 398447      1           4        1905        0  1041905           7
# 398448      1           5        1905        1  1051905          10
# ```

# In[22]:

def match_visibility(df_delayed, df_visibility, inplace=False):
    
    if not inplace:
        # we don't want to change the original data frame
        result = df_delayed.copy()
    
    for idx, row in result.iterrows():
        # find the row where the difference between the two times is minimum
        matched = (row['Time'] - df_visibility['Time']).idxmin()
        
        # used the index we found to extract the visibility and insert it into the result
        result.loc[idx, 'Visibility'] = df_visibility.loc[matched, 'Visibility']
        
    return result

local_visi = match_visibility(time_delayed, time_visi)

print(local_visi.head())


# Now we will split the data set into training and test sets. We will train on two columns, `CRSDepTime` and `Visibility`, so let's drop those columns.

# In[23]:

local_visi = local_visi.drop(['Month', 'DayofMonth', 'Time'], axis=1)
print(local_visi.head())


# ## Split
# 
# This function is the same function from [Problem 3.1](https://github.com/lcdm-uiuc/info490-sp17/blob/master/Week3/assignments/w3p1.ipynb). You can copy-paste your answer. I'll try not to make you write this again in the future.

# In[ ]:

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
    A 4-tuple of pandas.DataFrames
    '''

    #Create a new dataframe without the test column and then train it 
    newdf=df.drop(test_column, axis=1)
    X_train, X_test, y_train, y_test = train_test_split(newdf, df[test_column], 
                                                        test_size = test_size, random_state=random_state)

    
    return X_train, X_test, y_train, y_test


# We split `local_visi` into 80:20 training and test sets.

# In[ ]:

X_train, X_test, y_train, y_test = split(
    df=local_visi,
    test_column=['Delayed'],
    test_size=0.2,
    random_state=check_random_state(0)
    )


# In the following code cell, we test if the returned DataFrames have the correct columns and lengths.

# ## Train a Decision Trees model
# 
# - Write a function named `fit_and_predict()` that trains a **Decision Trees** model. Use default parameters. Don't forget that we have to pass an instance of check_random_state() to the train_test_split() function for reproducibility.

# In[24]:

def fit_and_predict(X_train, y_train, X_test, random_state):
    '''
    Fits Decision Trees.
    
    Parameters
    ----------
    X: A pandas.DataFrame. Training attributes.
    y: A pandas.DataFrame. Truth labels.
    
    Returns
    -------
    A numpy array.
    '''

    #fit and predict
    dtc = tree.DecisionTreeClassifier()
    dtc.fit(X_train, y_train)
    prediction = dtc.predict(X_test)


    return prediction


# In[25]:

y_pred = fit_and_predict(X_train, y_train, X_test, random_state=check_random_state(0))
accuracy = accuracy_score(y_test, y_pred)
print('The accuracy score is {:0.2f}.'.format(accuracy))


# In[ ]:

assert_is_instance(y_pred, np.ndarray)
assert_equal(len(y_pred), len(y_test))
assert_almost_equal(accuracy, 0.819190107608)

