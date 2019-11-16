
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

# ## Problem 14.2. Spark DataFrames
# 
# In this problem, we will use the Spark DataFrame to perform basic data processing tasks.

# In[1]:

import pyspark
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql.types import StructField, StructType, IntegerType, FloatType, StringType
import pandas as pd

from nose.tools import assert_equal, assert_is_instance
from pandas.util.testing import assert_frame_equal


# We run Spark in [local mode](http://spark.apache.org/docs/latest/programming-guide.html#local-vs-cluster-modes) from within our Docker container.

# In[2]:

sc = SparkContext('local[*]')


# We create a new RDD by reading in the data as a textfile. We use the ratings data from [MovieLens](http://grouplens.org/datasets/movielens/latest/).

# In[3]:

csv_path = '/home/data_scientist/data/ml-latest-small/ratings.csv'
text_file = sc.textFile(csv_path)


# - Write a function that creates a Spark DataFrame from `text_file`. For example, running
# 
# ```python
# >>> df = create_df(text_file)
# >>> df.show()
# ```
# 
# should give
# 
# ```
# +------+-------+------+----------+
# |userId|movieId|rating| timestamp|
# +------+-------+------+----------+
# |     1|     16|   4.0|1217897793|
# |     1|     24|   1.5|1217895807|
# |     1|     32|   4.0|1217896246|
# |     1|     47|   4.0|1217896556|
# |     1|     50|   4.0|1217896523|
# |     1|    110|   4.0|1217896150|
# |     1|    150|   3.0|1217895940|
# |     1|    161|   4.0|1217897864|
# |     1|    165|   3.0|1217897135|
# |     1|    204|   0.5|1217895786|
# |     1|    223|   4.0|1217897795|
# |     1|    256|   0.5|1217895764|
# |     1|    260|   4.5|1217895864|
# |     1|    261|   1.5|1217895750|
# |     1|    277|   0.5|1217895772|
# |     1|    296|   4.0|1217896125|
# |     1|    318|   4.0|1217895860|
# |     1|    349|   4.5|1217897058|
# |     1|    356|   3.0|1217896231|
# |     1|    377|   2.5|1217896373|
# +------+-------+------+----------+
# only showing top 20 rows
# ```

# In[4]:

def create_df(rdd):
    '''
    Parameters
    ----------
    rdd: A pyspark.rdd.RDD instance.
    
    Returns
    -------
    A pyspark.sql.dataframe.DataFrame instance.
    '''
    #Creates new RDD transforming text_file into RDD with columns of appropiate data types
    rdd = (rdd
           .map(lambda l: l.split(","))  
           .filter(lambda row: 'userId' not in row) 
           .map(lambda x: (int(x[0]), int(x[1]), float(x[2]), int(x[3]))) )
    
    # Create and initialize a new SQL Context
    sqlContext = SQLContext(sc)
    #Column labels
    schemaString = "userId movieId rating timestamp"
    #Label types
    fieldTypes = [IntegerType(), IntegerType(), FloatType(), IntegerType()]
    #Initialization for the schema
    f_data = [StructField(field_name, field_type, True) for field_name, field_type in zip(schemaString.split(), fieldTypes)]
    #Create schema
    schema =  StructType(f_data)
    #Create dataframe with the data and schema
    df = sqlContext.createDataFrame(rdd, schema)
    
    return df


# In[5]:

df = create_df(text_file)
df.show()


# In[6]:

assert_is_instance(df, pyspark.sql.dataframe.DataFrame)

# convert the Spark dataframe to Pandas dataframe
df_pd = pd.read_csv(csv_path)
assert_frame_equal(df.toPandas(), df_pd)


# - Select from the Spark DataFrame only the rows whose rating is greater than 3.
# - After filtering, return only two columns: `movieId` and `rating`.
# 
# ```python
# >>> favorable = filter_favorable_ratings(df)
# >>> favorable.show()
# ```
# 
# ```
# +-------+------+
# |movieId|rating|
# +-------+------+
# |     16|   4.0|
# |     32|   4.0|
# |     47|   4.0|
# |     50|   4.0|
# |    110|   4.0|
# |    161|   4.0|
# |    223|   4.0|
# |    260|   4.5|
# |    296|   4.0|
# |    318|   4.0|
# |    349|   4.5|
# |    457|   4.0|
# |    480|   3.5|
# |    527|   4.5|
# |    589|   3.5|
# |    590|   3.5|
# |    593|   5.0|
# |    608|   3.5|
# |    648|   3.5|
# |    724|   3.5|
# +-------+------+
# only showing top 20 rows
# ```

# In[7]:

def filter_favorable_ratings(df):
    '''
    Selects rows whose rating is greater than 3.
    
    Parameters
    ----------
    A pyspark.sql.dataframe.DataFrame instance.

    Returns
    -------
    A pyspark.sql.dataframe.DataFrame instance.

    '''
    #Filter only ratings >3
    df = df.filter(df['rating'] > 3).select(df['movieId'], df['rating'])
    
    return df


# In[8]:

favorable = filter_favorable_ratings(df)
favorable.show()


# In[9]:

assert_is_instance(favorable, pyspark.sql.dataframe.DataFrame)

favorable_pd = df_pd.loc[df_pd['rating'] > 3.0, ['movieId', 'rating']].reset_index(drop=True)
assert_frame_equal(favorable.toPandas(), favorable_pd)


# - Write a function that, given a `movieId`, computes the number of reviews for that movie.

# In[10]:

def find_n_reviews(df, movie_id):
    '''
    Finds the number of reviews for a movie.
    
    Parameters
    ----------
    movie_id: An int.
    
    Returns
    -------
    n_reviews: An int.
    '''
    #Counts reviews for a movue
    n_reviews=df.filter(df['movieId']==movie_id).count()
    
    return n_reviews


# In[11]:

n_toy_story = find_n_reviews(favorable, 1)
print(n_toy_story)


# In[12]:

assert_is_instance(n_toy_story, int)

test = [find_n_reviews(favorable, n) for n in range(1, 6)]
test_pd = favorable_pd.groupby('movieId').size()[:5].tolist()
assert_equal(test, test_pd)


# ## Cleanup
# 
# We must stop the SparkContext in order to release the spark resources before existing this Notebook.

# In[13]:

sc.stop()


# In[ ]:



