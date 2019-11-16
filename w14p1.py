
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
# 4. Make sure that you save your work (in the menubar, select _File_  → _Save and CheckPoint_)
# 
# 5. You are allowed to submit an assignment multiple times, but only the most recent submission will be graded.

# ## Problem 14.1. Spark
# 
# In this problem, we will perform basic data processing tasks within Spark using the concept of Resilient Distributed Datasets (RDDs).

# In[1]:

import pyspark
from pyspark import SparkConf, SparkContext

from nose.tools import assert_equal, assert_is_instance


# We run Spark in [local mode](http://spark.apache.org/docs/latest/programming-guide.html#local-vs-cluster-modes) from within our own server Docker container.

# In[2]:

sc = SparkContext('local[*]')


# We create a new RDD by reading in the data as a text file. We use the ratings data from [MovieLens](http://grouplens.org/datasets/movielens/latest/).

# In[3]:

text_file = sc.textFile('/home/data_scientist/data/ml-latest-small/ratings.csv')

assert_is_instance(text_file, pyspark.rdd.RDD)


# ## Part 1
# - Write a function that creates a new RDD by transforming `text_file` into an RDD with columns of appropriate data types.
# - The function accepts a `pyspark.rdd.RDD` instance (e.g., `text_file` in the above code cell) and returns another RDD instance, `pyspark.rdd.PipelinedRDD`.
# - `ratings.csv` contains a header row which should not be included in your output instance. Use the `head` command or otherwise to inspect the file.
# 

# In[4]:

def read_ratings_csv(rdd):
    '''
    Creates an RDD by transforming `ratings.csv`
    into columns with appropriate data types.
    
    Parameters
    ----------
    rdd: A pyspark.rdd.RDD instance.
    
    Returns
    -------
    A pyspark.rdd.PipelinedRDD instance.
    '''
    
    
        
    #Creates new RDD transforming text_file into RDD with columns of appropiate data types
    rdd = (rdd
           .map(lambda l: l.split(","))  
           .filter(lambda row: 'userId' not in row) 
           .map(lambda x: (int(x[0]), int(x[1]), float(x[2]), int(x[3]))) )
   
    
    return rdd


# In[5]:

ratings = read_ratings_csv(text_file)
print(ratings.take(3))


# In[6]:

assert_is_instance(ratings, pyspark.rdd.PipelinedRDD)
assert_equal(ratings.count(), 105339)
assert_equal(len(ratings.first()), 4)
assert_equal(
    ratings.take(5),
    [(1, 16, 4.0, 1217897793),
     (1, 24, 1.5, 1217895807),
     (1, 32, 4.0, 1217896246),
     (1, 47, 4.0, 1217896556),
     (1, 50, 4.0, 1217896523)]
    )


# ## Part 2
# For simplicity, we might want to restrict our analysis to only favorable ratings, which, since the movies are rated on a five-star system, we take to mean ratings greater than three. So
# 
# - Write a function that selects rows whose rating is greater than 3.

# In[7]:

def filter_favorable_ratings(rdd):
    '''
    Selects rows whose rating is greater than 3.
    
    Parameters
    ----------
    rdd: A pyspark.rdd.RDD instance.
    
    Returns
    -------
    A pyspark.rdd.PipelinedRDD instance.
    '''
    #Filter out ratings less than 3
    rdd = rdd.filter(lambda rating: rating[2] > 3.0)
    
    return rdd


# In[8]:

favorable = filter_favorable_ratings(ratings)


# In[9]:

assert_is_instance(favorable, pyspark.rdd.PipelinedRDD)
assert_equal(favorable.count(), 64160)


# ## Part 3
# We might also want to select only those movies that have been reviewed by multiple people.
# 
# - Write a function that returns the number of reviews for a given movie.

# In[10]:

def find_n_reviews(rdd, movie_id):
    '''
    Finds the number of reviews for a movie.
    
    Parameters
    ----------
    rdd: A pyspark.rdd.RDD instance.
    movie_id: An int.
    
    Returns
    -------
    An int.
    '''
    #Count the number of movies reviewed by multiple people
    n_reviews = rdd.filter(lambda reviews:reviews[1] == movie_id).count()
    
    return n_reviews


# In[11]:

n_toy_story = find_n_reviews(favorable, 1)
print(n_toy_story)


# In[12]:

assert_is_instance(n_toy_story, int)

test = [find_n_reviews(favorable, n) for n in range(5)]
assert_equal(test, [0, 172, 44, 18, 3])


# ## Cleanup
# 
# We must stop the SparkContext in order to release the spark resources before existing this Notebook.

# In[13]:

sc.stop()


# In[ ]:



