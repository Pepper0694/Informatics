
# coding: utf-8

# # Week 7 Problem 3
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
# # Problem 7.3. Text Mining
# In this problem, we use the Brown corpus to perform text mining tasks, such as n-grams, stemming, and clustering.

# In[42]:

import numpy as np
import pandas as pd
import string
import collections

from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans

import nltk
from nltk.corpus import brown
from nltk.stem.snowball import EnglishStemmer

from nose.tools import (
    assert_equal,
    assert_is_instance,
    assert_almost_equal,
    assert_true
)
from numpy.testing import assert_array_equal


# We will analyze the NLTK Brown corpus. The Brown Corpus was the first million-word electronic corpus of English, created in 1961 at Brown University. This corpus contains text from 500 sources, and the sources have been categorized by genre, such as news, editorial, and so on. See the [NLTK docs](http://www.nltk.org/book/ch02.html#brown-corpus) for more information.
# 
# ```python
# >>> print( len( brown.fileids() ) ) # total number of files
# 500
# 
# >>> print( brown.categories() ) # all genres in corpus
# ['adventure', 'belles_lettres', 'editorial', 'fiction', 'government', 'hobbies', 'humor', 'learned', 'lore', 'mystery', 'news', 'religion', 'reviews', 'romance', 'science_fiction']
# ```
# 
# In the Brown corpus, there are 500 files of 15 genres. We could access the raw data and the genre of each file. What we are going to do is make predictions of a file's genre based on its raw data. Since the sample size is not very large, especially for each genre, there is only about 30 files on average. Therefore, we would like to remove genres with few files first.
# 
# ## Data Preprocessing
# 
# In the following cell, write a function named `select_genres()` that returns raw data (using `raw()`) and genres of files (using `categories()`) whose genres appear more than `n` times in the corpus, where `n` is some integer.  For example, if `n`=70, then you first need to find the genres that have more than 70 files; in this case, they are `learned` or `belles_lettres`; the output should be a tuple of two 1d numpy arrays: raw data of files with selected genres, genres of files with selected genres. Each element in the two output arrays should correspond to the same file. 

# In[43]:

def select_genres(n):
    '''
    Selects genres with more than n files. Returns raw data and the genre of each file
    in the selected genres as two 1d numpy arrays.
    
    Parameters
    ----------
    n: An integer.
    
    Returns
    -------
    A tuple of (raw, genres)
    raw: A 1d numpy array.
    genres: A 1d numpy array.
    '''
    genres=[]
    raw=[]
    #Creates arrays of the genres and raw data for genres with more than n files
    for file in brown.fileids():
        
        for k in brown.categories(file):
            
            if len(brown.fileids(k))>n:
                genres.append(k)
                raw.append(brown.raw(file))
      
    
    return raw, genres


# In[44]:

t1_raw, t1_genres = select_genres(70)
assert_equal(np.shape(t1_raw), (155,))
assert_equal(np.shape(t1_genres), (155,))
assert_array_equal(t1_genres, ['belles_lettres']*75+['learned']*80)
assert_equal(t1_raw[5][:50], 'Die/fw-at Frist/fw-nn ist/fw-bez um/fw-rb ,/, und/')
assert_equal(t1_raw[120][120:160], 'agricultural/jj areas/nns in/in the/at w')

t2_raw, t2_genres = select_genres(29)
assert_equal(np.shape(t2_raw), (313,))
assert_equal(np.shape(t2_genres), (313,))
assert_array_equal(t2_genres, ['news']*44+['hobbies']*36+['lore']*48+['belles_lettres']*75+['government']*30+['learned']*80)
assert_equal(t2_raw[300][-80:], " is/bez not/* generally/rb used/vbn over-hand/rb ,/, but/cc under/rb ''/'' ./.\n\n")
assert_equal(t2_raw[249][490:530], 's from/in the/at cortex/nn to/in the/at ')


# ## Training and Testing Sets
# 
# Run the cell below to split selected data (We'll use `n`=27) into training and testing sets with a test size of 0.3.

# In[45]:

t_raw, t_genres = select_genres(27)
t_X_train, t_X_test, t_y_train, t_y_test = train_test_split(t_raw, 
                                                            t_genres, 
                                                            random_state=check_random_state(0), 
                                                            test_size=0.3)


# ## n-grams
# 
# - Use unigrams, bigrams, and trigrams,
# - Build a pipeline by using `TfidfVectorizer` and `KNeighborsClassifier`,
# - Name the first step `tf` and the second step `knc`,
# - Use English stop words,
# - Convert all words into lower case so that the model does not depend on cases,
# - Impose a minimum feature term that requires a term to be present in at least 3 documents, 
# - Set a maximum frequency of 70%, such that any term occurring in more than 70% of all documents will be ignored, and
# - Set the rest parameters to default for both `TfidfVectorizer` and `KNeighborsClassifier`.

# In[46]:

def ngram(X_train, y_train, X_test):
    '''
    Creates a document term matrix and uses KNC classifier to make document classifications.
    Uses unigrams, bigrams, and trigrams.
    
    Parameters
    ----------
    X_train: A 1d numpy array of strings.
    y_train: A 1d numpy array of strings.
    X_test: A 1d numpy array of strings.
    
    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A 1d numpy array.
    '''
    #Creates pipeline with the parameters defined in the problems
    clf = Pipeline([('tf',TfidfVectorizer()),('knc',KNeighborsClassifier())])
    clf.set_params(tf__stop_words = 'english',
                   tf__ngram_range=(1,3),
                   tf__max_df = 0.7,
                   tf__min_df = 3,
                   tf__lowercase=True,
                   )
    #Fits and predicts
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test)
    
    
    return clf, y_pred


# In[47]:

clf1, y_pred1 = ngram(t_X_train, t_y_train, t_X_test)
score1 = accuracy_score(y_pred1, t_y_test)
print("KNC prediction accuracy = {0:5.1f}%".format(100.0 * score1))


# In[48]:

assert_is_instance(clf1, Pipeline)
assert_is_instance(y_pred1, np.ndarray)
tf1 = clf1.named_steps['tf']
assert_is_instance(tf1, TfidfVectorizer)
assert_is_instance(clf1.named_steps['knc'], KNeighborsClassifier)
assert_equal(tf1.stop_words, 'english')
assert_equal(tf1.ngram_range, (1, 3))
assert_equal(tf1.min_df, 3)
assert_equal(tf1.max_df, 0.7)
assert_equal(len(y_pred1), len(t_y_test))
assert_array_equal(y_pred1[:5], ['belles_lettres', 'government', 'romance', 'belles_lettres', 'government'])
assert_array_equal(y_pred1[-5:], ['government', 'lore', 'government', 'learned', 'adventure'])
assert_almost_equal(score1, 0.52500000000000002)               


# ## Stemming
# 
# - Modify the `tokenize` function from [Introduction to Text Mining notebook](../../notebooks/intro2tm.ipynb). Use [Snowball](http://www.nltk.org/api/nltk.stem.html#module-nltk.stem.snowball) stemming algorithm instead of Porter Stemmer,
# - Use unigrams, bigrams, and trigrams, 
# - Build a pipeline by using `TfidfVectorizer` and `KNeighborsClassifier`,
# - Name the first step `tf` and the second step `knc`,
# - Use English stop words,
# - Convert all words into lower case so that the model does not depend on cases,
# - Impose a minimum feature term that requires a term to be present in at least 3 documents, 
# - Set a maximum frequency of 70%, such that any term occurring in more than 70% of all documents will be ignored, and
# - Set the rest parameters to default for both `TfidfVectorizer` and `KNeighborsClassifier`.

# In[59]:

def tokenize(text):
    '''
    Converts text into tokens. Same function as in the "introduction to text mining" notebook.
    Uses Snowball Stemmer.
    
    Parameters
    ----------
    text: a string.
    
    Returns
    -------
    tokens: a map object.
    '''
    #Converts text to tokens
    tokens = nltk.word_tokenize(text)
    tokens = [token for token in tokens if token not in string.punctuation]

    stemmer = EnglishStemmer()
    stems = map(stemmer.stem, tokens)
    
    return stems


# In[60]:

def stem(X_train, y_train, X_test):
    '''
    Creates a document term matrix and uses KNC classifier to make document classifications.
    Uses the Snowball stemmer.
    
    Parameters
    ----------
    X_train: A 1d numpy array of strings.
    y_train: A 1d numpy array of strings.
    X_test: A 1d numpy array of strings.
    
    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A 1d numpy array.
    '''
    #Creates pipeline with the parameters defined in the problems
    tools = [('tf', TfidfVectorizer()), ('knc', KNeighborsClassifier())]
    clf = Pipeline(tools)
    clf.set_params(tf__stop_words = 'english',
                   tf__ngram_range = (1,3),
                   tf__max_df = 0.7,
                   tf__min_df = 3,
                   tf__lowercase=True,
                   tf__tokenizer=tokenize)
    #Fits and predicts based off of the model              
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    return clf, y_pred


# In[61]:

# use sliced arrays to save execution time
# you should try use original data and compare predcition accurary
clf2, y_pred2 = stem(t_X_train[:100], t_y_train[:100], t_X_test[:50])
score2 = accuracy_score(y_pred2, t_y_test[:50])
print("KNC prediction accuracy = {0:5.1f}%".format(100.0 * score2))


# In[62]:

assert_is_instance(clf2, Pipeline)
assert_is_instance(y_pred2, np.ndarray)
tf2 = clf2.named_steps['tf']
assert_is_instance(tf2, TfidfVectorizer)
assert_is_instance(clf2.named_steps['knc'], KNeighborsClassifier)
assert_equal(tf2.stop_words, 'english')
assert_equal(tf2.ngram_range, (1, 3))
assert_equal(tf2.min_df, 3)
assert_equal(tf2.max_df, 0.7)

assert_equal(len(y_pred2), 50)
assert_array_equal(y_pred2[:5], ['lore', 'learned', 'romance', 'belles_lettres', 'learned'])
assert_array_equal(y_pred2[-5:], ['fiction', 'romance', 'belles_lettres', 'romance', 'learned'])
assert_almost_equal(score2, 0.41999999999999998 )


# ## Clustering Analysis
# 
# - Build a pipeline by using `TfidfVectorizer` and `KMeans`,
# - Name the first step `tf` and the second step `km`,
# - Use unigrams only,
# - Use English stop words, 
# - Convert all words into lower case so that the model does not depend on cases,
# - Impose a minimum feature term that requires a term to be present in at least 3 documents, 
# - Set a maximum frequency of 70%, such that any term occurring in more than 70% of all documents will be ignored,
# - Set the number of clusters equal to `k`,
# - Set the rest parameters to default for both `TfidfVectorizer` and `KNeighborsClassifier`, and 
# - Identify the most frequently used words for each cluster.

# In[74]:

def get_top_tokens(X_train, y_train, X_test, random_state, k, n):
    '''
    First, applies clustering analysis to a feature matrix.
    Then, identifies the most frequently used words in "icluster".
    
    Parameters
    ----------
    X_train: A 1d numpy array of strings.
    y_train: A 1d numpy array of strings.
    X_test: A 1d numpy array of strings.
    random_state: A np.random.RandomState instance for KMeans.
    k: An int. The number of clusters.
    n: An int. Specifies how many tokens for each cluster should be returned.
    
    Returns
    -------
    clf: A Pipeline instance.
    tokens: A 2d numpy array of strings with shape of (n_clusters, n_tokens of each cluster)
    '''
    #Creates the pipeline and fits it
    clf = Pipeline([('tf',TfidfVectorizer()),('km',KMeans())])
    clf.set_params(tf__stop_words = 'english',
                   tf__max_df = 0.7,
                   tf__min_df = 3,
                   tf__lowercase=True,
                   km__n_clusters = k,
                   km__random_state = random_state)
    clf.fit(X_train)
    km = clf.named_steps['km']
    tf = clf.named_steps['tf']
    #Finds the center of the clusters
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    tokens=[]
    terms = tf.get_feature_names()
    #Finds the most frequent words for each cluster
    for idx in range(k):
        for jdx in order_centroids[idx, :n]:
            tokens.append(terms[jdx])
    #Reshapes the array        
    tokens=np.array(tokens).reshape(k,n)        
    return clf, tokens


# In[75]:

k3 = len(np.unique(t_genres))
n3 = 5
clf3, tokens3 = get_top_tokens(t_X_train, t_y_train, t_X_test, check_random_state(0), k3, n3)
print('Top {} tokens per cluster:'.format(n3))
print('-'*45)
for i in range(k3):
    print("Cluster {0}: {1}".format(i, ' '.join(tokens3[i])))


# In[76]:

assert_is_instance(clf3, Pipeline)
tf3 = clf3.named_steps['tf']
assert_is_instance(tf3, TfidfVectorizer)
km3 = clf3.named_steps['km']
assert_is_instance(km3, KMeans)
assert_equal(tf3.stop_words, 'english')
assert_equal(tf3.ngram_range, (1, 1))
assert_equal(tf3.min_df, 3)
assert_equal(tf3.max_df, 0.7)
assert_equal(km3.n_clusters, k3)
assert_equal(np.shape(tokens3), (9, 5))
assert_array_equal(tokens3, [['fw', 'nil', 'bridge', 'pont', 'nps'],
                             ['men', 'man', 'said', 'eyes', 'dod'],
                             ['hl', 'costs', 'shelter', 'foam', 'foods'],
                             ['college', 'mrs', 'students', 'school', 'education'],
                             ['said', 'dod', 'uh', 'll', 'bem'],
                             ['hl', 'nps', 'state', 'president', 'law'],
                             ['hl', 'year', 'tax', 'sales', '1960'],
                             ['af', 'hl', 'temperature', 'pressure', 'fig'],
                             ['nc', 'fw', 'man', 'human', 'experience']])


# In[77]:

clf4, tokens4 = get_top_tokens(t_X_train, t_y_train, t_X_test, check_random_state(0), k3, 3)
assert_array_equal(tokens4[0], ['fw', 'nil', 'bridge'])
assert_array_equal(tokens4[6], ['hl', 'year', 'tax'])


# In[ ]:



