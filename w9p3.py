
# coding: utf-8

# # Week 9 Problem 3
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
# # Problem 9.3. NLP: Semantic Analysis
# In this problem, we explore semantic analysis.

# In[1]:

import string
import time
import numpy as np
import gensim
from nltk.tokenize import WordPunctTokenizer

from nose.tools import (
    assert_equal,
    assert_is_instance,
    assert_almost_equal,
    assert_true
    )

from numpy.testing import assert_array_equal


# --------------
# 
# # Wordnet
# We use the Wordnet synonym rings.

# In[2]:

from nltk.corpus import wordnet as wn


# - Find how many entries a word has in the wordnet synset.

# In[3]:

def find_number_of_entries(word):
    '''
    Finds the number of entries in the wordnet synset.
    
    Parameters
    ----------
    word: A string.
    
    Returns
    -------
    An int.
    '''
    #Finds the number of entries a word has in wordnet
    the_synsets = wn.synsets(word)
    result = len(the_synsets)
    
    return result


# In[4]:

the_word = 'love'
n_entries = find_number_of_entries(the_word)
print('{0} total entries in synonym ring for {1}. '.format(n_entries, the_word))


# In[5]:

the_word = 'live'
n_entries = find_number_of_entries(the_word)
print('{0} total entries in synonym ring for {1}. '.format(n_entries, the_word))


# In[6]:

assert_is_instance(find_number_of_entries('love'), int)
assert_equal(find_number_of_entries('love'), 10)
assert_equal(find_number_of_entries('live'), 19)


# ## Word Similarities
# - Compute the path similarity for the input words. 
# - Use the first noun synset (with number 01) for each word. 
# - You could assume all input words have at least one noun synset.

# In[7]:

def get_path_similarity(word1, word2):
    '''
    Computes the path similarity between word1 and word1.
    
    Parameters
    ----------
    word1: A string.
    word2: A string.
    
    Returns
    -------
    A float.
    '''
    #Computes path similarity for input words
    word1 = wn.synset(word1+'.n.01')
    word2 = wn.synset(word2+'.n.01')
    result = wn.path_similarity(word1,word2)
    
    return result


# In[8]:

# Now we print similarity measures.
fmt_str = '{1} to {2}: {0:4.3f}'

print('Path Similarity:')
print(40*'-')
print(fmt_str.format(get_path_similarity('excess', 'surplus'), 'excess', 'surplus'))
print(fmt_str.format(get_path_similarity('trade', 'economy'), 'trade', 'economy'))
print(fmt_str.format(get_path_similarity('mean', 'average'), 'mean', 'average'))
print(fmt_str.format(get_path_similarity('import', 'export'), 'mean', 'average'))
print(fmt_str.format(get_path_similarity('excess', 'excess'), 'excess', 'excess'))


# In[9]:

assert_is_instance(get_path_similarity('excess', 'surplus'), float)
assert_almost_equal(get_path_similarity('excess', 'surplus'), 1.0)
assert_almost_equal(get_path_similarity('trade', 'economy'), 0.1)
assert_almost_equal(get_path_similarity('mean', 'average'), 0.5)
assert_almost_equal(get_path_similarity('import', 'export'), 0.3333333333333333)
assert_almost_equal(get_path_similarity('excess', 'excess'), 1.0)


# ------
# 
# # Word2Vec
# In the second half of this problem, we use the NLTK reuters corpus to build a word2vec model.

# In[10]:

from nltk.corpus import reuters
sentences = reuters.sents()[:20000] # use a sample size smaller than corpus


# ## Word2Vec model
# - Build a Word2Vec model from sentences in the corpus.
# - Set the maximum distance between the current and predicted word within a sentence to 10.
# - Ignore all words with total frequency lower than 6.

# In[11]:

def get_model(sentences):
    '''
    Builds a Word2Vec model from sentences in corpus.
    
    Parameters
    ----------
    sentences: A list of lists(sentences); each sentence is a list of strings(words).
    
    Returns
    -------
    A Word2Vec instance.
    '''
    #Creates wird2vec model fri=om sentences with specified parameters
    model = gensim.models.Word2Vec(sentences, window=10, min_count=6)
    
    
    return model


# The following cell would take about 30 seconds to complete.

# In[12]:

start_time = time.clock()
model = get_model(sentences)
print(time.clock() - start_time, "seconds")


# In[13]:

assert_is_instance(model, gensim.models.Word2Vec)
assert_equal(model.window, 10)
assert_equal(model.min_count, 6)


# ## Cosine Similarity
# Compute Cosine Similarities.

# In[14]:

def get_cosine_similarity(model, word1, word2):
    '''
    Computes cosine similarity between "word1" and "word2" using a Word2Vec model.
    
    Parameters
    ----------
    model: A gensim.Word2Vec model.
    word1: A string.
    word2: A string.
    
    Returns
    -------
    A float.
    '''
    #Computes cosine similarities of word1 and word2 with the previous model
    similarity = model.similarity(word1, word2)
    
    return similarity


# In[15]:

# Now we print similarity measures.
fmt_str = '{1} to {2}: {0:4.3f}'

print('Cosine Similarity:')
print(40*'-')
print(fmt_str.format(get_cosine_similarity(model, 'excess', 'surplus'), 'excess', 'surplus'))
print(fmt_str.format(get_cosine_similarity(model, 'trade', 'economy'), 'trade', 'economy'))
print(fmt_str.format(get_cosine_similarity(model, 'mean', 'average'), 'mean', 'average'))
print(fmt_str.format(get_cosine_similarity(model, 'import', 'export'), 'mean', 'average'))
print(fmt_str.format(get_cosine_similarity(model, 'excess', 'excess'), 'excess', 'excess'))


# In[16]:

assert_is_instance(get_cosine_similarity(model, 'excess', 'surplus'), float)
assert_almost_equal(get_cosine_similarity(model, 'excess', 'surplus'), model.similarity('excess', 'surplus'))
assert_almost_equal(get_cosine_similarity(model, 'trade', 'economy'), model.similarity('trade', 'economy'))
assert_almost_equal(get_cosine_similarity(model, 'mean', 'average'), model.similarity('mean', 'average'))
assert_almost_equal(get_cosine_similarity(model, 'import', 'export'), model.similarity('import', 'export'))
assert_almost_equal(get_cosine_similarity(model, 'excess', 'excess'), 1.0)


# ## Most similar words
# Find the top 5 most similar words, where "price", "economy", and "trade" contribute positively towards the similarity, and "law" and "legal" contribute negatively.

# In[21]:

def find_most_similar_words(model):
    '''
    Find the top 5 most similar words,
    where "price", "economy", and "trade" contribute positively towards the similarity,
    and "law" and "legal" contribute negatively.
    
    Parameters
    ----------
    model: A gensim.Word2Vec model.
    
    Returns
    -------
    A list of tuples (word, similarty).
    word: A string.
    similarity: A float.
    '''
    #Finds 5 most similar words with the specified words as positive and the
    #specified words as negative
    result = model.most_similar(positive=['price', 'economy','trade'], 
                                negative=['law','legal'], topn=5)
    
    return result


# In[22]:

print('{0:14s}: {1}'.format('Word', 'Cosine Similarity'))
print(40*'-')
for val in find_most_similar_words(model):
    print('{0:14s}: {1:6.3f}'.format(val[0], val[1]))


# In[23]:

assert_is_instance(find_most_similar_words(model), list)
assert_true(all(isinstance(t[0], str) for t in find_most_similar_words(model)))
assert_true(all(isinstance(t[1], float) for t in find_most_similar_words(model)))
assert_equal(len(find_most_similar_words(model)), 5)
words = [t[0] for t in model.most_similar(positive=['price', 'economy', 'trade'], negative=['law', 'legal'], topn=5)]
similarities = [t[1] for t in model.most_similar(positive=['price', 'economy', 'trade'], negative=['law', 'legal'], topn=5)]
assert_equal([t[0] for t in find_most_similar_words(model)], words)
assert_almost_equal(find_most_similar_words(model)[0][1], similarities[0])
assert_almost_equal(find_most_similar_words(model)[1][1], similarities[1])
assert_almost_equal(find_most_similar_words(model)[2][1], similarities[2])
assert_almost_equal(find_most_similar_words(model)[3][1], similarities[3])
assert_almost_equal(find_most_similar_words(model)[4][1], similarities[4])


# In[ ]:



