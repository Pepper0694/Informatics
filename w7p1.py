
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
# 
# # Problem 7.1. Text Analysis.
# 
# For this problem, we will be performing text analysis on NLTK's Webtext corpus.

# In[30]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import scipy as sp
import re
import requests

import nltk

from nltk.corpus import webtext

from nose.tools import (
    assert_equal,
    assert_is_instance,
    assert_almost_equal,
    assert_true,
    assert_false
    )


# ## Tokenize the text
# Before we can really do some analysis on this corpus, we must first tokenize it.  In the `tokenize` function, you will be extracting the raw text from our NLTK corpus for one certain file within, identified by the fileID; you should reference [NLTK](http://www.nltk.org/book/ch02.html#reuters-corpus) for help.  After you have accessed this raw text, you should tokenize the text, making sure to cast everything to lower case and stripping out any errant punctuation marks and spaces.  Additionally, this should only return words that are made of one or more alphanumerical characters.

# In[31]:

def tokenize(corpus, fileID):
    '''
    Tokenizes the, casting all words to lower case, stripping out punctuation marks, spaces,
    and words not made of one or more alphanumerical characters.
    
    Parameters
    ----------
    corpus: An NLTK corpus
    fileID: A string
    
    Returns
    -------
    words: a list of strings
    '''
    #Use regex to remove punctuation marks and spaces. Only returns words with >= 1 alphanumerical characters
    pattern = re.compile(r'[^\w\s]')
    
    text=webtext.raw(fileID)
    words = [word.lower() for word in nltk.word_tokenize(re.sub(pattern, ' ', text))]
    return words


# In[32]:

monty = tokenize(webtext, 'grail.txt')
assert_is_instance(monty, list)
assert_equal(len(monty), 11608)

assert_true(all(isinstance(w, str) for w in monty))
assert_true(all(all(not c.isupper() for c in w) for w in monty))
assert_true(all(any(c.isalnum() for c in w) for w in monty))

assert_equal(monty[8:13], ['whoa', 'there', 'clop', 'clop', 'clop'])
assert_equal(monty[20:45], ['it', 'is', 'i', 'arthur', 'son', 'of', 'uther', 'pendragon',                            'from', 'the', 'castle', 'of', 'camelot', 'king', 'of', 'the',                            'britons', 'defeator', 'of', 'the', 'saxons', 'sovereign', 'of', 'all', 'england'])

pirates= tokenize(webtext, 'pirates.txt')
assert_is_instance(pirates, list)
assert_equal(len(pirates), 17153)

assert_true(all(isinstance(w, str) for w in pirates))
assert_true(all(all(not c.isupper() for c in w) for w in pirates))
assert_true(all(any(c.isalnum() for c in w) for w in pirates))

assert_equal(pirates[100:110], ['the', 'left', 'in', 'the', 'barn', 'where', 'the', 'marines', 'enter', 'liz'])
assert_equal(pirates[-10:], ['left', 'shoulder', 'faces', 'the', 'camera', 'and', 'snarls', 'scene', 'end', 'credits'])


# ## Count words
# Here, we want to find the number of tokens, the number of words and lexical diversity of one of the list of strings found with the previous function definition.

# In[35]:

def count_words(word_ls):
    '''
    Computes the the number of token, number of words, and lexical diversity.
    
    Parameters
    ----------
    word_ls: A list of of strings.
    
    Returns
    -------
    A 3-tuple of (num_tokens, num_words, lex_div) called tup
    num_tokens: An int. The number of tokens in "words".
    num_words: An int. The number of words in "words".
    lex_div: A float. The lexical diversity of "words".
    '''
    #Frequency distribution
    frequdist = nltk.FreqDist(word_ls)
    #Number of words
    num_words = len(word_ls)
    #Number of tokens
    num_tokens = len(freqdist)
    #Lexical diversity
    lex_div = num_words / num_tokens
    #Makes the variables into a tuple
    tup=(num_tokens,num_words,lex_div)
    return tup


# In[36]:

monty_tokens, monty_words, mld = count_words(monty)
assert_is_instance(monty_tokens, int)
assert_is_instance(monty_words, int)
assert_is_instance(mld, float)
assert_equal(monty_tokens, 1823)
assert_equal(monty_words, 11608)
assert_almost_equal(mld, 6.3675260559517275)

pirate_tokens, pirate_words, pld = count_words(pirates)
assert_is_instance(pirate_tokens, int)
assert_is_instance(pirate_words, int)
assert_is_instance(pld, float)
assert_equal(pirate_tokens, 2731)
assert_equal(pirate_words, 17153)
assert_almost_equal(pld, 6.280849505675577)


# ## Most common words
# Now that we have tokens, we can find the most common words used in each list of strings found with `tokenize`.

# In[20]:

def most_common(words, num_top_words):
    '''
    Takes the output of tokenize and find the most common words within that list,
    returning a list of tuples containing the most common words and their number 
    of occurances.
    
    Parameters
    ----------
    words: A list of strings
    num_top_words:  An int. The number of most common words (and tuples) 
                    that will be returned.
    
    Returns
    -------
    top_words:  A list of tuples, where each tuple contains a word and
                its number of occurances.
    '''
    #Frequency distribution
    counts = nltk.FreqDist(words)
    #The most frequent word
    top_words = counts.most_common(num_top_words)
    return top_words


# In[21]:

yarr = most_common(pirates, 5)

assert_is_instance(yarr, list)
assert_true(all(isinstance(t, tuple) for t in yarr))
assert_true(all(isinstance(t, str) for t, f in yarr))
assert_true(all(isinstance(f, int) for t, f in yarr))

assert_equal(len(most_common(pirates, 10)), 10)
assert_equal(len(most_common(pirates, 20)), 20)
assert_equal(yarr, [('the', 1073), ('jack', 470), ('a', 434), ('to', 372), ('of', 285)])

shrubbery = most_common(monty, 5)
assert_is_instance(shrubbery, list)
assert_true(all(isinstance(t, tuple) for t in shrubbery))
assert_true(all(isinstance(t, str) for t, f in shrubbery))
assert_true(all(isinstance(f, int) for t, f in shrubbery))

assert_equal(len(most_common(monty, 15)), 15)
assert_equal(len(most_common(monty, 37)), 37)
assert_equal(shrubbery, [('the', 334), ('you', 265), ('arthur', 261), ('i', 260), ('a', 238)])


# ## Hapaxes
# Now, we find the hapaxes from the list of strings we made with `tokenize`.

# In[22]:

def hapax(words):
    '''
    Finds all hapaxes from the "words" list of strings.    
    
    Parameters
    ----------
    words: A list of strings
    
    Returns
    -------
    hapax: A list of strings
    
    '''
    #Frequency distribution
    freqdist = nltk.FreqDist(words)
    #Finds the hapaxes
    hapax = freqdist.hapaxes()
    
    return hapax


# In[23]:

assert_is_instance(hapax(monty), list)
assert_true(all(isinstance(w, str) for w in hapax(monty)))
assert_equal(len(hapax(monty)), 977)
assert_equal(sorted(hapax(monty))[-5:],['zhiv', 'zone', 'zoo', 'zoop', 'zoosh'])

assert_is_instance(hapax(pirates), list)
assert_true(all(isinstance(w, str) for w in hapax(pirates)))
assert_equal(len(hapax(pirates)), 1433)
assert_equal(sorted(hapax(pirates))[-5:],['yeah', 'yep', 'yours', 'yourselves', 'zooming'])


# ## Long words
# 
# Finally, we find the words within the output of `tokenize` longer than a given length.

# In[26]:

def long_words(words, length=10):
    '''
    Finds all words in "words" longer than "length".
    
    Parameters
    ----------
    corpus: An list of strings.
    length: An int. Default: 10
    
    Returns
    -------
    A list of strings.
    '''
    #Finds all words longer than length with a list comprehension
    long_words = [word for word in words if len(word) > length]
    
    return long_words


# In[27]:

monty_l = long_words(monty, 12)
assert_is_instance(monty_l, list)
assert_true(all(isinstance(w, str) for w in monty_l))    
assert_equal(len(monty_l), 6)
assert_equal(
    set(monty_l),
    set(['unfortunately', 'understanding', 'oooohoohohooo', 'indefatigable', 'camaaaaaargue', 'automatically'])
    )
assert_equal(len(long_words(monty,10)), 68)
assert_equal(len(long_words(monty,11)), 37)


pirate_l = long_words(pirates, 13)
assert_is_instance(pirate_l, list)
assert_true(all(isinstance(w, str) for w in monty_l))    
assert_equal(len(pirate_l), 5)
assert_equal(
    set(pirate_l),
    set(['simultanenously', 'responsibility', 'reconciliatory', 'incapacitorially', 'enthusiastically']))
assert_equal(len(long_words(pirates,10)), 107)
assert_equal(len(long_words(pirates,12)), 29)


# In[ ]:




# In[ ]:



