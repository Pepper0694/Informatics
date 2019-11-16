
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
# # Problem 9.1: NLP: Basic Concepts
# 
# For this problem, we will be delving into part of speech tagging and some basic text analysis.  You will be analyzing text from Monty Python and the Holy Grail.

# In[1]:

import re
import requests
import nltk
import pprint
import collections

from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.collocations import BigramCollocationFinder, BigramAssocMeasures, TrigramCollocationFinder, TrigramAssocMeasures
from nltk.tag import DefaultTagger, UnigramTagger
from nltk.corpus import treebank
from nltk.corpus import webtext

from nose.tools import assert_equal, assert_is_instance, assert_true


# In[2]:

monty = webtext.raw('grail.txt')
assert_is_instance(monty, str)
assert_equal(len(monty), 65003)


# ## Tokenize
# In this function, you will tokenize the given input text.  The function word_tokenize() might prove helpful in this instance.

# In[3]:

def tokenize(text_str):
    '''
    Tokenizes the text string by words.
    
    Parameters
    ----------
    text: A string
    
    Returns
    -------
    A list of strings
    '''
    #tokenize a string of words
    tokens=word_tokenize(text_str)
    return tokens


# In[4]:

tok = tokenize(monty)
assert_is_instance(tok,list)
assert_true(all(isinstance(t, str) for t in tok))
assert_equal(len(tok), 16450)
assert_equal(tok[:10], ['SCENE', '1', ':', '[', 'wind', ']', '[', 'clop', 'clop', 'clop'])
assert_equal(tok[51:55], ['King', 'of', 'the', 'Britons'])
assert_equal(tok[507:511], ['African', 'swallows', 'are', 'non-migratory'])


# ## Collocations: bigrams
# 
# Here you will make a function that will use NLTK to grab the x best bi-grams, where x is a positive integer.  You should be using pointwise mutual information in order to do this.

# In[5]:

def x_bigrams(tokens, x):
    '''
    Find the x best bi-grams given tokens (a list of strings) and x which will 
    tell you how many bi-grams to return.
    
    Parameters
    ----------
    tokens: A list of strings
    x: An integer
    
    
    Returns
    -------
    ls_bigram: A list of tuples, with the tuples being of the form (str, str).
    '''
    #Finds bigrams than finds the x best ones
    bigramass = BigramAssocMeasures()
    finder = BigramCollocationFinder.from_words(tokens)
    ls_bigram = finder.nbest(bigramass.pmi, x)
    
    return ls_bigram


# In[6]:

bigrams = x_bigrams(tok, 20)
assert_is_instance(bigrams, list)
assert_true(all(isinstance(b, tuple) for b in bigrams))
assert_true(len(bigrams), 20)
assert_equal(bigrams, [("'Til", 'Recently'), ("'To", 'whoever'),
                       ('Anybody', 'armed'), ('Attila', 'raised'),
                       ('Badon', 'Hill'), ('Bon', 'magne'), ('Chapter', 'Two'),
                       ('Clark', 'Gable'), ('Divine', 'Providence'),
                       ('Great', 'scott'), ('Most', 'kind'),
                       ('Olfin', 'Bedwere'), ('Recently', 'Said'),
                       ('Thou', 'hast'), ('Thy', 'mer'), ('Too', 'late'),
                       ('Uther', 'Pendragon'), ('absolutely', 'necessary'),
                       ('advancing', 'behaviour'),
                       ('anarcho-syndicalist', 'commune')])


# ## Collocations: trigrams
# 
# Now you will repeat the previous function, but instead of bi-grams, you will be finding the x best tri-grams, again using PMI.

# In[7]:

def x_trigrams(tokens, x):
    '''
    Find the x best tri-grams given tokens (a list of strings) and x which will 
    tell you how many tri-grams to return.
    
    Parameters
    ----------
    tokens: A list of strings
    x: An integer
    
    
    Returns
    -------
    tri_list: A list of tuples, with the tuples being of the 
    form (str, str, str).
    
    '''
    #Finds bigrams than finds the x best ones
    trigramass = TrigramAssocMeasures()
    finder = TrigramCollocationFinder.from_words(tokens)
    tri_list = finder.nbest(trigramass.pmi, x)
    
    return tri_list


# In[8]:

trigrams = x_trigrams(tok, 5)
assert_is_instance(trigrams, list)
assert_true(all(isinstance(t, tuple) for t in trigrams))
assert_true(len(trigrams), 5)
assert_equal(trigrams, [("'Til", 'Recently', 'Said'),
                        ("'To", 'whoever', 'finds'), 
                        ('Thou', 'hast', 'vouchsafed'),
                        ('basic', 'medical', 'training'),
                        ('dorsal', 'guiding', 'feathers')])


# ## Part of Speech Tagging
# Now that we have a good handle on our best bi- and tri-grams, let's change gears and try to do some POS tagging.  For this function, we will only do rudimentary POS tagging.  You should find the use of `pos_tag` to be helpful here.  You should write your code so that if default is true, you use the default tagger, but if it is false, then the Universal tagger should be used.

# In[9]:

def tagging(tokens, default = True):
    '''
    Performs POS tagging with the tagger determined by the boolean 'default'.    
    
    Parameters
    ----------
    tokens: a list of strings
    default: a boolean 
    
    Returns
    -------
    tagged: a list of tuples, with the tuples taking the form of (str, str)
    '''
    #POS tagging with default settings
    if default==True:
        tagged = pos_tag(tokens)
    #POS with typset universal    
    else:
        tagged = pos_tag(tokens, tagset='universal')
    return tagged


# In[10]:

uni = tagging(tok, default = False)
assert_is_instance(uni, list)
assert_true(all(isinstance(u, tuple) for u in uni))
assert_true(len(uni), 16450)
assert_equal(uni[745:760], [('DEAD', 'NOUN'), ('PERSON', 'NOUN'),
                            (':', '.'), ('I', 'PRON'), ("'m", 'VERB'),
                            ('not', 'ADV'), ('dead', 'ADJ'), ('!', '.'),
                            ('CART-MASTER', 'NOUN'), (':', '.'),
                            ('What', 'PRON'), ('?', '.'), ('CUSTOMER', 'NOUN'),
                            (':', '.'), ('Nothing', 'NOUN')])

not_uni = tagging(tok)
assert_is_instance(not_uni, list)
assert_true(all(isinstance(n, tuple) for n in not_uni))
assert_true(len(not_uni), 16450)
assert_equal(not_uni[1503:1525], [('We', 'PRP'), ("'re", 'VBP'), ('an', 'DT'),
                                  ('anarcho-syndicalist', 'JJ'),
                                  ('commune', 'NN'), ('.', '.'), ('We', 'PRP'),
                                  ('take', 'VBP'), ('it', 'PRP'), ('in', 'IN'),
                                  ('turns', 'VBZ'), ('to', 'TO'), ('act', 'VB'),
                                  ('as', 'IN'), ('a', 'DT'), ('sort', 'NN'),
                                  ('of', 'IN'), ('executive', 'JJ'),
                                  ('officer', 'NN'), ('for', 'IN'),
                                  ('the', 'DT'), ('week', 'NN')])


# ## Tagged Text extraction
# 
# Finally, we will create a function that will only return the Nouns or Adjectives from our tokens.  It will be helpful to use regular expressions in this case.  Also, you should utilize the "tagging" function that you just made above. Additionally, your function should return the "n" most common words (and their occurances) in ext_tag. In order to find the most common words and their occurances, please consider using Counter.

# In[11]:

def tag_tx_ext(tokens, n):
    '''
    Takes in tokens and returns a list of tokens that are either nouns
    or adjectives as well as a list of tuples of the most common adjectives
    or nouns with their number of occurances.
    
    Parameters
    ----------
    tokens: A list of strings.
    n: An integer.
    
    Returns
    -------
    A tuple of ext_tag and common where these two arguments have the following
    structure:
    ext_tag: A list of strings.
    common: A list of tuples of the form (str, int)
    '''
    #Import the module to count the occurances of the most common words
    from collections import Counter
    #Find all adjectives and nouns
    rgxs = re.compile(r"(JJ|NN)")
    ptgs=tagging(tokens)
    #Counts the most common adjectives and nouns
    ext_tag = [tkn[0] for tkn in ptgs if re.match(rgxs, tkn[1])]
    common=Counter([word for word in ext_tag]).most_common(n)
    
    
    return ext_tag, common


# In[12]:

ex_tags, com = tag_tx_ext(tok, 13)
assert_is_instance(ex_tags, list)
assert_true(all(isinstance(ex, str) for ex in ex_tags))
assert_true(len(ex_tags), 5323)
assert_equal(ex_tags[603:620], ['BLACK', 'KNIGHT', 'Aagh', 'GREEN', 'KNIGHT',
                                '[', 'King', 'Arthur', 'music', ']', 'Ooh',
                                '[', 'music', ']', 'BLACK','KNIGHT', 'Aaagh'])

assert_equal(ex_tags[1000:1010], ['Burn', 'BEDEVERE', 'Quiet', 'Quiet', 'Quiet',
                                  'Quiet', 'ways', 'witch', 'VILLAGER', 'Are'])

assert_is_instance(com, list)
assert_true(all(isinstance(c, tuple) for c in com))
assert_true(len(com), 13)
assert_equal(com, [(']', 296), ('[', 285), ('ARTHUR', 220), ('LAUNCELOT', 71),
                   ('KNIGHT', 68), ('GALAHAD', 67), ('FATHER', 63),
                   ('BEDEVERE', 60), ('HEAD', 54), ('GUARD', 53),
                   ('Sir', 51), ('VILLAGER', 47), ('boom', 45)])


# In[ ]:



