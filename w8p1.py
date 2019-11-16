
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
# # Problem 8.1: Social Media, Email.
# 
# For this problem, we will be doing basic analysis and data extraction on emails located in our data file.

# In[1]:

import os
import email as em
from email import policy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.utils import check_random_state

from nose.tools import assert_equal, assert_is_instance, assert_true
from numpy.testing import assert_array_equal, assert_almost_equal


# ## Get email info
# In this function, you will write a function to do the following things:
# - Open an email, given a datapath
# - Return who the message was to, who it was from, and the subject line.
# - If "key" is true, then also return a list of keys. If "key" is false, an empty list is returned instead.

# In[2]:

def email_info(datapath, key=False):
    '''
    Extracts the "To", "From" and "Subject" information from a given email.
    Also returns either a list of header keys or an empty list, depending on the
    value of "key".
    
    Parameters
    ----------
    datapath: A string
    key: A boolean
    
    Returns
    -------
    tofromsub = a list of strings
    headkey = a list of strings or an empty list
    '''
    with open(datapath) as f:
        earmuffs=em.message_from_file(f, policy=policy.default)
    earmuffs_to = earmuffs['to']
    earmuffs_from = earmuffs['from']
    earmuffs_subject = earmuffs['subject']   
    tofromsub=[earmuffs_to,earmuffs_from,earmuffs_subject]
    if key==True:
        headkey=earmuffs.keys()
    else:
        headkey=[]
    return tofromsub, headkey


# In[3]:

dat1 = '/home/data_scientist/data/email/ham/00001.1a31cc283af0060967a233d26548a6ce'
emailhead1, headkey1 = email_info(dat1)
assert_is_instance(emailhead1, list)
assert_is_instance(headkey1, list)
assert_equal(headkey1, [])
assert_equal(emailhead1, ['Chris Garrigues <cwg-dated-1030314468.7c7c85@DeepEddy.Com>', 'Robert Elz <kre@munnari.OZ.AU>', 'Re: New Sequences Window'])
assert_equal(len(headkey1), 0)
assert_equal(len(emailhead1), 3)

dat2 = '/home/data_scientist/data/email/spam/00001.317e78fa8ee2f54cd4890fdc09ba8176'
emailhead2, headkey2 = email_info(dat2, key = True)
assert_is_instance(emailhead2, list)
assert_is_instance(headkey2, list)
assert_equal(headkey2, ['Return-Path', 'Delivered-To' , 'Received', 'Received',                        'Received', 'Received', 'Received',                         'X-Authentication-Warning', 'Received', 'Message-Id',                        'Date', 'To', 'From', 'MIME-Version', 'Content-Type',                        'Subject', 'Sender', 'Errors-To', 'X-Mailman-Version',                        'Precedence', 'List-Id', 'X-Beenthere'])
assert_equal(emailhead2, ['ilug@linux.ie', 'Start Now <startnow2002@hotmail.com>', '[ILUG] STOP THE MLM INSANITY'])
assert_equal(len(headkey2), 22)
assert_equal(len(emailhead2), 3)


# ## Many Payloads
# Now, we want to grab many emails from the same directory so that we can perform analysis and training on them later. This function should take four arguments: a string called "path" that states the base directory that you want to extract all of your payloads from, two integers, stating on which number message to start and stop at, inclusive, and a boolean that states if you want to look at ham or spam.  You should use os.walk(). For example, if beg = 50 and end = 75, you should iterate through the 50th message through the 75th, inclusive. 

# In[4]:

def many_payloads(path, beg, end, ham = True):
    '''
    Captures the payloads of the emails specified between beg and end,
    and appends the payloads into a list called payloads.
    
    Parameters
    ----------
    path: A string
    beg: An integer
    end: An integer
    ham: A boolean
    
    Returns
    -------
    payloads: A list of strings.
    '''
    #Create array of payloads of emails from beg to end
    payloads=[]
    if ham==True:
        band_camp='ham'
    else:
        band_camp='spam'
    for root, dirs, files in os.walk(os.path.join(path,band_camp)):
        
        for file in files[beg:end+1]:
            #file_path=path+'/'+file
            with open(os.path.join(root,file), encoding='ISO-8859-1') as fin:
                el_guapo_salsa=em.message_from_file(fin, policy=policy.default)
                for part in el_guapo_salsa.walk():
                    if part.get_content_type()=='text/plain':
                        who_is_kaiser_salsa = part.get_payload(None, decode=True)
                payloads.append(who_is_kaiser_salsa.decode(encoding='ISO-8859-1'))
    return payloads


# In[5]:

ham = many_payloads('/home/data_scientist/data/email', 100, 600, ham = True)
assert_is_instance(ham, list)
assert_true(all(isinstance(h, str) for h in ham))
assert_equal(len(ham), 501)
assert_true(ham[7].startswith("I've got some really interesting wav files here."))
assert_true(ham[53].startswith('On Tue, Jul 30, 2002 at 11:28:11AM +0200, David Neary mentioned:'))

spam = many_payloads('/home/data_scientist/data/email', 100, 600, ham = False)
assert_is_instance(spam, list)
assert_true(all(isinstance(s, str) for s in spam))
assert_equal(len(spam), 501)
assert_true(spam[365].startswith("1916eEph3-937NQem2852GQnA3-l25"))
assert_true(spam[-1].startswith('Your mortgage has been approved.'))


# ## Convert to arrays
# In order to use scikit learn, we need to convert our ham and spam arrays to numpy arrays (pos_emails and neg_emails), and then create label arrays for each previous list (spam or ham) where the ham label array (pos_labels) should be filled with ones and be the length of pos_emails, and the spam label array (neg_labels) should be filled with zeros and be the length of the neg_emails.

# In[6]:

def to_arrays(ham, spam):
    '''
    Converts ham and spam to arrays, and also creates two label arrays:
    one filled with zeros for spam and one filled with ones for ham. 
    
    Parameters
    ----------
    spam: A list of strings
    ham: A list of strings
    
    Returns
    -------
    A tuple of four arrays
    '''
    pos_emails=np.array(ham)
    neg_emails=np.array(spam)
    pos_labels = np.ones(pos_emails.shape[0])
    neg_labels = np.zeros(neg_emails.shape[0])
    
    
    return pos_emails, neg_emails, pos_labels, neg_labels


# In[7]:

pos_emails, neg_emails, pos_labels, neg_labels = to_arrays(ham, spam)

assert_is_instance(pos_emails, np.ndarray)
assert_is_instance(neg_emails, np.ndarray)
assert_is_instance(pos_labels, np.ndarray)
assert_is_instance(neg_labels, np.ndarray)

assert_array_equal(pos_emails, ham)
assert_array_equal(neg_emails, spam)

assert_array_equal(pos_labels, [1] * len(ham))
assert_array_equal(neg_labels, [0] * len(spam))

assert_true(pos_emails[0].startswith("Use the GUI and don't delete files, use the other option, whats it called"))
assert_true(neg_emails[60].startswith("RECIEVE ALL CHANNELS ON YOUR SATELLITE SYSTEM! 1-888-406-4246"))


# In[8]:

# Freeing up some memory
get_ipython().magic('xdel ham')
get_ipython().magic('xdel spam')


# ## Training and testing sets
# In order to perform some analysis on this data, we need to split and then contatenate the pos_emails and neg_emails together (and do the same for the pos_labels and neg_labels) so as to create a training and testing array for both X and y.  The "split" variable will tell you where to split your arrays.  For example, If split is 300, the training set will consist of the first 300 emails in pos_emails plus the first 300 emails in neg_emails, and the rest of the emails go into the test set.  The same will be true for the label arrays.

# In[9]:

def test_train(pos_emails, neg_emails, pos_labels, neg_labels, split):
    '''
    Splits the emails and labels into training and testing sets.    
    
    Parameters
    ----------
    pos_emails: A numpy array of strings
    neg_emails: A numpy array of strings
    pos_labels: A numpy array of ints or floats
    neg_labels: A numpy array of ints or floats
    split: an int 
    
    Returns
    -------
    A tuple of four numpy arrays: X_train, X_test, y_train, y_test.
    '''
    X_train = np.concatenate((pos_emails[:split], 
                          neg_emails[:split]), axis = 0)

    X_test = np.concatenate((pos_emails[split:],
                         neg_emails[split:]), axis = 0)

    y_train = np.concatenate((pos_labels[:split], 
                          neg_labels[:split]), axis = 0)

    y_test = np.concatenate((pos_labels[split:],
                         neg_labels[split:]), axis = 0)
    
    return X_train, X_test, y_train, y_test


# In[10]:

X_train, X_test, y_train, y_test = test_train(
    pos_emails, neg_emails, pos_labels, neg_labels, split=400
    )

assert_is_instance(X_train, np.ndarray)
assert_is_instance(X_test, np.ndarray)
assert_is_instance(y_train, np.ndarray)
assert_is_instance(y_test, np.ndarray)

assert_array_equal(X_train[:400], pos_emails[:400])
assert_array_equal(X_train[400:], neg_emails[:400])

assert_array_equal(X_test[:len(pos_emails) - 400], pos_emails[400:])
assert_array_equal(X_test[len(pos_emails) - 400:], neg_emails[400:])

assert_array_equal(y_train[:400], pos_labels[:400])
assert_array_equal(y_train[400:], neg_labels[:400])

assert_array_equal(y_test[:len(pos_labels) - 400], pos_labels[400:])
assert_array_equal(y_test[len(pos_labels) - 400:], neg_labels[400:])


# In[11]:

# Freeing up some more memory
get_ipython().magic('xdel pos_emails')
get_ipython().magic('xdel neg_emails')


# ## Spam classification
# 
# Finally, we will use our training and testing sets to identify spam correctly.
# - Use unigrams and bigrams,
# - Build a pipeline by using TfidfVectorizer and LinearSVC,
# - Name the first step tf and the second step svc,
# - Use default parameters for both TfidfVectorizer and LinearSVC, and
# - Use English stop words.

# In[12]:

def fit_and_predict(X_train, y_train, X_test, random_state):
    '''
    Creates a document term matrix and uses SVM classifier to make document classifications.
    Uses English stop words.
    
    Parameters
    ----------
    X_train: A list of strings.
    y_train: A list of strings.
    X_test: A list of strings.
    random_state: A np.random.RandomState instance.
    
    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A Pipeline instance.
    y_pred: A numpy array.    '''
    quacamole = [('tf', TfidfVectorizer()), ('svc',LinearSVC())]
    clf = Pipeline(quacamole)

    # Lowercase, bigrams, stop words.
    clf.set_params(tf__stop_words = 'english', 
                tf__ngram_range=(1,2), 
                tf__lowercase=True)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    
    return clf, y_pred


# In[13]:

clf, y_pred = fit_and_predict(X_train, y_train, X_test, random_state=check_random_state(0))
score = accuracy_score(y_test, y_pred)
print("SVC prediction accuracy = {0:5.1f}%".format(100.0 * score))

assert_is_instance(clf, Pipeline)
assert_is_instance(y_pred, np.ndarray)
tf = clf.named_steps['tf']
assert_is_instance(tf, TfidfVectorizer)
assert_is_instance(clf.named_steps['svc'], LinearSVC)
assert_equal(tf.ngram_range, (1, 2))
assert_equal(tf.stop_words, 'english')
assert_equal(len(y_pred), len(y_test))
assert_array_equal(y_pred[:10], [1] * 10)
assert_array_equal(y_pred[-10:], [0] * 10)
assert_almost_equal(score, 0.7277227722772277)


# In[ ]:



