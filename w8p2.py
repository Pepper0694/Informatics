
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

# # Problem 8.2. Social Media: Twitter
# 
# In this problem, we will use the twitter API to extract a set of tweets, and perform a sentiment analysis on twitter data to classify tweets as positive or negative.

# In[1]:

import numpy as np
import nltk
import tweepy as tw

from sklearn.utils import check_random_state
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from nose.tools import assert_equal, assert_is_instance, assert_true, assert_almost_equal
from numpy.testing import assert_array_equal


# ## Twitter Application
# 
# - Create a Twitter Application, and save the credentials into a file at `/home/data_scientist/twitter.cred`. 
# 
# `twitter.cred` must have the following four credentials in order on separate lines:
# 
# ```
# Access Token
# Access Token Secret
# Consumer Key
# Consumer Secret
# ```
# 
# Once you have stored your credientials, run the following code cells (you don't have to write any code in this section) and check if you are able to use the API.

# In[2]:

def connect_twitter_api(cred_file):
    
    # Order: Access Token, Access Token Secret, Consumer Key, Consumer SecretAccess
    with open(cred_file) as fin:
        tokens = [line.rstrip('\n') for line in fin if not line.startswith('#')]

    auth = tw.OAuthHandler(tokens[2], tokens[3])
    auth.set_access_token(tokens[0], tokens[1])

    return tw.API(auth)


# In[3]:

# Do NOT change file path or name of twitter.cred
api = connect_twitter_api('/home/data_scientist/twitter.cred')
assert_equal(api.get_user('katyperry').screen_name, 'katyperry')
assert_equal(api.get_user('justinbieber').created_at.strftime('%Y %m %d %H %M %S'), '2009 03 28 16 41 22')
assert_equal(api.get_user('BarackObama').name, 'Barack Obama')


# ## Positive and negative tweets
# 
# We will first train a model on the NLTK twitter corpus, and use it to classify a set of tweets fetched from the Twitter API.

# In[4]:

from nltk.corpus import twitter_samples as tws


# - Write a function that creates a training set from the NLTK twitter corpus. Positive tweets are in `positive_tweets.json`, while negative tweets are in `negative_tweets.json`. The `data` and `targets` ararys should be one-dimensional numpy arrays, where the first half are the positive tweets and the second half are the negative tweets. Every positive tweets should be assigned a numerical label of 1 in `targets`, and negative tweets 0.

# In[5]:

def get_pos_neg_tweets(corpus):
    '''
    Creates a training set from twitter_samples corpus.
    
    Parameters
    ----------
    corpus: The nltk.corpus.twitter_samples corpus.
    
    Returns
    -------
    A tuple of (data, targets)
    
    '''
    #create training set NLTK from twitter corpus. Separates positive and
    #negative tweets
    wrong_doug = np.array(corpus.strings('positive_tweets.json'))
    julia_gulia = np.array(corpus.strings('negative_tweets.json'))

    moo_moo_mr_cow = np.ones(wrong_doug.shape[0])
    power_wagon = np.zeros(julia_gulia.shape[0])

    targets = np.concatenate((moo_moo_mr_cow, power_wagon), axis=0)
    data = np.concatenate((wrong_doug, julia_gulia), axis = 0)
    return data, targets


# In[6]:

data, targets = get_pos_neg_tweets(tws)


# In[7]:

assert_is_instance(data, np.ndarray)
assert_is_instance(targets, np.ndarray)
assert_equal(len(data), 10000)
assert_equal(len(targets), 10000)
assert_array_equal(
    data[:5],
    [
        '#FollowFriday @France_Inte @PKuchly57 @Milipol_Paris for being top '
            'engaged members in my community this week :)',
        '@Lamb2ja Hey James! How odd :/ Please call our Contact Centre on 02392441234'
            ' and we will be able to assist you :) Many thanks!',
        '@DespiteOfficial we had a listen last night :) As You Bleed is an amazing '
            'track. When are you in Scotland?!',
        '@97sides CONGRATS :)',
        'yeaaaah yippppy!!!  my accnt verified rqst has succeed got a blue tick mark '
            'on my fb profile :) in 15 days'
        ]
    )
assert_array_equal(
    data[5000:5005],
    [
        'hopeless for tmr :(',
        "Everything in the kids section of IKEA is so cute. Shame I'm nearly 19 in "
            "2 months :(",
        '@Hegelbon That heart sliding into the waste basket. :(',
        '“@ketchBurning: I hate Japanese call him "bani" :( :(”\n\nMe too',
        'Dang starting next week I have "work" :('
        ]
    )
assert_array_equal(targets[:5000], [1] * 5000)
assert_array_equal(targets[5000:], [0] * 5000)


# ## Training
# 
# We train on 80% of the data, and test the performance on the remaining 20%.

# In[8]:

X_train, X_test, y_train, y_test = train_test_split(
    data, targets, test_size=0.2, random_state=check_random_state(0)
    )


# - Use unigrams, bigrams, and trigrams.
# - Build a pipeline by using TfidfVectorizer and RandomForestClassifier,
# - Name the first step tf and the second step rf,
# - Use default parameters for both TfidfVectorizer and RandomForestClassifier, and
# - Use English stop words,

# In[9]:

def train(X_train, y_train, X_test, random_state):
    '''
    Creates a document term matrix and uses Random Forest classifier to make document classifications.
    
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
    y_pred: A numpy array.
    '''
    #Fit and predict off of clf
    mini_masterpiece = [('tf', TfidfVectorizer(stop_words='english', ngram_range=(1,3))), 
             ('rf', RandomForestClassifier(random_state=random_state))]
    clf = Pipeline(mini_masterpiece)

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return clf, y_pred


# In[10]:

clf, y_pred = train(X_train, y_train, X_test, random_state=check_random_state(0))
score = accuracy_score(y_pred, y_test)
print("RF prediction accuracy = {0:5.1f}%".format(100.0 * score))


# In[11]:

assert_is_instance(clf, Pipeline)
assert_is_instance(y_pred, np.ndarray)
tf = clf.named_steps['tf']
assert_is_instance(tf, TfidfVectorizer)
assert_is_instance(clf.named_steps['rf'], RandomForestClassifier)
assert_equal(tf.stop_words, 'english')
assert_equal(tf.ngram_range, (1, 3))
assert_equal(len(y_pred), len(y_test))
assert_array_equal(y_pred[:10], [0, 1, 1, 0, 1, 0, 0, 0, 1, 1])
assert_array_equal(y_pred[-10:], [0, 0, 1, 1, 1, 0, 0, 0, 1, 0])
assert_almost_equal(score, 0.723)


# ## User timeline
# 
# - Use Tweepy's [user_timeline()](http://docs.tweepy.org/en/latest/api.html#API.user_timeline) to extract 20 tweets from a specified user. Specify the `since_id` parameter for reproducibility.

# In[12]:

def get_timeline(user, since_id, max_id):
    '''
    Fetches 20 tweets from "user".
    
    Parameters
    ----------
    user: A string. The ID or screen name of a Twitter user.
    since_id: An int. Returns only statuses with an ID greater than (that is, more recent than) the specified ID.
    max_id: An int. Returns only statuses with an ID less than (that is, older than) or equal to the specified ID..
    
    Returns
    -------
    A list of integers.
    '''
    #Get tweets from a user
    timeline = api.user_timeline(user, since_id=since_id, max_id = max_id)
    return timeline


# In[13]:

timeline1 = get_timeline('TheDemocrats', since_id =735495811841298432  ,max_id=  837326424117817346 )


# In[14]:

timeline2 = get_timeline('GOP', since_id=  734118726795042817, max_id =834928725183586308)


# In[15]:

assert_equal(
    [t.id for t in timeline1],
    [837326424117817346,
    837320785794564096,
    837100935457488901,
    837072132290949120,
    837052049317588994,
    837037094686031873,
    837011510446682113,
    836990720640749569,
    836966617632342016,
    836803847762886656,
    836801033133260800,
    836785330455998464,
    836779989068558336,
    836777446821281792,
    836771651694039042,
    836769183375458306,
    836768477943836673,
    836768098275495936,
    836767367107645441,
    836766862566387712]
    )
assert_equal(
    [t.id for t in timeline2],
    [834928725183586308,
    834898073121861635,
    834896078055014401,
    834856145974153216,
    834851413746413568,
    834807922995712000,
    834790862815125504,
    834779274175512576,
    834756446608879618,
    834531775883911168,
    834513243087523841,
    834497491517263872,
    834205062805196800,
    834058475785351168,
    833738748894535680,
    833699482852327425,
    833682386953056256,
    833489605559275520,
    833371582366175233,
    833368197160136706])


# ## Make predictions
# 
# - Use the RandomForestClassifier to classify each tweet in timeline as a positive tweet or a negative tweet.

# In[16]:

def predict(clf, timeline):
    '''
    Uses a classifier ("clf") to classify each tweet in
    "timeline" as a positive tweet or a negative tweet.
    
    Parameters
    ----------
    clf: A Pipeline instance.
    timeline: A tweepy.models.ResultSet instance.
    
    Returns
    -------
    A numpy array.
    '''
    #Classify tweets in a timeline as positive with RFC
    kids_livewell=[]
    for tweet in timeline:
        kids_livewell.append(tweet.text)

    
    y_pred = clf.predict(kids_livewell)
    
    return y_pred


# In[17]:

pred1 = predict(clf, timeline1)
print('{} has {} positive tweets and {} negative tweets.'.format(
    'The Democrats account', (pred1 == 1).sum(), (pred1 == 0).sum()))


# In[18]:

assert_is_instance(pred1, np.ndarray)
assert_array_equal(
     pred1,
     [0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  1.,  
      0.,  0.,  0.,  0.,  0.,  0.,  0.,  0., 1.,  0.]
    )


# In[19]:

pred2 = predict(clf, timeline2)
print('{} has {} positive tweets and {} negative tweets.'.format(
    'The GOP account', (pred2 == 1).sum(), (pred2 == 0).sum()))


# In[20]:

assert_is_instance(pred2, np.ndarray)
assert_array_equal(
     pred2,
     [1.,  1.,  1.,  1.,  0. , 1.,  0.,  0.,  1.,  0.,  
      1.,  1.,  1.,  1.,  1.,  1.,  1.,  0., 0.,  1.]
    )

