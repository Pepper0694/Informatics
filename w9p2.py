
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

# # Problem 9.2. NLP: Topic Modeling.
# 
# In this problem, we explore the concept of topic modeling.

# In[1]:

import numpy as np
import pandas as pd
import json

from scipy.sparse.csr import csr_matrix
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cross_validation import check_random_state
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from gensim.matutils import Sparse2Corpus
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel

from nose.tools import assert_equal, assert_is_instance, assert_true, assert_almost_equal
from numpy.testing import assert_array_equal, assert_array_almost_equal


# We will be using the reuters data (nltk.corpus.reuters). The X_train, X_test, y_train, and y_test have already been preprocessed and saved in JSON format, for convenience in the `/home/data_scientist/data/misc` directory. Using the code below, we will fetch them for use.

# In[2]:


def load_reuters(name):
    fpath = '/home/data_scientist/data/misc/reuters_{}.json'.format(name)
    with open(fpath) as f:
        reuters = json.load(f)
    return reuters

X_train, X_test, y_train, y_test = map(load_reuters, ['X_train', 'X_test', 'y_train', 'y_test'])


# ## Document term matrix
# 
# - Use TfidfVectorizer to create a document term matrix for both `X_train` and `X_test`.
# - Use English stop words.
# - Use unigrams and bigrams.
# - Ignore terms that have a document frequency strictly lower than 2.
# - Build a vocabulary that only consider the top 20,000 features ordered by term frequency across the corpus.

# In[3]:

def get_document_term_matrix(train_data, test_data):
    '''
    Uses TfidfVectorizer to create a document term matrix for "X_train" and "X_test".
    
    Paramters
    ---------
    train_data: A list of strings
    test_data:A list of strings
    
    Returns
    -------
    A 3-tuple of (model, train_matrix, test_matrix).
    model: A TfidfVectorizer instance
    train_matrix: A scipy.csr_matrix
    test_matrix: A scipy.csr_matrix
    '''
    #Create the TfidfVectorizer model with the parameters specified
    model = TfidfVectorizer(stop_words = 'english',
                     lowercase=True,
                     min_df=2,
                     max_features=20000,
                        ngram_range=(1,2))
    #Fit the model then transform it to a train and test matrix
    train_matrix = model.fit_transform(train_data)
    test_matrix = model.transform(test_data)
    return model, train_matrix, test_matrix


# In[4]:

cv, train_data, test_data = get_document_term_matrix(X_train, X_test)


# In[5]:

assert_is_instance(cv, TfidfVectorizer)
assert_is_instance(train_data, csr_matrix)
assert_is_instance(test_data, csr_matrix)
assert_equal(cv.stop_words, 'english')
assert_equal(cv.ngram_range, (1, 2))
assert_equal(cv.min_df, 2)
assert_equal(cv.max_features, 20000)
assert_equal(train_data.data.size, 588963)
assert_array_almost_equal(
    train_data.data[:5],
    [0.0375267,   0.0401517,   0.03477509,  0.0474274,   0.03217005]
    )
assert_equal(test_data.data.size, 210403)
assert_array_almost_equal(
    test_data.data[:5],
    [ 0.02399319,  0.04801429,  0.04859632,  0.0403796,   0.0403796]
    )


# ## Non-negative matrix factorization
# 
# - Apply non-negative matrix factorization (NMF) to compute topics in `train_data`.
# - Use 60 topics.
# - Normalize the transformed data to have unit probability.

# In[6]:

def apply_nmf(data, random_state):
    '''
    Applies non-negative matrix factorization (NMF) to compute topics.
    
    Parameters
    ----------
    data: A csr_matrix
    random_state: A RandomState instance for NMF
    
    Returns
    -------
    A tuple of (nmf, transformed_data)
    nmf: An sklearn.NMF instance
    transformed_data: A numpy.ndarray
    '''
    #Apply non-negative matrix factorization on train_data with specified parameters
    nmf = NMF(n_components = 60, max_iter=200, random_state=random_state).fit(data)
    transformed_data = nmf.transform(data)
    #Normalize the data
    transformed_data = normalize(transformed_data, norm='l1', axis=1)
    return nmf, transformed_data


# In[7]:

nmf, td_norm = apply_nmf(train_data, random_state=check_random_state(0))


# In[8]:

assert_is_instance(nmf, NMF)
assert_is_instance(td_norm, np.ndarray)
assert_equal(nmf.n_components, 60)
assert_equal(nmf.max_iter, 200)
assert_equal(td_norm.shape, (7769, 60))
assert_array_almost_equal(
    td_norm[0, :5],
    [0. ,         0.08515023,  0.01682892,  0.,          0.02451052]
    )
assert_array_almost_equal(
    td_norm[-1, -5:],
    [  0.,          0.,          0.,         0.00342309,  0.        ]
    )


# ## Topic-based Classification
# 
# - Train a LinearSVC classifier on the topics in the training data sample of the reuters data set.
# - Use default parameters for the LinearSVC classifier. Don't forget to set the `random_state` parameter.
# - Compute the topics, by using the previously created NMF model, for the test data and compute classifications from these topic models. 

# In[9]:

def classify_topics(nmf, X_train, y_train, X_test, random_state):
    '''
    
    Paramters
    ---------
    nmf: An sklearn.NMF model.
    X_train: A numpy array.
    y_train: A numpy array.
    X_test: A scipy csr_matrix.
    random_state: A RandomState instance for LinearSVC Classifier.
    
    Returns
    -------
    A tuple of (clf, y_pred)
    clf: A LinearSVC instance.
    y_pred: A numpy array.
    '''
    #Create model of LinearSVC with default parameters
    clf = LinearSVC(random_state=random_state)
    #Fit and predict
    clf.fit(X_train, y_train)
    y_pred = clf.predict(nmf.transform(X_test))
    
    return clf, y_pred


# In[10]:

clf, ts_preds = classify_topics(
    nmf, nmf.transform(train_data), y_train, test_data, check_random_state(0)
    )


# In[11]:

assert_is_instance(clf, LinearSVC)
assert_is_instance(ts_preds, np.ndarray)
assert_equal(len(ts_preds), len(y_test))
assert_array_equal(ts_preds[:5], ['trade', 'grain', 'crude', 'earn', 'crude'])
assert_array_equal(ts_preds[-5:], ['acq', 'dlr', 'crude', 'grain', 'acq'])


# ## Topic Modeling with Gensim
# 
# - Use the gensim library to perform topic modeling of the reuters data. First transform a sparse matrix into a gensim corpus, and then construct a vocabulary dictionary. Finally, create a  Latent Dirichlet allocation (LDA) model with 20 topics for the reuters text, and return 5 most significant words for each topic.
# - You should specify three parameters in `LdaModel()`: `corpus`, `id2word`, and `num_topics`. Use default values for all other paramters. Ignore any warnings about `passes` or `iterations`.

# In[12]:

def get_topics(cv, train_data):
    '''
    Uses gensim to perform topic modeling.
    
    Paramters
    ---------
    cv: A TfidfVectorizer instance.
    train_data: A scipy csr_matrix.
    
    Returns
    -------
    A list of strings (functions of the most important terms in each topic).
    '''
    #Create the gensim corpus from train data
    td_gensim = Sparse2Corpus(train_data)
    #Create vocab dictionary
    tmp_dct = dict((idv, word) for word, idv in cv.vocabulary_.items())
    dct = Dictionary.from_corpus(td_gensim, id2word=tmp_dct)
    #Create LDA model with specified parameters
    lda_gs = LdaModel(corpus=td_gensim, id2word=dct, num_topics=20)
    topics = lda_gs.top_topics(corpus=td_gensim, num_words=5)
    return topics


# In[13]:

topics = get_topics(cv, train_data)

for idx, (lst, val) in enumerate(topics):
    print('Topic {0}'.format(idx))
    print(35*('-'))
    for i, z in lst:
        print('    {0:20s}: {1:5.4f}'.format(z, i))
    print(35*('-'))


# In[14]:

assert_is_instance(topics, list)
assert_equal(len(topics), 20)

for topic, score in topics:
    assert_is_instance(topic, list)
    assert_is_instance(score, float)
    assert_equal(len(topic), 5)
    for v, k in topic:
        assert_is_instance(k, str)
        assert_is_instance(v, float)


# ## Pipeline of FeatureUnion and Logistic Regression
# 
# - Build a pipeline by using [FeatureUnion](http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html) of  and  `LinearSVC`. 
# - A FeatureUnion helps process data in parallel and can be considered pipelines themselves (check this [resource](http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html) for an overview on how FeatureUnion is used in NLP).
# - The first step of the pipeline should use the FeatureUnion with the name `features`. The first component of the FeatureUnion should contain the `CountVectorizer`, using the name `cv` , followed by the `TfidfVectorizer` using the name `tf`. 
# - The second step of the pipeline should be `LinearSVC` with the name `svc`.
# - Do not use stop words.

# In[17]:

def cv_tfidf_svc_pipe(X_train, y_train, X_test, random_state):
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
    y_pred: A numpy array.
    '''
    #Create the feature union for the pipeline
    union=[('cv', CountVectorizer()), ('tf',  TfidfVectorizer())]
    #Create the pipeline
    tools = [('features', FeatureUnion(union)), ('svc',  LinearSVC(random_state=random_state))]
    clf = Pipeline(tools)
    
    #Fit and predict based off of the model
    clf = clf.fit(X_train, y_train)
    predicted = clf.predict(X_test)
    return clf, predicted


# In[18]:

clf2, y_pred2 = cv_tfidf_svc_pipe(X_train, y_train, X_test, random_state=check_random_state(0))
score2 = accuracy_score(y_pred2, y_test)
print("SVC prediction accuracy = {0:3.1f}%".format(100.0 * score2))


# In[19]:

assert_is_instance(clf2, Pipeline)
assert_is_instance(y_pred2, np.ndarray)
assert_is_instance(clf2.named_steps['features'], FeatureUnion)
assert_is_instance(clf2.named_steps['svc'], LinearSVC)
assert_is_instance(clf2.named_steps['features'].transformer_list[0][1], CountVectorizer)
assert_is_instance(clf2.named_steps['features'].transformer_list[1][1], TfidfVectorizer)
assert_equal(len(y_pred2), len(y_test))
assert_array_equal(y_pred2[:5], ['trade', 'grain', 'crude', 'corn', 'palm-oil'])
assert_array_equal(y_pred2[-5:], ['acq', 'dlr', 'earn', 'ipi', 'gold'])
assert_almost_equal(score2, 0.884067572044, 3)


# In[ ]:



