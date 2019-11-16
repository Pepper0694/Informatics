
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
# # Problem 6.1. Recommender Systems.
# 
# For this problem, we will be creating a single user recommender system that will consider unfavorable as well as favorable ratings with the help of the [Book-Crossing Dataset](http://www2.informatik.uni-freiburg.de/~cziegler/BX/).  The Book-Crossing Dataset was created by Cai-Nicolas Ziegler in 2004 and used data from the [Book-Crossing Website](http://www.bookcrossing.com/).  These datafiles contain information about the books in circulation, ratings that users have given the books, and information on the users themselves.
# 
# As usual, we first import the modules needed, and we will also be creating a sandbox directory within our current directory to hold our datafiles for the Book-Crossing Dataset.

# In[1]:

import numpy as np
import numpy.ma as ma

import pandas as pd

# We do this to ignore several specific Pandas warnings
import warnings
warnings.filterwarnings("ignore")
import os

from nose.tools import assert_equal, assert_almost_equal

data_dir = '/home/data_scientist/data/book-crossing'


# ## Preparing the data
# Now that the data has been downloaded, I will translate the CSV files into Pandas Dataframes.

# In[2]:

ratings_file = os.path.join(data_dir, 'BX-Book-Ratings.csv')
books_file = os.path.join(data_dir, 'BX-Books.csv')
users_file = os.path.join(data_dir, 'BX-Users.csv')

ratings = pd.read_csv(ratings_file, sep=';', encoding = 'latin1')
books = pd.read_csv(books_file, sep=';', error_bad_lines = False, encoding = 'latin1')
users = pd.read_csv(users_file, sep=';', encoding = 'latin1')


# Since the dataset is so large, we can only look at about half of the ratings.  Now, let's look at the structure of the dataframes so that we may gain some intuition about how we can best use them.

# In[3]:

ratings = ratings[0:500000]
print(len(ratings))
ratings.head()


# In[4]:

books.tail()


# ## Making the Book/Rating Dataframe
# In order to find what rating a certain user gave a movie, we should combine the books and ratings dataframe on their common column (ISBN) in order to create a new table.    

# In[5]:

def combine(bks, rtgs):
    '''
    Combines the books and ratings dataframes on their common column: their
    ISBN numbers.  We then return the newly combined dataframe.
    
    Parameters
    ----------
    bks: A pandas.Dataframe
    rtgs: A pandas.Dataframe
    
    Returns
    -------
    A pandas.Dataframe
    '''
    #Combines the two df's on the column ISBN
    bk_rt=bks.merge(rtgs, on='ISBN')
    return bk_rt


# In[6]:

bk_rating = combine(books, ratings)
assert_equal(bk_rating['Book-Title'].value_counts().head().tolist(),[1088, 548, 407, 361, 349])
assert_equal(list(bk_rating), ['ISBN','Book-Title', 'Book-Author',                               'Year-Of-Publication', 'Publisher',                               'Image-URL-S', 'Image-URL-M', 'Image-URL-L',                               'User-ID', 'Book-Rating'])
assert_equal(bk_rating['Book-Title'].value_counts().head().index.tolist(),             ['Wild Animus', 'The Lovely Bones: A Novel', 'The Da Vinci Code',              'A Painted House', 'The Nanny Diaries: A Novel'])


# ## Pivot Table
# Now that we have our books-ratings dataframe, we now need to create a pivot table which will allow us to compare users' recommendations so that we might find book recommendations.  In order to create a manageable pivot table, we will limit our list of books to only those with 150 ratings or more. We will say that if a book was given a rating of over 5, then the rating is favorable, but if it is 5 or less, then the rating should be considered unfavorable.  Favorable ratings will be assigned the value of one. Additionally, your function should allow you to assign unfavorable ratings the value of either zero or negative one.  For example, if zeroto1 is True, then you should scale your ratings from 0 to 1; however if it is False, then you should scale your ratings to -1 to 1   You will then return a numpy array that will serve as the comparison array for the recommender system below.

# In[7]:

def pivot(rtngs, rating_count = 150, zeroto1=True):
    '''
    Takes the ratings dataframe and reduces it by removing books with less than the rating_count.
    It then makes a pivot table containing ratings indexed by user id and ISBN, which is returned.  Finally, the ratings are
    then converted into a matrix of 0 and 1 or -1 and 1, depending on the value of zeroto1.
    
    Parameters
    ----------
    rtgs: A pandas.Dataframe
    rating_count: An integer
    zeroto1: A Boolean
    
    Returns
    -------
    A Numpy array and a Pandas Dataframe
    
    
    '''
    #Groups rtngs by ISBN and counts row with the same ISBN
    grouped_rtngs=rtngs.groupby('ISBN')['ISBN'].transform('count')
    #Cuts rtngs to only those with a count greater than 150
    rtngs=rtngs.iloc[grouped_rtngs[grouped_rtngs> rating_count].index]
    
    #Creates pivot for some reason pivot_table crashed my notebook everytime
    pivot_df=rtngs.pivot(values='Book-Rating', index='User-ID', 
                            columns='ISBN') 
    
    if zeroto1==True:
        #Normalizes
        ratings_matrix= pivot_df.applymap(lambda x: 1 if x>5 else 0).as_matrix()
    else:
        #Scales to [-1,1]
        ratings_matrix= pivot_df.applymap(lambda x: 1 if x>5 else -1).as_matrix()
    
    return ratings_matrix, pivot_df


# In[8]:

zero_ratings_matrix, pivot_df_zero = pivot(ratings)
assert_equal(type(np.array([])), type(zero_ratings_matrix))
assert_equal(1, np.max(zero_ratings_matrix))
assert_equal(0, np.min(zero_ratings_matrix))
assert_equal((6393, 64), zero_ratings_matrix.shape)

ratings_matrix, pivot_df = pivot(ratings, zeroto1=False)
assert_equal(1, np.max(ratings_matrix))
assert_equal(-1, np.min(ratings_matrix))
assert_equal((6393, 64), ratings_matrix.shape)
assert_equal(type(np.array([])), type(ratings_matrix))


# As we get to the actual recommendations, we will rely heavily the cosine_similarity in order to make our preditions.  The code for it is below and does not need modification.

# In[9]:

def cosine_similarity(u, v):
    return(np.dot(u, v)/np.sqrt((np.dot(u, u) * np.dot(v, v))))


# Below, I have made a user which is guaranteed to have a match in the ratings matrix.  

# In[10]:

x = ratings_matrix
y = np.zeros(ratings_matrix.shape[1], dtype= np.int32)
book = np.where(x[200,:] == 1)[0]
y[4] = 1
y[36] = 1
y[44] = 1
y[30] = -1
y[20] = 1

pivot_df.tmp_idx = np.array(range(x.shape[0]))


# ## Single user Recommendations
# Finally, let us find recommendations for a single user given our new user.  In this function, we will find the index of the existing user with the most similar likes and dislikes, as well as the cosine similarity between this existing user and the new user, y.  You should use cosine_similarity to find the most similar existing user.  Additionally, you should be outputting the mask array, which will help us later in finding the books that should be recommended.

# In[11]:

def similar_user(x, y):
    '''
    Takes the array of user ratings as well as a new user and outputs the most similar user's
    index in the x array which can be used to find the userID of the most similar user. Should
    also output the cosine_similarity of the new user and the most similar user, as well as the 
    mask array.
    
    Parameters
    ----------
    x: Numpy array
    y: Numpy array
    
    Returns
    -------
    idx: integer
    cos: float
    bk_vec: Numpy array
    '''
    #Creates an array of similar users
    sims = np.apply_along_axis(cosine_similarity, 1, x, y)
    #Computes the best user
    cos = np.nanmax(sims)

    # Finds index of the best matching user
    idx = np.where(sims==cos)[0][0]
    # Now we subtract the vectors
    # (any negative value is a movie to recommend)
    bk_vec = y - x[idx]

    # We want a mask aray, so we zero out any recommended movie.
    bk_vec[bk_vec >= 0] = 1
    bk_vec[bk_vec < 0] = 0
    
    return idx, cos, bk_vec


# In[12]:

id, cos, bk_vec = similar_user(x, y)
assert_equal(64, len(bk_vec))
assert_almost_equal(0.167705098312, cos)
assert_equal(11676, pivot_df[pivot_df.tmp_idx == id].index[0])


# ## List of Recommendations
# Now that we have created all of this frame work, we need to finally find the list of recommended books for the books with ratings that ranged from -1 to 1.  You should do this with the assistance of your masked array, your pivot dataframe, your books dataframe and the index of the most similar user.

# In[13]:

def find_books(pivot_df, idx, books, bk_vec):
    '''
    Uses the inputs to create a list of books that are recommended to the new user.
    
    Parameters
    ----------
    pivot_df: A pandas Dataframe
    idx: An integer
    books: A pandas Dataframe
    bk_vec: A numpy Array
    
    Returns
    -------
    bk_ls: A list
    
    '''
    
    # Get the columns (ISBNs) for the current user
    pivot_ISBNs = pivot_df[pivot_df.tmp_idx == idx].columns
    
    # Now make a masked array to find ISBNs to recommend
    ma_bk_idx = ma.array(pivot_ISBNs, mask = bk_vec)
    bk_idx = ma_bk_idx[~ma_bk_idx.mask]
    
    # Creates DataFrame of the ISBNs of interest
    bk_df = books.ix[books.ISBN.isin(bk_idx)].dropna()
    #Creates list of book titles that match the ISBNs
    bk_ls = bk_df['Book-Title'].values.tolist()
    return bk_ls


# In[14]:

book_list = find_books(pivot_df, id, books, bk_vec)
assert_equal(33, len(book_list))
assert_equal("Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))", book_list[23])
assert_equal('Confessions of a Shopaholic (Summer Display Opportunity)', book_list[15])
assert_equal(['The Testament', 'Wild Animus', 'The Street Lawyer', 'The Five People You Meet in Heaven', 'A Painted House', 'The Perfect Storm : A True Story of Men Against the Sea', 'Empire Falls', 'The Red Tent (Bestselling Backlist)', 'The Nanny Diaries: A Novel', 'Life of Pi', "Where the Heart Is (Oprah's Book Club (Paperback))", 'The Da Vinci Code', 'Me Talk Pretty One Day', 'SHIPPING NEWS', 'Jurassic Park', 'Confessions of a Shopaholic (Summer Display Opportunity)', 'A Prayer for Owen Meany', 'Good in Bed', 'Summer Sisters', 'The Reader', 'The Brethren', '1st to Die: A Novel', 'Snow Falling on Cedars', "Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))", "Tuesdays with Morrie: An Old Man, a Young Man, and Life's Greatest Lesson", 'Interview with the Vampire', 'The Notebook', 'The Nanny Diaries: A Novel', 'Midwives: A Novel', 'The Divine Secrets of the Ya-Ya Sisterhood: A Novel', 'Harry Potter and the Chamber of Secrets (Book 2)', 'The Bridges of Madison County', 'House of Sand and Fog'], book_list)


# In[ ]:




# In[ ]:



