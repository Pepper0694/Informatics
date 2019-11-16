
# coding: utf-8

# # Week 12 Problem 3
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
# # Problem 12.3. Apache Pig
# 
# In this problem, we will run Pig to compute the average rating for each book in the book-crossing data set.

# In[16]:

from nose.tools import assert_equal, assert_almost_equal


# ---
# ### Raw Data Preview
# 
# First, let's have a look at the data in case you don't remember them from w6p1 assignment:

# In[17]:

get_ipython().system('head -5 $HOME/data/book-crossing/BX-Book-Ratings.csv')


# In[18]:

get_ipython().system('head -5 $HOME/data/book-crossing/BX-Books.csv')


# ---
# ### Data Preprocessing
# To make the messy data easier to be processed later, the bash script here does the following:
# - Removes the header line;
# - Removes the quotation marks for data in each field (otherwise there might be problems with numbers);
# - Cut the last three columns of `BX-Books.csv`, which are image urls and publishers that we don't need for this problem;
# - Saves the output as `ratings.csv`, `books.csv` to the current directory (same directory as this notebook);
# - Displays the first 10 lines of each output csv file.
# 
# The columns in the processed file is:
# - Columns of ratings.csv: **User-ID; ISBN; Book-Rating**
# - Columns of books.csv: **ISBN; Book-Title; Book-Author; Year-Of-Publication**
# 

# In[12]:

get_ipython().run_cell_magic('bash', '', '#Removes header/quotations\n#Cuts last 3 columns\n#Saves output to ratings.csv, books.csv\nsed \'s/"//g\' $HOME/data/book-crossing/BX-Book-Ratings.csv | sed \'1d\' > ratings.csv\nsed \'s/"//g\' $HOME/data/book-crossing/BX-Books.csv | cut -d\';\' -f -4 | sed \'1d\' > books.csv\n\necho\necho \'***** Ratings File *****\'\nhead ratings.csv\n\necho\necho \'***** Books File *****\'\nhead books.csv')


# -----
# ### Pig Latin: Average
# 
# Write a Pig script that:
# 
# - Imports `ratings.csv` and `books.csv` (note that these two files are seperated by semicolon),
# - Groups all reviews by ISBN and uses [AVG](https://pig.apache.org/docs/r0.7.0/piglatin_ref2.html#AVG) to compute the average rating for each book,
# - Joins the averaged rating dataset and the book dataset on the ISBN column, 
# - Sorts the joined dataset by book title using default ascending string order, and
# - Uses the DUMP command to display the first 10 rows.
# 
# The resulting schema should contain six columns:
# 
# ```
# (ISBN and average rating from calculated ratings.csv, ISBN, book title, book author, publish year from books.csv)
# ```
# 
# For example, the second line should be (the first line is harder to read since its title has commas):
# 
# ```
# (0964147726,0.0,0964147726, Always Have Popsicles,Rebecca Harvin,1994)
# ```
# 
# Some hints for debugging:
# 
# - Don't rush to the end; do and check one step at a time.
# - Use operations that display output wisely, e.g. DESCRIBE, ILLUSTRATE.
# - Before you use DUMP, make sure that you are trying to display a small number of rows, instead of all rows at a time. Otherwise your notebook might crash.
# - Take advantage of the debugging cell provided before the assertion cell.

# In[33]:

get_ipython().run_cell_magic('writefile', 'average.pig', "#Groups reviews by ISBN using AVG for the average rating of each book\nratings = LOAD 'ratings.csv' USING PigStorage(';')\n    AS (userID:chararray, movieID:int, rating:double) ;\n    \nbooks = LOAD 'books.csv' USING PigStorage(';')\n    AS (ISBN:chararray, title:chararray, genre:chararray, year:int) ;\n#Group by averaged rating on ISBN\ntemp = GROUP ratings BY bkISBN;\n\naverage_ratings = FOREACH temp GENERATE group AS ISBN, AVG(ratings.rating);\n\n\nresult = JOIN average_ratings by bkISBN, books by ISBN;\n\ntop_rows = LIMIT result 10;\nDUMP top_rows;")


# In[34]:

average_ratings = get_ipython().getoutput('pig -x local -f average.pig 2> pig_stderr.log')
print('\n'.join(average_ratings))


# To debug, uncomment and run the following code cell.

# In[31]:

#!cat pig_stderr.log


# ----
# ### Tests

# In[32]:

answer = [
    '(0590567330,2.25,0590567330, A Light in the Storm: The Civil War Diary of Amelia Martin, Fenwick Island, Delaware, 1861 (Dear America),Karen Hesse,1999)',
    '(0964147726,0.0,0964147726, Always Have Popsicles,Rebecca Harvin,1994)',
    '(0942320093,0.0,0942320093, Apple Magic (The Collector\'s series),Martina Boudreau,1984)',
    '(0310232546,8.0,0310232546, Ask Lily (Young Women of Faith: Lily Series, Book 5),Nancy N. Rue,2001)',
    '(0962295701,0.0,0962295701, Beyond IBM: Leadership Marketing and Finance for the 1990s,Lou Mobley,1989)',
    '(0439188970,0.0,0439188970, Clifford Visita El Hospital (Clifford El Gran Perro Colorado),Norman Bridwell,2000)',
    '(0399151788,10.0,0399151788, Dark Justice,Jack Higgins,2004)',
    '(0786000015,0.0,0786000015, Deceived,Carla Simpson,1994)',
    '(006250746X,5.0,006250746X, Earth Prayers From around the World: 365 Prayers, Poems, and Invocations for Honoring the Earth,Elizabeth Roberts,1991)',
    '(1566869250,5.0,1566869250, Final Fantasy Anthology: Official Strategy Guide (Brady Games),David Cassady,1999)'
    ]

a1 = [a.split(',') for a in answer]
a2 = [a.split(',') for a in average_ratings]

for irow, row in enumerate(answer):
    for icol in [0, 2, 3]:
        assert_equal(a1[irow][icol], a2[irow][icol])
    #float numbers in column 1
    assert_almost_equal(float(a1[irow][1]), float(a2[irow][1]))
    if irow in [0, 3, 8]: 
        continue
    for icol in [4, 5]:
        assert_equal(a1[irow][icol], a2[irow][icol])


# ---
# ### Cleanup

# In[ ]:

get_ipython().run_cell_magic('bash', '', '# Remove pig log files\nrm -f pig*.log\n\n# Remove our pig scripts\nrm -f *.pig\n\n# Remove csv files\nrm books.csv ratings.csv')

