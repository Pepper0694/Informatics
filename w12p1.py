
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

# # Problem 12.1. Intro to Hadoop.
# In this problem set, you will be doing simple exercises using Hadoop.  Before you start, however, you should be aware of the following: __you MUST delete YOUR CODE HERE in order for your code to work (comments beginning with # are NOT kosher for command-line statements!!)__. 
# 
# When you comment your code for this assignment, please make the comments either above or below any command-line statements (lines starting with !).

# In[18]:

get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from nose.tools import assert_equal, assert_true, assert_is_instance
from numpy.testing import assert_array_almost_equal, assert_almost_equal


# First, we make sure that namenodes and datanodes are stopped, formatted and started, and make sure to get rid of any spurious files that might gunk up our system.

# In[19]:

get_ipython().system('$HADOOP_PREFIX/sbin/stop-dfs.sh')
get_ipython().system('$HADOOP_PREFIX/sbin/stop-yarn.sh')
get_ipython().system('rm -rf /tmp/*')
get_ipython().system('echo "Y" | $HADOOP_PREFIX/bin/hdfs namenode -format 2> /dev/null')
get_ipython().system('$HADOOP_PREFIX/etc/hadoop/hadoop-env.sh')
get_ipython().system('$HADOOP_PREFIX/sbin/start-dfs.sh')
get_ipython().system('$HADOOP_PREFIX/sbin/start-yarn.sh')
get_ipython().system('$HADOOP_PREFIX/bin/hdfs dfsadmin -safemode leave')
get_ipython().system('$HADOOP_PREFIX/bin/hdfs dfs -mkdir -p /user/$NB_USER')


# ## Part 1: Exploring files and the system
# First, let's start out by listing the contents of the directory `/user/` in HDFS.  When you do this, you will pipe the output into a file called temp1.txt so that you may pass the assertion tests below (the easiest way to do this piping is with the `>temp1.txt` statement after your command-line statement). 

# In[20]:

get_ipython().system('$HADOOP_PREFIX/bin/hdfs dfs -ls / >temp1.txt')
#get contents and pipe it to temp1


# In[21]:

res1 = get_ipython().getoutput('cat temp1.txt')

assert_is_instance(res1, list)
assert_is_instance(res1[0], str)
assert_is_instance(res1[1], str)
assert_true(res1[1], "Found 1 items")
assert_true(res1[1][:40], "drwxr-xr-x   - data_scientist supergroup")


# ## Part 2 Free space: 
# Now, let's issue a Hadoop command that allows us to see the free space available to us, making sure to make it human readable. Like before, you will pipe the output into a file called temp2.txt so that you may pass the assertion tests below (this piping can be done by putting `>temp2.txt` after your command-line statement). 

# In[22]:

get_ipython().system('$HADOOP_PREFIX/bin/hdfs dfs -df -h >temp2.txt')
#pipe free space info to temp2


# In[23]:

res2 = get_ipython().getoutput('cat temp2.txt')
assert_is_instance(res2, list)
assert_is_instance(res2[0], str)
assert_is_instance(res2[1], str)
assert_true(len(res2), 2)
assert_true(res2[0], "Filesystem                                             Size  Used  Available  Use%")
assert_true(res2[1][:46], "hdfs://info490rb.studentspace.cs.illinois.edu:")


# ## Part 3: Version
# Next, let's get the version information of Hadoop that we are running, making sure to pipe the output into the vers.txt file provided.

# In[24]:

get_ipython().system('$HADOOP_PREFIX/bin/hdfs version >vers.txt')
#Pipe version of hadoop to vers.txt


# In[25]:

vers = get_ipython().getoutput('cat vers.txt')
assert_true(all(isinstance(w, str) for w in vers))
assert_true(vers[0], 'Hadoop 2.7.2')
assert_true(vers[3], 'Compiled with protoc 2.5.0')
assert_true(len(vers), 6)


# ## Cleaning up files
# Run this cell before restarting and rerunning your code!

# In[26]:

get_ipython().system('rm temp1.txt')
get_ipython().system('rm temp2.txt')
get_ipython().system('rm vers.txt')


# ## New directory for Hadoop
# Here, I'm creating a new directory for Hadoop so that we are ready for the next two coding parts.

# In[27]:

get_ipython().run_cell_magic('bash', '', '#!/usr/bin/env bash\n\nDIR=$HOME/hadoop_assign\n\n# Delete if exists\nif [ -d "$DIR" ]; then\n    rm -rf "$DIR"\nfi\n\n# Now make the directory\nmkdir "$DIR"\n\nls -la $DIR')


# ## Part 4: Copying a book into a directory
# For these final two coding sections, we will be dealing with the script for Monty Python and the Holy Grail.
# 
# For this section, you must copy the file grail.txt from here:  
# 
# `/home/data_scientist/data/nltk_data/corpora/webtext/` 
# 
# into your hadoop_assign directory that you just created in above your $HOME directory. Please use `cp` and do not use Hadoop commands here or else you might fail the assertion tests.

# In[28]:

cp /home/data_scientist/data/nltk_data/corpora/webtext/grail.txt $HOME/hadoop_assign/
#compy grail.txt to hadoop_assign


# In[29]:

get_ipython().system('ls $HOME/hadoop_assign >copy.txt')
copy = get_ipython().getoutput('cat copy.txt')
assert_is_instance(copy[0], str)
assert_is_instance(copy, list)
assert_true(len(copy), 1)
assert_true(copy[0], 'grail.txt')


# ## Part 5: Making a new directory and copying a book in Hadoop
# Finally, we will do two things: we will create a new directory called `grail/in` using Hadoop, and then we will put grail.txt (located in `$HOME/hadoop_assign/`) into `grail/in`, once again using Hadoop.

# In[30]:

get_ipython().system('$HADOOP_PREFIX/bin/hdfs dfs -mkdir -p grail/in')
get_ipython().system('$HADOOP_PREFIX/bin/hdfs dfs -put $HOME/hadoop_assign/grail.txt grail/in')
#Create new directory grail/in and put grail.txt into it


# In[31]:

had_grail = get_ipython().getoutput('$HADOOP_PREFIX/bin/hdfs dfs -count -h grail/in/*')
had_grail = had_grail[0].split()
assert_is_instance(had_grail, list)
assert_true(all(isinstance(w, str) for w in had_grail))
assert_true(had_grail, ['0', '1', '63.5', 'K', 'grail/in/grail.txt'])


# ### Clean up 
# Please run this before you restart and run your assignment!

# In[32]:

get_ipython().system('rm copy.txt')
get_ipython().system('$HADOOP_PREFIX/bin/hdfs dfs -rm -r -f grail')
get_ipython().system('rm -rf $HOME/hadoop_assign')
get_ipython().system('$HADOOP_PREFIX/sbin/stop-dfs.sh')
get_ipython().system('$HADOOP_PREFIX/sbin/stop-yarn.sh')


# In[ ]:




# In[ ]:



