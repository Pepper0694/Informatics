
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

# ## Problem 12.2. MapReduce.
# 
# In this problem, we will use Hadoop Streaming to execute a MapReduce code written in Python.

# In[31]:

import os
from nose.tools import assert_equal, assert_true


# We will use the [airline on-time performance data](http://stat-computing.org/dataexpo/2009/), but before proceeding, recall that the data set is encoded in `latin-1`. However, the Python 3 interpreter expects the standard input and output to be in `utf-8` encoding. Thus, we have to explicitly state that the Python interpreter should use `latin-1` for all IO operations, which we can do by setting the Python environment variable `PYTHONIOENCODING` equal to `latin-1`. We can set the environment variables of the current IPython kernel by modifying the `os.environ` dictionary.

# In[32]:

os.environ['PYTHONIOENCODING'] = 'latin-1'


# Let's use the shell to check if the variable is set correctly. If you are not familiar with the following syntax (i.e., Python variable = ! shell command), [this notebook](https://github.com/UI-DataScience/info490-fa15/blob/master/Week4/assignment/unix_ipython.ipynb) from the previous semester might be useful.

# In[33]:

python_io_encoding = get_ipython().getoutput('echo $PYTHONIOENCODING')
assert_equal(python_io_encoding.s, 'latin-1')


# ## Mapper
# 
# Write a Python script that
#   - Reads data from `STDIN`,
#   - Skips the first line (The first line of `2001.csv` is the header that has the column titles.)
#   - Outputs to `STDOUT` the `Origin` and `AirTime` columns separated with a tab.

# In[34]:

get_ipython().run_cell_magic('writefile', 'mapper.py', '#!/usr/bin/env python3\n\nimport sys\n#read data from stdin\nwith sys.stdin as fin:\n    # skip header\n    next(fin)  \n    with sys.stdout as fout:\n        #Output origin and airtime with tab delimiter\n        for line in fin:        \n            line = line.strip()\n            words = line.split(\',\')\n            fout.write("{0}\\t{1}\\n".format(words[16], words[13]))')


# We need make the file executable.

# In[35]:

get_ipython().system(' chmod u+x mapper.py')


# Before testing the mapper code on the entire data set, let's first create a small file and test our code on this small data set.

# In[36]:

get_ipython().system(' head -n 50 $HOME/data/2001.csv > 2001.csv.head')
map_out_head = get_ipython().getoutput('./mapper.py < 2001.csv.head')
print('\n'.join(map_out_head))


# In[37]:

assert_equal(
    map_out_head,
    ['BWI\t60','BWI\t64','BWI\t80','BWI\t66','BWI\t62','BWI\t61',
     'BWI\t61','BWI\t60','BWI\t52','BWI\t62','BWI\t62','BWI\t55',
     'BWI\t60','BWI\t61','BWI\t63','PHL\t53','PHL\t54','PHL\t55',
     'PHL\t53','PHL\t50','PHL\tNA','PHL\t57','PHL\t48','PHL\t56',
     'PHL\t55','PHL\t55','PHL\t55','PHL\t55','PHL\t49','PHL\t75',
     'PHL\t49','PHL\t50','PHL\t49','PHL\tNA','PHL\t46','PHL\tNA',
     'PHL\t51','PHL\t53','PHL\t52','PHL\t52','PHL\t54','PHL\t56',
     'PHL\t55','PHL\t51','PHL\t49','PHL\t49','CLT\t82','CLT\t82',
     'CLT\t78']
    )


# ## Reducer
# 
# Write a Python script that
# 
#   - Reads key-value pairs from `STDIN`,
#   - Computes the minimum and maximum air time for flights, with respect to each origin airport,
#   - Outputs to `STDOUT` the airports and the minimum and maximum air time for flights at each airport, separated with tabs.
#   
# For example,
# 
# ```shell
# $ ./mapper.py < 2001.csv.head | sort -n -k 1 | ./reducer.py
# ```
# 
# should give
# 
# ```
# BWI	52	80
# CLT	78	82
# PHL	46	75
# ```

# In[38]:

get_ipython().run_cell_magic('writefile', 'reducer.py', '#!/usr/bin/env python3\n\nimport sys\n#Read key value pairs from stdin\nwith sys.stdin as fin:\n    with sys.stdout as fout:\n        current_word = None\n        current_min = None\n        current_max = None\n        word = None\n        for line in fin:\n            \n            word = line.split(\'\\t\')[0]\n            airt = line.split(\'\\t\')[1]\n            if airt != "NA\\n":\n                airt = int(airt)\n                #Compute min and max airtimes\n                if current_word == None:\n                    current_min = current_max = airt\n                    current_word = word\n                elif word == current_word:\n                    current_min = min(current_min, airt)\n                    current_max = max(current_max, airt)\n                else:\n                    fout.write(\'%s\\t%d\\t%d\\n\' % (current_word, current_min, current_max))\n                    current_min = current_max = airt\n                    current_word = word\n                    #output to stdout the max/min for flights in that airport\n        else:\n            if current_word == word:\n                fout.write(\'%s\\t%d\\t%d\\n\' % (current_word, current_min, current_max))')


# In[39]:

get_ipython().system(' chmod u+x reducer.py')


# In[40]:

red_head_out = get_ipython().getoutput('./mapper.py < 2001.csv.head | sort -n -k 1 | ./reducer.py')
print('\n'.join(red_head_out))


# In[41]:

assert_equal(red_head_out, ['BWI\t52\t80','CLT\t78\t82','PHL\t46\t75'])


# If the previous tests on the smaller data set were successful, we can run the mapreduce on the entire data set.

# In[42]:

mapred_out = get_ipython().getoutput('./mapper.py < $HOME/data/2001.csv | sort -n -k 1 | ./reducer.py')
print('\n'.join(mapred_out[:10]))


# In[43]:

assert_equal(len(mapred_out), 231)
assert_equal(mapred_out[:5], ['ABE\t16\t180', 'ABI\t28\t85', 'ABQ\t15\t264', 'ACT\t19\t81', 'ACY\t33\t33'])
assert_equal(mapred_out[-5:], ['TYS\t11\t177', 'VPS\t28\t123', 'WRG\t5\t38', 'XNA\t33\t195', 'YAK\t28\t72'])


# ## HDFS: Reset
# 
# We will do some cleaning up before we run Hadoop streaming. Let's first stop the [namenode and datanodes](https://hadoop.apache.org/docs/r1.2.1/hdfs_design.html).

# In[44]:

get_ipython().system(' $HADOOP_PREFIX/sbin/stop-dfs.sh')
get_ipython().system(' $HADOOP_PREFIX/sbin/stop-yarn.sh')


# If there are any temporary files created during the previous Hadoop operation, we want to clean them up.

# In[45]:

get_ipython().system(' rm -rf /tmp/*')


# We will simply [format the namenode](https://wiki.apache.org/hadoop/GettingStartedWithHadoop#Formatting_the_Namenode) and delete all files in our HDFS. Note that our HDFS is in an ephemeral Docker container, so all data will be lost anyway when the Docker container is shut down.

# In[46]:

get_ipython().system(' echo "Y" | $HADOOP_PREFIX/bin/hdfs namenode -format 2> /dev/null')


# After formatting the namenode, we restart the namenode and datanodes.

# In[47]:

get_ipython().system('$HADOOP_PREFIX/etc/hadoop/hadoop-env.sh')
get_ipython().system('$HADOOP_PREFIX/sbin/start-dfs.sh')
get_ipython().system('$HADOOP_PREFIX/sbin/start-yarn.sh')


# Sometimes when the namenode is restarted, it enteres Safe Mode, not allowing any changes to the file system. We do want to make changes, so we manually leave Safe Mode.

# In[48]:

get_ipython().system(' $HADOOP_PREFIX/bin/hdfs dfsadmin -safemode leave')


# ## HDFS: Create directory
# 
# - Create a new directory in HDFS at `/user/data_scientist`.

# In[49]:

get_ipython().system('$HADOOP_PREFIX/bin/hdfs dfs -mkdir -p /user/data_scientist')
#Create new directory in data_scientist


# In[50]:

ls_user = get_ipython().getoutput('$HADOOP_PREFIX/bin/hdfs dfs -ls /user/')
print('\n'.join(ls_user))


# In[51]:

assert_true('/user/data_scientist' in ls_user.s)


# - Create a new directory in HDFS at `/user/data_scientist/wc/in`

# In[52]:

# Create a new directory in HDFS at `/user/data_scientist/wc/in`

get_ipython().system('$HADOOP_PREFIX/bin/hdfs dfs -mkdir -p /user/data_scientist/wc/in')


# In[53]:

ls_wc = get_ipython().getoutput('$HADOOP_PREFIX/bin/hdfs dfs -ls wc')
print('\n'.join(ls_wc))


# In[54]:

assert_true('wc/in' in ls_wc.s)


# ## HDFS: Copy
# 
# - Copy `/home/data_scientist/data/2001.csv` from local file system into our new HDFS directory `wc/in`.

# In[55]:

# Copy `/home/data_scientist/data/2001.csv` from local file system into our new HDFS directory `wc/in`.

get_ipython().system('$HADOOP_PREFIX/bin/hdfs dfs -put /home/data_scientist/data/2001.csv /user/data_scientist/wc/in/2001.csv')


# In[56]:

ls_wc_in = get_ipython().getoutput('$HADOOP_PREFIX/bin/hdfs dfs -ls wc/in')
print('\n'.join(ls_wc_in))


# In[57]:

assert_true('wc/in/2001.csv' in ls_wc_in.s)


# ## Python Hadoop Streaming
# 
# - Run `mapper.py` and `reducer.py` via Hadoop Streaming.
# - Use `/usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar`.
# - We need to pass the `PYTHONIOENCODING` environment variable to our Hadoop streaming task. To find out how to set `PYTHONIOENCODING` to `latin-1` in a Hadoop streaming task, use the `--help` and `-info` options.

# In[58]:

get_ipython().run_cell_magic('bash', '', '$HADOOP_PREFIX/bin/hdfs dfs -rm -r -f wc/out\n\n#This kept crashing the server everytime I ran it. I think tooo many people are trying to run it    \n$HADOOP_PREFIX/bin/hadoop jar /usr/local/hadoop/share/hadoop/tools/lib/hadoop-streaming-2.7.2.jar \\\n    -files mapper.py,reducer.py \\\n    -input wc/in \\\n    -output wc/out -mapper mapper.py -reducer reducer.py \\\n    -cmdenv PYTHONIOENCODING=latin-1')


# In[ ]:

ls_wc_out = get_ipython().getoutput('$HADOOP_PREFIX/bin/hdfs dfs -ls wc/out')
print('\n'.join(ls_wc_out))


# In[ ]:

assert_true('wc/out/_SUCCESS' in ls_wc_out.s)
assert_true('wc/out/part-00000' in ls_wc_out.s)


# In[ ]:

stream_out = get_ipython().getoutput('$HADOOP_PREFIX/bin/hdfs dfs -cat wc/out/part-00000')
print('\n'.join(stream_out[:10]))


# In[ ]:

assert_equal(mapred_out, stream_out)


# ## Cleanup

# In[ ]:

get_ipython().system(' $HADOOP_PREFIX/bin/hdfs dfs -rm -r -f -skipTrash wc/out')

