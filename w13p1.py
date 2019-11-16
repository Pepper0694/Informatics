
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

# ## Problem 13.1. MongoDB
# 
# In this problem, we work with MongoDB from a Python program by using the pymongo database driver.

# In[1]:

import os
import datetime
import json
import bson
import pymongo as pm

from nose.tools import assert_equal, assert_true, assert_is_instance


# Here, we will be using historical weather data from [Weather Underground](http://www.wunderground.com/) to create a database. This dataset will be from January 1, 2001, collected from O'Hare (KORD). To make life easier for you, I've imported the data for you below:

# In[2]:

fpath = '/home/data_scientist/data/weather'
fname = 'weather_kord_2001_0101.json'

with open(os.path.join(fpath, fname)) as f:
    weather_json = json.load(f)

assert_is_instance(weather_json, dict)
assert_equal(set(weather_json.keys()), set(['current_observation', 'response', 'history']))
assert_true('observations' in weather_json['history'])


# In[3]:

observations = weather_json['history']['observations']
print('There are {} dictionaries in the list.'.format(len(observations)))
print('The first element is\n{}'.format(observations[0]))

assert_is_instance(observations, list)
assert_true(all(isinstance(o, dict) for o in observations))


# We connect to the course MongoDB cloud computing system, hosted by NCSA's Nebula cloud.

# In[4]:

client = pm.MongoClient("mongodb://141.142.211.6:27017")


# Since we are using a shared resource without authentication, we use your netid to create a database for each student.

# In[5]:

# Filename containing user's netid
fname = '/home/data_scientist/users.txt'
with open(fname, 'r') as fin:
    netid = fin.readline().rstrip()

# We will delete our working directory if it exists before recreating.
dbname = 'assignment-{0}'.format(netid)

if dbname in client.database_names():
    client.drop_database(dbname)

print('Existing databases:', client.database_names())

assert_true(dbname not in client.database_names())


# ## Inserting Data
# 
# - Create a new collection using the name `collection_name` and add new documents `data` to our MongoDB collection
# - Return a list of object IDs as a validation of the insertion process.

# In[6]:

def insert_data(db, collection_name, data):
    '''
    Creates a new collection using the name "collection_name" 
    and adds new documents `data` to our MongoDB collection.
    
    Parameters
    ----------
    data: A list of dictionaries.
    db: A pymongo.database.Database instance.
    collection_name: Name of new MongoDB collection.
    
    Returns
    -------
    A list of bson.ObjectId
    '''
    
    coll = db[collection_name]
   
     # for each item, insert it into the db, and get the corrsponding inserted_id
    inserted_ids = list(map(lambda item: coll.insert_one(item).inserted_id, data)) 
    
    return inserted_ids


# In[7]:

inserted_ids = insert_data(client[dbname], '0101', observations)

print("New weather ID: ", inserted_ids)
print('Existing databases:', client.database_names())
print('Existing collections:', client[dbname].collection_names())


# In[8]:

assert_is_instance(inserted_ids, list)
assert_true(all(isinstance(i, bson.objectid.ObjectId) for i in inserted_ids))

assert_true(dbname in client.database_names())
assert_true('0101' in client[dbname].collection_names())
assert_equal(client[dbname]['0101'].count(), len(observations))


# ## Retrieving Data
# 
# - Find all documents that have a given weather `condition` (e.g., `conds == "Clear"` or `conds == "Partly Cloudy"`)
# - Return the `_id` values of all documents that match the search query.

# In[9]:

def retrieve_data(collection, condition):
    '''
    Finds all documents that have a given weather `condition`
    and return the `_id` values of all documents that match the search query.
    
    Parameters
    ----------
    collection: A pymongo.Collection instance.
    condition: A string, e.g., "Clear", "Partly Cloudy", "Overcast".
    
    Returns
    -------
    A list of bson.ObjectId
    '''
    
    
    #Get ids of all docs that match the given weather condition
    return [item['_id'] for item in collection.find({"conds": condition})]


# In[10]:

clear_ids = retrieve_data(client[dbname]['0101'], 'Clear')
print(clear_ids)


# In[11]:

assert_is_instance(clear_ids, list)
assert_true(all(isinstance(i, bson.objectid.ObjectId) for i in clear_ids))

conds = {obs['conds'] for obs in observations}
for cond in conds:
    r = retrieve_data(client[dbname]['0101'], cond)
    n = [obs['_id'] for obs in observations if obs['conds'] == cond]
    assert_equal(len(r), len(n))
    assert_equal(set(r), set(n))


# ## Modifying Data
# 
# - Find all documents whose `conds` value is `"Clear"` and change the `conds` attribute to `Cloudy`.
# - Return the number of documents modified as a validation of the process.

# In[12]:

def modify_data(collection):
    '''
    Finds all documents whose "conds" value is "Clear"
    and change the "conds" attribute to "Cloudy".

    Parameters
    ----------
    collection: A pymongo.Collection instance.
    
    Returns
    -------
    An int. The number of documents modified.
    '''
    #Finds clear for the weather condition
    count = len([item for item in collection.find({'conds':'Clear'})])
    #Change clear to cloudy
    collection.update_many({'conds':'Clear'}, {'$set':{'conds': 'Cloudy'}})
    
    return count


# In[13]:

n_modified = modify_data(client[dbname]['0101'])
print('{0} records modified.'.format(n_modified))


# In[14]:

assert_equal(
    n_modified,
    len([obs['_id'] for obs in observations if obs['conds'] == 'Clear'])
    )

conds = [obs['conds'] for obs in observations]
for cond in conds:
    if cond != 'Clear' and cond != 'Cloudy':
        r = retrieve_data(client[dbname]['0101'], cond)
        n = [obs['_id'] for obs in observations if obs['conds'] == cond]
        assert_equal(len(r), len(n))
        assert_equal(set(r), set(n))


# ## Advanced Querying
# 
# - Find all documents with `visi` equal to `"10.0"` and sort the documents by `conds`.
# - Return a list of `conds` as a validation of the process.

# In[15]:

def query(collection):
    '''
    Finds all documents with "visi" equal to `"10.0"
    and sort the documents by "conds".
    
    Parameters
    ----------
    collection: A pymongo.Collection instance.

    Returns
    -------
    A list of strings.
    '''
    #Finds all documents that have a visibility of 10 and sorts by condition
    return [item['conds'] for item in collection.find({"visi": {'$eq': '10.0'}}).sort('conds')]


# In[16]:

query_conds = query(client[dbname]['0101'])
print(query_conds)


# In[17]:

modified_conds = [obs['conds'] for obs in observations if obs['visi'] == '10.0']
modified_conds = ['Cloudy' if cond == 'Clear' else cond for cond in modified_conds]
modified_conds = sorted(modified_conds)
assert_equal(query_conds, modified_conds)


# ## Deleting Data
# 
# - Delete all documents whose `conds` attribute is equal to `"Cloudy"` from our collection.
# - Return the number of documents deleted as a validation of the process.

# In[18]:

def delete_data(collection):
    '''
    Deletes all documents whose "conds" == "Cloudy".
    
    Paramters
    ---------
    collection: A pymongo.Collection instance.

    Returns
    -------
    An int. The number of documents deleted.
    '''
    #Find the documents where condition is cloudy
    count = len([item for item in collection.find({'conds':'Cloudy'})])
    #Deletes them
    collection.delete_many({'conds': 'Cloudy'})
    
    return count


# In[19]:

n_deleted = delete_data(client[dbname]['0101'])
print('{0} records deleted.'.format(n_deleted))


# In[20]:

deleted_obs = [obs for obs in modified_conds if obs == 'Cloudy']
assert_equal(n_deleted, len(deleted_obs))

for cond in set(conds):
    if cond != 'Clear' and cond != 'Cloudy':
        r = retrieve_data(client[dbname]['0101'], cond)
        n = [obs['_id'] for obs in observations if obs['conds'] == cond]
        assert_equal(len(r), len(n))
        assert_equal(set(r), set(n))


# ## Cleanup
# 
# When you are done or if you want to start over with a clean database, run the following code cell.
# 
# PLEASE MAKE SURE TO RUN THIS BEFORE RESTARTING AND RUNNING YOUR CODE!!!

# In[1]:

if dbname in client.database_names():
    client.drop_database(dbname)
    
assert_true(dbname not in client.database_names())


# In[ ]:



