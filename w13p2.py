
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

# # Problem 13.2. Cassandra
# 
# In this problem, we use the Cassandra Python database driver to execute CQL (Cassandra Query Language) queries.

# In[1]:

import os
import json
import cassandra
from cassandra.cluster import Cluster
from cassandra.policies import WhiteListRoundRobinPolicy
from cassandra.query import dict_factory
from cassandra.cqlengine import connection, management

from nose.tools import assert_equal, assert_true, assert_is_instance


# We use the historical weather data from [Weather Underground](http://www.wunderground.com/) to create a database.

# In[2]:

fpath = '/home/data_scientist/data/weather'
fname = 'weather_kord_2001_1231.json'

with open(os.path.join(fpath, fname)) as f:
    weather_json = json.load(f)


# For simplicity, we use only two attributes, `wspdi` and `wdire`.

# In[3]:

observations = weather_json['history']['observations']
observations = [{key: value for key, value in obs.items() if key in ['wdire', 'wspdi']} for obs in observations]

print('There are {} dictionaries in the list.'.format(len(observations)))
print('The first element is {}'.format(observations[0]))


# We use the course Cassandra server on 141.142.211.105  on the default port of 9042:

# In[4]:

cassandra_ips = ['141.142.211.105 ']

# Establish a connection to Cassandra

# The Policy is necessary to allow Cassandra to run on Azure.
pcy = WhiteListRoundRobinPolicy(cassandra_ips)

# Create Connection
cluster = Cluster(contact_points=cassandra_ips, load_balancing_policy=pcy)
session = cluster.connect()

print('Cluster Name: {0}'.format(cluster.metadata.cluster_name))
for host in cluster.metadata.all_hosts():
    print('{0}: Host: {1} in {2}'          .format(host.datacenter, host.address, host.rack))


# To provide distinct environments for each student, each student will create their own keyspace in the shared Cassandra cluster by using their netids.

# In[5]:

# Filename containing user's netid
fname = '/home/data_scientist/users.txt'
with open(fname, 'r') as fin:
    netid = fin.readline().rstrip()

# We will delete our working directory if it exists before recreating.
ks_name = '{0}'.format(netid)

session.row_factory = dict_factory

connection.set_session(session)

# Explicitly set session hosts, this removes annoying warnings.
connection.session.hosts = cassandra_ips

# Drop Keyspace if it exists
if ks_name in cluster.metadata.keyspaces:
    management.drop_keyspace(ks_name)

# Create Keyspace
management.create_keyspace_simple(ks_name, 1)

# Set keyspace for this session
# Note: If keyspace exists in Cassandra instance, this is only line we need.
session.set_keyspace(ks_name)

# Display all non-system keyspaces.
# Do not change to a different keyspace!

keys = [val for val in sorted(cluster.metadata.keyspaces.keys()) if 'system' not in val]
for ks in keys:
    print(ks)

print('\nCQL Query to recreate this keyspace:')
print(40*'-')
print(cluster.metadata.keyspaces[ks_name].export_as_string())


# We first drop the table if it exists to ensure a clean slate before we create our new schema and insert data. Note that the table name we will be using in this problem is `weather`.

# In[6]:

def drop_table(session):
    '''
    Drops "weather" table if exists.
    
    Parameters
    ----------
    session: A cassandra.cluster.Session instance.
    
    Returns
    -------
    A cassandra.cluster.ResultSet instance.
    '''
    result = session.execute('DROP TABLE IF EXISTS weather;')
    return result

result = drop_table(session)

assert_is_instance(result, cassandra.cluster.ResultSet)
assert_equal(result.column_names, None)
assert_equal(len(result.current_rows), 0)


# ## Creating Table
# 
# - Craete a `weather` table that has the following 4 columns:
#   - `id` (`INT`)
#   - `date` (`TEXT`)
#   - `wdire` (`TEXT`)
#   - `wspdi` (`DOUBLE`)
# - We will use the `WHERE` clause on both `wdire` and `id` later in this problem, so create an appropriate primary key.

# In[7]:

def create_table(session):
    '''
    Creates a "weather" table with four attributes:
    id, date, wdire, and wspdi.
    
    Parameters
    ----------
    session: A cassandra.cluster.Session instance.
    
    Returns
    -------
    A cassandra.cluster.ResultSet instance.
    '''
    #Create Keyspace
    management.create_keyspace_simple(ks_name, 1)
    session.set_keyspace(ks_name)
    #Defines the weather table
    create_schema = '''
CREATE TABLE weather (
    id int,
    date text,
    wdire text,
    wspdi double,
    PRIMARY KEY(id, wdire)
);
'''
    #Creates weather table
    result = session.execute(create_schema)
    
    return result


# In[8]:

create_result = create_table(session)
print(create_result.response_future)


# In[9]:

result = session.execute('SELECT * FROM weather;')
assert_is_instance(result, cassandra.cluster.ResultSet)
assert_equal(set(result.column_names), {'date', 'id', 'wdire', 'wspdi'})
assert_equal(len(result.current_rows), 0)


# ## Inserting Data
# 
# - Add the weather data `observations` to our Cassandra database.
# - The `date` column should be `"1231"` for all rows.
# - The `id` column should start from 1, and `id == 1` should correspond to the first element of `data`, `id == 2` to the second element, and so on.

# In[13]:

def insert_data(session, data):
    '''
    Adds new rows to Cassandra database.
    
    Parameters
    ----------
    session: A cassandra.cluster.Session instance.
    data: A list of dictionaries.
    
    Returns
    -------
    A cassandra.cluster.ResultSet instance.
    '''
    
    insert_many = '''
    INSERT INTO weather (id, date, wdire, wspdi) 
    VALUES (:id, :date, :wdire, :wspdi) ;
    '''

    prepare = session.prepare(insert_many)
    for index, item in enumerate(data):
        result = session.execute(prepare, ((index + 1), 
                                                  '1231', item['wdire'],
                                                  float(item['wspdi'])))
    
    return result


# In[14]:

insert_result = insert_data(session, observations)
print(insert_result.response_future)


# In[15]:

result = session.execute('SELECT * FROM weather;')
assert_is_instance(result, cassandra.cluster.ResultSet)
assert_equal(len(result.current_rows), len(observations))
assert_equal(
    {row['wdire'] for row in result.current_rows},
    {obs['wdire'] for obs in observations}
    )
assert_equal(
    {str(row['wspdi']) for row in result.current_rows},
    {obs['wspdi'] for obs in observations}
    )
assert_true(all(row['date'] == '1231') for row in result.current_rows)
assert_equal(
    {row['id'] for row in result.current_rows},
    set(range(1, len(observations) + 1))
    )


# ## Retrieving Data
# 
# - Retrieve the `id` attribute of all rows where the `wdire` attribute of the `weather` table is equal to `direction` (e.g., "WSW", "West", "WNW", etc.).

# In[16]:

def retrieve_data(session, direction):
    '''
    Retrieves the "id" attribute of all rows
    where the "wdire" attribute of the "weather" table
    is equal to "direction"
    
    Parameters
    ----------
    session: A cassandra.cluster.Session instance.
    direction: A string, e.g., "WSW", "West", "WNW", etc.
    
    Returns
    -------
    A cassandra.cluster.ResultSet instance.

    '''
    #Get ids of rows where wdire is direction direction
    query = '''
SELECT id
FROM  weather 
WHERE wdire = %(dir)s 
ALLOW FILTERING
'''
    result = session.execute(query, {'dir': direction})
    
    return result


# In[17]:

retrieve_result = retrieve_data(session, 'WSW')
print(retrieve_result.response_future)
print("\nRESULTS")
print(40*'-')
for row in retrieve_result:
    print('id: {0}'.format(row['id']))
    print(40*'-')


# In[18]:

assert_is_instance(result, cassandra.cluster.ResultSet)
wdire = {obs['wdire'] for obs in observations}
for dire in wdire:
    r = [row['id'] for row in retrieve_data(session, dire).current_rows]
    n = [idx + 1 for idx, obs in enumerate(observations) if obs['wdire'] == dire]
    assert_equal(len(r), len(n))
    assert_equal(set(r), set(n))


# ## Modifying Data
# 
# - Change the `wspdi` value to 1.0 for all rows where the `wdire` attribute is equal to `"WSW"`.

# In[21]:

def modify(session):
    '''
    Changes "wspdi" to 1.0 if "wdire" is equal to "WSW".
    
    Parameters
    ----------
    session: A cassandra.cluster.Session instance.
    
    Returns
    -------
    A cassandra.cluster.ResultSet instance.
    '''
    #Get table
    table = retrieve_data(session, direction='WSW')
    #Define how the table should be updated
    ud_stmt = '''
    UPDATE weather
      SET wspdi = 1.0
      WHERE wdire = 'WSW' AND id = %(id)s;
    '''
    #Update the table
    for item in table:
        results = session.execute(ud_stmt, {'id': item['id']})
    
    return result


# In[22]:

modify_result = modify(session)
print(modify_result.response_future)

display_query = session.execute("SELECT * FROM weather;")
print("\nRESULTS")
print(40*'-')
for row in display_query:
    if row['wdire'] == 'WSW':
        print('id: {0}'.format(row['id']))
        print('wdire: {0}'.format(row['wdire']))
        print('wspdi: {0}'.format(row['wspdi']))
        print(40*'-')


# In[23]:

result = session.execute('SELECT * FROM weather;')

assert_equal(
    len([row for row in result.current_rows if row['wdire'] == 'WSW']),
    len([obs for obs in observations if obs['wdire'] == 'WSW'])
    )

wdire = {obs['wdire'] for obs in observations}

for dire in wdire:
    if dire != 'WSW' and dire != 'WNW':
        r = [str(row['wspdi']) for row in result.current_rows if row['wdire'] == dire]
        n = [obs['wspdi'] for obs in observations if obs['wdire'] == dire]
        assert_equal(len(r), len(n))
        assert_equal(set(r), set(n))


# ## Deleting Data
# 
# - Delete all rows where the `wdire` attribute is equal to `"WSW"`.

# In[26]:

def delete(session):
    '''
    Delete all rows where "wdire" is "WSW".
    
    Parameters
    ----------
    session: A cassandra.cluster.Session instance.
    
    Returns
    -------
    A cassandra.cluster.ResultSet instance.
    '''
    #Get table
    table = retrieve_data(session, direction='WSW')
    #Condition for deleting rows
    ud_stmt = '''
    DELETE FROM weather
       WHERE wdire = 'WSW' AND id = %(id)s;
    '''
    #Apply the deletion condition
    for item in table:
        results = session.execute(ud_stmt, {'id': item['id']})
    
    
    return result


# In[27]:

count_query_1 = session.execute("SELECT COUNT(*) FROM weather WHERE wdire = 'WSW' ALLOW FILTERING")
print((40*'-'),'\nPre-Delete\n')
print('Records(WSW) count = {0}'.format(count_query_1[0]['count'])) 
print(40*'-')
delete_result = delete(session)
print(delete_result.response_future)
count_query_2 = session.execute("SELECT COUNT(*) FROM weather WHERE wdire = 'WSW' ALLOW FILTERING")
print((40*'-'),'\nPost-Delete\n')
print('Records(WSW) count = {0}'.format(count_query_2[0]['count']))


# In[28]:

result = session.execute('SELECT * FROM weather;')

assert_equal(len([row for row in result.current_rows if row['wdire'] == 'WSW']), 0)

observations = [obs for obs in observations if obs['wdire'] != 'WSW']

for dire in wdire:
    r = [str(row['wspdi']) for row in result.current_rows if row['wdire'] == dire]
    n = [obs['wspdi'] for obs in observations if obs['wdire'] == dire]
    assert_equal(len(r), len(n))
    assert_equal(set(r), set(n))


# In[ ]:



