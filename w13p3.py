
# coding: utf-8

# # Week 13 Problem 3
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
# # Problem 13.3. Neo4J
# 
# In this problem, we will persist a NetworkX graph in Neo4J and then make queries using [CQL](https://www.tutorialspoint.com/neo4j/neo4j_cql_introduction.htm).

# In[1]:

import networkx as nx
from py2neo import authenticate, Graph, Node, Relationship
from py2neo.database import cypher

from nose.tools import assert_equal, assert_true, assert_is_instance


# First, let's get connected to the Neo4J database. 
# In the following code cell, we read in the current user's netid to obtain a unique database name for this Notebook.
# If you are not able to get connected, you should post in the forum and email TAs immediately. Try not to wait until the last minute, since there might be a lot of traffic that makes the server down.

# In[2]:

# Filename containing user's netid
fname = '/home/data_scientist/users.txt'
with open(fname, 'r') as fin:
    netid = fin.readline().rstrip()

# We will delete our working directory if it exists before recreating.
dbname = '{0}'.format(netid)

host_ip = '141.142.211.60:7474'
username = 'neo4j'
password = 'Lcdm#info490'

# First we authenticate
authenticate(host_port=host_ip, user=username, password=password)

# Now create database URL
db_url = 'http://{0}/db/{1}'.format(host_ip, dbname)

print('Creating connection to {0}'.format(db_url))
graph = Graph(db_url)

version = graph.dbms.kernel_version
print('Neo4J Kernel version {0}.{1}.{2}'.format(version[0], version[1], version[2]))


# We use the social network of [Florentine Families](https://en.wikipedia.org/wiki/Category:Families_of_Florence) data set. For more information, see [Week 10 Problem 2](../Week10/assignments/w10p2.ipynb).

# In[3]:

florentine_families = nx.florentine_families_graph()


# ## Persisting Graphs
# 
# Write a funtion named `persist_graph` that:
# - Gets all nodes and edges from the NetworkX graph (`florentine_families`), and adds them to the Neo4J database,
# - Provides a label `"families"` to all nodes,
# - Provides a name using the node name read from the NetworkX graph to all nodes, and
# - Creates a relationship of `"tied to"` for all edges.

# In[4]:

def persist_graph(neo_graph, nx_graph):
    '''
    Persists a NetworkX graph in Neo4J.
    All nodes are labeled "families".
    All edges have connection type "tied to".
    
    Parameters
    ----------
    neo_graph: A py2neo.database.Graph instance.
    nx_graph: A networkx.Graph instance.
    '''
    
    
    gr = neo_graph.begin()
    nodes = []
    #Create list of nodes where the name is a string of an int
    for node in nx_graph.nodes():
        nd = Node('families', name = str(node))
        nodes.append(nd)
        gr.create(nd)
    #Create the relationship between two nodes for each edge    
    for edge in nx_graph.edges():
        for i in nodes:
            if i["name"] == edge[0]:
                node1 = i
            for j in nodes:
                if j["name"] == edge[1]:
                    node2 = j
        gr.create(Relationship(node1, 'tied to', node2))
    #Make the changes happen    
    gr.commit()
        
    return None


# In[5]:

# clean out graph database
graph.delete_all()


# In[6]:

# execute the function
persist_graph(graph, florentine_families)


# In[7]:

# do a query to display all nodes and relationships in the database
for result in graph.run('START n=node(*) MATCH (n)-[r]->(m) RETURN n,r,m;'):
    print(result)


# In[8]:

# test nodes
assert_true(all(isinstance(n['name'], str) for n in graph.find('families')))
node_names = [n['name'] for n in graph.find('families')]
assert_equal(len(node_names), len(florentine_families.nodes()))
assert_equal(set(node_names), set(florentine_families.nodes()))


# In[9]:

# test relationships
edges = [e for e in graph.match(rel_type='tied to')]
start_nodes = [e.start_node()['name'] for e in edges]
end_nodes = [e.end_node()['name'] for e in edges]

assert_equal(len(edges), len(florentine_families.edges()))
assert_equal(set(start_nodes), {e[0] for e in florentine_families.edges()})
assert_equal(set(end_nodes), {e[1] for e in florentine_families.edges()})


# ## Querying Graphs
# 
# Write a funtion named `query_graph` that returns a CQL query string. The CQL query does the following:
# - Finds the two nodes: `"Medici"` and `"Guadagni"`,
# - Creates a new relationship `"business friend of"` between the two nodes, using `"Medici"` as start node and `"Guadagni"` as end node, and
# - Returns the relationship record just created.

# In[10]:

def query_graph():
    '''
    Constructs a CQL string that makes a query to the Neo4J database.
    Finds nodes "Medici" and "Guadagni" and makes a new relationship 
      "business friend of" between these two nodes.
    
    Ruturns
    ----------
    cql: A string.
    '''
    #Creates new relationship "business friend of" between Medici and Guadagni
    cql = 'MATCH (a),(b)     WHERE a.name = "Medici" AND b.name = "Guadagni"     CREATE (a)‐[r:`business friend of`]‐>(b)     RETURN r'
    
    return cql


# In[11]:

# run the query to add the new relationship to the database
cql = query_graph()
for result in (graph.run(cql)):
    print(result)


# In[12]:

# do a query to display all nodes and relationships in the database
for result in graph.run('START n=node(*) MATCH (n)-[r]->(m) RETURN n,r,m;'):
    print(result)


# In[13]:

# tests
assert_equal(type(cql), str)

new_edge = [e for e in graph.match(rel_type='business friend of')]
new_edge_start = [e.start_node()['name'] for e in new_edge]
new_edge_end = [e.end_node()['name'] for e in new_edge]

assert_equal(len(new_edge), 1)
assert_equal(new_edge_start[0], 'Medici')
assert_equal(new_edge_end[0], 'Guadagni')


# ## Cleanup

# In[14]:

# clean out graph database
graph.delete_all()

