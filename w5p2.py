
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

# # Problem 5.2. Clustering.
# 
# In this problem, we will continue from where we left off in Problem 5.1, and apply k-means clustering algorithm on Delta Airline's aircrafts.

# In[1]:

get_ipython().magic('matplotlib inline')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn

from sklearn.utils import check_random_state
from sklearn.cluster import KMeans

from nose.tools import assert_equal, assert_is_instance, assert_true, assert_is_not
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_almost_equal


# I saved the `reduced` array (the first 10 principal components of the Delta Airline data set) from Problem 8.1 as a `npy` file.
# 
# ```python
# >>> np.save('delta_reduced.npy', reduced)
# ```
# 
# This file is in `/home/data_scientist/data/misc`. We will load this file as a Numpy array and start from there.

# In[2]:

reduced = np.load('/home/data_scientist/data/misc/delta_reduced.npy')


# # k-means
# 
# - Write a function named `cluster()` that fits a k-means clustering algorithm, and returns a tuple `(sklearn.cluster.k_means_.KMeans, np.array)`. The second element of the tuple is a 1-d array that contains the predictions of k-means clustering, i.e. which cluster each data point belongs to.
# 
# Use default values for all parameters in `KMeans()` execept for `n_clusters` and `random_state`.

# In[3]:

def cluster(array, random_state, n_clusters=4):
    '''
    Fits and predicts k-means clustering on "array"
    
    Parameters
    ----------
    array: A numpy array
    random_state: Random seed, e.g. check_random_state(0)
    n_clusters: The number of clusters. Default: 4
    
    Returns
    -------
    A tuple (sklearn.KMeans, np.ndarray)
    '''
    #Fits a k-means clustering algorithm to array
    model=KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters=model.fit_predict(array)
    
    return model, clusters


# In[4]:

k_means_t, cluster_t = cluster(reduced, random_state=check_random_state(1), n_clusters=5)

assert_is_instance(k_means_t, sklearn.cluster.k_means_.KMeans)
assert_is_instance(cluster_t, np.ndarray)
assert_equal(k_means_t.n_init, 10)
assert_equal(k_means_t.n_clusters, 5)
assert_equal(len(cluster_t), len(reduced))
assert_true((cluster_t < 5).all()) # n_cluster = 5 so labels should be between 0 and 5
assert_true((cluster_t >= 0).all())
labels_gold = -1. * np.ones(len(reduced), dtype=np.int)
mindist = np.empty(len(reduced))
mindist.fill(np.infty)
for i in range(5):
    dist = np.sum((reduced - k_means_t.cluster_centers_[i])**2., axis=1)
    labels_gold[dist < mindist] = i
    mindist = np.minimum(dist, mindist)
assert_true((mindist >= 0.0).all())
assert_true((labels_gold != -1).all())
assert_array_equal(labels_gold, cluster_t)


# ## The elbow method
# 
# Now, we would like to apply the k-means clustering technique, but how do we determine k, the number of clusters?
# 
# The simplest method is [the elbow method](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set#The_Elbow_Method). But what criterion should we use, i.e. what should go on the y-axis?
# 
# According to [scikit-learn documentation](http://scikit-learn.org/stable/modules/clustering.html#k-means),
# 
# ```
# The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance,
# minimizing a criterion known as the inertia or within-cluster sum-of-squares.
# ```
# 
# The scikit-learn documentation on [sklearn.cluster.KMeans](http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn-cluster-kmeans) says that `sklearn.cluster.KMeans` has the inertia value in the `inertia_` attribute. So we can vary the number of clusters in `KMeans`, plot `KMeans.inertia_` as a function of the number of clusters, and pick the "elbow" in the plot.
# 
# ![](./images/elbow.png)
# 
# Always use `check_random_state(0)` to seed the random number generator.

# In[5]:

def plot_inertia(array, start=1, end=10):
    '''
    Increase the number of clusters from "start" to "end" (inclusive).
    Finds the inertia of k-means clustering for different k.
    Plots inertia as a function of the number of clusters.

    
    Parameters
    ----------
    array: A numpy array.
    start: An int. Default: 1
    end: An int. Default: 10
    
    Returns
    -------
    A matplotlib.Axes instance.
    '''
    
    
    clusters=[]
    inertias=[]
    #Increments the number of clusters and calculates the new k
    for i in range(start,end+1):
        clusters.append(i)
        kmeans, _ =cluster(array, check_random_state(0), n_clusters=i)
        inertias.append(kmeans.inertia_)
    
    #Initializes the plot
    _, ax=plt.subplots()
    ax.set(xlabel='Number of clusters', ylabel='Inertia', title='The elbow method',
           xlim=(1,10), ylim=(0,1600)) 
    #Plots clusters vs inertias
    plt.plot(clusters, inertias)
    
    return ax


# In[6]:

inertia = plot_inertia(reduced)


# In[7]:

assert_is_instance(inertia, mpl.axes.Axes)
assert_true(len(inertia.lines) >= 1)

xdata, ydata = inertia.lines[0].get_xydata().T

for i in range(1, 11):
    k_means_t, cluster_t = cluster(reduced, random_state=check_random_state(0), n_clusters=i)
    assert_array_equal(xdata[i - 1], i)
    assert_almost_equal(ydata[i - 1], k_means_t.inertia_)

assert_is_not(len(inertia.title.get_text()), 0,
    msg="Your plot doesn't have a title.")
assert_is_not(inertia.xaxis.get_label_text(), '',
    msg="Change the x-axis label to something more descriptive.")
assert_is_not(inertia.yaxis.get_label_text(), '',
    msg="Change the y-axis label to something more descriptive.")


# ## Pair Grid
# 
# - Write a function named `plot_pair()` that uses [seaborn.PairGrid](http://stanford.edu/~mwaskom/software/seaborn/generated/seaborn.PairGrid.html#) to visualize the clusters in terms of first four principal components. The plots on the diagonal should be histograms of corresponding attributes, and the off-diagonal should be scatter plots.
# 
# ![](./images/pca_pair_plot.png)

# In[8]:

def plot_pair(reduced, clusters):
    '''
    Uses seaborn.PairGrid to visualize the data distribution
    when axes are the first four principal components.
    Diagonal plots are histograms. The off-diagonal plots are scatter plots.
    
    Parameters
    ----------
    reduced: A numpy array. Comes from importing delta_reduced.npy
    
    Returns
    -------
    A seaborn.axisgrid.PairGrid instance.
    '''
    #Initializes the new df
    df=reduced[:,0:4]
    df=pd.DataFrame(df, columns=['PCA1','PCA2','PCA3','PCA4'])
    df['clusters']=clusters
    
    #Creates a pairgrid plot of the new df
    ax=sns.PairGrid(df, hue='clusters', vars=['PCA1','PCA2','PCA3','PCA4'])
    ax.map_diag(plt.hist)
    ax.map_offdiag(plt.scatter)
    return ax


# In[9]:

k_means, clusters = cluster(reduced, random_state=check_random_state(0), n_clusters=4)
pg = plot_pair(reduced, clusters)


# We can see that the one outlier is in its own cluster, there’s 3 or 4 in the other and the remainder are split into two clusters of greater size.

# In[10]:

assert_is_instance(pg.fig, plt.Figure)
assert_true(len(pg.data.columns) >= 4)

for ax in pg.diag_axes:
    assert_equal(len(ax.patches), 4 * 10) # 4 clusters with 10 patches in each histogram

for i, j in zip(*np.triu_indices_from(pg.axes, 1)):
    ax = pg.axes[i, j]
    x_out, y_out = ax.collections[0].get_offsets().T
    x_in = reduced[clusters == 0, j] # we only check the first cluster
    y_in = reduced[clusters == 0, i]
    assert_array_equal(x_in, x_out)
    assert_array_equal(y_in, y_out)

for i, j in zip(*np.tril_indices_from(pg.axes, -1)):
    ax = pg.axes[i, j]
    x_in = reduced[clusters == 0, j]
    y_in = reduced[clusters == 0, i]
    x_out, y_out = ax.collections[0].get_offsets().T
    assert_array_equal(x_in, x_out)
    assert_array_equal(y_in, y_out)

for i, j in zip(*np.diag_indices_from(pg.axes)):
    ax = pg.axes[i, j]
    assert_equal(len(ax.collections), 0)


# ## More discussion
# 
# You don't have to write any code in this section, but here's one interpretaion of what we have done.
# 
# Let's take a closer look at each cluster.

# In[11]:

df = pd.read_csv('/home/data_scientist/data/delta.csv', index_col='Aircraft')
df['Clusters'] = clusters
df['Aircraft'] = df.index
df_grouped = df.groupby('Clusters').mean()
print(df_grouped.Accommodation)


# In[12]:

print(df_grouped['Length (ft)'])


# Cluster 3 has only one aircraft:

# In[13]:

clust3 = df[df.Clusters == 3]
print(clust3.Aircraft)


# Airbus A319 VIP is not one of Delta Airline's regular fleet and is one of Airbus corporate jets.
# 
# Cluster 2 has four aircrafts.

# In[14]:

clust2 = df[df.Clusters == 2]
print(clust2.Aircraft)


# These are small aircrafts and only have economy seats.

# In[15]:

cols_seat = ['First Class', 'Business', 'Eco Comfort', 'Economy']
print(df.loc[clust2.index, cols_seat])


# Next, we look at Cluster 1.

# In[16]:

clust1 = df[df.Clusters == 1]
print(clust1.Aircraft)


# These aircrafts do not have first class seating.

# In[17]:

print(df.loc[clust1.index, cols_seat])


# Finally, cluster 0 has the following aircrafts:

# In[18]:

clust0 = df[df.Clusters == 0]
print(clust0.Aircraft)


# The aircrafts in cluster 0 (except for one aircraft) have first class seating but no business class.

# In[19]:

print(df.loc[clust0.index, cols_seat])


# In[ ]:



