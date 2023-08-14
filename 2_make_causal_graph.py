'''
Script for running the PC algorithm on alerts data
Can be run as a nohup instead of the notebook
'''


import random
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import csv
from datetime import datetime, timedelta
from causallearn.utils.cit import chisq
from causallearn.search.ConstraintBased import PC
from copy import deepcopy
import networkx as nx
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn import feature_selection      ## mutual information
import pickle
import os
import shutil



df_all_alerts_parsed = pd.read_csv('./Alerts_data/all_alerts_parsed.csv', index_col=0)
df_all_alerts_parsed['time_r'] = (pd.to_datetime(df_all_alerts_parsed['created_at']).round('15T')).astype(str)
no_of_alerts = len(df_all_alerts_parsed['time_r'])

alert_map = pd.read_csv('./Alerts_data/alert_id_mapping.csv', index_col=0)
unique_alerts = alert_map['id'].to_numpy()

'''
# In[5]:


alert_map


# In[8]:


df_all_alerts_parsed = df_all_alerts_parsed.sort_values(by='created_at')


# In[33]:


df_all_alerts_parsed['created_at']


# In[35]:


## dts contains 15 minute interval timestamps for times correspodning to the alert data we have
def datetime_range(start, end, delta):
    current = start
    while current < end:
        yield current
        current += delta

dts = dict()
for i,dt in enumerate(datetime_range(datetime(2020, 2, 26, 12), datetime(2022, 7, 20, 12), timedelta(minutes=15))):
    dts[dt.strftime('%Y-%m-%d %H:%M:00+00:00')] = i


# In[36]:


time_series = np.zeros((len(dts), len(unique_alerts)))          ## quite efficient implementation now
time_series = np.int_(time_series)

for j in range(no_of_alerts):
    i = df_all_alerts_parsed['alert_id'].iloc[j]
    time = df_all_alerts_parsed['time_r'].iloc[j]
    time_series[dts[time],i] = 1


# In[11]:


### Not ingnoring all zero rows completely, keeping a percentage of zero rows. 


# In[37]:


new_time_series = []
for i in range(time_series.shape[0]):
    sumi = 0
    for j in range(time_series.shape[1]):
        sumi += time_series[i, j]
    if sumi != 0:
        new_time_series.append(list(time_series[i,:]))
    else:
        if random.uniform(0,1) < 0.05 :             ## 5% of zero rows on average
            new_time_series.append(list(time_series[i,:]))


# In[38]:


## new_time_series is a time series 2d array consisting alerts time series data, having all non-zero rows along with a percentage(5) of zero rows
new_time_series = np.array(new_time_series)


# In[39]:


print(new_time_series.shape)


# In[40]:


pd.DataFrame(new_time_series).to_csv("./causal_graph_data/binary_series_" + str(len(unique_alerts)) +"_filtered.csv")


# In[41]:


pd.DataFrame(new_time_series)
'''

# #### PC algo

# In[60]:


df_binary_series = pd.read_csv("./causal_graph_data/binary_series_" + str(len(unique_alerts)) +"_filtered.csv", index_col=0) # also reads the index column this way

df_binary_series_reduced = df_binary_series.drop('cso_exists', axis=1)

df_binary_series_numpy = df_binary_series_reduced.to_numpy()



# In[68]:


# to get this causal graph, run the lines below, to build_again, pass True
def build_cg(build_again = False):
    global cg
    if build_again:
        cg = PC.pc(df_binary_series_numpy, 0.05, chisq, verbose=False, show_progress=True)
        with open('./causal_graph_data/cg_binary_pc_filtered.cg', 'wb') as f:
            pickle.dump(cg, f)
    else:
        f = open('./causal_graph_data/cg_binary_pc_filtered.cg', 'rb')
        cg = pickle.load(f)

build_cg(True)


# In[ ]:


cg.to_nx_graph() ## edges of G are as wanted, undirected edges taken twice in both the directions


# #### Adding nodes to CG

# #### Adding weights to the edges

# Using p-score to add the weights, test - ssr based chi2 test
# 
# For later uses, adding a list of (lag,p-score) pairs
# 
# Since, time intervals are of 15 minutes, assuming lags to be confined within an hour, so maxlag = 10 which corresponds to lag of 150 minutes should be sufficient. Also since dependencis between alerts might be different, taking a universal lag doesn't make sense.
# 
# 
# 
# Since this is the description of the Granger Causality
# ![image.png](attachment:f9fc083e-6793-46f2-bd5d-d7ba87b2f84e.png)
# 
# In causal graph, arrow from a->b taking it as a causes b

# In[ ]:


alert_id_mapping = pd.read_csv('./Alerts_data/alert_id_mapping.csv')
alert_id_dict = {}        ## key: id value: name
for i,alert in enumerate(alert_id_mapping['alert_name'].values):
    alert_id_dict[i] = alert


# In[ ]:


## for export creates a causal graph with unweighted edges
def create_cg_nx(g, max_lag):
    G = nx.DiGraph()
    kwargs = {}
    kwargs['color'] = 'cyan'
    kwargs['type'] = 'alert'
    for node in list(g.nx_graph.nodes()):
        kwargs['label'] = alert_id_dict[node]
        G.add_node(node, **kwargs)
    color_dict = {'b':'blue', 'g': 'green', 'r': 'red'}
    kwargs = {}
    kwargs['label'] = 'causes'
    for edge in list(g.nx_graph.edges(data = True)):
        kwargs['color'] = color_dict[edge[2]['color']]
        G.add_edge(edge[0], edge[1], **kwargs)
    return G


# In[ ]:


## for export creates a causal graph with weighted edges, weight - mutual information
def create_cg_nx_mutual_info(g, max_lag):
    G = nx.DiGraph()
    kwargs = {}
    kwargs['color'] = 'cyan'
    kwargs['type'] = 'alert'
    for node in list(g.nx_graph.nodes()):
        kwargs['label'] = alert_id_dict[node]
        G.add_node(node, **kwargs)
    color_dict = {'b':'blue', 'g': 'green', 'r': 'red'}
    kwargs = {}
    kwargs['label'] = 'causes'
    for edge in list(g.nx_graph.edges(data = True)):
        kwargs['color'] = color_dict[edge[2]['color']]
        lst = []
        mi = feature_selection.mutual_info_classif(np.reshape(df_binary_series_numpy[:, edge[0]], (-1,1)), df_binary_series_numpy[:, edge[1]])
        lst.append((0, mi[0]))
        for k in range(1, max_lag + 1):
            mi = feature_selection.mutual_info_classif(np.reshape(df_binary_series_numpy[:-k, edge[0]], (-1,1)), df_binary_series_numpy[k:, edge[1]])
            lst.append((k, mi[0]))
        # result = grangercausalitytests(df_binary_series_numpy[:,[edge[1],edge[0]]],10,True, False)
        # for i in result:
        #     lst.append((i-1 , 1 - result[i][0]['ssr_ftest'][1]))
        kwargs['weights'] = lst
        G.add_edge(edge[0], edge[1], **kwargs)
    return G


# In[ ]:


## for export creates a causal graph with weighted edges, weight - p value for ftest
def create_cg_nx_ftest(g, max_lag):
    G = nx.DiGraph()
    kwargs = {}
    kwargs['color'] = 'cyan'
    kwargs['type'] = 'alert'
    for node in list(g.nx_graph.nodes()):
        kwargs['label'] = alert_id_dict[node]
        G.add_node(node, **kwargs)
    color_dict = {'b':'blue', 'g': 'green', 'r': 'red'}
    kwargs = {}
    kwargs['label'] = 'causes'
    for edge in list(g.nx_graph.edges(data = True)):
        kwargs['color'] = color_dict[edge[2]['color']]
        lst = []
        result = grangercausalitytests(df_binary_series_numpy[:,[edge[1],edge[0]]],10,True, False)
        for i in result:
            lst.append((i-1 , 1 - result[i][0]['ssr_ftest'][1]))
        kwargs['weights'] = lst
        G.add_edge(edge[0], edge[1], **kwargs)
    return G


# In[ ]:


## for export creates a causal graph with weighted edges, weight - p value for chi2test
def create_cg_nx_chi2test(g, max_lag):
    G = nx.DiGraph()
    kwargs = {}
    kwargs['color'] = 'cyan'
    kwargs['type'] = 'alert'
    for node in list(g.nx_graph.nodes()):
        kwargs['label'] = alert_id_dict[node]
        G.add_node(node, **kwargs)
    color_dict = {'b':'blue', 'g': 'green', 'r': 'red'}
    kwargs = {}
    kwargs['label'] = 'causes'
    for edge in list(g.nx_graph.edges(data = True)):
        kwargs['color'] = color_dict[edge[2]['color']]
        lst = []
        result = grangercausalitytests(df_binary_series_numpy[:,[edge[1],edge[0]]],10,True, False)
        for i in result:
            lst.append((i-1 , 1 - result[i][0]['ssr_chi2test'][1]))
        kwargs['weights'] = lst
        G.add_edge(edge[0], edge[1], **kwargs)
    return G


# In[ ]:


max_lag = 10


# In[ ]:


G_unweighted = create_cg_nx(cg, max_lag)
nx.write_gpickle(G_unweighted, './causal_graph_data/cg_unweighted_filtered.gpickle')


# In[ ]:


G_mutual_info = create_cg_nx_mutual_info(cg, max_lag)
nx.write_gpickle(G_mutual_info, './causal_graph_data/cg_weighted_mutual_info_filtered.gpickle')


# In[ ]:


G_ftest = create_cg_nx_ftest(cg, max_lag)
nx.write_gpickle(G_ftest , './causal_graph_data/cg_weighted_ftest_filtered.gpickle')


# In[ ]:


G_chi2test = create_cg_nx_chi2test(cg, max_lag)
nx.write_gpickle(G_chi2test, './causal_graph_data/cg_weighted_chi2test_filtered.gpickle')

