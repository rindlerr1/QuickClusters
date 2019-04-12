#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 07:36:13 2019

@author: Home
"""

import numpy as np
from numpy.random import randint
import pandas as pd
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.cluster import silhouette_score
from fastdtw import fastdtw


camp_1 = []
camp_2 = []
camp_3 = []
camp_4 = []
camp_5 = []
camp_6 = []
camp_7 = []
camp_8 = []



for i in range(0, 500):
    camp_1.append(randint(0,100)/100)
    camp_2.append(randint(70,90)/100)
    camp_3.append(randint(0,100)/100)
    camp_4.append(randint(0,100)/100)
    camp_5.append(randint(0,100)/100)
    camp_6.append(randint(0,100)/100)
    camp_7.append(randint(0,100)/100)
    camp_8.append(randint(10,50)/100)

    

df = pd.DataFrame({'Camp_1':camp_1,
                   'Camp_2':camp_2,
                   'Camp_3':camp_3,
                   'Camp_4':camp_4,
                   'Camp_5':camp_5,
                   'Camp_6':camp_6,
                   'Camp_7':camp_7,
                   'Camp_8': camp_8})
    
df['Camp_3'] = df['Camp_2']*.9



df['Camp_9'] = df['Camp_8']*1.1
df['Camp_10'] = df['Camp_8']*1.15
    

titles = [x for x in df.columns]


table = df



def scale_distance(table):
    titles = [x for x in table.columns]
    
    values = []
    for i in range(0, len(table.columns)):
        for q in range(0, len(table)):
            values.append(table[table.columns[i]][q])
    values = np.array(values).reshape(-1,1)

    scaler = StandardScaler()
    scaler.fit(values)
    scaled_values = scaler.transform(values)


    data = [list() for x in titles]
    count = 0
    for i in range(0, len(table.columns)):
        for q in range(0, len(table)):
            data[i].append(scaled_values[count][0])
            count += 1

    scaled_table = pd.DataFrame(data=data).transpose()
    scaled_table.columns = titles

    return scaled_table


table_1= scale_distance(table)






def distance_matrix(table):
    campaigns = [x for x in table.columns]
    
    table = table[campaigns]
    table = table.reset_index()
    
    #container for all distance values
    distance_list = []
    for x in campaigns:
        distance_list.append(list())
    
    #calaculate all distances and load into container
    for i in range(0, len(campaigns)):
        for q in range(0, len(campaigns)):
            if campaigns[i] != campaigns[q]:
                camp_x = table[['index', campaigns[i]]].values
                camp_y = table[['index', campaigns[q]]].values
                distance, path = fastdtw(camp_x, camp_y, dist=euclidean)
                distance_list[i].append(distance)
            elif campaigns[i] == campaigns[q]:
                distance_list[i].append(0.0)
                
    #create output matrix of distance            
    df_list = []
    for i in range(0, len(campaigns)):
        mini_df = pd.DataFrame({campaigns[i]:distance_list[i]})
        df_list.append(mini_df)
    
    output = pd.concat(df_list, axis=1)
    
    return output

table_2 = distance_matrix(table_1)




def db_gridsearch(table):
    #eps_list = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    max_val = 0
    for i in range(len(table)):
        for q in range(0, len(table)):
            if table[table.columns[i]][q] > max_val:
               max_val = table[table.columns[i]][q]
               
    eps_list = []
    for i in range(1, 11):
        eps_list.append(int(i*(max_val/10)))
        
        
        
    min_samples_list = [x for x in range(1, len(table.columns)+1)]
    eps_param= []
    samples_param = []
    scores = []
    X = table.values  
    for i in range(0, len(eps_list)):
        for q in range(0, len(min_samples_list)):
            dbscan = DBSCAN(metric='precomputed', eps=eps_list[i], min_samples=min_samples_list[q])
            clusters = dbscan.fit_predict(X)
            eps_param.append(eps_list[i])
            samples_param.append(min_samples_list[q])
            try:
                score = silhouette_score(X, clusters)                
                scores.append(score)
            except ValueError:
                scores.append(np.nan)
                
    
    performance_table = pd.DataFrame({'EPS':eps_param,
                                      'Samples':samples_param,
                                      'Scores':scores})     
                
    top_perf = performance_table.sort_values('Scores',ascending=False).reset_index(drop=True)
    
    
    
    
    print(top_perf.head(1))
    
    return performance_table

table_3 = db_gridsearch(table_2)

def db_cluster(table, eps, samples):
    X = table.values
    dbscan = DBSCAN(metric='precomputed', eps=eps, min_samples=samples)
    clusters = dbscan.fit_predict(X)
    score = silhouette_score(X, clusters)
    print(clusters)
    print(score)
    return clusters, score
    
c , s = db_cluster(table_2, 380, 2)
