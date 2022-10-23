# -*- coding: utf-8 -*-
"""
Created on Sun Oct 23 22:27:31 2022

@author: ANANDAN RAJU
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
mall=pd.read_csv('Mall_Customers.csv')

mall.isnull().sum()
mall.columns[mall.isna().any()]
mall.rename(columns={'Annual Income (k$)':'Annual_Income','Spending Score (1-100)':'Spending_Score'},inplace=True)

age=mall['Age'].values
spg_s=mall['Spending_Score'].values
X=np.array(list(zip(age,spg_s)))

from sklearn.cluster import KMeans

wcs=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(X)
    wcs.append(km.inertia_)
    
plt.plot(range(1,11),wcs,color='red',marker='8')
plt.title('Optimised K Value')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

model=KMeans(n_clusters=4,random_state=0)
y_means=model.fit_predict(X)

a=model.cluster_centers_[:,0]
b=model.cluster_centers_[:,1]

plt.scatter(X[y_means==0,0],X[y_means==0,1],c='yellow',label='C1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],c='blue',label='C2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],c='orange',label='C3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],c='green',label='C4')
fig=plt.scatter(a,b,s=100,marker='s',label='centroids',c='red')
plt.title('Age Vs Spending Score Analysis')
plt.xlabel('Age')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


al_inc=mall['Annual_Income'].values
spg_s=mall['Spending_Score'].values
X=np.array(list(zip(al_inc,spg_s)))

wcs=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=0)
    km.fit(X)
    wcs.append(km.inertia_)
    
plt.plot(range(1,11),wcs,color='blue',marker='+')
plt.title('Optimised K Value')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

model=KMeans(n_clusters=4,random_state=0)
y_means=model.fit_predict(X)

c=model.cluster_centers_[:,0]
d=model.cluster_centers_[:,1]

plt.scatter(X[y_means==0,0],X[y_means==0,1],c='yellow',label='C1')
plt.scatter(X[y_means==1,0],X[y_means==1,1],c='blue',label='C2')
plt.scatter(X[y_means==2,0],X[y_means==2,1],c='orange',label='C3')
plt.scatter(X[y_means==3,0],X[y_means==3,1],c='green',label='C4')
fig=plt.scatter(c,d,s=100,marker='s',label='centroids',c='red')
plt.title('Annual Income Spend Analysis')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()
