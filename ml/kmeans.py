#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import warnings as wr
wr.filterwarnings('ignore')

#importing the dataset
df=pd.read_csv('iris.csv')
print(df.head())

#extracting the features
x=df.iloc[:,:-1].values

sse=[]
for k in  range(1,11):
    km=KMeans(n_clusters=k,random_state=2)
    km.fit(x)
    sse.append(km.inertia_)

sns.set_style('whitegrid')
g=sns.lineplot(x=range(1,11),y=sse)

g.set(xlabel="Number of cluster (k)",
      ylabel="Sum Squared Error",
      title='Elbow Method')
plt.show()

kmeans=KMeans(n_clusters=3,random_state=2)
kmeans.fit(x)

print(kmeans.cluster_centers_)

pred=kmeans.fit_predict(x)
print(pred)

#plotting the cluster centers with datapoints
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(x[:,0],x[:,1],c=pred,cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
    center=center[:2]
    plt.scatter(center[0],center[1],marker='^',c='red')
plt.xlabel('Petal length (cm)')
plt.ylabel('petal width (cm)')

plt.subplot(1,2,2)
plt.scatter(x[:,2],x[:,3],c=pred,cmap=cm.Accent)
plt.grid(True)
for center in kmeans.cluster_centers_:
    center=center[2:4]
    plt.scatter(center[0],center[1],marker='^',c='red')
plt.xlabel('Sepal length (cm)')
plt.ylabel('sepal width (cm)')
plt.show()
