import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
wine=load_wine()
wine.feature_names
df=pd.DataFrame(wine.data)
df.columns=wine.feature_names
print(df.head())
df.insert(13,'outcome',wine.target,True)
print(df.head())
x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,y_train,x_test,y_test=train_test_split(x,y,test_size=0.3,random_state=None)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.fit_transform([x_test])
from sklearn.decomposition import PCA
pca=PCA(n_components=5)
x_train_pca=pca.fit_transform(x_train)
x_test_pca=pca.fit_transform(x_test)
