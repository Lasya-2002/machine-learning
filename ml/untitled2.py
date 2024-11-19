import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
def sklearn_to_frame(sklearn_dataset):
    df=pd.DataFrame(data=sklearn_dataset.data,columns=sklearn_dataset.feature_names)
    df['target']=pd.Series(data=sklearn_dataset.target)
    return df

df_wine=sklearn_to_frame(datasets.load_wine())
print(df_wine.head())
x=df_wine.iloc[:,:-1].values
y=df_wine.iloc[:,-1].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=None)
print(x_train.shape,'training x size')
print(x_test.shape,'testing x size')
print(y_train.shape,'training y size')
print(y_test.shape,'testing y size')
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train_scaling=sc.fit_transform(x_train)
x_test_scaling=sc.transform(x_test)

from sklearn.decomposition import PCA
pca=PCA(n_components=5)
x_train_pca=pca.fit_transform(x_train_scaling)

x_test_pca=pca.transform(x_test_scaling)
print(pca.components_)
print(sum(pca.explained_variance_ratio_))

#explained variance ratio visualization
nums=np.arange(14)
var_ratio=[]
for num in nums:
    pca=PCA(n_components=num)
    x_pca=pca.fit(x_train_scaling)
    var_ratio.append(np.sum(pca.explained_variance_ratio_))


#visulaizing EVR graph
plt.figure(figsize=(4,2),dpi=150)
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.title('Explained Variance Ratio')
plt.xlabel('n_components')
plt.ylabel('Explained Variance')
plt.show()

#scree plot
pc_values=np.arange(pca.n_components_)+1
plt.grid()
plt.plot(pc_values,pca.explained_variance_ratio_,'-o',linewidth=2,color='blue')
plt.xlabel('principal components')
plt.ylabel('varaince explained')
plt.title('Scree plot')
plt.show()


