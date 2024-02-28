#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#import the dataset
df=pd.read_csv('iris.csv')
print(df.info())
X=df.iloc[:,:-1].values

#scaling the features
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X=sc.fit_transform(X)

#principal component analysis
from sklearn.decomposition import PCA
pca=PCA(n_components=2)
X_pca=pca.fit_transform(X)
print(sum(pca.explained_variance_ratio_))

#explained variance ratio
nums=np.arange(4)
var_ratio=[]
for num in nums:
    pca=PCA(n_components=num)
    pca.fit(X)
    var_ratio.append(np.sum(pca.explained_variance_ratio_))

#display the graph
plt.figure(figsize=(4,2),dpi=150)
plt.grid()
plt.plot(nums,var_ratio,marker='o')
plt.xlabel('n_components')
plt.ylabel('Explained Variance Ratio')
plt.title('n_components vs explained variance ratio')
plt.show()

#scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.grid()
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

