import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings as wr
wr.filterwarnings('ignore')

#importing the dataset
from sklearn import datasets
def sklearn_to_frame(sklearn_dataset):
    df=pd.DataFrame(data=sklearn_dataset.data,columns=sklearn_dataset.feature_names)
    df['target']=pd.Series(data=sklearn_dataset.target)
    return df
df1=sklearn_to_frame(datasets.load_wine())

print(df1.head())
print(df1.isnull().sum())
x=df1.iloc[:,:-1].values
y=df1.iloc[:,-1].values

sns.pairplot(data=df1)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_scaled=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)
print(x_train.shape,y_train.shape)

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=10)
knn.fit(x_train,y_train)

y_pred=knn.predict(x_test)

#confusion matrix and classification report
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_pred,y_test)
print(cm)

print(classification_report(y_pred,y_test))

#confusion matrix visualization
plt.figure(figsize=(10,8))
ax=sns.heatmap(cm,annot=True,cmap='Blues')
ax.set_title('confusion matrix')
ax.set_xlabel('predicted values')
ax.set_ylabel('actual values')
plt.show()










