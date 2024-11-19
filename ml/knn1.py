#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

#importing the dataset
df=pd.read_csv('diabetes.csv')
df.head()

#dataset information
print(df.dtypes=='object')
df.info()

#dataset dimensions
df.shape

#statistics of the numeric attributes in the dataset
df.describe()

#checking the missing values
df.isnull().sum()

#extracting the target feature
x=df.drop('Outcome',axis=1)
y=df['Outcome']

#feature scaling by standardization
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
ds=sc.fit_transform(x)
data=pd.DataFrame(ds,columns=x.columns)
print(data.head())

#pairplot for interaction among the features
sns.pairplot(df,hue='Outcome')

#splitting the dataset into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(data,y,test_size=0.30,
                                               random_state=10)

#dimesions of features and the target feature
print(x.shape,y.shape)

#dimensions of the training and testing data
print('x_train:  ',x_train.shape)
print('y_train:  ',y_train.shape)
print('x_test:  ',x_test.shape)
print('y_test:  ',y_test.shape)

#importing K nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=9)

#training the model
knn.fit(x_train,y_train)

#performance evaluation
pred=knn.predict(x_test)
print(pred)

#accuracy
print(knn.score(x_test,y_test)*100)

#PCA for increasing the accuracy of the model
from sklearn.decomposition import PCA
pca=PCA(0.95)
x_pca=pca.fit_transform(x_train)
print(x_train)
print(pca.n_components_)

x_train_pca,x_test_pca,y_train,y_test=train_test_split(data,y,
                                            test_size=0.30,random_state=30)

from sklearn.neighbors import KNeighborsClassifier
knn_pca=KNeighborsClassifier(n_neighbors=9)
knn_pca.fit(x_train_pca,y_train)
knn_pca.score(x_test_pca,y_test)
pred_pca=knn_pca.predict(x_test_pca)
print(pred_pca)

#classification evaluation metrics
from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,pred_pca)
print(cm)

#confusion matrix visualization
plt.figure(figsize=(10,8))
ax=sns.heatmap(cm,annot=True,cmap='Blues')
ax.set_title('Confusion Matrix with labels \n\n')
ax.set_xlabel('\nPredicted labels')
ax.set_ylabel('\nActual values')
plt.show()

#classification report of KNN
print(classification_report(y_test,pred))

#Classification report after pca
print(classification_report(y_test,pred_pca))

#elbow method for optimal k value
#cross validation
from sklearn.model_selection import cross_val_score

#accuracy rate calculation
accuracy_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x,y,cv=10)
    accuracy_rate.append(score.mean())
    
#error rate calculation for plotting the graph
error_rate=[]
for i in range(1,40):
    knn=KNeighborsClassifier(n_neighbors=i)
    score=cross_val_score(knn,x,y,cv=10)
    error_rate.append(1-score.mean())
    
#plottimg the relation between the K value and error rate
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue',marker='o',
         linestyle='dashed',markerfacecolor='red',markersize=10)
plt.title('error-rate vs k-value')
plt.xlabel('k value')
plt.ylabel('error-rate')
plt.show()

#Checking for the accuracy at k=1
knn=KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train,y_train)
y_pred1=knn.predict(x_test)
print('At K=1\n')
print(confusion_matrix(y_pred1,y_test),'\n')
print(classification_report(y_pred1,y_test))

#Checking the accuracy of the model at the most optimal k 
#value from the elbow method i.e. K=18
knn=KNeighborsClassifier(n_neighbors=18)
knn.fit(x_train,y_train)
y_pred1=knn.predict(x_test)
print('At K=18\n')
print(confusion_matrix(y_pred1,y_test),'\n')
print(classification_report(y_pred1,y_test))

knn_pca=KNeighborsClassifier(n_neighbors=18)
knn_pca.fit(x_train_pca,y_train)
knn_pca.score(x_test_pca,y_test)
pred_pca=knn_pca.predict(x_test_pca)
print(pred_pca)
print(classification_report(y_test,pred_pca))

