#importing the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

#importing the dataset
df=pd.read_csv('iris.csv')
print(df.head())

#Exploratory Data Analysis
print(df.isnull().sum(),'\n')
print(df.info(),'\n')
print(df.describe(),'\n')
print('the dataset shape is',df.shape,'\n')

#Data Visualization 
#Count plot
count_iris=df['species'].value_counts()
plt.figure(figsize=(10,8))
plt.bar(count_iris.index, count_iris,color='blue')
plt.xlabel('species')
plt.ylabel('Count')
plt.title('Count plot of the target')
plt.show()

#KDE plot
sns.set_style('darkgrid')
numerical_columns=df.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(14,len(numerical_columns)*3))
for idx, feature in enumerate(numerical_columns,1):
    plt.subplot(len(numerical_columns),2,idx)
    sns.histplot(df[feature],kde=True)
    plt.title(f"{feature}| skewness:{round(df[feature].skew(),2)}")

plt.tight_layout()
plt.show()

#Bivariate analysis
#pair plots
sns.set_palette('pastel')
sns.pairplot(df)
plt.suptitle('Pair Plot Of dataframe')
plt.show()

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report
#Support Vector Machine
svm=SVC(C=1.0,gamma=0.5,kernel='linear')
svm.fit(x_train,y_train)
y_pred_svm=svm.predict(x_test)
cm_svm=confusion_matrix(y_test,y_pred_svm)
rep_svm=classification_report(y_test,y_pred_svm)

x1=df.iloc[:,:2].values
y1=df.iloc[:,-1].values
svm1=SVC(C=1.0,gamma=0.5,kernel='linear')
svm1.fit(x1,y1)
plt.xlabel('Sepal width')
plt.ylabel('Sepal length')

# Plot Decision Boundary
from sklearn.inspection import DecisionBoundaryDisplay

DecisionBoundaryDisplay.from_estimator(
		svm1,
		x1,
		response_method="predict",
		cmap=plt.cm.Spectral,
		alpha=0.8,
		xlabel=df.columns[0],
		ylabel=df.columns[1],
        )
plt.scatter(x1[:,0],x1[:,1],edgecolors='k')
plt.show()

#Decision Tree Classifier
dt=DecisionTreeClassifier(criterion='gini',random_state=200)
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(x_test)
cm_dt=confusion_matrix(y_test,y_pred_dt)
rep_dt=classification_report(y_test,y_pred_dt)

from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(dt,feature_names=df.columns,rounded=True,max_depth=5)
plt.show()
#KMeans Clustering
sse=[]
for i in range(1,11):
    km=KMeans(n_clusters=i,random_state=2)
    km.fit(x)
    sse.append(km.inertia_)
sns.set_style('whitegrid')
g=sns.lineplot(x=range(1,11),y=sse)
g.set(xlabel='Number of clusters k',ylabel='Sum Squared Error',title='Elbow Method')
plt.show()

kmeans=KMeans(n_clusters=3,random_state=2)
km_predict=kmeans.fit_predict(x)

#visualizing the clusters
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.scatter(x[:,0],x[:,1],c=km_predict,cmap=cm.Accent)
plt.grid(True)
for centers in kmeans.cluster_centers_:
    centers=centers[:2]
    plt.scatter(centers[0],centers[1],marker='^',c='red')
plt.xlabel('Sepal width (cm)')
plt.ylabel('Sepal length(cm)')

plt.subplot(1,2,2)
plt.scatter(x[:,2],x[:,3],c=km_predict,cmap=cm.Accent)
plt.grid(True)
for centers in kmeans.cluster_centers_:
    centers=centers[2:4]
    plt.scatter(centers[0],centers[1],marker='^',c='red')
plt.xlabel('Petal width (cm)')
plt.ylabel('Petal length(cm)')


