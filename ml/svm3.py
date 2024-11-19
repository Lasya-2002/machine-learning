import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

df=pd.read_csv('iris.csv')
print(df.head())

print(df.info(),'\n')
print(df.shape,'\n')
print(df.dtypes=='object','\n')
print(df.describe(),'\n')
print(df.isnull().sum())

count_class=df['species'].value_counts()
plt.figure(figsize=(10,8))
plt.bar(count_class.index,count_class,color='red')
plt.xlabel('Classes')
plt.ylabel('Count')
plt.title('Count Plot')
plt.show()

sns.set_style('darkgrid')
numerical_columns=df.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(14,len(numerical_columns)*3))
for idx, feature in enumerate(numerical_columns,1):
    plt.subplot(len(numerical_columns),2,idx)
    sns.histplot(df[feature],kde=True)
    plt.title(f"{feature}| skewness:{round(df[feature].skew(),2)}")
plt.tight_layout()
plt.show()

sns.pairplot(data=df,hue='species')
plt.show()

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)

from sklearn.svm import SVC
svm=SVC(C=1.0,kernel='linear',gamma=0.5)
svm.fit(x_train,y_train)

y_pred=svm.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm)
rep=classification_report(y_test,y_pred)
print(rep)

x1=df.iloc[:,:2].values
y1=df.iloc[:,-1].values

from sklearn.svm import SVC
svm1=SVC(kernel='linear',C=1.0,gamma=0.5)
svm1.fit(x1,y1)

# Plot Decision Boundary
# Scatter plot

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
plt.scatter(x[:,0],x[:,1],edgecolors='k')
plt.show()





