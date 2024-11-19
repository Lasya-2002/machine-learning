#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

#importing the dataset
df=pd.read_csv('C:/Users/mvsla/OneDrive/Desktop/ml/titanic_train.csv')
print(df.head())

#Exploratory data analysis
print(df.info(),'\n')
print(df.shape,'\n')
print(df.dtypes=='object','\n')
print(df.describe(),'\n')
print(df.isnull().sum())

print(df['Embarked'].value_counts())
embarked_mode=df['Embarked'].mode()
print(embarked_mode)

df['Embarked'].replace(str(embarked_mode),inplace=True)
print(df.isnull().sum())
df['Age'].fillna(df['Age'].mean(),inplace=True)

df['Cabin'].value_counts()
df['Cabin'].fillna(df['Cabin'].mode,inplace=True)

#seaborn plots 
sns.set_style('darkgrid')
numerical_columns=df.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(14,len(numerical_columns)*3))
for idx, feature in enumerate(numerical_columns,1):
    plt.subplot(len(numerical_columns),2,idx)
    sns.histplot(df[feature],kde=True)
    plt.title(f"{feature}| skewness:{round(df[feature].skew(),2)}")

plt.tight_layout()
plt.show()

sns.pairplot(df,hue='Survived')
plt.show()

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['Embarked'] = le.fit_transform(df['Embarked'])
df['Sex'] = le.fit_transform(df['Sex'])
df = df[df['Sex'] != 0]
df=df[df['Embarked']!=0]
df = df.reset_index(drop=True)

x=df.drop(['PassengerId','Ticket','SibSp','Parch','Name','Cabin','Survived'],axis=1)
y=df['Survived']
print(x.head())

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_scaled=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.30,random_state=20)

#fitting an svm
from sklearn.svm import SVC 
svm = SVC(kernel="linear", gamma=0.5, C=1.0)
svm.fit(x_train,y_train)

y_pred=svm.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_pred,y_test)
print(cm)

report=classification_report(y_test,y_pred)
print(report)

#confusion matrix visualization
plt.figure(figsize=(10,8))
ax=sns.heatmap(cm,annot=True,cmap='Blues')
ax.set_title('Confusion Matrix with labels \n\n')
ax.set_xlabel('\nPredicted labels')
ax.set_ylabel('\nActual values')
plt.show()
