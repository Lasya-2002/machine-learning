#importing the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

#importing the dataset
df=pd.read_csv('daily_weather.csv')
print(df.head(),'\n')
print(df.shape,'\n')
print(df.isnull().sum())

#Data preprocessing
print(df.columns)
del df['number']
print(df.head(2))

clean_df=df.copy()
clean_df=clean_df.dropna()
clean_df.info()
clean_df.isnull().sum()
clean_df['high_humidity_label']=(clean_df['relative_humidity_3pm']>24.99)*1
print(clean_df['high_humidity_label'])
print(clean_df.head())

sns.pairplot(data=clean_df,hue='high_humidity_label')

plt.figure(figsize=(10,8))
sns.heatmap(clean_df.corr(),cmap='Pastel2',annot=True,fmt='.2f')
plt.title('correlation heatmap')


del clean_df['relative_humidity_3pm']
clean_df.describe()

x=clean_df.iloc[:,:-1].values
y=clean_df.iloc[:,-1].values

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_scaled=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x_scaled,y,test_size=0.30,random_state=100)

print(x_train.shape,'\n',x_test.shape,'\n',y_train.shape,'\n',y_test.shape)

from sklearn.tree import DecisionTreeClassifier
humidity_classifier=DecisionTreeClassifier(criterion='gini',max_depth=10)
humidity_classifier.fit(x_train,y_train)

print(type(humidity_classifier))

y_pred=humidity_classifier.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
print('The confusion matrix representation is: ',cm)

acc=accuracy_score(y_test,y_pred)*100
print('Accuracy is :',acc)

rep=classification_report(y_test,y_pred)
print('The classifictaion report is:',rep)

#tree plot
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(humidity_classifier,rounded=True,max_depth=5)
plt.show()

