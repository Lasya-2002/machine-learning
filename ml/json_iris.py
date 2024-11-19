import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

df=pd.read_json('iris.json')
print(df.head(5))

print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df.describe())

#count plot
count_plots=df['species'].value_counts()
plt.figure(figsize=(10,8))
plt.bar(count_plots.index,count_plots,color='blue')
plt.title('count plot for class balance')
plt.xlabel('count of classes')
plt.ylabel('classes')
plt.show()

#bivariate analysis
#creating the kernel density plots
sns.set_style('darkgrid')
numerical_columns=df.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(14,len(numerical_columns)*3))
for idx, feature in enumerate(numerical_columns,1):
    plt.subplot(len(numerical_columns),2,idx)
    sns.histplot(df[feature],kde=True)
    plt.title(f"{feature}| skewness:{round(df[feature].skew(),2)}")

plt.tight_layout()
plt.show()

sns.pairplot(data=df,palette='pastel',hue='species')
plt.show()

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)

y_pred=lr.predict(x_test)

from sklearn.metrics import confusion_matrix,classification_report
cm=confusion_matrix(y_test,y_pred)
print(cm)

rep=classification_report(y_test,y_pred)
print(rep)
