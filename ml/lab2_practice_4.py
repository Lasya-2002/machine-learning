import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

df=pd.read_csv('USA_Housing.csv')
print(df.head())
print(df.isnull().sum())
print(df.dtypes)
print(df.info())
print(df.describe())

df=df.drop('Address',axis=1)

sns.set_style('darkgrid')
numerical_columns=df.select_dtypes(include=['int64','float64']).columns
plt.figure(figsize=(14,len(numerical_columns)*3))
for idx,feature in enumerate(numerical_columns,1):
    plt.subplot(len(numerical_columns),2,idx)
    sns.histplot(df[feature],kde=True)
    plt.title(f"{feature}| skewness:{round(df[feature].skew(),2)}")
plt.tight_layout()
plt.show()

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

sns.set_style('darkgrid')
sns.set_palette('pastel')
sns.pairplot(data=df,hue='Price')
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(data=df.corr(),annot=True,fmt='.2f',cmap='Pastel2')
plt.show()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)

from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)

from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
print('R Squared value:',r2_score(y_test,y_pred))
print('Mean Squared Error:',mean_squared_error(y_test,y_pred))
print('Mean absolute error: ',mean_absolute_error(y_test,y_pred))