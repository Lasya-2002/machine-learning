import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings as wr
wr.filterwarnings('ignore')

df=pd.read_csv('Salary_dataset.csv')
print(df.head())

print(df.isnull().sum())
print(df.info())
print(df.dtypes)
print(df.describe())

df=df.drop(df.columns[[0]],axis=1)

plt.figure(figsize=(10,8))
sns.scatterplot(x='YearsExperience',y='Salary',data=df)
plt.show()

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

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
print('R squared value is :',r2_score(y_test,y_pred))
print('MSE is :',mean_squared_error(y_test,y_pred))
print('MAE is :',mean_absolute_error(y_test,y_pred))

print('Intercept of regression line is :',lr.intercept_)
print('Coefficient is the regression line is :',lr.coef_)

x1=[[1.65]]
print(lr.predict(x1))

y1=lr.coef_*1.65+lr.intercept_
print(y1)

plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,lr.predict(x_train),color='blue')
plt.title('Salary vs Experience test set results')
plt.xlabel('years of experience')
plt.ylabel('Salary')
plt.show()